import sys
import math
import re
import time
import json
import urllib.request
import urllib.error
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from ..common.logger import setup_logger
from ..common.paths import data_path
from ..common import kv_store

log = setup_logger('basketball')

BASKETBALL_VERSION = '2026-08-12-joint-line-movement-v2'
BASKETBALL_HISTORY_KEY = 'basketball_prediction_history'
BASKETBALL_HISTORY_LIMIT = 500

BASE_URL = 'https://trade.500.com'
SCHEDULE_URL = f'{BASE_URL}/jclq/'
MATCH_DETAIL_URL = f'{BASE_URL}/fenxi/lqdata-'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Referer': SCHEDULE_URL,
}

BET_TYPES = {
    'spf': {'name': '胜负', 'description': '预测比赛胜负结果'},
    'rqspf': {'name': '让分胜负', 'description': '主队让分后的胜负'},
    'dx': {'name': '大小分', 'description': '预测总得分是否超过预设分数'},
}

LEAGUE_PROFILES = {
    'NBA': {'avg_total': 220.0, 'home_win_rate': 0.57},
    'CBA': {'avg_total': 190.0, 'home_win_rate': 0.55},
    'NCAAB': {'avg_total': 150.0, 'home_win_rate': 0.58},
    '欧洲篮球': {'avg_total': 165.0, 'home_win_rate': 0.54},
    'WNBA': {'avg_total': 160.0, 'home_win_rate': 0.56},
    '美职女篮': {'avg_total': 170.0, 'home_win_rate': 0.55},
}

def fetch(url, encoding='utf-8', referer=None):
    headers = {**HEADERS, 'Referer': referer} if referer else HEADERS
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        log.warning(f"HTTP Error {e.code} for {url}")
        return None
    except urllib.error.URLError as e:
        log.warning(f"URL Error {e} for {url}")
        return None

    for enc in [encoding, 'gbk', 'gb2312', 'utf-8']:
        try:
            result = raw.decode(enc, errors='replace')
            result = result.encode('utf-8', errors='replace').decode('utf-8')
            return result
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode('utf-8', errors='replace')

def odds_to_prob(odds):
    if odds is None or odds <= 0:
        return 0.0
    return 1.0 / odds

def calc_implied_prob(home_odds, away_odds):
    p_home = odds_to_prob(home_odds)
    p_away = odds_to_prob(away_odds)
    total = p_home + p_away + 1e-9
    return p_home / total, p_away / total


def _confidence_from_probs(p1, p2):
    gap = abs(float(p1) - float(p2))
    return 'high' if gap > 0.15 else ('medium' if gap > 0.08 else 'low')


def _official_pick_status(bet_type, pick_prob, confidence):
    """Separate a model opinion from an accuracy-tracked official pick."""
    thresholds = {'spf': 0.56, 'rqspf': 0.55, 'dx': 0.54}
    if confidence == 'low':
        return {'playable': False, 'official': False, 'skip_reason': 'low_confidence'}
    if float(pick_prob or 0) < thresholds.get(bet_type, 0.56):
        return {'playable': False, 'official': False, 'skip_reason': 'probability_below_threshold'}
    return {'playable': True, 'official': True, 'skip_reason': None}


def _movement_accuracy_gate(bet_type, status, movement, confirmation):
    """Accuracy-first admission for spread and totals official picks."""
    if bet_type not in ('rqspf', 'dx'):
        return status
    if not movement or not movement.get('available'):
        return {**status, 'playable': False, 'official': False,
                'skip_reason': 'movement_unavailable'}
    if movement.get('stale'):
        return {**status, 'playable': False, 'official': False,
                'skip_reason': 'movement_stale'}
    if int(movement.get('samples', 0) or 0) < 2:
        return {**status, 'playable': False, 'official': False,
                'skip_reason': 'movement_samples_insufficient'}
    if not confirmation.get('confirmed'):
        return {**status, 'playable': False, 'official': False,
                'skip_reason': 'movement_conflicts_with_model'}
    return status


def _elo_predictions(match):
    """Return ELO estimates with a conservative trust score for known teams."""
    try:
        from .elo import get_elo_system
        elo = get_elo_system()
        home, away = match.get('home', ''), match.get('away', '')
        league = match.get('league', '')
        win = elo.predict_win_prob(home, away, league)
        margin = elo.predict_margin(home, away, league)
        total = elo.predict_total_score(home, away, league)
        games = min(int(win.get('home_games', 0)), int(win.get('away_games', 0)))
        # Do not let cold-start ratings dilute the much more informative market.
        trust = min(1.0, games / 20.0)
        return win, margin, total, trust
    except Exception as exc:
        log.warning(f"篮球 ELO 预测不可用: {exc}")
        return {}, {}, {}, 0.0


def _calibrate_pick(bet_type, p1, p2, league, confidence):
    """Calibrate the selected outcome while preserving a normalized two-way market."""
    pick_first = p1 >= p2
    raw_pick = p1 if pick_first else p2
    try:
        from .calibration import get_calibrator
        calibrated = get_calibrator().calibrate(bet_type, raw_pick, league, confidence)
    except Exception as exc:
        log.warning(f"篮球历史校准不可用: {exc}")
        calibrated = raw_pick
    # A weak calibrator may reduce conviction, but must not silently reverse the
    # selected side; reversal requires fresh market/model evidence.
    calibrated = max(0.5001, min(0.95, float(calibrated)))
    return ((calibrated, 1.0 - calibrated) if pick_first
            else (1.0 - calibrated, calibrated)), round(raw_pick, 4)

def analyze_spf(match, movement=None):
    home_odds = match.get('spf_home')
    away_odds = match.get('spf_away')
    
    if home_odds is None or away_odds is None:
        return {
            'available': False,
            'reason': 'missing_odds',
            'home_prob': 0.5,
            'away_prob': 0.5,
            'recommendation': None,
            'confidence': 'low',
        }
    
    p_home, p_away = calc_implied_prob(home_odds, away_odds)
    
    league = match.get('league', '')
    profile = LEAGUE_PROFILES.get(league, {'home_win_rate': 0.55})
    home_bias = profile['home_win_rate']
    
    # The closing market is the strongest baseline. League prior is deliberately
    # small and ELO only participates after both teams have enough history.
    p_home = p_home * 0.9 + home_bias * 0.1
    market_home_prob = p_home
    elo_win, _, _, elo_trust = _elo_predictions(match)
    elo_weight = 0.30 * elo_trust
    if elo_win:
        p_home = p_home * (1.0 - elo_weight) + elo_win['home_prob'] * elo_weight
    p_away = 1.0 - p_home
    # 赔率实时走势微调：聪明钱追捧的一侧小幅提升概率（仅增强，不反转）
    if movement and movement.get('available') and movement.get('side') != 'flat':
        from .odds_movement import apply_movement
        p_home, p_away = apply_movement(p_home, p_away, movement)
    confidence = _confidence_from_probs(p_home, p_away)
    (p_home, p_away), raw_pick_prob = _calibrate_pick(
        'spf', p_home, p_away, league, confidence)
    recommendation = '主胜' if p_home > p_away else '客胜'
    pick_prob = max(p_home, p_away)
    status = _official_pick_status('spf', pick_prob, confidence)
    # 聪明钱是否确认本方推荐
    line_movement = None
    sharp_confirmed = False
    sc = {'confirmed': False, 'reason': 'movement_unavailable', 'boost': 0.0}
    if movement and movement.get('available'):
        from .odds_movement import sharp_confirmation
        sc = sharp_confirmation(movement, recommendation)
        line_movement = {**movement, 'recommendation': recommendation,
                         'confirmed': sc['confirmed'], 'reason': sc['reason']}
        if sc['confirmed'] and sc['boost'] > 0:
            sharp_confirmed = True
            if confidence == 'low':
                confidence = 'medium'
            elif confidence == 'medium' and sc['boost'] >= 1.0:
                confidence = 'high'
        status = _movement_accuracy_gate('rqspf', status, movement, sc)
    
    return {
        'available': True,
        'home_prob': round(p_home, 4),
        'away_prob': round(p_away, 4),
        'home_odds': home_odds,
        'away_odds': away_odds,
        'recommendation': recommendation,
        'confidence': confidence,
        'pick_prob': round(pick_prob, 4),
        'raw_pick_prob': raw_pick_prob,
        'market_home_prob': round(market_home_prob, 4),
        'elo_home_prob': elo_win.get('home_prob') if elo_win else None,
        'elo_trust': round(elo_trust, 3),
        'line_movement': line_movement,
        'sharp_confirmed': sharp_confirmed,
        **status,
    }

def analyze_rqspf(match, movement=None):
    handicap = match.get('handicap')
    rq_home_odds = match.get('rqspf_home')
    rq_away_odds = match.get('rqspf_away')
    
    if rq_home_odds is None or rq_away_odds is None:
        return {
            'available': False,
            'reason': 'missing_rqspf_odds',
            'handicap': handicap,
            'home_prob': 0.5,
            'away_prob': 0.5,
            'recommendation': None,
            'confidence': 'low',
        }
    
    p_home, p_away = calc_implied_prob(rq_home_odds, rq_away_odds)
    
    market_home_prob = p_home
    _, elo_margin, _, elo_trust = _elo_predictions(match)
    try:
        handicap_value = float(handicap)
    except (TypeError, ValueError):
        handicap_value = 0.0
    if elo_margin:
        adjusted_margin = float(elo_margin.get('expected_margin', 0)) + handicap_value
        elo_home_prob = 1.0 / (1.0 + math.exp(-adjusted_margin / 8.5))
        elo_weight = 0.25 * elo_trust
        p_home = p_home * (1.0 - elo_weight) + elo_home_prob * elo_weight
        p_away = 1.0 - p_home
    else:
        elo_home_prob = None
    # 赔率实时走势微调（让分侧：home=让胜, away=让负）
    if movement and movement.get('available') and movement.get('side') != 'flat':
        from .odds_movement import apply_movement
        p_home, p_away = apply_movement(p_home, p_away, movement)
    confidence = _confidence_from_probs(p_home, p_away)
    (p_home, p_away), raw_pick_prob = _calibrate_pick(
        'rqspf', p_home, p_away, match.get('league', ''), confidence)
    recommendation = '让胜' if p_home > p_away else '让负'
    pick_prob = max(p_home, p_away)
    status = _official_pick_status('rqspf', pick_prob, confidence)
    line_movement = None
    sharp_confirmed = False
    sc = {'confirmed': False, 'reason': 'movement_unavailable', 'boost': 0.0}
    if movement and movement.get('available'):
        from .odds_movement import sharp_confirmation
        sc = sharp_confirmation(movement, recommendation)
        line_movement = {**movement, 'recommendation': recommendation,
                         'confirmed': sc['confirmed'], 'reason': sc['reason']}
        if sc['confirmed'] and sc['boost'] > 0:
            sharp_confirmed = True
            if confidence == 'low':
                confidence = 'medium'
            elif confidence == 'medium' and sc['boost'] >= 1.0:
                confidence = 'high'
    status = _movement_accuracy_gate('rqspf', status, movement, sc)
    
    return {
        'available': True,
        'handicap': handicap,
        'home_prob': round(p_home, 4),
        'away_prob': round(p_away, 4),
        'home_odds': rq_home_odds,
        'away_odds': rq_away_odds,
        'recommendation': recommendation,
        'confidence': confidence,
        'pick_prob': round(pick_prob, 4),
        'raw_pick_prob': raw_pick_prob,
        'market_home_prob': round(market_home_prob, 4),
        'elo_margin': elo_margin.get('expected_margin') if elo_margin else None,
        'elo_home_prob': round(elo_home_prob, 4) if elo_home_prob is not None else None,
        'elo_trust': round(elo_trust, 3),
        'line_movement': line_movement,
        'sharp_confirmed': sharp_confirmed,
        **status,
    }

def analyze_daxiao(match, movement=None):
    total_line = match.get('total_line')
    over_odds = match.get('dx_over')
    under_odds = match.get('dx_under')
    
    if over_odds is None or under_odds is None:
        return {
            'available': False,
            'reason': 'missing_dx_odds',
            'total_line': total_line,
            'over_prob': 0.5,
            'under_prob': 0.5,
            'recommendation': None,
            'confidence': 'low',
        }
    
    p_over, p_under = calc_implied_prob(over_odds, under_odds)
    
    league = match.get('league', '')
    profile = LEAGUE_PROFILES.get(league, {'avg_total': 200.0})
    avg_total = profile['avg_total']
    
    if total_line is not None and avg_total:
        line_diff = total_line - avg_total
        if line_diff > 5:
            p_under += 0.05
            p_over -= 0.05
        elif line_diff < -5:
            p_over += 0.05
            p_under -= 0.05
    
    total = p_over + p_under + 1e-9
    p_over /= total
    p_under /= total
    
    market_over_prob = p_over
    _, _, elo_total, elo_trust = _elo_predictions(match)
    expected_total = elo_total.get('expected_total') if elo_total else None
    if expected_total is not None and total_line is not None:
        elo_over_prob = 1.0 / (1.0 + math.exp(-(float(expected_total) - float(total_line)) / 12.0))
        elo_weight = 0.20 * elo_trust
        p_over = p_over * (1.0 - elo_weight) + elo_over_prob * elo_weight
        p_under = 1.0 - p_over
    else:
        elo_over_prob = None
    # 赔率实时走势微调（大小分：over=主侧, under=客侧）
    if movement and movement.get('available') and movement.get('side') != 'flat':
        from .odds_movement import apply_movement
        p_over, p_under = apply_movement(p_over, p_under, movement)
    confidence = _confidence_from_probs(p_over, p_under)
    (p_over, p_under), raw_pick_prob = _calibrate_pick(
        'dx', p_over, p_under, league, confidence)
    recommendation = '大分' if p_over > p_under else '小分'
    pick_prob = max(p_over, p_under)
    status = _official_pick_status('dx', pick_prob, confidence)
    line_movement = None
    sharp_confirmed = False
    sc = {'confirmed': False, 'reason': 'movement_unavailable', 'boost': 0.0}
    if movement and movement.get('available'):
        from .odds_movement import sharp_confirmation
        sc = sharp_confirmation(movement, recommendation)
        line_movement = {**movement, 'recommendation': recommendation,
                         'confirmed': sc['confirmed'], 'reason': sc['reason']}
        if sc['confirmed'] and sc['boost'] > 0:
            sharp_confirmed = True
            if confidence == 'low':
                confidence = 'medium'
            elif confidence == 'medium' and sc['boost'] >= 1.0:
                confidence = 'high'
    status = _movement_accuracy_gate('dx', status, movement, sc)
    
    return {
        'available': True,
        'total_line': total_line,
        'over_prob': round(p_over, 4),
        'under_prob': round(p_under, 4),
        'over_odds': over_odds,
        'under_odds': under_odds,
        'recommendation': recommendation,
        'confidence': confidence,
        'pick_prob': round(pick_prob, 4),
        'raw_pick_prob': raw_pick_prob,
        'market_over_prob': round(market_over_prob, 4),
        'elo_total': expected_total,
        'elo_over_prob': round(elo_over_prob, 4) if elo_over_prob is not None else None,
        'elo_trust': round(elo_trust, 3),
        'line_movement': line_movement,
        'sharp_confirmed': sharp_confirmed,
        **status,
    }

def fetch_basketball_schedule(date=None):
    if date is None:
        date = time.strftime('%Y-%m-%d')
    
    url = f'{BASE_URL}/jclq/?playid=313&g=2&date={date}'
    log.info(f"抓取篮球赛程: {date}")
    
    try:
        content = fetch(url, encoding='gbk', referer=SCHEDULE_URL)
        if not content:
            log.warning("未获取到篮球赛程内容")
            return []
        
        matches = []
        
        tr_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL)
        td_pattern = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL)
        span_pattern = re.compile(r'<span[^>]*>([^<]*)</span>')
        
        all_trs = tr_pattern.findall(content)
        
        for tr in all_trs:
            tds = td_pattern.findall(tr)
            if len(tds) < 7:
                continue
            
            try:
                num_cell = tds[0]
                num = re.sub(r'<[^>]*>', '', num_cell).strip()
                
                if not num or not re.match(r'^[\u4e00-\u9fa5a-zA-Z]', num):
                    continue
                
                league_cell = tds[1]
                league = re.sub(r'<[^>]*>', '', league_cell).strip()
                
                time_cell = tds[2]
                match_time = re.sub(r'<[^>]*>', '', time_cell).strip()
                
                team_cell = tds[3]
                team_text = re.sub(r'<[^>]*>', '', team_cell).strip()
                
                vs_idx = team_text.find('VS')
                if vs_idx == -1:
                    vs_idx = team_text.find('vs')
                if vs_idx == -1:
                    vs_idx = team_text.find('对')
                
                if vs_idx == -1:
                    continue
                
                home_team = team_text[:vs_idx].strip()
                away_team = team_text[vs_idx+2:].strip()
                
                home_team = re.sub(r'^\[\w+\d*\]', '', home_team).strip()
                away_team = re.sub(r'\[\w+\d*\]$', '', away_team).strip()
                
                if not home_team or not away_team:
                    continue
                
                sf_cell = tds[4]
                sf_spans = span_pattern.findall(sf_cell)
                spf_home = float(sf_spans[0]) if len(sf_spans) >= 1 and sf_spans[0].replace('.', '').isdigit() else None
                spf_away = float(sf_spans[1]) if len(sf_spans) >= 2 and sf_spans[1].replace('.', '').isdigit() else None
                
                rfsf_cell = tds[5]
                rfsf_spans = span_pattern.findall(rfsf_cell)
                rqspf_home = float(rfsf_spans[0]) if len(rfsf_spans) >= 1 and rfsf_spans[0].replace('.', '').isdigit() else None
                handicap = rfsf_spans[1] if len(rfsf_spans) >= 2 else None
                rqspf_away = float(rfsf_spans[2]) if len(rfsf_spans) >= 3 and rfsf_spans[2].replace('.', '').isdigit() else None
                
                dxf_cell = tds[6]
                dxf_spans = span_pattern.findall(dxf_cell)
                dx_over = float(dxf_spans[0]) if len(dxf_spans) >= 1 and dxf_spans[0].replace('.', '').isdigit() else None
                total_line = float(dxf_spans[1]) if len(dxf_spans) >= 2 and dxf_spans[1].replace('.', '').isdigit() else None
                dx_under = float(dxf_spans[2]) if len(dxf_spans) >= 3 and dxf_spans[2].replace('.', '').isdigit() else None
                
                match_date = date
                if match_time and len(match_time) >= 5:
                    month_day = match_time[:5]
                    match_date = f"{date[:4]}-{month_day}"
                
                match = {
                    'id': f"{match_date}_{home_team}_{away_team}",
                    'date': match_date,
                    'time': match_time,
                    'num': num,
                    'league': league,
                    'home': home_team,
                    'away': away_team,
                    'handicap': handicap,
                    'rqspf_home': rqspf_home,
                    'rqspf_away': rqspf_away,
                    'spf_home': spf_home,
                    'spf_away': spf_away,
                    'total_line': total_line,
                    'dx_over': dx_over,
                    'dx_under': dx_under,
                    'status': 'not_started',
                }
                
                matches.append(match)
            except Exception as e:
                log.warning(f"解析篮球比赛失败: {e}")
                continue
        
        if not matches:
            next_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            log.info(f"今日无比赛，尝试获取明日赛程: {next_date}")
            url_next = f'{BASE_URL}/jclq/?playid=313&g=2&date={next_date}'
            content_next = fetch(url_next, encoding='gbk', referer=SCHEDULE_URL)
            if content_next:
                all_trs_next = tr_pattern.findall(content_next)
                for tr in all_trs_next:
                    tds = td_pattern.findall(tr)
                    if len(tds) < 7:
                        continue
                    try:
                        num_cell = tds[0]
                        num = re.sub(r'<[^>]*>', '', num_cell).strip()
                        if not num or not re.match(r'^[\u4e00-\u9fa5a-zA-Z]', num):
                            continue
                        league_cell = tds[1]
                        league = re.sub(r'<[^>]*>', '', league_cell).strip()
                        time_cell = tds[2]
                        match_time = re.sub(r'<[^>]*>', '', time_cell).strip()
                        team_cell = tds[3]
                        team_text = re.sub(r'<[^>]*>', '', team_cell).strip()
                        vs_idx = team_text.find('VS')
                        if vs_idx == -1:
                            vs_idx = team_text.find('vs')
                        if vs_idx == -1:
                            vs_idx = team_text.find('对')
                        if vs_idx == -1:
                            continue
                        home_team = team_text[:vs_idx].strip()
                        away_team = team_text[vs_idx+2:].strip()
                        home_team = re.sub(r'^\[\w+\d*\]', '', home_team).strip()
                        away_team = re.sub(r'\[\w+\d*\]$', '', away_team).strip()
                        if not home_team or not away_team:
                            continue
                        sf_cell = tds[4]
                        sf_spans = span_pattern.findall(sf_cell)
                        spf_home = float(sf_spans[0]) if len(sf_spans) >= 1 and sf_spans[0].replace('.', '').isdigit() else None
                        spf_away = float(sf_spans[1]) if len(sf_spans) >= 2 and sf_spans[1].replace('.', '').isdigit() else None
                        rfsf_cell = tds[5]
                        rfsf_spans = span_pattern.findall(rfsf_cell)
                        rqspf_home = float(rfsf_spans[0]) if len(rfsf_spans) >= 1 and rfsf_spans[0].replace('.', '').isdigit() else None
                        handicap = rfsf_spans[1] if len(rfsf_spans) >= 2 else None
                        rqspf_away = float(rfsf_spans[2]) if len(rfsf_spans) >= 3 and rfsf_spans[2].replace('.', '').isdigit() else None
                        dxf_cell = tds[6]
                        dxf_spans = span_pattern.findall(dxf_cell)
                        dx_over = float(dxf_spans[0]) if len(dxf_spans) >= 1 and dxf_spans[0].replace('.', '').isdigit() else None
                        total_line = float(dxf_spans[1]) if len(dxf_spans) >= 2 and dxf_spans[1].replace('.', '').isdigit() else None
                        dx_under = float(dxf_spans[2]) if len(dxf_spans) >= 3 and dxf_spans[2].replace('.', '').isdigit() else None
                        match_date = next_date
                        if match_time and len(match_time) >= 5:
                            month_day = match_time[:5]
                            match_date = f"{next_date[:4]}-{month_day}"
                        match = {
                            'id': f"{match_date}_{home_team}_{away_team}",
                            'date': match_date,
                            'time': match_time,
                            'num': num,
                            'league': league,
                            'home': home_team,
                            'away': away_team,
                            'handicap': handicap,
                            'rqspf_home': rqspf_home,
                            'rqspf_away': rqspf_away,
                            'spf_home': spf_home,
                            'spf_away': spf_away,
                            'total_line': total_line,
                            'dx_over': dx_over,
                            'dx_under': dx_under,
                            'status': 'not_started',
                        }
                        matches.append(match)
                    except Exception as e:
                        log.warning(f"解析明日篮球比赛失败: {e}")
                        continue
        
        now = datetime.now()
        for match in matches:
            if match['time'] and match['date']:
                try:
                    match_datetime = datetime.strptime(f"{match['date']} {match['time']}", '%Y-%m-%d %H:%M')
                    if match_datetime < now - timedelta(hours=3):
                        match['status'] = 'finished'
                    elif match_datetime < now + timedelta(hours=1):
                        match['status'] = 'in_progress'
                    else:
                        match['status'] = 'not_started'
                except ValueError:
                    pass
        
        # 仅保留「未开赛」比赛：已开赛(kickoff<=now)/已完结的不进入列表与逐场分析，
        # 既减少前端渲染量，也避免对已开赛比赛做无意义的模型分析（提速）。
        def _started(m):
            try:
                dt = datetime.strptime(f"{m['date']} {m['time']}", '%Y-%m-%d %H:%M')
                return dt <= now
            except (ValueError, KeyError, TypeError):
                return False
        matches = [m for m in matches if not _started(m)]

        log.info(f"获取到 {len(matches)} 场未开赛篮球比赛")
        return matches
    
    except Exception as e:
        log.error(f"抓取篮球赛程失败: {e}")
        return []

def _try_okooo_movement_for_500(matches, date):
    """为 500 源的比赛尽力从澳客获取更丰富的盘路走势（按队名匹配）。

    返回 {500_match_id: movement_dict}。任何失败都回退空 dict（调用方再用
    kv_store 历史兜底）。
    """
    try:
        from .okooo import fetch_okooo_basketball_schedule, prefetch_market_bundles
        from .odds_movement import build_movement_for_match

        def norm(t):
            return re.sub(r'\[.*?\]', '', str(t or '')).strip()

        ok_matches = fetch_okooo_basketball_schedule(date)
        if not ok_matches:
            return {}
        ok_map = {}
        for om in ok_matches:
            ok_map[(norm(om.get('home', '')), norm(om.get('away', '')))] = om

        ids = [str(om['id']) for om in ok_matches if om.get('id')]
        bundles = prefetch_market_bundles(ids) if ids else {}

        result = {}
        for m in matches:
            key = (norm(m.get('home', '')), norm(m.get('away', '')))
            om = ok_map.get(key)
            if not om:
                continue
            bid = str(om.get('id'))
            bundle = bundles.get(bid, {})
            mv = build_movement_for_match(om, okooo_bundle=bundle)
            result[m.get('id')] = mv
        return result
    except Exception as exc:
        log.warning(f"澳客走势增强失败(回退kv历史): {exc}")
        return {}


def _build_movement_map(matches, source, date):
    """为每场构建 {spf, rqspf, dx} 的 movement 字典。

    - 500 源：先轮询累积快照，再用 kv_store 历史计算；并尽力从澳客补充更
      丰富的盘路（按队名匹配）。
    - okooo 源：比赛自带 rf_trend / dx_trend，再补充 ml bundle（胜负走势）。
    """
    from .odds_movement import build_movement_for_match, get_odds_history, track_basketball_odds

    kv_history = None
    if source != 'okooo':
        try:
            track_basketball_odds(date)
        except Exception:
            pass
        kv_history = get_odds_history()

    okooo_bundles = {}
    if source == 'okooo':
        ids = [str(m['id']) for m in matches if m.get('id')]
        try:
            from .okooo import prefetch_market_bundles
            okooo_bundles = prefetch_market_bundles(ids)
        except Exception as exc:
            log.warning(f"澳客 bundle 抓取失败: {exc}")
            okooo_bundles = {}
    else:
        okooo_bundles = _try_okooo_movement_for_500(matches, date)

    out = {}
    for m in matches:
        mid = m.get('id')
        # okooo 源：matches 已含 rf_trend/dx_trend，直接用 build
        if source == 'okooo':
            mv = build_movement_for_match(m, okooo_bundle=okooo_bundles.get(str(mid)))
        else:
            # 优先用澳客增强结果；缺失的玩法用 kv 历史补齐
            ok_mv = okooo_bundles.get(mid)
            if ok_mv:
                mv = ok_mv
                if kv_history:
                    fallback = build_movement_for_match(m, kv_history=kv_history)
                    for k in ('spf', 'rqspf', 'dx'):
                        if not mv.get(k) and fallback.get(k):
                            mv[k] = fallback[k]
            else:
                mv = build_movement_for_match(m, kv_history=kv_history)
        out[mid] = mv
    return out


def generate_basketball_recommendations(date=None, bet_types=None, source='500',
                                        use_movement=True):
    if date is None:
        date = time.strftime('%Y-%m-%d')
    
    if bet_types is None:
        bet_types = ['spf', 'rqspf', 'dx']
    
    if source == 'okooo':
        try:
            from .okooo import fetch_okooo_basketball_schedule
            matches = fetch_okooo_basketball_schedule(date)
        except Exception as exc:
            log.warning(f"澳客篮球赛程不可用，回退500: {exc}")
            matches = fetch_basketball_schedule(date)
    else:
        matches = fetch_basketball_schedule(date)
    
    movement_map = {}
    if use_movement and matches:
        try:
            movement_map = _build_movement_map(matches, source, date)
        except Exception as exc:
            log.warning(f"篮球赔率走势构建失败(回退无走势): {exc}")
            movement_map = {}
    
    results = []
    for match in matches:
        mv = movement_map.get(match.get('id'), {}) or {}
        result = {
            'match': match,
            'spf': None,
            'rqspf': None,
            'dx': None,
        }
        
        if 'spf' in bet_types:
            result['spf'] = analyze_spf(match, mv.get('spf'))
        
        if 'rqspf' in bet_types:
            result['rqspf'] = analyze_rqspf(match, mv.get('rqspf'))
        
        if 'dx' in bet_types:
            result['dx'] = analyze_daxiao(match, mv.get('dx'))

        if match.get('status') == 'in_progress':
            for bet_type in ('spf', 'rqspf', 'dx'):
                section = result.get(bet_type)
                if isinstance(section, dict) and section.get('available'):
                    section.update({
                        'playable': False,
                        'official': False,
                        'skip_reason': 'match_already_started',
                    })

        from .odds_movement import describe_market_movement
        result['market_analysis'] = describe_market_movement(mv, result)
        
        results.append(result)
    
    # 统计走势命中情况，便于后续评估优化效果
    movement_stats = {'sharp_confirmed': 0, 'with_movement': 0, 'steam': 0}
    for r in results:
        for bt in ('spf', 'rqspf', 'dx'):
            b = r.get(bt) or {}
            if b.get('line_movement'):
                movement_stats['with_movement'] += 1
                if b.get('sharp_confirmed'):
                    movement_stats['sharp_confirmed'] += 1
                if (b.get('line_movement') or {}).get('steam'):
                    movement_stats['steam'] += 1

    payload = {
        'date': date,
        'count': len(results),
        'results': results,
        'version': BASKETBALL_VERSION,
        'source': source,
        'use_movement': use_movement,
        'movement_stats': movement_stats,
    }
    try:
        from .records import save_predictions, get_prediction_stats
        if results:
            save_predictions(date, results, BASKETBALL_VERSION)
        payload['history_stats'] = get_prediction_stats()
    except Exception as exc:
        log.warning(f"篮球预测记录保存失败: {exc}")
    return payload

def find_value_bets(results, threshold=0.05):
    """价值投注筛选，并融入赔率实时走势信号。

    在模型 edge（概率 - 0.5）基础上叠加「聪明钱确认」增益 movement_edge，
    优先推荐资金流向与模型方向一致的场次（steam/强确认优先）。
    """
    value_bets = []
    for r in results:
        match = r['match']
        if r['spf'] and r['spf']['available'] and r['spf'].get('playable', True):
            _append_value_bet(value_bets, r['spf'], '胜负',
                              f"{match['home']} vs {match['away']}", threshold,
                              max(r['spf']['home_prob'], r['spf']['away_prob']))
        
        if r['rqspf'] and r['rqspf']['available'] and r['rqspf'].get('playable', True):
            _append_value_bet(value_bets, r['rqspf'], '让分胜负',
                              f"{match['home']} vs {match['away']} ({match['handicap']})", threshold,
                              max(r['rqspf']['home_prob'], r['rqspf']['away_prob']))
        
        if r['dx'] and r['dx']['available'] and r['dx'].get('playable', True):
            _append_value_bet(value_bets, r['dx'], '大小分',
                              f"{match['home']} vs {match['away']} (总分{match['total_line']})", threshold,
                              max(r['dx']['over_prob'], r['dx']['under_prob']))

    value_bets.sort(key=lambda x: (-x['movement_edge'], -x['edge']))
    return value_bets[:20]


def _append_value_bet(value_bets, bet, label, match_label, threshold, prob):
    """向价值投注列表追加一项，并叠加聪明钱确认增益。"""
    edge = prob - 0.5
    if edge <= threshold:
        return
    sharp = bool(bet.get('sharp_confirmed', False))
    movement_edge = edge + (0.04 if sharp else 0.0)
    value_bets.append({
        'type': label,
        'match': match_label,
        'recommendation': bet.get('recommendation'),
        'edge': round(edge, 4),
        'movement_edge': round(movement_edge, 4),
        'prob': round(prob, 4),
        'sharp_confirmed': sharp,
        'movement': bet.get('line_movement'),
    })

def summarize_basketball_history(limit=50):
    history = kv_store.get(BASKETBALL_HISTORY_KEY, [])
    if isinstance(history, list):
        return history[-limit:]
    return []
