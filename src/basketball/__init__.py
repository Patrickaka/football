import sys
import math
import re
import time
import json
import urllib.request
import urllib.error
import random
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from ..common.logger import setup_logger
from ..common.paths import data_path
from ..common import kv_store

log = setup_logger('basketball')

BASKETBALL_VERSION = '2026-07-13-v1'
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

def analyze_spf(match):
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
    
    p_home = p_home * 0.7 + home_bias * 0.3
    p_away = 1.0 - p_home
    
    recommendation = '主胜' if p_home > p_away else '客胜'
    confidence = 'high' if abs(p_home - p_away) > 0.15 else ('medium' if abs(p_home - p_away) > 0.08 else 'low')
    
    return {
        'available': True,
        'home_prob': round(p_home, 4),
        'away_prob': round(p_away, 4),
        'home_odds': home_odds,
        'away_odds': away_odds,
        'recommendation': recommendation,
        'confidence': confidence,
    }

def analyze_rqspf(match):
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
    
    recommendation = '让胜' if p_home > p_away else '让负'
    confidence = 'high' if abs(p_home - p_away) > 0.15 else ('medium' if abs(p_home - p_away) > 0.08 else 'low')
    
    return {
        'available': True,
        'handicap': handicap,
        'home_prob': round(p_home, 4),
        'away_prob': round(p_away, 4),
        'home_odds': rq_home_odds,
        'away_odds': rq_away_odds,
        'recommendation': recommendation,
        'confidence': confidence,
    }

def analyze_daxiao(match):
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
    
    if total_line and avg_total:
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
    
    recommendation = '大分' if p_over > p_under else '小分'
    confidence = 'high' if abs(p_over - p_under) > 0.15 else ('medium' if abs(p_over - p_under) > 0.08 else 'low')
    
    return {
        'available': True,
        'total_line': total_line,
        'over_prob': round(p_over, 4),
        'under_prob': round(p_under, 4),
        'over_odds': over_odds,
        'under_odds': under_odds,
        'recommendation': recommendation,
        'confidence': confidence,
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
        
        matches = [m for m in matches if m['status'] != 'finished']
        
        log.info(f"获取到 {len(matches)} 场未完结篮球比赛")
        return matches
    
    except Exception as e:
        log.error(f"抓取篮球赛程失败: {e}")
        return []

def generate_basketball_recommendations(date=None, bet_types=None, source='500'):
    if date is None:
        date = time.strftime('%Y-%m-%d')
    
    if bet_types is None:
        bet_types = ['spf', 'rqspf', 'dx']
    
    matches = fetch_basketball_schedule(date)
    
    results = []
    for match in matches:
        result = {
            'match': match,
            'spf': None,
            'rqspf': None,
            'dx': None,
        }
        
        if 'spf' in bet_types:
            result['spf'] = analyze_spf(match)
        
        if 'rqspf' in bet_types:
            result['rqspf'] = analyze_rqspf(match)
        
        if 'dx' in bet_types:
            result['dx'] = analyze_daxiao(match)
        
        results.append(result)
    
    return {
        'date': date,
        'count': len(results),
        'results': results,
        'version': BASKETBALL_VERSION,
    }

def find_value_bets(results, threshold=0.05):
    value_bets = []
    for r in results:
        match = r['match']
        if r['spf'] and r['spf']['available']:
            edge = max(r['spf']['home_prob'], r['spf']['away_prob']) - 0.5
            if edge > threshold:
                value_bets.append({
                    'type': '胜负',
                    'match': f"{match['home']} vs {match['away']}",
                    'recommendation': r['spf']['recommendation'],
                    'edge': round(edge, 4),
                    'prob': max(r['spf']['home_prob'], r['spf']['away_prob']),
                })
        
        if r['rqspf'] and r['rqspf']['available']:
            edge = max(r['rqspf']['home_prob'], r['rqspf']['away_prob']) - 0.5
            if edge > threshold:
                value_bets.append({
                    'type': '让分胜负',
                    'match': f"{match['home']} vs {match['away']} ({match['handicap']})",
                    'recommendation': r['rqspf']['recommendation'],
                    'edge': round(edge, 4),
                    'prob': max(r['rqspf']['home_prob'], r['rqspf']['away_prob']),
                })
        
        if r['dx'] and r['dx']['available']:
            edge = max(r['dx']['over_prob'], r['dx']['under_prob']) - 0.5
            if edge > threshold:
                value_bets.append({
                    'type': '大小分',
                    'match': f"{match['home']} vs {match['away']} (总分{match['total_line']})",
                    'recommendation': r['dx']['recommendation'],
                    'edge': round(edge, 4),
                    'prob': max(r['dx']['over_prob'], r['dx']['under_prob']),
                })
    
    value_bets.sort(key=lambda x: -x['edge'])
    return value_bets[:20]

def summarize_basketball_history(limit=50):
    history = kv_store.get(BASKETBALL_HISTORY_KEY, [])
    if isinstance(history, list):
        return history[-limit:]
    return []