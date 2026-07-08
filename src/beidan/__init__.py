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

log = setup_logger('beidan')

BEIDAN_VERSION = '2026-07-08-v5'
BEIDAN_HISTORY_KEY = 'beidan_prediction_history'
BEIDAN_HISTORY_LIMIT = 500

BASE_URL = 'https://odds.500.com'
SCHEDULE_URL = f'{BASE_URL}/index_jczq.shtml'
DC_SCHEDULE_URL = f'{BASE_URL}/index_zqdc.shtml'
MATCH_DETAIL_URL = f'{BASE_URL}/fenxi/shuju-'

OKOOO_BASE = 'https://www.okooo.com'
OKOOO_DANCHANG_URL = f'{OKOOO_BASE}/danchang/'
OKOOO_MATCH_URL = f'{OKOOO_BASE}/soccer/match/'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Referer': SCHEDULE_URL,
}

OKOOO_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
    'Sec-Ch-Ua': '"Not/A)Brand";v="8", "Chromium";v="126", "Google Chrome";v="126"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"Windows"',
    'Sec-Ch-Ua-Platform-Version': '"10.0.0"',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'Referer': OKOOO_DANCHANG_URL,
    'Host': 'www.okooo.com',
}

_okooo_session = requests.Session()
_okooo_session.headers.update(OKOOO_HEADERS)
_okooo_session.verify = False

_okooo_waf_blocked = False
_okooo_waf_blocked_time = 0

def _mark_okooo_waf_blocked():
    global _okooo_waf_blocked, _okooo_waf_blocked_time
    _okooo_waf_blocked = True
    _okooo_waf_blocked_time = time.time()

def _is_okooo_waf_blocked():
    global _okooo_waf_blocked, _okooo_waf_blocked_time
    if not _okooo_waf_blocked:
        return False
    if time.time() - _okooo_waf_blocked_time > 60:
        _okooo_waf_blocked = False
        return False
    return True

def _init_okooo_session():
    global _okooo_session
    if _is_okooo_waf_blocked():
        log.info("okooo WAF已拦截，跳过session初始化")
        return
    try:
        log.info("初始化okooo session...")
        _okooo_session.get('https://www.okooo.com/', timeout=10)
        time.sleep(0.5)
        _okooo_session.get(OKOOO_DANCHANG_URL, timeout=10)
        log.info("okooo session初始化完成")
    except Exception as e:
        log.warning(f"初始化okooo session失败: {e}")

_init_okooo_session()

BET_TYPES = {
    'spf': {'name': '胜平负', 'description': '预测比赛胜负平结果'},
    'rqspf': {'name': '让球胜平负', 'description': '主队让球后的胜负平'},
    'bifen': {'name': '比分', 'description': '预测具体比分'},
    'zjq': {'name': '总进球', 'description': '预测总进球数'},
    'bqc': {'name': '半全场', 'description': '预测半场和全场结果'},
}

MAX_GOALS = 7

LEAGUE_PROFILES = {
    '英超': {'avg_goals': 2.8, 'draw_rate': 0.28},
    '西甲': {'avg_goals': 2.7, 'draw_rate': 0.27},
    '德甲': {'avg_goals': 3.1, 'draw_rate': 0.24},
    '意甲': {'avg_goals': 2.5, 'draw_rate': 0.30},
    '法甲': {'avg_goals': 2.6, 'draw_rate': 0.26},
    '欧冠': {'avg_goals': 2.7, 'draw_rate': 0.25},
    '欧联': {'avg_goals': 2.6, 'draw_rate': 0.26},
    '世界杯': {'avg_goals': 2.6, 'draw_rate': 0.27},
    '欧洲杯': {'avg_goals': 2.5, 'draw_rate': 0.28},
    '美洲杯': {'avg_goals': 2.7, 'draw_rate': 0.25},
    '亚洲杯': {'avg_goals': 2.4, 'draw_rate': 0.28},
    '中超': {'avg_goals': 2.5, 'draw_rate': 0.29},
    '日职联': {'avg_goals': 2.4, 'draw_rate': 0.30},
    '韩职': {'avg_goals': 2.5, 'draw_rate': 0.28},
    '巴西甲': {'avg_goals': 2.7, 'draw_rate': 0.25},
    '阿根廷甲级联赛': {'avg_goals': 2.6, 'draw_rate': 0.26},
}

def poisson_pmf(k, mu):
    if mu <= 0:
        return 0.0 if k > 0 else 1.0
    return (mu ** k) * math.exp(-mu) / math.factorial(k)

def euro_implied_lambdas(p_home, p_draw, p_away, target_total):
    supremacy = (p_home - p_away) / (p_home + p_draw + p_away + 1e-9)
    lam_home = target_total * (0.5 + supremacy * 0.35)
    lam_away = target_total * (0.5 - supremacy * 0.35)
    return max(0.01, lam_home), max(0.01, lam_away)

def calibrate_draw_probability(p_home, p_draw, p_away, handicap,
                               home_draw_rate=0.25, away_draw_rate=0.25,
                               league_draw_rate=0.25):
    ref_draw_rate = (home_draw_rate + away_draw_rate + league_draw_rate) / 3
    if handicap is not None:
        try:
            if isinstance(handicap, str):
                handicap = float(handicap.replace('(', '').replace(')', ''))
            if abs(handicap) >= 1.0:
                ref_draw_rate *= 0.8
            elif abs(handicap) >= 0.5:
                ref_draw_rate *= 0.95
        except (ValueError, TypeError):
            pass
    
    total = p_home + p_draw + p_away + 1e-9
    current_draw_rate = p_draw / total
    
    if current_draw_rate < ref_draw_rate * 0.8:
        p_draw *= 1.2
    elif current_draw_rate > ref_draw_rate * 1.3:
        p_draw *= 0.9
    
    total_new = p_home + p_draw + p_away + 1e-9
    return p_home / total_new, p_draw / total_new, p_away / total_new

def predict_scores_by_poisson(home_prob, draw_prob, away_prob, league='', handicap=0):
    league_profile = LEAGUE_PROFILES.get(league, {'avg_goals': 2.6, 'draw_rate': 0.27})
    avg_goals = league_profile['avg_goals']
    
    p_home, p_draw, p_away = calibrate_draw_probability(
        home_prob, draw_prob, away_prob, handicap,
        league_draw_rate=league_profile['draw_rate']
    )
    
    lam_home, lam_away = euro_implied_lambdas(p_home, p_draw, p_away, avg_goals)
    
    score_probs = {}
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            prob = poisson_pmf(h, lam_home) * poisson_pmf(a, lam_away)
            if prob > 1e-6:
                score_probs[(h, a)] = prob
    
    total_prob = sum(score_probs.values()) + 1e-9
    score_probs = {k: v / total_prob for k, v in score_probs.items()}
    
    sorted_scores = sorted(score_probs.items(), key=lambda x: -x[1])
    
    top3 = []
    for (h, a), prob in sorted_scores[:3]:
        top3.append({
            'score': f"{h}-{a}",
            'probability': prob,
            'home_goals': h,
            'away_goals': a,
        })
    
    return {
        'top3': top3,
        'score_probs': score_probs,
        'lambda_home': lam_home,
        'lambda_away': lam_away,
        '1x2_prob': {'H': p_home, 'D': p_draw, 'A': p_away},
    }

def parse_beidan_handicap(handicap):
    if handicap is None:
        return None
    if isinstance(handicap, (int, float)):
        return float(handicap)

    text = str(handicap).strip()
    if not text:
        return None
    text = text.replace('（', '(').replace('）', ')')
    match = re.search(r'[-+]?\d+(?:\.\d+)?', text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None

def rqspf_probs_from_score_probs(score_probs, handicap):
    handicap_value = parse_beidan_handicap(handicap)
    if handicap_value is None:
        return {}, {'available': False, 'reason': 'missing_handicap'}

    probs = {'让胜': 0.0, '让平': 0.0, '让负': 0.0}
    top_scores = []
    for (home_goals, away_goals), prob in score_probs.items():
        adjusted_margin = home_goals + handicap_value - away_goals
        if adjusted_margin > 0:
            label = '让胜'
        elif adjusted_margin < 0:
            label = '让负'
        else:
            label = '让平'
        probs[label] += prob
        top_scores.append({
            'score': f"{home_goals}-{away_goals}",
            'handicap_score': f"{home_goals + handicap_value:g}-{away_goals}",
            'result': label,
            'probability': prob,
        })

    total = sum(probs.values())
    if total > 0:
        probs = {key: value / total for key, value in probs.items()}

    top_scores.sort(key=lambda item: -item['probability'])
    return probs, {
        'available': True,
        'handicap': handicap_value,
        'top_scores': top_scores[:5],
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
            result = raw.decode(enc)
            result = result.encode('utf-8', errors='replace').decode('utf-8')
            return result
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode('utf-8', errors='replace')

def fetch_json(url, referer=None):
    try:
        content = fetch(url, encoding='utf-8', referer=referer)
        if content:
            return json.loads(content)
    except json.JSONDecodeError as e:
        log.error(f"JSON decode error: {e}")
    return None

def fetch_okooo(url, encoding='utf-8', referer=None, max_retries=2):
    global _okooo_session
    
    if _is_okooo_waf_blocked():
        log.debug(f"okooo WAF已拦截，跳过请求: {url}")
        return None
    
    for attempt in range(max_retries):
        try:
            headers = {}
            if referer:
                headers['Referer'] = referer
            
            resp = _okooo_session.get(url, headers=headers, timeout=30)
            
            if resp.status_code == 403 or resp.status_code == 503:
                log.warning(f"WAF拦截 {resp.status_code} for {url}, marking as blocked")
                _mark_okooo_waf_blocked()
                _okooo_session = requests.Session()
                _okooo_session.headers.update(OKOOO_HEADERS)
                _okooo_session.verify = False
                return None
            
            if resp.status_code != 200:
                log.warning(f"HTTP Error {resp.status_code} for {url}")
                if resp.status_code >= 500 and attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
            
            try:
                resp.encoding = 'gb2312'
                result = resp.text
            except:
                result = resp.content.decode('gb2312', errors='replace')
            
            if 'aliyun_waf' in result and '<title></title>' in result:
                log.warning(f"WAF拦截 detected for {url}, marking as blocked")
                _mark_okooo_waf_blocked()
                _okooo_session = requests.Session()
                _okooo_session.headers.update(OKOOO_HEADERS)
                _okooo_session.verify = False
                return None
            
            return result
            
        except requests.RequestException as e:
            log.warning(f"Request Error {e} for {url} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    log.error(f"Failed to fetch {url} after {max_retries} attempts")
    return None

def fetch_okooo_schedule(date=None):
    if date is None:
        date = time.strftime('%Y-%m-%d')
    
    url = f'{OKOOO_DANCHANG_URL}?date={date}'
    log.info(f"抓取okooo北单赛程: {date}")
    
    try:
        html = fetch_okooo(url, referer=OKOOO_DANCHANG_URL)
        if not html:
            log.warning(f"okooo页面返回为空(WAF拦截或网络错误)，尝试500.com备用数据源")
            fallback_matches = fetch_beidan_schedule(date, source='dc')
            if not fallback_matches:
                fallback_matches = fetch_beidan_schedule(date, source='jczq')
            for m in fallback_matches:
                m['source'] = '500.com'
            return fallback_matches
        
        log.debug(f"okooo页面HTML长度: {len(html)}")
        matches = []
        
        table_pattern = re.compile(r'<table[^>]*>(.*?)</table>', re.DOTALL)
        tables = table_pattern.findall(html)
        log.debug(f"okooo页面找到 {len(tables)} 个table标签")
        
        if len(tables) < 2:
            log.warning(f"okooo页面未找到比赛表格，找到 {len(tables)} 个table标签，尝试500.com备用数据源")
            return fetch_beidan_schedule(date)
        
        main_table = tables[1]
        tr_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL)
        tr_list = tr_pattern.findall(main_table)
        
        current_date = date
        
        for tr in tr_list:
            td_pattern = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL)
            td_list = td_pattern.findall(tr)
            
            if len(td_list) < 6:
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', tr)
                if date_match:
                    current_date = date_match.group(1)
                continue
            
            num_match = re.search(r'<span class="xh"><i>(\d+)</i></span>', td_list[0])
            league_match = re.search(r'href="//www\.okooo\.com/soccer/league/\d+/"[^>]*>([^<]+)</a>', td_list[0])
            match_id_match = re.search(r'/soccer/match/(\d+)', tr)
            
            time_str = re.sub(r'<[^>]+>', '', td_list[1]).strip()
            mtime_match = re.search(r'mTime="([^"]+)"', td_list[1])
            
            score = re.sub(r'<[^>]+>', '', td_list[5]).strip()
            
            home_match = re.search(r'<span class="homenameobj[^>]*title="([^"]+)"[^>]*>([^<]+)</span>', td_list[2])
            away_match = re.search(r'<span class="awaynameobj[^>]*title="([^"]+)"[^>]*>([^<]+)</span>', td_list[2])
            handicap_match = re.search(r'<span class="handicapobj[^>]*>([^<]+)</span>', td_list[2])
            
            odds_pattern = re.findall(r'<em[^>]*>([\d.]+)</em>', td_list[2])
            
            if not home_match or not away_match:
                continue
            
            num = num_match.group(1) if num_match else ''
            league = league_match.group(1) if league_match else ''
            match_id = match_id_match.group(1) if match_id_match else f'{current_date.replace("-", "")}_{num}'
            
            home_name = home_match.group(2).strip()
            away_name = away_match.group(2).strip()
            handicap_info = handicap_match.group(1).strip() if handicap_match else None
            
            spf_sp = float(odds_pattern[0]) if len(odds_pattern) >= 1 else None
            spf_s = float(odds_pattern[1]) if len(odds_pattern) >= 2 else None
            spf_f = float(odds_pattern[2]) if len(odds_pattern) >= 3 else None
            
            if mtime_match:
                match_time = mtime_match.group(1)
                date_time_match = re.match(r'(\d{2}-\d{2})\s+(\d{2}:\d{2})', match_time)
                if date_time_match:
                    match_date = f"{current_date[:4]}-{date_time_match.group(1)}"
                    match_time = date_time_match.group(2)
                else:
                    match_date = current_date
            else:
                date_time_match = re.match(r'(\d{2}-\d{2})\s+(\d{2}:\d{2})', time_str)
                if date_time_match:
                    match_date = f"{current_date[:4]}-{date_time_match.group(1)}"
                    match_time = date_time_match.group(2)
                else:
                    match_date = current_date
                    match_time = time_str
            
            status = 'finished' if score and score != '-' else 'not_started'
            
            matches.append({
                'id': match_id,
                'home': home_name,
                'away': away_name,
                'num': num,
                'date': match_date,
                'time': match_time,
                'league': league,
                'spf_sp': spf_sp,
                'spf_s': spf_s,
                'spf_f': spf_f,
                'rqspf_sp': None,
                'rqspf_s': None,
                'rqspf_f': None,
                'handicap': handicap_info,
                'status': status,
                'source': 'okooo',
            })
        
        matches = [m for m in matches if m['status'] != 'finished']
        
        if not matches:
            log.warning(f"okooo未找到未完结比赛，尝试500.com备用数据源")
            return fetch_beidan_schedule(date)
        
        log.info(f"okooo获取到 {len(matches)} 场未完结北单比赛")
        return matches
    
    except Exception as e:
        log.error(f"抓取okooo北单赛程失败: {e}，尝试500.com备用数据源")
        fallback_matches = fetch_beidan_schedule(date)
        for m in fallback_matches:
            m['source'] = '500.com'
        return fallback_matches

def fetch_okooo_asian_history(match_id):
    urls = [
        f'{OKOOO_MATCH_URL}{match_id}/ah/',
        f'{OKOOO_MATCH_URL}{match_id}/hodds/',
        f'{OKOOO_MATCH_URL}{match_id}/odds/',
        f'{OKOOO_MATCH_URL}{match_id}/history/',
    ]
    log.info(f"抓取okooo亚盘赔率变化: match_id={match_id}")
    
    for url in urls:
        try:
            html = fetch_okooo(url, referer=OKOOO_DANCHANG_URL)
            if not html:
                continue
            
            html_clean = html.replace('display:none', '').replace('display: none', '')
            
            asian_history = []
            
            tables = re.findall(r'<table[^>]*>(.*?)</table>', html_clean, re.DOTALL)
            for table in tables:
                rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table, re.DOTALL)
                if len(rows) < 2:
                    continue
                
                headers = re.findall(r'<th[^>]*>(.*?)</th>', rows[0], re.DOTALL)
                header_text = ''.join(headers)
                
                if '亚盘' in header_text or '让球' in header_text or '盘口' in header_text:
                    for row in rows[1:]:
                        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
                        if len(cells) >= 4:
                            try:
                                time_val = cells[0].strip()
                                handicap_val = cells[1].strip()
                                home_val = cells[2].strip()
                                away_val = cells[3].strip()
                                
                                time_val = re.sub(r'<[^>]+>', '', time_val).strip()
                                handicap_val = re.sub(r'<[^>]+>', '', handicap_val).strip()
                                home_val = re.sub(r'<[^>]+>', '', home_val).strip()
                                away_val = re.sub(r'<[^>]+>', '', away_val).strip()
                                
                                if not time_val or len(time_val) > 8:
                                    continue
                                if not handicap_val or len(handicap_val) > 20:
                                    continue
                                
                                asian_history.append({
                                    'time': time_val,
                                    'handicap': handicap_val,
                                    'home_odds': float(home_val) if home_val.replace('.', '').isdigit() else None,
                                    'away_odds': float(away_val) if away_val.replace('.', '').isdigit() else None,
                                })
                            except Exception:
                                continue
            
            if asian_history:
                log.info(f"从 {url} 获取到 {len(asian_history)} 条亚盘赔率记录")
                return {'history': asian_history}
            
            script_pattern = re.compile(r'<script[^>]*>(.*?)</script>', re.DOTALL)
            scripts = script_pattern.findall(html_clean)
            for script in scripts:
                if len(script) < 50:
                    continue
                if '亚盘' not in script and 'asian' not in script.lower() and 'AH' not in script:
                    continue
                
                data_patterns = re.findall(r'(\d{2}:\d{2})\s*,\s*["\']?([^\s"\',]+)["\']?\s*,\s*([\d.]+)\s*,\s*([\d.]+)', script)
                for time_val, handicap_val, home_val, away_val in data_patterns:
                    try:
                        asian_history.append({
                            'time': time_val,
                            'handicap': handicap_val,
                            'home_odds': float(home_val),
                            'away_odds': float(away_val),
                        })
                    except Exception:
                        continue
                
                if asian_history:
                    break
            
            if asian_history:
                log.info(f"从 {url} 通过脚本获取到 {len(asian_history)} 条亚盘赔率记录")
                return {'history': asian_history}
        
        except Exception as e:
            log.warning(f"抓取okooo亚盘赔率变化失败({url}): {e}")
            continue
    
    log.info(f"未获取到亚盘赔率变化数据")
    return {'history': []}

def fetch_okooo_goals_history(match_id):
    urls = [
        f'{OKOOO_MATCH_URL}{match_id}/goals/',
    ]
    log.info(f"抓取okooo总进球赔率变化: match_id={match_id}")
    
    for url in urls:
        try:
            html = fetch_okooo(url, referer=OKOOO_DANCHANG_URL)
            if not html:
                continue
            
            html_clean = html.replace('display:none', '').replace('display: none', '')
            
            goals_history = []
            
            tables = re.findall(r'<table[^>]*>(.*?)</table>', html_clean, re.DOTALL)
            for table in tables:
                rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table, re.DOTALL)
                if len(rows) < 2:
                    continue
                
                headers = re.findall(r'<th[^>]*>(.*?)</th>', rows[0], re.DOTALL)
                header_text = ''.join(headers)
                
                if '进球' in header_text or '大小球' in header_text:
                    for row in rows[1:]:
                        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
                        if len(cells) >= 3:
                            try:
                                time_val = cells[0].strip()
                                goals_val = cells[1].strip()
                                over_val = cells[2].strip()
                                under_val = cells[3].strip() if len(cells) >= 4 else None
                                
                                time_val = re.sub(r'<[^>]+>', '', time_val).strip()
                                goals_val = re.sub(r'<[^>]+>', '', goals_val).strip()
                                over_val = re.sub(r'<[^>]+>', '', over_val).strip()
                                under_val = re.sub(r'<[^>]+>', '', under_val).strip() if under_val else None
                                
                                if not time_val or len(time_val) > 8:
                                    continue
                                
                                goals_history.append({
                                    'time': time_val,
                                    'line': goals_val,
                                    'over_odds': float(over_val) if over_val.replace('.', '').isdigit() else None,
                                    'under_odds': float(under_val) if under_val and under_val.replace('.', '').isdigit() else None,
                                })
                            except Exception:
                                continue
            
            if goals_history:
                log.info(f"从 {url} 获取到 {len(goals_history)} 条总进球赔率记录")
                return {'history': goals_history}
            
            script_pattern = re.compile(r'<script[^>]*>(.*?)</script>', re.DOTALL)
            scripts = script_pattern.findall(html_clean)
            for script in scripts:
                if len(script) < 50:
                    continue
                if '进球' not in script and 'goals' not in script.lower() and 'total' not in script.lower():
                    continue
                
                data_patterns = re.findall(r'(\d{2}:\d{2})\s*,\s*["\']?([^\s"\',]+)["\']?\s*,\s*([\d.]+)\s*,\s*([\d.]+)', script)
                for time_val, line_val, over_val, under_val in data_patterns:
                    try:
                        goals_history.append({
                            'time': time_val,
                            'line': line_val,
                            'over_odds': float(over_val),
                            'under_odds': float(under_val),
                        })
                    except Exception:
                        continue
                
                if goals_history:
                    break
            
            if goals_history:
                log.info(f"从 {url} 通过脚本获取到 {len(goals_history)} 条总进球赔率记录")
                return {'history': goals_history}
        
        except Exception as e:
            log.warning(f"抓取okooo总进球赔率变化失败({url}): {e}")
            continue
    
    log.info(f"未获取到总进球赔率变化数据")
    return {'history': []}

def fetch_okooo_cs_history(match_id):
    urls = [
        f'{OKOOO_MATCH_URL}{match_id}/cs/',
    ]
    log.info(f"抓取okooo比分赔率变化: match_id={match_id}")
    
    for url in urls:
        try:
            html = fetch_okooo(url, referer=OKOOO_DANCHANG_URL)
            if not html:
                continue
            
            html_clean = html.replace('display:none', '').replace('display: none', '')
            
            cs_history = []
            
            tables = re.findall(r'<table[^>]*>(.*?)</table>', html_clean, re.DOTALL)
            for table in tables:
                rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table, re.DOTALL)
                if len(rows) < 2:
                    continue
                
                headers = re.findall(r'<th[^>]*>(.*?)</th>', rows[0], re.DOTALL)
                header_text = ''.join(headers)
                
                if '比分' in header_text or 'cs' in header_text.lower():
                    for row in rows[1:]:
                        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
                        if len(cells) >= 3:
                            try:
                                time_val = cells[0].strip()
                                score_val = cells[1].strip()
                                odds_val = cells[2].strip()
                                
                                time_val = re.sub(r'<[^>]+>', '', time_val).strip()
                                score_val = re.sub(r'<[^>]+>', '', score_val).strip()
                                odds_val = re.sub(r'<[^>]+>', '', odds_val).strip()
                                
                                if not time_val or len(time_val) > 8:
                                    continue
                                if not score_val or '-' not in score_val:
                                    continue
                                
                                cs_history.append({
                                    'time': time_val,
                                    'score': score_val,
                                    'odds': float(odds_val) if odds_val.replace('.', '').isdigit() else None,
                                })
                            except Exception:
                                continue
            
            if cs_history:
                log.info(f"从 {url} 获取到 {len(cs_history)} 条比分赔率记录")
                return {'history': cs_history}
            
            script_pattern = re.compile(r'<script[^>]*>(.*?)</script>', re.DOTALL)
            scripts = script_pattern.findall(html_clean)
            for script in scripts:
                if len(script) < 50:
                    continue
                if '比分' not in script and 'cs' not in script.lower() and 'score' not in script.lower():
                    continue
                
                data_patterns = re.findall(r'(\d{2}:\d{2})\s*,\s*["\']?([^\s"\',]+)["\']?\s*,\s*([\d.]+)', script)
                for time_val, score_val, odds_val in data_patterns:
                    try:
                        cs_history.append({
                            'time': time_val,
                            'score': score_val,
                            'odds': float(odds_val),
                        })
                    except Exception:
                        continue
                
                if cs_history:
                    break
            
            if cs_history:
                log.info(f"从 {url} 通过脚本获取到 {len(cs_history)} 条比分赔率记录")
                return {'history': cs_history}
        
        except Exception as e:
            log.warning(f"抓取okooo比分赔率变化失败({url}): {e}")
            continue
    
    log.info(f"未获取到比分赔率变化数据")
    return {'history': []}

def adjust_probs_by_asian(home_win_prob, draw_prob, away_win_prob, asian_history):
    if not asian_history or len(asian_history) < 2:
        return home_win_prob, draw_prob, away_win_prob
    
    recent_changes = asian_history[-5:]
    
    home_odds_changes = []
    away_odds_changes = []
    
    for i in range(1, len(recent_changes)):
        prev = recent_changes[i-1]
        curr = recent_changes[i]
        
        if prev.get('home_odds') and curr.get('home_odds'):
            home_odds_changes.append(curr['home_odds'] - prev['home_odds'])
        if prev.get('away_odds') and curr.get('away_odds'):
            away_odds_changes.append(curr['away_odds'] - prev['away_odds'])
    
    home_trend = sum(home_odds_changes) / len(home_odds_changes) if home_odds_changes else 0
    away_trend = sum(away_odds_changes) / len(away_odds_changes) if away_odds_changes else 0
    
    adjustment_factor = 0.15
    
    if home_trend > 0.02:
        home_win_prob *= (1 - adjustment_factor)
        away_win_prob *= (1 + adjustment_factor * 0.5)
    elif home_trend < -0.02:
        home_win_prob *= (1 + adjustment_factor)
        away_win_prob *= (1 - adjustment_factor * 0.5)
    
    if away_trend > 0.02:
        away_win_prob *= (1 - adjustment_factor)
        home_win_prob *= (1 + adjustment_factor * 0.5)
    elif away_trend < -0.02:
        away_win_prob *= (1 + adjustment_factor)
        home_win_prob *= (1 - adjustment_factor * 0.5)
    
    total_prob = home_win_prob + draw_prob + away_win_prob
    if total_prob > 0:
        home_win_prob /= total_prob
        draw_prob /= total_prob
        away_win_prob /= total_prob
    
    return home_win_prob, draw_prob, away_win_prob

def analyze_asian_trend(asian_history):
    if not asian_history:
        return {'direction': 'stable', 'strength': 0}
    
    recent = asian_history[-5:]
    if len(recent) < 2:
        return {'direction': 'stable', 'strength': 0}
    
    home_changes = []
    away_changes = []
    
    for i in range(1, len(recent)):
        prev = recent[i-1]
        curr = recent[i]
        
        if prev.get('home_odds') and curr.get('home_odds'):
            home_changes.append(curr['home_odds'] - prev['home_odds'])
        if prev.get('away_odds') and curr.get('away_odds'):
            away_changes.append(curr['away_odds'] - prev['away_odds'])
    
    avg_home_change = sum(home_changes) / len(home_changes) if home_changes else 0
    avg_away_change = sum(away_changes) / len(away_changes) if away_changes else 0
    
    strength = abs(avg_home_change) + abs(avg_away_change)
    
    if avg_home_change < -0.03:
        direction = 'home_backing'
    elif avg_home_change > 0.03:
        direction = 'home_laying'
    elif avg_away_change < -0.03:
        direction = 'away_backing'
    elif avg_away_change > 0.03:
        direction = 'away_laying'
    else:
        direction = 'stable'
    
    return {
        'direction': direction,
        'strength': round(strength, 4),
        'avg_home_change': round(avg_home_change, 4),
        'avg_away_change': round(avg_away_change, 4),
    }

def analyze_cs_trend(cs_history):
    if not cs_history or len(cs_history) < 2:
        return {'direction': 'stable', 'strength': 0, 'hot_scores': []}
    
    recent = cs_history[-10:]
    
    score_odds_map = {}
    for entry in recent:
        score = entry.get('score')
        odds = entry.get('odds')
        if score and odds:
            if score not in score_odds_map:
                score_odds_map[score] = []
            score_odds_map[score].append(odds)
    
    hot_scores = []
    for score, odds_list in score_odds_map.items():
        avg_odds = sum(odds_list) / len(odds_list)
        trend = odds_list[-1] - odds_list[0] if len(odds_list) >= 2 else 0
        hot_scores.append({
            'score': score,
            'avg_odds': round(avg_odds, 2),
            'trend': 'down' if trend < -0.1 else ('up' if trend > 0.1 else 'stable'),
            'current_odds': odds_list[-1],
        })
    
    hot_scores.sort(key=lambda x: x['current_odds'])
    
    return {
        'direction': 'active' if len(hot_scores) > 0 else 'stable',
        'strength': len(hot_scores),
        'hot_scores': hot_scores[:5],
    }

def enhance_scores_with_cs(score_prediction, cs_history):
    if not cs_history or len(cs_history) < 2:
        return score_prediction
    
    recent = cs_history[-5:]
    
    cs_odds_map = {}
    for entry in recent:
        score = entry.get('score')
        odds = entry.get('odds')
        if score and odds:
            cs_odds_map[score] = odds
    
    if not cs_odds_map:
        return score_prediction
    
    enhanced_scores = []
    for score_item in score_prediction['top3']:
        if isinstance(score_item, dict):
            score = score_item.get('score')
            prob = score_item.get('probability', 0)
        else:
            score = score_item[0]
            prob = score_item[1]
        if not score:
            continue
        
        if score in cs_odds_map:
            cs_odds = cs_odds_map[score]
            cs_prob = 1.0 / cs_odds
            enhanced_prob = (prob + cs_prob) / 2
            enhanced_scores.append((score, enhanced_prob, 'cs_enhanced'))
        else:
            enhanced_scores.append((score, prob, 'poisson'))
    
    for score, odds in cs_odds_map.items():
        if score not in [s[0] for s in enhanced_scores]:
            cs_prob = 1.0 / odds
            enhanced_scores.append((score, cs_prob * 0.5, 'cs_new'))
    
    enhanced_scores.sort(key=lambda x: -x[1])
    
    score_prediction['top3'] = [
        {
            'score': s[0],
            'probability': s[1],
            'source': s[2],
            'home_goals': int(s[0].split('-')[0]) if '-' in s[0] and s[0].split('-')[0].isdigit() else None,
            'away_goals': int(s[0].split('-')[1]) if '-' in s[0] and s[0].split('-')[1].isdigit() else None,
        }
        for s in enhanced_scores[:3]
    ]
    
    return score_prediction

def calculate_goals_factor(goals_history):
    if not goals_history or len(goals_history) < 2:
        return 1.0
    
    recent = goals_history[-10:]
    
    total_over_odds = 0
    total_under_odds = 0
    count = 0
    
    for entry in recent:
        over_odds = entry.get('over_odds')
        under_odds = entry.get('under_odds')
        if over_odds and under_odds:
            total_over_odds += over_odds
            total_under_odds += under_odds
            count += 1
    
    if count == 0:
        return 1.0
    
    avg_over = total_over_odds / count
    avg_under = total_under_odds / count
    
    if avg_over < avg_under:
        return 1.2
    elif avg_over > avg_under + 0.5:
        return 0.85
    else:
        return 1.0

def adjust_zjq_by_goals(zjq_probs, goals_history):
    if not goals_history or len(goals_history) < 2:
        return zjq_probs
    
    recent = goals_history[-5:]
    
    over_trend = 0
    under_trend = 0
    count = 0
    
    for i in range(1, len(recent)):
        prev = recent[i-1]
        curr = recent[i]
        if prev.get('over_odds') and curr.get('over_odds'):
            over_trend += curr['over_odds'] - prev['over_odds']
            count += 1
        if prev.get('under_odds') and curr.get('under_odds'):
            under_trend += curr['under_odds'] - prev['under_odds']
    
    if count == 0:
        return zjq_probs
    
    over_trend_avg = over_trend / count
    
    if over_trend_avg < -0.05:
        for key in ['3', '4', '5', '6', '7+']:
            zjq_probs[key] = zjq_probs.get(key, 0) * 1.2
        for key in ['0', '1', '2']:
            zjq_probs[key] = zjq_probs.get(key, 0) * 0.85
    elif over_trend_avg > 0.05:
        for key in ['0', '1', '2']:
            zjq_probs[key] = zjq_probs.get(key, 0) * 1.2
        for key in ['3', '4', '5', '6', '7+']:
            zjq_probs[key] = zjq_probs.get(key, 0) * 0.85
    
    return zjq_probs

def analyze_goals_trend(goals_history):
    if not goals_history or len(goals_history) < 2:
        return {'direction': 'stable', 'strength': 0}
    
    recent = goals_history[-10:]
    
    over_changes = []
    under_changes = []
    
    for i in range(1, len(recent)):
        prev = recent[i-1]
        curr = recent[i]
        if prev.get('over_odds') and curr.get('over_odds'):
            over_changes.append(curr['over_odds'] - prev['over_odds'])
        if prev.get('under_odds') and curr.get('under_odds'):
            under_changes.append(curr['under_odds'] - prev['under_odds'])
    
    avg_over_change = sum(over_changes) / len(over_changes) if over_changes else 0
    avg_under_change = sum(under_changes) / len(under_changes) if under_changes else 0
    
    strength = abs(avg_over_change) + abs(avg_under_change)
    
    if avg_over_change < -0.05:
        direction = 'over_backing'
    elif avg_over_change > 0.05:
        direction = 'over_laying'
    elif avg_under_change < -0.05:
        direction = 'under_backing'
    elif avg_under_change > 0.05:
        direction = 'under_laying'
    else:
        direction = 'stable'
    
    return {
        'direction': direction,
        'strength': round(strength, 4),
        'avg_over_change': round(avg_over_change, 4),
        'avg_under_change': round(avg_under_change, 4),
    }

def calculate_asian_goal_factor(asian_history):
    if not asian_history or len(asian_history) < 2:
        return 1.0
    
    recent = asian_history[-10:]
    
    total_odds_sum = 0
    count = 0
    
    for entry in recent:
        home_odds = entry.get('home_odds')
        away_odds = entry.get('away_odds')
        if home_odds and away_odds:
            total_odds_sum += home_odds + away_odds
            count += 1
    
    if count == 0:
        return 1.0
    
    avg_total_odds = total_odds_sum / count
    
    if avg_total_odds < 3.6:
        return 1.3
    elif avg_total_odds < 4.0:
        return 1.15
    elif avg_total_odds < 4.4:
        return 1.0
    elif avg_total_odds < 4.8:
        return 0.9
    else:
        return 0.75

def fetch_beidan_schedule(date=None, source='jczq'):
    if date is None:
        date = time.strftime('%Y-%m-%d')
    
    if source == 'dc':
        url = DC_SCHEDULE_URL
        referer = DC_SCHEDULE_URL
    else:
        url = SCHEDULE_URL
        referer = SCHEDULE_URL
    
    log.info(f"抓取北单赛程({source}): {date}")
    
    try:
        html = fetch(url, referer=referer)
        if not html:
            return []
        
        matches = []
        
        title_pat = re.compile(
            r'shuju-(\d+)\.shtml.*?title="([^"]+?)VS([^"]+?)'
            r'(?:数据|盘口|百家|欧赔|亚赔|亚盘|指数|对比|分析)[^"]*"',
            re.DOTALL
        )
        for m in title_pat.finditer(html):
            match_id = m.group(1).strip()
            home_name = m.group(2).strip()
            away_name = m.group(3).strip()
            for suffix in ['百家', '欧赔', '亚赔', '亚盘', '数据', '盘口', '指数', '对比', '分析', '百家欧赔', '百家亚盘']:
                if home_name.endswith(suffix):
                    home_name = home_name[:-len(suffix)].strip()
                if away_name.endswith(suffix):
                    away_name = away_name[:-len(suffix)].strip()
            if home_name and away_name and match_id:
                matches.append({
                    'id': match_id,
                    'home': home_name,
                    'away': away_name,
                    'num': '',
                    'date': date,
                    'time': '',
                    'league': '',
                    'spf_sp': None,
                    'spf_s': None,
                    'spf_f': None,
                    'rqspf_sp': None,
                    'rqspf_s': None,
                    'rqspf_f': None,
                    'handicap': None,
                    'status': 'not_started',
                })
        
        match_time_map = {}
        time_patterns = [
            r'<td[^>]*?rowspan="2"[^>]*?>(\d{2}-\d{2}\s+\d{2}:\d{2})</td>.*?'
            r'shuju-(\d+)\.shtml',
            r'shuju-(\d+)\.shtml.*?(\d{2}-\d{2}\s+\d{2}:\d{2})',
        ]
        for pat in time_patterns:
            time_row_pat = re.compile(pat, re.DOTALL)
            for m in time_row_pat.finditer(html):
                if m.group(1).isdigit():
                    match_id = m.group(1)
                    time_val = m.group(2)
                else:
                    match_id = m.group(2)
                    time_val = m.group(1)
                if match_id not in match_time_map:
                    match_time_map[match_id] = time_val
        
        match_num_map = dict(
            re.findall(r'value="(\d+)"\s*/>\s*(周[一二三四五六日]\d{3})', html)
        )
        
        match_league_map = {}
        league_blocks = re.split(r'<a[^>]*href="//liansai\.500\.com/zuqiu-\d+/"[^>]*>([^<]+)</a>', html)
        current_league = ''
        for i, block in enumerate(league_blocks):
            if i % 2 == 1:
                current_league = block.strip()
            else:
                match_ids_in_block = re.findall(r'shuju-(\d+)\.shtml', block)
                for match_id in match_ids_in_block:
                    if match_id not in match_league_map:
                        match_league_map[match_id] = current_league
        
        now = datetime.now()
        
        for match in matches:
            match_id = match['id']
            if match_id in match_time_map:
                time_val = match_time_map[match_id]
                date_match = re.match(r'(\d{2}-\d{2})\s+(\d{2}:\d{2})', time_val)
                if date_match:
                    match['time'] = date_match.group(2)
                    time_date_str = date_match.group(1)
                    if time_date_str != date[5:]:
                        match['date'] = f"{date[:4]}-{time_date_str}"
                else:
                    match['time'] = time_val
            if match_id in match_league_map:
                match['league'] = match_league_map[match_id].strip()
            if match_id in match_num_map:
                match['num'] = match_num_map[match_id]
            
            match_datetime = None
            try:
                if match['date'] and match['time']:
                    match_datetime = datetime.strptime(f"{match['date']} {match['time']}", '%Y-%m-%d %H:%M')
            except ValueError:
                pass
            
            if match_datetime:
                if match_datetime < now - timedelta(hours=3):
                    match['status'] = 'finished'
                elif match_datetime < now + timedelta(hours=1):
                    match['status'] = 'in_progress'
                else:
                    match['status'] = 'not_started'
            else:
                try:
                    match_date = datetime.strptime(match['date'], '%Y-%m-%d')
                    if match_date < now.date():
                        match['status'] = 'finished'
                except ValueError:
                    pass
        
        matches = [m for m in matches if m['status'] != 'finished']
        
        log.info(f"获取到 {len(matches)} 场未完结北单比赛")
        return matches
    
    except Exception as e:
        log.error(f"抓取北单赛程失败: {e}")
        return []

def fetch_jczq_schedule(date=None):
    return fetch_beidan_schedule(date, source='jczq')

def fetch_zqdc_schedule(date=None):
    return fetch_beidan_schedule(date, source='dc')

def fetch_beidan_bifen(date=None):
    if date is None:
        date = time.strftime('%Y-%m-%d')
    
    url = f'{BASE_URL}/football/jc/data/ssq_match_info.jsp?date={date}&gameType=bifen'
    log.info(f"抓取北单比分数据: {date}")
    
    try:
        content = fetch(url, referer=SCHEDULE_URL)
        if not content:
            return {}
        
        result = {}
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('|')
            if len(parts) < 2:
                continue
            
            match_id = parts[0]
            odds = {}
            
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    score = parts[i]
                    try:
                        odd = float(parts[i + 1])
                        odds[score] = odd
                    except ValueError:
                        pass
            
            if odds:
                result[match_id] = odds
        
        return result
    
    except Exception as e:
        log.error(f"抓取北单比分数据失败: {e}")
        return {}

def fetch_beidan_zjq(date=None):
    if date is None:
        date = time.strftime('%Y-%m-%d')
    
    url = f'{BASE_URL}/football/jc/data/ssq_match_info.jsp?date={date}&gameType=zjq'
    log.info(f"抓取北单总进球数据: {date}")
    
    try:
        content = fetch(url, referer=SCHEDULE_URL)
        if not content:
            return {}
        
        result = {}
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('|')
            if len(parts) < 8:
                continue
            
            match_id = parts[0]
            try:
                zjq_odds = {
                    '0': float(parts[1]) if parts[1] else None,
                    '1': float(parts[2]) if parts[2] else None,
                    '2': float(parts[3]) if parts[3] else None,
                    '3': float(parts[4]) if parts[4] else None,
                    '4': float(parts[5]) if parts[5] else None,
                    '5': float(parts[6]) if parts[6] else None,
                    '6': float(parts[7]) if parts[7] else None,
                    '7+': float(parts[8]) if len(parts) > 8 else None,
                }
                result[match_id] = zjq_odds
            except Exception as e:
                log.warning(f"解析总进球数据失败: {line} - {e}")
        
        return result
    
    except Exception as e:
        log.error(f"抓取北单总进球数据失败: {e}")
        return {}

def fetch_beidan_bqc(date=None):
    if date is None:
        date = time.strftime('%Y-%m-%d')
    
    url = f'{BASE_URL}/football/jc/data/ssq_match_info.jsp?date={date}&gameType=bqc'
    log.info(f"抓取北单半全场数据: {date}")
    
    try:
        content = fetch(url, referer=SCHEDULE_URL)
        if not content:
            return {}
        
        result = {}
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('|')
            if len(parts) < 10:
                continue
            
            match_id = parts[0]
            try:
                bqc_odds = {
                    '胜胜': float(parts[1]) if parts[1] else None,
                    '胜平': float(parts[2]) if parts[2] else None,
                    '胜负': float(parts[3]) if parts[3] else None,
                    '平胜': float(parts[4]) if parts[4] else None,
                    '平平': float(parts[5]) if parts[5] else None,
                    '平负': float(parts[6]) if parts[6] else None,
                    '负胜': float(parts[7]) if parts[7] else None,
                    '负平': float(parts[8]) if parts[8] else None,
                    '负负': float(parts[9]) if parts[9] else None,
                }
                result[match_id] = bqc_odds
            except Exception as e:
                log.warning(f"解析半全场数据失败: {line} - {e}")
        
        return result
    
    except Exception as e:
        log.error(f"抓取北单半全场数据失败: {e}")
        return {}

def calculate_implied_probability(odds_dict):
    if not odds_dict:
        return {}
    
    prob_sum = sum(1 / o for o in odds_dict.values() if o and o > 0)
    if prob_sum == 0:
        return {}
    
    return {k: (1 / v) / prob_sum for k, v in odds_dict.items() if v and v > 0}

def _actual_spf_from_record(record):
    actual = record.get('actual') if isinstance(record.get('actual'), dict) else {}
    settlement = record.get('settlement') if isinstance(record.get('settlement'), dict) else {}

    direct = (
        record.get('actual_spf')
        or actual.get('spf')
        or settlement.get('spf')
        or settlement.get('actual_spf')
    )
    if direct in ('胜', '平', '负'):
        return direct

    score = (
        record.get('actual_score')
        or actual.get('score')
        or actual.get('actual_score')
        or settlement.get('score')
        or settlement.get('actual_score')
    )
    if not score or '-' not in str(score):
        return None
    try:
        home_goals, away_goals = map(int, str(score).split('-', 1))
    except ValueError:
        return None
    if home_goals > away_goals:
        return '胜'
    if home_goals < away_goals:
        return '负'
    return '平'

def _actual_zjq_from_record(record):
    actual = record.get('actual') if isinstance(record.get('actual'), dict) else {}
    settlement = record.get('settlement') if isinstance(record.get('settlement'), dict) else {}

    direct = (
        record.get('actual_zjq')
        or actual.get('zjq')
        or settlement.get('zjq')
        or settlement.get('actual_zjq')
    )
    if direct is not None:
        direct = str(direct)
        if direct in {'0', '1', '2', '3', '4', '5', '6', '7+'}:
            return direct

    score = (
        record.get('actual_score')
        or actual.get('score')
        or actual.get('actual_score')
        or settlement.get('score')
        or settlement.get('actual_score')
    )
    if not score or '-' not in str(score):
        return None
    try:
        home_goals, away_goals = map(int, str(score).split('-', 1))
    except ValueError:
        return None
    total_goals = home_goals + away_goals
    return '7+' if total_goals >= 7 else str(total_goals)

def apply_beidan_history_calibration(probabilities, bet_type, league=None, min_samples=8, limit=200):
    """Use settled Beidan snapshots as a conservative reliability correction."""
    if not probabilities:
        return probabilities, {'applied': False, 'reason': 'empty_probabilities'}

    records = _load_beidan_history()
    if not records:
        return probabilities, {'applied': False, 'reason': 'no_history'}

    expected = {str(k): 0.0 for k in probabilities}
    actuals = {str(k): 0.0 for k in probabilities}
    samples = 0

    for record in records[:limit]:
        if not record.get('settled'):
            continue
        section = record.get(bet_type) if isinstance(record.get(bet_type), dict) else {}
        past_probs = section.get('probabilities') if isinstance(section.get('probabilities'), dict) else {}
        if not past_probs:
            continue

        if bet_type == 'spf':
            actual = _actual_spf_from_record(record)
        elif bet_type == 'zjq':
            actual = _actual_zjq_from_record(record)
        else:
            actual = None
        if actual not in expected:
            continue

        league_weight = 1.25 if league and record.get('league') == league else 1.0
        for option in expected:
            expected[option] += float(past_probs.get(option, 0.0) or 0.0) * league_weight
        actuals[actual] += league_weight
        samples += league_weight

    if samples < min_samples:
        return probabilities, {
            'applied': False,
            'reason': 'insufficient_settled_samples',
            'sample_count': round(samples, 3),
            'min_samples': min_samples,
        }

    factors = {}
    prior = 6.0
    option_count = max(len(expected), 1)
    prior_each = prior / option_count
    for option in expected:
        ratio = (actuals[option] + prior_each) / max(expected[option] + prior_each, 1e-9)
        factors[option] = max(0.86, min(1.16, ratio))

    adjusted = {
        option: float(probabilities.get(option, 0.0) or 0.0) * factors.get(option, 1.0)
        for option in probabilities
    }
    total = sum(adjusted.values())
    if total <= 0:
        return probabilities, {'applied': False, 'reason': 'zero_adjusted_total'}

    adjusted = {option: value / total for option, value in adjusted.items()}
    return adjusted, {
        'applied': True,
        'sample_count': round(samples, 3),
        'factors': {k: round(v, 6) for k, v in factors.items()},
        'actuals': {k: round(v, 3) for k, v in actuals.items()},
        'expected': {k: round(v, 3) for k, v in expected.items()},
    }

def assess_recommendation_quality(probabilities, prediction=None, context=None):
    """Classify narrow calls so the UI can avoid treating them as strong picks."""
    if not probabilities:
        return {
            'level': 'unknown',
            'label': '无数据',
            'confidence': 0,
            'lead': 0,
            'top2': [],
            'advice': '跳过',
            'avoid_single': True,
        }

    ranked = sorted(
        ((str(k), float(v)) for k, v in probabilities.items() if v is not None),
        key=lambda x: -x[1]
    )
    if not ranked:
        return {
            'level': 'unknown',
            'label': '无数据',
            'confidence': 0,
            'lead': 0,
            'top2': [],
            'advice': '跳过',
            'avoid_single': True,
        }

    top_key, top_prob = ranked[0]
    if prediction and prediction != top_key:
        for key, prob in ranked:
            if key == prediction:
                top_key, top_prob = key, prob
                break

    second_prob = ranked[1][1] if len(ranked) > 1 else 0.0
    lead = top_prob - second_prob
    top2 = [{'option': key, 'probability': prob} for key, prob in ranked[:2]]

    context = context or {}
    asian_direction = context.get('asian_direction')
    score_consistency = context.get('score_consistency') or {}
    conflict = False
    if asian_direction in ('home_backing', 'away_laying') and top_key == '负':
        conflict = True
    elif asian_direction in ('away_backing', 'home_laying') and top_key == '胜':
        conflict = True
    if score_consistency.get('conflict'):
        conflict = True

    if top_prob >= 0.46 and lead >= 0.08 and not conflict:
        level, label, advice = 'strong', '强推荐', f'主推 {top_key}'
    elif top_prob >= 0.40 and lead >= 0.045 and not conflict:
        level, label, advice = 'medium', '可参考', f'主推 {top_key}'
    elif lead < 0.035 or conflict:
        level, label = 'split', '分歧较大'
        combo = '/'.join(item['option'] for item in top2)
        advice = f'建议双选 {combo}' if len(top2) >= 2 else f'谨慎 {top_key}'
    else:
        level, label, advice = 'low', '低置信', f'谨慎 {top_key}'

    return {
        'level': level,
        'label': label,
        'prediction': top_key,
        'confidence': round(top_prob, 6),
        'lead': round(lead, 6),
        'top2': top2,
        'advice': advice,
        'avoid_single': level in ('low', 'split', 'unknown'),
        'conflict': conflict,
        'score_consistency': score_consistency,
    }

def _score_result_label(score):
    if isinstance(score, dict):
        home_goals = score.get('home_goals')
        away_goals = score.get('away_goals')
        if home_goals is None or away_goals is None:
            score_text = str(score.get('score', ''))
        else:
            try:
                home_goals = int(home_goals)
                away_goals = int(away_goals)
                return '胜' if home_goals > away_goals else ('负' if home_goals < away_goals else '平')
            except (TypeError, ValueError):
                score_text = str(score.get('score', ''))
    else:
        score_text = str(score[0]) if score else ''

    if '-' not in score_text:
        return None
    left, right = score_text.split('-', 1)
    try:
        home_goals = int(left)
        away_goals = int(right)
    except ValueError:
        return None
    return '胜' if home_goals > away_goals else ('负' if home_goals < away_goals else '平')

def assess_score_consistency(scores, prediction):
    if not scores or not prediction:
        return {'available': False, 'conflict': False}

    weights = {'胜': 0.0, '平': 0.0, '负': 0.0}
    top_result = None
    total_weight = 0.0
    for idx, score in enumerate(scores[:3]):
        result = _score_result_label(score)
        if not result:
            continue
        if top_result is None:
            top_result = result
        if isinstance(score, dict):
            prob = score.get('probability')
        else:
            prob = score[1] if len(score) > 1 else None
        weight = float(prob) if prob is not None else max(0.1, 1.0 - idx * 0.25)
        weights[result] += weight
        total_weight += weight

    if total_weight <= 0:
        return {'available': False, 'conflict': False}

    agreement = weights.get(prediction, 0.0) / total_weight
    conflict = top_result != prediction and agreement < 0.45
    return {
        'available': True,
        'conflict': conflict,
        'top_score_result': top_result,
        'agreement': round(agreement, 6),
        'result_weights': {k: round(v / total_weight, 6) for k, v in weights.items()},
    }

def build_zjq_group_recommendation(zjq_probs):
    if not zjq_probs:
        return {'groups': [], 'primary': None}

    definitions = [
        ('small', '小球组', ['0', '1', '2']),
        ('middle', '中位组', ['2', '3']),
        ('big', '大球组', ['3', '4', '5', '6', '7+']),
    ]
    groups = []
    for key, label, options in definitions:
        prob = sum(float(zjq_probs.get(opt, 0) or 0) for opt in options)
        groups.append({
            'key': key,
            'label': label,
            'options': options,
            'probability': round(prob, 6),
            'advice': f"{label} {'/'.join(options)}",
        })

    groups.sort(key=lambda item: -item['probability'])
    primary = groups[0] if groups else None
    return {
        'groups': groups,
        'primary': primary,
    }

def _beidan_record_key(match):
    return '|'.join(str(match.get(k, '')) for k in ('date', 'num', 'home', 'away'))

def _load_beidan_history():
    data = kv_store.load(BEIDAN_HISTORY_KEY, [])
    return data if isinstance(data, list) else []

def _save_beidan_history(records):
    records = sorted(records, key=lambda r: r.get('created_at', ''), reverse=True)
    kv_store.save(BEIDAN_HISTORY_KEY, records[:BEIDAN_HISTORY_LIMIT])

def _compact_beidan_record(match, source):
    spf = match.get('spf') or {}
    zjq = match.get('zjq') or {}
    return {
        'key': _beidan_record_key(match),
        'source': source,
        'date': match.get('date'),
        'num': match.get('num'),
        'time': match.get('time'),
        'league': match.get('league'),
        'home': match.get('home'),
        'away': match.get('away'),
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'updated_at': datetime.now().isoformat(timespec='seconds'),
        'settled': False,
        'spf': {
            'prediction': spf.get('prediction'),
            'confidence': spf.get('confidence'),
            'quality': spf.get('quality'),
            'probabilities': spf.get('probabilities'),
            'score_consistency': spf.get('score_consistency'),
        },
        'zjq': {
            'prediction': zjq.get('prediction'),
            'confidence': zjq.get('confidence'),
            'quality': zjq.get('quality'),
            'goal_groups': zjq.get('goal_groups'),
            'probabilities': zjq.get('probabilities'),
        },
    }

def save_beidan_prediction_snapshot(result):
    if not isinstance(result, dict) or 'error' in result:
        return {'saved': 0, 'total': 0}

    records = _load_beidan_history()
    by_key = {r.get('key'): r for r in records if r.get('key')}
    saved = 0
    for match in result.get('recommendations') or []:
        key = _beidan_record_key(match)
        if not key.strip('|'):
            continue
        compact = _compact_beidan_record(match, result.get('source'))
        if key in by_key and by_key[key].get('settled'):
            compact['settled'] = True
            compact['actual'] = by_key[key].get('actual')
            compact['settlement'] = by_key[key].get('settlement')
            compact['created_at'] = by_key[key].get('created_at') or compact['created_at']
        by_key[key] = {**by_key.get(key, {}), **compact}
        saved += 1

    _save_beidan_history(list(by_key.values()))
    return {'saved': saved, 'total': len(by_key)}

def summarize_beidan_history(limit=200):
    records = _load_beidan_history()
    recent = sorted(records, key=lambda r: r.get('created_at', ''), reverse=True)[:limit]
    levels = {}
    for record in recent:
        for bet_type in ('spf', 'zjq'):
            level = ((record.get(bet_type) or {}).get('quality') or {}).get('level') or 'unknown'
            levels[level] = levels.get(level, 0) + 1

    settled = [r for r in recent if r.get('settled')]
    return {
        'total_records': len(records),
        'recent_records': len(recent),
        'settled_records': len(settled),
        'pending_records': len(recent) - len(settled),
        'quality_levels': levels,
        'latest': recent[:30],
    }

_ouzhi_cache = {}

def fetch_ouzhi_odds(match_id):
    match_id = str(match_id)
    if match_id in _ouzhi_cache:
        return _ouzhi_cache[match_id]
    
    url = f'{BASE_URL}/fenxi1/json/ouzhi.php?fid={match_id}&cid=0&type=europe&r=1'
    referer = f'{BASE_URL}/fenxi/ouzhi-{match_id}.shtml'
    
    try:
        series = fetch_json(url, referer=referer)
        if not isinstance(series, list) or len(series) == 0:
            _ouzhi_cache[match_id] = None
            return None
        
        close = series[0]
        if not isinstance(close, (list, tuple)) or len(close) < 3:
            _ouzhi_cache[match_id] = None
            return None
        
        result = {
            'home': float(close[0]),
            'draw': float(close[1]),
            'away': float(close[2]),
        }
        _ouzhi_cache[match_id] = result
        return result
    except Exception as e:
        log.warning(f"获取欧赔数据失败 match_id={match_id}: {e}")
        _ouzhi_cache[match_id] = None
        return None

def _clear_ouzhi_cache():
    global _ouzhi_cache
    _ouzhi_cache = {}

def analyze_spf(match, asian_data=None, cs_data=None):
    result = {
        'match_id': match['id'],
        'num': match['num'],
        'home': match['home'],
        'away': match['away'],
        'league': match['league'],
        'time': match['time'],
        'type': 'spf',
    }
    
    odds_data = fetch_ouzhi_odds(match['id'])
    
    if not odds_data:
        if match.get('spf_sp') and match.get('spf_s') and match.get('spf_f'):
            odds_data = {
                'home': match['spf_sp'],
                'draw': match['spf_s'],
                'away': match['spf_f'],
            }
            result['odds_source'] = 'okooo_main'
        else:
            result['error'] = '欧赔数据不可用'
            return result
    
    if odds_data:
        odds = {
            '胜': odds_data['home'],
            '平': odds_data['draw'],
            '负': odds_data['away'],
        }
        
        probs = calculate_implied_probability(odds)
        
        if probs:
            result['odds'] = odds
            result['margin'] = sum(probs.values()) - 1.0
            
            home_win_prob = probs.get('胜', 0.33)
            draw_prob = probs.get('平', 0.33)
            away_win_prob = probs.get('负', 0.34)
            
            if asian_data and asian_data.get('history'):
                home_win_prob, draw_prob, away_win_prob = adjust_probs_by_asian(
                    home_win_prob, draw_prob, away_win_prob, asian_data['history']
                )
                result['asian_adjusted'] = True

            model_probs = {
                '胜': home_win_prob,
                '平': draw_prob,
                '负': away_win_prob,
            }
            prob_total = sum(model_probs.values())
            if prob_total > 0:
                model_probs = {k: v / prob_total for k, v in model_probs.items()}

            model_probs, calibration_meta = apply_beidan_history_calibration(
                model_probs,
                'spf',
                league=match.get('league')
            )
            result['history_calibration'] = calibration_meta

            result['probabilities'] = model_probs
            result['raw_probabilities'] = probs
            result['prediction'] = max(model_probs, key=model_probs.get)
            result['confidence'] = model_probs[result['prediction']]
            
            score_prediction = predict_scores_by_poisson(
                home_win_prob,
                draw_prob,
                away_win_prob,
                league=match['league'],
                handicap=match.get('handicap', 0)
            )
            
            if cs_data and cs_data.get('history'):
                score_prediction = enhance_scores_with_cs(score_prediction, cs_data['history'])
                result['cs_adjusted'] = True
            
            result['scores'] = score_prediction['top3']
            result['lambda_home'] = score_prediction['lambda_home']
            result['lambda_away'] = score_prediction['lambda_away']
            
            if asian_data and asian_data.get('history'):
                result['asian_trend'] = analyze_asian_trend(asian_data['history'])
            
            quality_context = {}
            if result.get('asian_trend'):
                quality_context['asian_direction'] = result['asian_trend'].get('direction')
            result['score_consistency'] = assess_score_consistency(
                result['scores'],
                result['prediction']
            )
            quality_context['score_consistency'] = result['score_consistency']
            result['quality'] = assess_recommendation_quality(
                model_probs,
                result['prediction'],
                quality_context
            )
            
            if cs_data and cs_data.get('history'):
                result['cs_trend'] = analyze_cs_trend(cs_data['history'])
    else:
        result['error'] = '欧赔数据不可用'
    
    return result

def analyze_rqspf(match):
    result = {
        'match_id': match['id'],
        'num': match['num'],
        'home': match['home'],
        'away': match['away'],
        'league': match['league'],
        'time': match['time'],
        'handicap': match['handicap'],
        'type': 'rqspf',
    }
    
    handicap_value = parse_beidan_handicap(match.get('handicap'))
    if handicap_value is None:
        result['error'] = '让球值不可用，无法计算让球胜平负'
        return result

    odds_data = fetch_ouzhi_odds(match['id'])

    if not odds_data:
        if match.get('spf_sp') and match.get('spf_s') and match.get('spf_f'):
            odds_data = {
                'home': match['spf_sp'],
                'draw': match['spf_s'],
                'away': match['spf_f'],
            }
            result['odds_source'] = 'okooo_main'
        else:
            result['error'] = '欧赔数据不可用，无法计算让球胜平负'
            return result

    odds = {
        '胜': odds_data['home'],
        '平': odds_data['draw'],
        '负': odds_data['away'],
    }
    probs = calculate_implied_probability(odds)
    if not probs:
        result['error'] = '欧赔概率不可用，无法计算让球胜平负'
        return result

    score_prediction = predict_scores_by_poisson(
        probs.get('胜', 0.33),
        probs.get('平', 0.33),
        probs.get('负', 0.34),
        league=match.get('league', ''),
        handicap=handicap_value
    )
    rq_probs, rq_meta = rqspf_probs_from_score_probs(
        score_prediction['score_probs'],
        handicap_value
    )
    if not rq_probs:
        result['error'] = '让球胜平负概率计算失败'
        result['rqspf_meta'] = rq_meta
        return result

    result['spf_odds'] = odds
    result['odds'] = {}
    result['raw_spf_probabilities'] = probs
    result['probabilities'] = rq_probs
    result['prediction'] = max(rq_probs, key=rq_probs.get)
    result['confidence'] = rq_probs[result['prediction']]
    result['lambda_home'] = score_prediction['lambda_home']
    result['lambda_away'] = score_prediction['lambda_away']
    result['rqspf_meta'] = rq_meta
    result['quality'] = assess_recommendation_quality(
        rq_probs,
        result['prediction'],
        {}
    )
    result['scores'] = [
        {
            'score': item['score'],
            'handicap_score': item['handicap_score'],
            'result': item['result'],
            'probability': item['probability'],
        }
        for item in rq_meta.get('top_scores', [])
    ]
    
    return result

def analyze_bifen(match, bifen_odds):
    result = {
        'match_id': match['id'],
        'num': match['num'],
        'home': match['home'],
        'away': match['away'],
        'league': match['league'],
        'time': match['time'],
        'type': 'bifen',
    }
    
    if match['id'] not in bifen_odds:
        result['error'] = '比分数据不可用'
        return result
    
    odds = bifen_odds[match['id']]
    probs = calculate_implied_probability(odds)
    
    if probs:
        result['odds'] = odds
        result['probabilities'] = probs
        
        sorted_scores = sorted(probs.items(), key=lambda x: -x[1])
        result['top3'] = sorted_scores[:3]
        result['prediction'] = sorted_scores[0][0]
        result['confidence'] = sorted_scores[0][1]
    
    return result

def analyze_zjq(match, zjq_odds=None, asian_data=None, goals_data=None):
    result = {
        'match_id': match['id'],
        'num': match['num'],
        'home': match['home'],
        'away': match['away'],
        'league': match['league'],
        'time': match['time'],
        'type': 'zjq',
    }
    
    odds_data = fetch_ouzhi_odds(match['id'])
    
    if not odds_data:
        if match.get('spf_sp') and match.get('spf_s') and match.get('spf_f'):
            odds_data = {
                'home': match['spf_sp'],
                'draw': match['spf_s'],
                'away': match['spf_f'],
            }
            result['odds_source'] = 'okooo_main'
        else:
            result['error'] = '欧赔数据不可用，无法计算总进球'
            return result
    
    home_odds, draw_odds, away_odds = odds_data['home'], odds_data['draw'], odds_data['away']
    
    home_prob = 1 / home_odds
    draw_prob = 1 / draw_odds
    away_prob = 1 / away_odds
    total_prob = home_prob + draw_prob + away_prob
    
    home_prob_norm = home_prob / total_prob
    draw_prob_norm = draw_prob / total_prob
    away_prob_norm = away_prob / total_prob
    
    league_profile = LEAGUE_PROFILES.get(match.get('league'), {'avg_goals': 2.6, 'draw_rate': 0.27})
    avg_goals = league_profile['avg_goals']
    
    if asian_data and asian_data.get('history'):
        asian_factor = calculate_asian_goal_factor(asian_data['history'])
        avg_goals *= asian_factor
        result['asian_goal_factor'] = asian_factor
    
    if goals_data and goals_data.get('history'):
        goals_factor = calculate_goals_factor(goals_data['history'])
        avg_goals *= goals_factor
        result['goals_factor'] = goals_factor
    
    mu1 = avg_goals * (0.5 + 0.05 * (home_prob_norm - away_prob_norm))
    mu2 = avg_goals * (0.5 - 0.05 * (home_prob_norm - away_prob_norm))
    
    if mu1 < 0.1:
        mu1 = 0.1
    if mu2 < 0.1:
        mu2 = 0.1
    
    zjq_probs = {}
    for n in range(0, 8):
        if n == 0:
            prob = math.exp(-mu1 - mu2)
        elif n == 1:
            prob = (mu1 + mu2) * math.exp(-mu1 - mu2)
        elif n == 2:
            prob = (mu1**2 + 2*mu1*mu2 + mu2**2) * math.exp(-mu1 - mu2) / 2
        elif n == 3:
            prob = (mu1**3 + 3*mu1**2*mu2 + 3*mu1*mu2**2 + mu2**3) * math.exp(-mu1 - mu2) / 6
        elif n == 4:
            prob = (mu1**4 + 4*mu1**3*mu2 + 6*mu1**2*mu2**2 + 4*mu1*mu2**3 + mu2**4) * math.exp(-mu1 - mu2) / 24
        elif n == 5:
            prob = math.exp(-mu1 - mu2) * sum(mu1**(5-i) * mu2**i / (math.factorial(5-i) * math.factorial(i)) for i in range(6))
        elif n == 6:
            prob = math.exp(-mu1 - mu2) * sum(mu1**(6-i) * mu2**i / (math.factorial(6-i) * math.factorial(i)) for i in range(7))
        else:
            prob = max(0, 1 - sum(zjq_probs.values()))
        
        zjq_probs[str(n)] = prob if n < 7 else prob
    
    zjq_probs['7+'] = max(0, 1 - sum(zjq_probs.get(str(i), 0) for i in range(7)))
    
    if goals_data and goals_data.get('history'):
        zjq_probs = adjust_zjq_by_goals(zjq_probs, goals_data['history'])
        result['goals_adjusted'] = True

    market_zjq_odds = (zjq_odds or {}).get(match['id']) if isinstance(zjq_odds, dict) else None
    if market_zjq_odds:
        market_probs = calculate_implied_probability(market_zjq_odds)
        if market_probs:
            blended = {}
            for key in set(zjq_probs) | set(market_probs):
                model_prob = zjq_probs.get(key, 0)
                market_prob = market_probs.get(key, 0)
                blended[key] = model_prob * 0.55 + market_prob * 0.45
            zjq_probs = blended
            result['odds'] = market_zjq_odds
            result['market_probabilities'] = market_probs
            result['market_adjusted'] = True
    
    total_zjq = sum(zjq_probs.values())
    if total_zjq > 0:
        zjq_probs = {k: v / total_zjq for k, v in zjq_probs.items()}

    zjq_probs, calibration_meta = apply_beidan_history_calibration(
        zjq_probs,
        'zjq',
        league=match.get('league')
    )
    result['history_calibration'] = calibration_meta
    
    result['probabilities'] = zjq_probs
    result['mu_home'] = mu1
    result['mu_away'] = mu2
    
    sorted_counts = sorted(zjq_probs.items(), key=lambda x: -x[1])
    result['top3'] = sorted_counts[:3]
    result['prediction'] = sorted_counts[0][0]
    result['confidence'] = sorted_counts[0][1]
    result['quality'] = assess_recommendation_quality(
        zjq_probs,
        result['prediction'],
        {}
    )
    result['goal_groups'] = build_zjq_group_recommendation(zjq_probs)
    
    over25_prob = sum(zjq_probs.get(str(i), 0) for i in [3, 4, 5, 6]) + zjq_probs.get('7+', 0)
    result['over25_prob'] = over25_prob
    result['under25_prob'] = 1 - over25_prob
    
    if goals_data and goals_data.get('history'):
        result['goals_trend'] = analyze_goals_trend(goals_data['history'])
    
    return result

def analyze_bqc(match, bqc_odds):
    result = {
        'match_id': match['id'],
        'num': match['num'],
        'home': match['home'],
        'away': match['away'],
        'league': match['league'],
        'time': match['time'],
        'type': 'bqc',
    }
    
    if match['id'] not in bqc_odds:
        result['error'] = '半全场数据不可用'
        return result
    
    odds = bqc_odds[match['id']]
    probs = calculate_implied_probability(odds)
    
    if probs:
        result['odds'] = odds
        result['probabilities'] = probs
        
        sorted_results = sorted(probs.items(), key=lambda x: -x[1])
        result['top3'] = sorted_results[:3]
        result['prediction'] = sorted_results[0][0]
        result['confidence'] = sorted_results[0][1]
        
        half_probs = {}
        full_probs = {}
        for key, prob in probs.items():
            half = key[0]
            full = key[1]
            half_probs[half] = half_probs.get(half, 0) + prob
            full_probs[full] = full_probs.get(full, 0) + prob
        
        result['half_probabilities'] = half_probs
        result['full_probabilities'] = full_probs
    
    return result

def _candidate_beidan_dates(date, allow_fallback=True, days=2):
    dates = [date]
    if not allow_fallback:
        return dates
    try:
        base = datetime.strptime(date, '%Y-%m-%d')
    except (TypeError, ValueError):
        return dates
    for offset in range(1, days + 1):
        candidate = base + timedelta(days=offset)
        dates.append(f'{candidate.year:04d}-{candidate.month:02d}-{candidate.day:02d}')
    return dates

def _fetch_beidan_matches_with_fallback(date, source, allow_date_fallback=True):
    sources = [source]
    if source != 'okooo':
        sources.append('okooo')

    attempts = []
    for candidate_source in sources:
        for candidate_date in _candidate_beidan_dates(date, allow_date_fallback):
            if candidate_source == 'okooo':
                matches = fetch_okooo_schedule(candidate_date)
            else:
                matches = fetch_beidan_schedule(candidate_date, source=candidate_source)

            attempts.append({
                'source': candidate_source,
                'date': candidate_date,
                'match_count': len(matches),
            })
            if matches:
                return matches, {
                    'requested_source': source,
                    'requested_date': date,
                    'source': candidate_source,
                    'date': candidate_date,
                    'source_fallback': candidate_source != source,
                    'date_fallback': candidate_date != date,
                    'attempts': attempts,
                }

    log.warning(f"所有数据源均返回0场比赛，尝试重置okooo session并重试")
    _okooo_session = requests.Session()
    _okooo_session.headers.update(OKOOO_HEADERS)
    _okooo_session.verify = False
    global _okooo_waf_blocked
    _okooo_waf_blocked = False
    
    for candidate_date in _candidate_beidan_dates(date, allow_date_fallback):
        matches = fetch_okooo_schedule(candidate_date)
        attempts.append({
            'source': 'okooo_retry',
            'date': candidate_date,
            'match_count': len(matches),
            'retry': True,
        })
        if matches:
            log.info(f"okooo重试成功，获取到{len(matches)}场比赛")
            return matches, {
                'requested_source': source,
                'requested_date': date,
                'source': 'okooo',
                'date': candidate_date,
                'source_fallback': True,
                'date_fallback': candidate_date != date,
                'attempts': attempts,
                'retry_success': True,
            }

    log.error(f"所有数据源均失败，返回空结果。尝试记录: {attempts}")
    return [], {
        'requested_source': source,
        'requested_date': date,
        'source': source,
        'date': date,
        'source_fallback': False,
        'date_fallback': False,
        'attempts': attempts,
    }

def generate_beidan_recommendations(date=None, bet_types=None, source='okooo', save_history=True):
    if bet_types is None:
        bet_types = ['spf', 'rqspf']
    
    allow_date_fallback = not date
    date = date or time.strftime('%Y-%m-%d')
    matches, match_meta = _fetch_beidan_matches_with_fallback(
        date,
        source,
        allow_date_fallback=allow_date_fallback
    )
    
    if not matches:
        return {
            'error': '未获取到比赛数据',
            'date': date,
            'source': source,
            'attempts': match_meta.get('attempts', []),
        }
    date = match_meta.get('date', date)
    source = match_meta.get('source', source)
    
    bifen_odds = {}
    zjq_odds = {}
    bqc_odds = {}
    
    if 'bifen' in bet_types and source != 'okooo':
        bifen_odds = fetch_beidan_bifen(date)
    if 'zjq' in bet_types and source != 'okooo':
        zjq_odds = fetch_beidan_zjq(date)
    if 'bqc' in bet_types and source != 'okooo':
        bqc_odds = fetch_beidan_bqc(date)
    
    _clear_ouzhi_cache()
    
    match_ids = [str(m['id']) for m in matches]
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        ouzhi_futures = {executor.submit(fetch_ouzhi_odds, mid): mid for mid in match_ids}
        for future in as_completed(ouzhi_futures):
            pass
    
    match_odds_cache = {mid: fetch_ouzhi_odds(mid) for mid in match_ids}
    
    actual_source = matches[0].get('source', source) if matches else source
    if actual_source == 'okooo' and bet_types:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            for match in matches:
                if 'spf' in bet_types or 'zjq' in bet_types:
                    futures[executor.submit(fetch_okooo_asian_history, match['id'])] = ('asian', match['id'])
                if 'zjq' in bet_types:
                    futures[executor.submit(fetch_okooo_goals_history, match['id'])] = ('goals', match['id'])
                if 'spf' in bet_types:
                    futures[executor.submit(fetch_okooo_cs_history, match['id'])] = ('cs', match['id'])
            
            asian_cache = {}
            goals_cache = {}
            cs_cache = {}
            
            for future in as_completed(futures):
                data_type, match_id = futures[future]
                try:
                    data = future.result()
                    if data_type == 'asian':
                        asian_cache[match_id] = data
                    elif data_type == 'goals':
                        goals_cache[match_id] = data
                    elif data_type == 'cs':
                        cs_cache[match_id] = data
                except Exception as e:
                    log.warning(f"并行获取数据失败 {data_type} {match_id}: {e}")
    else:
        asian_cache = {}
        goals_cache = {}
        cs_cache = {}
    
    recommendations = []
    
    for match in matches:
        rec = {
            'num': match['num'],
            'date': match['date'],
            'time': match['time'],
            'league': match['league'],
            'home': match['home'],
            'away': match['away'],
            'handicap': match['handicap'],
        }
        
        asian_data = asian_cache.get(match['id'])
        goals_data = goals_cache.get(match['id'])
        cs_data = cs_cache.get(match['id'])
        
        if asian_data and asian_data.get('history'):
            rec['asian'] = asian_data
        if goals_data and goals_data.get('history'):
            rec['goals'] = goals_data
        if cs_data and cs_data.get('history'):
            rec['cs'] = cs_data
        
        if 'spf' in bet_types:
            rec['spf'] = analyze_spf(match, asian_data, cs_data)
        
        if 'rqspf' in bet_types:
            rec['rqspf'] = analyze_rqspf(match)
        
        if 'bifen' in bet_types:
            rec['bifen'] = analyze_bifen(match, bifen_odds)
        
        if 'zjq' in bet_types:
            rec['zjq'] = analyze_zjq(match, zjq_odds, asian_data, goals_data)
        
        if 'bqc' in bet_types:
            rec['bqc'] = analyze_bqc(match, bqc_odds)
        
        recommendations.append(rec)
    
    result = {
        'date': date,
        'total_matches': len(matches),
        'pending_matches': len(recommendations),
        'recommendations': recommendations,
        'source': source,
        'match_fetch': match_meta,
        'history_summary': summarize_beidan_history(limit=200),
    }
    if save_history:
        result['history_save'] = save_beidan_prediction_snapshot(result)
        result['history_summary'] = summarize_beidan_history(limit=200)
    return result

def find_value_bets(date=None, threshold=0.05, source='okooo'):
    result = generate_beidan_recommendations(date, bet_types=['spf', 'rqspf', 'zjq'], source=source)
    
    if 'error' in result:
        return result
    
    value_bets = []
    
    for match in result['recommendations']:
        for bet_type in ['spf', 'rqspf', 'zjq']:
            if bet_type not in match:
                continue
            
            data = match[bet_type]
            if 'probabilities' not in data:
                continue
            
            probs = data['probabilities']
            
            if bet_type == 'zjq':
                odds_map = data.get('odds') or {}
                for key, prob in probs.items():
                    if key == '7+':
                        continue
                    odd = odds_map.get(key)
                    if odd and odd > 0:
                        implied_prob = 1 / odd
                        edge = prob - implied_prob
                        if edge > threshold:
                            value_bets.append({
                                'num': match['num'],
                                'home': match['home'],
                                'away': match['away'],
                                'type': bet_type,
                                'option': key,
                                'probability': prob,
                                'odd': odd,
                                'implied_probability': implied_prob,
                                'edge': edge,
                            })
            else:
                for key, prob in probs.items():
                    odd = data['odds'].get(key)
                    if odd and odd > 0:
                        implied_prob = 1 / odd
                        edge = prob - implied_prob
                        if edge > threshold:
                            value_bets.append({
                                'num': match['num'],
                                'home': match['home'],
                                'away': match['away'],
                                'type': bet_type,
                                'option': key,
                                'probability': prob,
                                'odd': odd,
                                'implied_probability': implied_prob,
                                'edge': edge,
                            })
    
    return {
        'date': result['date'],
        'total_matches': result['total_matches'],
        'value_bets': sorted(value_bets, key=lambda x: -x['edge']),
    }

def print_recommendations(result):
    if 'error' in result:
        print(f"错误: {result['error']}")
        return
    
    print(f"📅 北单推荐 ({result['date']})")
    print(f"场次总数: {result['total_matches']}, 未开赛: {result['pending_matches']}")
    print("=" * 80)
    
    for match in result['recommendations']:
        print(f"\n⚽ [{match['num']}] {match['league']}")
        print(f"   {match['home']} VS {match['away']}")
        time_display = match['time'] if match['time'] else match['date']
        print(f"   时间: {time_display}")
        
        if match.get('handicap'):
            print(f"   让球: {'主队让' if match['handicap'] > 0 else '客队让'} {abs(match['handicap'])}球")
        
        for bet_type in ['spf', 'rqspf', 'bifen', 'zjq', 'bqc']:
            if bet_type not in match:
                continue
            
            data = match[bet_type]
            if 'error' in data:
                continue
            
            name = BET_TYPES.get(bet_type, {}).get('name', bet_type)
            print(f"\n   🎯 {name}:")
            
            if bet_type == 'bifen':
                for score, prob in data.get('top3', []):
                    odd = data['odds'].get(score)
                    print(f"      {score}: 概率 {prob:.2%}, 赔率 {odd}")
            elif bet_type == 'zjq':
                for count, prob in data.get('top3', []):
                    odd = data['odds'].get(count) if 'odds' in data else None
                    print(f"      {count}球: 概率 {prob:.2%}")
                print(f"      大球(>2.5): {data.get('over25_prob', 0):.2%}")
                print(f"      小球(<2.5): {data.get('under25_prob', 0):.2%}")
            elif bet_type == 'bqc':
                for bqc, prob in data.get('top3', []):
                    odd = data['odds'].get(bqc)
                    print(f"      {bqc}: 概率 {prob:.2%}, 赔率 {odd}")
            else:
                for key, prob in data.get('probabilities', {}).items():
                    odd = data['odds'].get(key)
                    print(f"      {key}: 概率 {prob:.2%}, 赔率 {odd}")
                print(f"      推荐: {data.get('prediction')} (置信度 {data.get('confidence', 0):.2%})")
        
        print("-" * 60)

def main():
    print("=" * 80)
    print("              北单足球彩票分析系统")
    print("=" * 80)
    
    source = input("\n请选择数据源 (dc=竞彩单场, jczq=竞彩足球): ").strip()
    if not source:
        source = 'dc'
    if source not in ['dc', 'jczq']:
        source = 'dc'
    
    date = input("请输入日期(格式: YYYY-MM-DD, 回车为今天): ").strip()
    if not date:
        date = time.strftime('%Y-%m-%d')
    
    print(f"\n正在获取 {date} 的{'竞彩单场' if source == 'dc' else '竞彩足球'}数据...")
    
    result = generate_beidan_recommendations(date, bet_types=['spf', 'zjq'], source=source)
    
    print_recommendations(result)

if __name__ == '__main__':
    main()
