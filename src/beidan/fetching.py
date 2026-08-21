# -*- coding: utf-8 -*-
"""北单数据抓取：500.com 与 okooo 页面、赛程"""

import sys
import math
import re
from collections import defaultdict
import time
import json
import urllib.request
import urllib.error
import random
import requests

from src.football.fetching import (
    FETCH_THROTTLE_SECONDS, RATE_LIMIT_STATUSES,
    _await_fetch_throttle, _await_rate_slot, _enter_fetch_throttle,
)
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from ..common.logger import setup_logger
from ..common.paths import data_path
from ..common import kv_store

log = setup_logger('beidan')

from .config import (
    DC_SCHEDULE_URL, HEADERS, OKOOO_DANCHANG_URL, OKOOO_HEADERS, OKOOO_MATCH_URL, SCHEDULE_URL, _is_okooo_waf_blocked, _mark_okooo_waf_blocked, _okooo_session,
)

# 北单与足球打的是同一个 odds.500.com。两边各跑一条预热线程，若各自独立发请求，
# 就会互相挤占源站配额、把对方推进 429。这里让 500.com 的请求共用足球那条限速
# 令牌流：请求前领号，撞限流则写进全局冷却，足球侧也会一起退避。
# okooo 走另一个域名和独立 session，不参与这条预算。
_SHARED_RATE_LIMIT_HOST = 'odds.500.com'


def _shares_football_rate_budget(url):
    return _SHARED_RATE_LIMIT_HOST in (url or '')


def fetch(url, encoding='utf-8', referer=None):
    headers = {**HEADERS, 'Referer': referer} if referer else HEADERS
    req = urllib.request.Request(url, headers=headers)
    throttled = _shares_football_rate_budget(url)
    if throttled:
        _await_fetch_throttle()
        _await_rate_slot()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        if throttled and e.code in RATE_LIMIT_STATUSES:
            _enter_fetch_throttle(FETCH_THROTTLE_SECONDS + random.uniform(0, 0.3))
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
            
            # 快速失败后走现有备用数据源，避免单个站点拖住整批预测。
            resp = _okooo_session.get(url, headers=headers, timeout=(5, 12))
            
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
            rqspf_sp = float(odds_pattern[3]) if len(odds_pattern) >= 6 else None
            rqspf_s = float(odds_pattern[4]) if len(odds_pattern) >= 6 else None
            rqspf_f = float(odds_pattern[5]) if len(odds_pattern) >= 6 else None
            
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
                'rqspf_sp': rqspf_sp,
                'rqspf_s': rqspf_s,
                'rqspf_f': rqspf_f,
                'rqspf_odds': (
                    {'让胜': rqspf_sp, '让平': rqspf_s, '让负': rqspf_f}
                    if all(value and value > 1.0 for value in (rqspf_sp, rqspf_s, rqspf_f))
                    else None
                ),
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
    log.debug(f"抓取okooo亚盘赔率变化: match_id={match_id}")
    
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
    
    log.debug(f"未获取到亚盘赔率变化数据")
    return {'history': []}


def fetch_okooo_goals_history(match_id):
    urls = [
        f'{OKOOO_MATCH_URL}{match_id}/goals/',
    ]
    log.debug(f"抓取okooo总进球赔率变化: match_id={match_id}")
    
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
    
    log.debug(f"未获取到总进球赔率变化数据")
    return {'history': []}


def fetch_okooo_cs_history(match_id):
    urls = [
        f'{OKOOO_MATCH_URL}{match_id}/cs/',
    ]
    log.debug(f"抓取okooo比分赔率变化: match_id={match_id}")
    
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
    
    log.debug(f"未获取到比分赔率变化数据")
    return {'history': []}


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


