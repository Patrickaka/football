#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
澳客竞彩篮球数据抓取与各家赔率分析
================================
赛程页：https://www.okooo.com/jingcailanqiu/hunhe/
明细页：/basketball/match/{id}/odds|ah|ou/

设计对齐北单澳客部分：
- session + WAF 降级
- 主盘口 SP + 各家共识
- 盘口/水位走势修正概率
"""

from __future__ import annotations

import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests

from ..common.logger import setup_logger

log = setup_logger('basketball.okooo')

OKOOO_BASE = 'https://www.okooo.com'
OKOOO_HUNHE_URL = f'{OKOOO_BASE}/jingcailanqiu/hunhe/'
OKOOO_MATCH_URL = f'{OKOOO_BASE}/basketball/match/'

OKOOO_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
    'Upgrade-Insecure-Requests': '1',
    'Referer': OKOOO_HUNHE_URL,
    'Host': 'www.okooo.com',
}

_okooo_session = requests.Session()
_okooo_session.headers.update(OKOOO_HEADERS)
_okooo_session.verify = False

_okooo_waf_blocked = False
_okooo_waf_blocked_time = 0.0

_market_cache: Dict[str, dict] = {}


def _mark_okooo_waf_blocked():
    global _okooo_waf_blocked, _okooo_waf_blocked_time
    _okooo_waf_blocked = True
    _okooo_waf_blocked_time = time.time()


def _is_okooo_waf_blocked() -> bool:
    global _okooo_waf_blocked, _okooo_waf_blocked_time
    if not _okooo_waf_blocked:
        return False
    if time.time() - _okooo_waf_blocked_time > 60:
        _okooo_waf_blocked = False
        return False
    return True


def _reset_okooo_session():
    global _okooo_session
    _okooo_session = requests.Session()
    _okooo_session.headers.update(OKOOO_HEADERS)
    _okooo_session.verify = False


def _init_okooo_session():
    if _is_okooo_waf_blocked():
        return
    try:
        _okooo_session.get(OKOOO_BASE + '/', timeout=10)
        time.sleep(0.3)
        _okooo_session.get(OKOOO_HUNHE_URL, timeout=10)
    except Exception as e:
        log.warning(f"初始化澳客篮球 session 失败: {e}")


try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

_init_okooo_session()


def fetch_okooo(url: str, referer: Optional[str] = None, max_retries: int = 2) -> Optional[str]:
    """抓取澳客页面（gbk），WAF 时返回 None。"""
    if _is_okooo_waf_blocked():
        return None

    for attempt in range(max_retries):
        try:
            headers = {}
            if referer:
                headers['Referer'] = referer
            # 篮球按比赛批量抓取；单场快速失败比阻塞整批更安全。
            resp = _okooo_session.get(url, headers=headers, timeout=(5, 12))

            if resp.status_code in (403, 503):
                log.warning(f"WAF拦截 {resp.status_code} for {url}")
                _mark_okooo_waf_blocked()
                _reset_okooo_session()
                return None

            if resp.status_code != 200:
                if resp.status_code >= 500 and attempt < max_retries - 1:
                    time.sleep(1.5)
                    continue
                return None

            try:
                resp.encoding = 'gb2312'
                result = resp.text
            except Exception:
                result = resp.content.decode('gb2312', errors='replace')

            if 'aliyun_waf' in result and '<title></title>' in result:
                log.warning(f"WAF页面 for {url}")
                _mark_okooo_waf_blocked()
                _reset_okooo_session()
                return None
            return result
        except requests.RequestException as e:
            log.warning(f"澳客请求失败 {url}: {e} ({attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(1.5)
    return None


def _strip_html(text: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', text or '')).strip()


def _safe_float(val) -> Optional[float]:
    try:
        if val is None or val == '':
            return None
        return float(str(val).replace(',', '').strip())
    except (TypeError, ValueError):
        return None


def parse_rflist(rflist: str) -> List[Dict]:
    """
    解析盘口历史：'07/14 11:47 1.75 (1.5) 1.65,07/14 10:17 1.60 (-1.5) 1.81'
    页面通常新→旧，返回按时间升序。
    """
    if not rflist:
        return []
    entries = []
    for part in str(rflist).split(','):
        part = part.strip()
        m = re.match(
            r'(\d{2}/\d{2})\s+(\d{2}:\d{2})\s+([\d.]+)\s+\(([+\-]?\d+(?:\.\d+)?)\)\s+([\d.]+)',
            part,
        )
        if not m:
            continue
        entries.append({
            'date': m.group(1),
            'time': m.group(2),
            'home_odds': float(m.group(3)),
            'line': float(m.group(4)),
            'away_odds': float(m.group(5)),
        })
    entries.reverse()
    return entries


def analyze_line_trend(history: List[Dict], kind: str = 'ah') -> Dict:
    """盘口走势：水位缩短=资金偏该侧。"""
    if not history or len(history) < 2:
        return {'direction': 'stable', 'strength': 0.0, 'kind': kind}

    first, last = history[0], history[-1]
    home_move = (last.get('home_odds') or 0) - (first.get('home_odds') or 0)
    away_move = (last.get('away_odds') or 0) - (first.get('away_odds') or 0)
    line_move = (last.get('line') or 0) - (first.get('line') or 0)
    strength = abs(home_move) + abs(away_move) + abs(line_move) * 0.05

    if home_move < -0.03 and away_move > 0.01:
        direction = 'home_backing'
    elif away_move < -0.03 and home_move > 0.01:
        direction = 'away_backing'
    elif kind == 'ou' and home_move < -0.03:
        direction = 'over_backing'
    elif kind == 'ou' and away_move < -0.03:
        direction = 'under_backing'
    elif abs(line_move) >= 1.0:
        direction = 'line_up' if line_move > 0 else 'line_down'
    else:
        direction = 'stable'

    return {
        'direction': direction,
        'strength': round(strength, 4),
        'home_move': round(home_move, 4),
        'away_move': round(away_move, 4),
        'line_move': round(line_move, 4),
        'kind': kind,
        'samples': len(history),
        'opening_line': first.get('line'),
        'current_line': last.get('line'),
    }


def adjust_two_way_by_trend(p_home: float, p_away: float, trend: Optional[Dict],
                            factor: float = 0.12) -> Tuple[float, float]:
    """按资金流向微调双边概率。"""
    if not trend or trend.get('direction') in (None, 'stable'):
        return p_home, p_away

    direction = trend['direction']
    strength = min(1.0, float(trend.get('strength') or 0) / 0.2)
    adj = factor * (0.5 + 0.5 * strength)

    if direction in ('home_backing', 'over_backing'):
        p_home *= (1 + adj)
        p_away *= (1 - adj * 0.6)
    elif direction in ('away_backing', 'under_backing'):
        p_away *= (1 + adj)
        p_home *= (1 - adj * 0.6)
    elif direction == 'line_up' and trend.get('kind') == 'ah':
        # handicap 是加在主队一侧的数值；数值上升代表客队方向增强。
        p_away *= (1 + adj * 0.5)
        p_home *= (1 - adj * 0.3)
    elif direction == 'line_down' and trend.get('kind') == 'ah':
        # 例如 -3.5 -> -5.5：主队让深，代表主队方向增强。
        p_home *= (1 + adj * 0.5)
        p_away *= (1 - adj * 0.3)

    total = p_home + p_away + 1e-9
    return p_home / total, p_away / total


def _parse_average_row(html: str, kind: str) -> Dict:
    """解析页脚平均值。"""
    out = {}
    m = re.search(
        r'平均值</td>\s*'
        r'<td[^>]*>\s*([+\-]?\d+(?:\.\d+)?)\s*</td>\s*'
        r'<td[^>]*>\s*([+\-]?\d+(?:\.\d+)?|\s*[+\-]?\d+(?:\.\d+)?\s*)\s*</td>\s*'
        r'<td[^>]*>\s*([+\-]?\d+(?:\.\d+)?)\s*</td>',
        html,
        re.S,
    )
    # ML 平均值：初主 初客 即主 即客
    if kind == 'ml':
        m2 = re.search(
            r'平均值</td>\s*'
            r'<td[^>]*>\s*([\d.]+)\s*</td>\s*'
            r'<td[^>]*>\s*([\d.]+)\s*</td>\s*'
            r'<td[^>]*>\s*([\d.]+)\s*</td>\s*'
            r'<td[^>]*>\s*([\d.]+)\s*</td>',
            html,
            re.S,
        )
        if m2:
            out = {
                'home_init': float(m2.group(1)),
                'away_init': float(m2.group(2)),
                'home': float(m2.group(3)),
                'away': float(m2.group(4)),
            }
        return out

    # AH/OU 行结构不完全一致，优先从各家行取均值
    return out


def _parse_book_rows(html: str, kind: str) -> List[Dict]:
    """从各家赔率表解析盘口行。"""
    books = []
    change_path = {'ml': 'odds', 'ah': 'ah', 'ou': 'ou'}[kind]
    trs = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.S)
    for tr in trs:
        if f'/{change_path}/change/' not in tr and f'/{change_path}/handicap/' not in tr:
            # ou pages often use /ou/change/
            if f'/{change_path}/' not in tr or 'change/' not in tr:
                continue
        cid_m = re.search(rf'/{change_path}/(?:change|handicap|line)/(\d+)/', tr)
        spans = [_safe_float(x) for x in re.findall(r'<span>([^<]+)</span>', tr)]
        nums = [x for x in spans if x is not None]
        title_m = re.search(r'<span title="([^"]+)"', tr)
        name = title_m.group(1) if title_m else ''

        if kind == 'ml':
            if len(nums) < 4:
                continue
            books.append({
                'company_id': cid_m.group(1) if cid_m else '',
                'name': name,
                'home_init': nums[0],
                'away_init': nums[1],
                'home': nums[2],
                'away': nums[3],
            })
        elif kind == 'ah':
            # 初：主/盘/客，即：主/盘/客
            plain_nums = [_safe_float(x) for x in re.findall(
                r'>([+\-]?\d+(?:\.\d+)?)<', tr
            )]
            plain_nums = [x for x in plain_nums if x is not None]
            # 过滤序号等：找含半盘/整数盘的切片
            if len(plain_nums) < 6:
                continue
            # 跳过左侧序号等，找第一个像赔率(~1.x-3.x)再跟盘口的位置
            start = 0
            for i in range(len(plain_nums) - 5):
                a, b, c = plain_nums[i], plain_nums[i + 1], plain_nums[i + 2]
                if 1.01 <= a <= 5.0 and abs(b) <= 40 and 1.01 <= c <= 5.0:
                    start = i
                    break
            chunk = plain_nums[start:start + 6]
            if len(chunk) < 6:
                continue
            books.append({
                'company_id': cid_m.group(1) if cid_m else '',
                'name': name,
                'home_init': chunk[0],
                'line_init': chunk[1],
                'away_init': chunk[2],
                'home': chunk[3],
                'line': chunk[4],
                'away': chunk[5],
            })
        else:  # ou
            plain_nums = [_safe_float(x) for x in re.findall(
                r'>([+\-]?\d+(?:\.\d+)?)<', tr
            )]
            plain_nums = [x for x in plain_nums if x is not None]
            if len(plain_nums) < 6:
                continue
            start = 0
            for i in range(len(plain_nums) - 5):
                a, b, c = plain_nums[i], plain_nums[i + 1], plain_nums[i + 2]
                if 1.01 <= a <= 5.0 and 100 <= b <= 280 and 1.01 <= c <= 5.0:
                    start = i
                    break
            chunk = plain_nums[start:start + 6]
            if len(chunk) < 6:
                continue
            books.append({
                'company_id': cid_m.group(1) if cid_m else '',
                'name': name,
                'over_init': chunk[0],
                'line_init': chunk[1],
                'under_init': chunk[2],
                'over': chunk[3],
                'line': chunk[4],
                'under': chunk[5],
            })
    return books


def _consensus_from_books(books: List[Dict], kind: str) -> Dict:
    if not books:
        return {'available': False, 'book_count': 0}

    def avg(key):
        vals = [b[key] for b in books if b.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    if kind == 'ml':
        home, away = avg('home'), avg('away')
        home_init, away_init = avg('home_init'), avg('away_init')
        out = {
            'available': True,
            'book_count': len(books),
            'home': round(home, 3) if home else None,
            'away': round(away, 3) if away else None,
            'home_init': round(home_init, 3) if home_init else None,
            'away_init': round(away_init, 3) if away_init else None,
            'home_move': round(home - home_init, 4) if home and home_init else 0.0,
            'away_move': round(away - away_init, 4) if away and away_init else 0.0,
        }
        if home and away and home > 0 and away > 0:
            p_h, p_a = 1 / home, 1 / away
            tot = p_h + p_a
            out['home_prob'] = round(p_h / tot, 4)
            out['away_prob'] = round(p_a / tot, 4)
        history = []
        if home_init and away_init:
            history.append({'home_odds': home_init, 'away_odds': away_init, 'line': 0})
        if home and away:
            history.append({'home_odds': home, 'away_odds': away, 'line': 0})
        out['trend'] = analyze_line_trend(history, 'ml')
        return out

    if kind == 'ah':
        home, away, line = avg('home'), avg('away'), avg('line')
        home_init, away_init, line_init = avg('home_init'), avg('away_init'), avg('line_init')
        out = {
            'available': True,
            'book_count': len(books),
            'home': round(home, 3) if home else None,
            'away': round(away, 3) if away else None,
            'line': round(line, 2) if line is not None else None,
            'line_init': round(line_init, 2) if line_init is not None else None,
            'home_move': round(home - home_init, 4) if home and home_init else 0.0,
            'away_move': round(away - away_init, 4) if away and away_init else 0.0,
            'line_move': round(line - line_init, 2) if line is not None and line_init is not None else 0.0,
        }
        if home and away and home > 0 and away > 0:
            p_h, p_a = 1 / home, 1 / away
            tot = p_h + p_a
            out['home_prob'] = round(p_h / tot, 4)
            out['away_prob'] = round(p_a / tot, 4)
        history = []
        if home_init and away_init and line_init is not None:
            history.append({'home_odds': home_init, 'away_odds': away_init, 'line': line_init})
        if home and away and line is not None:
            history.append({'home_odds': home, 'away_odds': away, 'line': line})
        out['trend'] = analyze_line_trend(history, 'ah')
        return out

    # ou
    over, under, line = avg('over'), avg('under'), avg('line')
    over_init, under_init, line_init = avg('over_init'), avg('under_init'), avg('line_init')
    out = {
        'available': True,
        'book_count': len(books),
        'over': round(over, 3) if over else None,
        'under': round(under, 3) if under else None,
        'line': round(line, 2) if line is not None else None,
        'line_init': round(line_init, 2) if line_init is not None else None,
        'over_move': round(over - over_init, 4) if over and over_init else 0.0,
        'under_move': round(under - under_init, 4) if under and under_init else 0.0,
        'line_move': round(line - line_init, 2) if line is not None and line_init is not None else 0.0,
    }
    if over and under and over > 0 and under > 0:
        p_o, p_u = 1 / over, 1 / under
        tot = p_o + p_u
        out['over_prob'] = round(p_o / tot, 4)
        out['under_prob'] = round(p_u / tot, 4)
    history = []
    if over_init and under_init and line_init is not None:
        history.append({'home_odds': over_init, 'away_odds': under_init, 'line': line_init})
    if over and under and line is not None:
        history.append({'home_odds': over, 'away_odds': under, 'line': line})
    out['trend'] = analyze_line_trend(history, 'ou')
    return out


def fetch_match_market_bundle(match_id: str, use_cache: bool = True) -> Dict:
    """抓取单场各家欧赔/让分/大小共识。"""
    match_id = str(match_id)
    if use_cache and match_id in _market_cache:
        return _market_cache[match_id]

    bundle = {
        'match_id': match_id,
        'ml': {'available': False},
        'ah': {'available': False},
        'ou': {'available': False},
    }

    pages = {
        'ml': f'{OKOOO_MATCH_URL}{match_id}/odds/',
        'ah': f'{OKOOO_MATCH_URL}{match_id}/ah/',
        'ou': f'{OKOOO_MATCH_URL}{match_id}/ou/',
    }

    for kind, url in pages.items():
        html = fetch_okooo(url, referer=OKOOO_HUNHE_URL)
        if not html:
            continue
        books = _parse_book_rows(html, kind)
        avg_row = _parse_average_row(html, kind)
        consensus = _consensus_from_books(books, kind)
        if avg_row and kind == 'ml' and avg_row.get('home'):
            # 页面平均值优先于简单算术平均
            consensus = {
                **consensus,
                'available': True,
                'home': avg_row['home'],
                'away': avg_row['away'],
                'home_init': avg_row.get('home_init'),
                'away_init': avg_row.get('away_init'),
                'source': 'page_avg',
            }
            h, a = avg_row['home'], avg_row['away']
            if h and a:
                p_h, p_a = 1 / h, 1 / a
                tot = p_h + p_a
                consensus['home_prob'] = round(p_h / tot, 4)
                consensus['away_prob'] = round(p_a / tot, 4)
            hi, ai = avg_row.get('home_init'), avg_row.get('away_init')
            if hi and ai and h and a:
                consensus['home_move'] = round(h - hi, 4)
                consensus['away_move'] = round(a - ai, 4)
                consensus['trend'] = analyze_line_trend([
                    {'home_odds': hi, 'away_odds': ai, 'line': 0},
                    {'home_odds': h, 'away_odds': a, 'line': 0},
                ], 'ml')
            if not consensus.get('book_count'):
                consensus['book_count'] = len(books)
        if consensus.get('available'):
            consensus['books_sample'] = books[:5]
            bundle[kind] = consensus

    _market_cache[match_id] = bundle
    return bundle


def clear_market_cache():
    global _market_cache
    _market_cache = {}


def prefetch_market_bundles(match_ids: List[str], max_workers: int = 6) -> Dict[str, dict]:
    clear_market_cache()
    result = {}
    ids = [str(i) for i in match_ids if i]
    if not ids:
        return result
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_match_market_bundle, mid, False): mid for mid in ids}
        for fut in as_completed(futures):
            mid = futures[fut]
            try:
                result[mid] = fut.result()
            except Exception as e:
                log.warning(f"各家赔率抓取失败 {mid}: {e}")
                result[mid] = {'match_id': mid, 'ml': {'available': False},
                               'ah': {'available': False}, 'ou': {'available': False}}
    return result


def _parse_match_row(tr: str, date_str: str) -> Optional[Dict]:
    tds = re.findall(r'<td[^>]*>(.*?)</td>', tr, re.S)
    if len(tds) < 7:
        return None

    match_id_m = re.search(r'/basketball/match/(\d+)/', tr)
    if not match_id_m:
        return None
    match_id = match_id_m.group(1)

    num_m = re.search(r'<i>(\d+)</i>', tds[0])
    league_m = re.search(
        r'href="[^"]*basketball/league/\d+/[^"]*"[^>]*title="([^"]+)"',
        tds[0],
    ) or re.search(
        r'href="[^"]*basketball/league/\d+/[^"]*"[^>]*>([^<]+)</a>',
        tds[0],
    )
    num = num_m.group(1) if num_m else ''
    league = (league_m.group(1) if league_m else '').strip()
    if not league:
        league_m2 = re.search(r'>(WNBA|NBA|CBA|NCAAB|欧篮联|美职女篮)</a>', tds[0])
        league = league_m2.group(1) if league_m2 else ''


    # The date section header can lag behind the actual row.  The row's time
    # cell carries the authoritative full datetime in its title attribute.
    row_datetime = re.search(r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})(?::\d{2})?', tr)
    if row_datetime:
        date_str = row_datetime.group(1)
        match_time = row_datetime.group(2)
    else:
        match_time = ''

    time_cell = _strip_html(tds[1])
    # e.g. "23:00 22:00" → 开赛 / 截止
    tm = re.search(r'(\d{2}:\d{2})', time_cell)
    if not match_time:
        match_time = tm.group(1) if tm else ''

    home_m = re.search(r'class="[^"]*duinameh[^"]*"[^>]*title="([^"]+)"', tds[2])
    away_titles = re.findall(r'class="[^"]*duinameh[^"]*"[^>]*title="([^"]+)"', tds[2])
    if home_m and len(away_titles) >= 2:
        home = home_m.group(1)
        away = away_titles[1]
    else:
        plain_teams = _strip_html(tds[2])
        vs_m = re.search(r'(.+?)\s+VS\s+(.+)', plain_teams, re.I)
        if not vs_m:
            # 已完场带比分
            score_m = re.search(r'(.+?)\s+(\d+\s*[-:]\s*\d+)\s+(.+)', plain_teams)
            if score_m:
                home, away = score_m.group(1), score_m.group(3)
            else:
                return None
        else:
            home, away = vs_m.group(1), vs_m.group(2)
        home = re.sub(r'\[.*?\]', '', home).strip()
        away = re.sub(r'\[.*?\]', '', away).strip()

    score_plain = _strip_html(tds[6])
    finished = bool(score_plain and score_plain != '-' and re.search(r'\d+\s*[-:]\s*\d+', score_plain))

    # 竞彩 SP：胜负 / 让分 / 大小
    sf_odds = re.findall(
        r'class="betObj[^"]*"[^>]*>\s*([\d.]+)\s*<', tds[4], re.S
    )
    if len(sf_odds) < 2:
        sf_odds = re.findall(r'mixsf[\s\S]*?>([\d.]+)<[\s\S]*?>([\d.]+)<', tds[4])
        sf_odds = list(sf_odds[0]) if sf_odds else []

    rf_blocks = re.findall(r'class="rbetObj[^"]*"[^>]*>\s*([\d.]+)\s*<', tds[4], re.S)
    hc_m = re.search(
        r'class="[^"]*rfsfrfzObj[^"]*"[^>]*rflist="([^"]*)"[^>]*>\s*([+\-]?\d+(?:\.\d+)?)\s*<',
        tds[4],
        re.S,
    )
    if not hc_m:
        hc_m = re.search(
            r'rflist="([^"]*)"[^>]*class="[^"]*rfsfrfzObj[^"]*"[^>]*>\s*([+\-]?\d+(?:\.\d+)?)',
            tds[4],
            re.S,
        )
    if not hc_m:
        # alternate attribute order
        hc_span = re.search(
            r'<span class="[^"]*rfsfrfzObj[^"]*"([^>]*)>([^<]+)</span>',
            tds[4],
            re.S,
        )
        if hc_span:
            attrs, txt = hc_span.group(1), hc_span.group(2).strip()
            rf_hist = re.search(r'rflist="([^"]*)"', attrs)
            hc_m_g = (rf_hist.group(1) if rf_hist else '', txt)
        else:
            hc_m_g = None
    else:
        hc_m_g = (hc_m.group(1), hc_m.group(2).strip())

    dx_odds = re.findall(r'class="dbetObj[^"]*"[^>]*>\s*([\d.]+)\s*<', tds[4], re.S)
    dx_span = re.search(
        r'<span class="[^"]*dxfjxzObj[^"]*"([^>]*)>([^<]+)</span>',
        tds[4],
        re.S,
    )
    dx_rflist, total_line = '', None
    if dx_span:
        dx_rflist_m = re.search(r'rflist="([^"]*)"', dx_span.group(1))
        dx_rflist = dx_rflist_m.group(1) if dx_rflist_m else ''
        total_line = _safe_float(dx_span.group(2))

    handicap = None
    rf_rflist = ''
    if hc_m_g:
        rf_rflist, handicap = hc_m_g[0], hc_m_g[1]

    spf_home = _safe_float(sf_odds[0]) if len(sf_odds) >= 1 else None
    spf_away = _safe_float(sf_odds[1]) if len(sf_odds) >= 2 else None
    rqspf_home = _safe_float(rf_blocks[0]) if len(rf_blocks) >= 1 else None
    rqspf_away = _safe_float(rf_blocks[1]) if len(rf_blocks) >= 2 else None
    dx_over = _safe_float(dx_odds[0]) if len(dx_odds) >= 1 else None
    dx_under = _safe_float(dx_odds[1]) if len(dx_odds) >= 2 else None

    rf_history = parse_rflist(rf_rflist)
    dx_history = parse_rflist(dx_rflist)

    home_score = away_score = None
    if finished:
        sm = re.search(r'(\d+)\s*[-:]\s*(\d+)', score_plain)
        if sm:
            home_score, away_score = int(sm.group(1)), int(sm.group(2))

    return {
        'id': match_id,
        'okooo_id': match_id,
        'date': date_str,
        'time': match_time,
        'num': num,
        'league': league,
        'home': home.strip(),
        'away': away.strip(),
        'handicap': handicap,
        'rqspf_home': rqspf_home,
        'rqspf_away': rqspf_away,
        'spf_home': spf_home,
        'spf_away': spf_away,
        'total_line': total_line,
        'dx_over': dx_over,
        'dx_under': dx_under,
        'status': 'finished' if finished else 'not_started',
        'home_score': home_score,
        'away_score': away_score,
        'source': 'okooo',
        'rf_history': rf_history,
        'dx_history': dx_history,
        'rf_trend': analyze_line_trend(rf_history, 'ah') if rf_history else None,
        'dx_trend': analyze_line_trend(dx_history, 'ou') if dx_history else None,
    }


def fetch_okooo_basketball_schedule(date: Optional[str] = None) -> List[Dict]:
    """抓取澳客竞彩篮球混合过关赛程。"""
    if date is None:
        date = time.strftime('%Y-%m-%d')

    log.info(f"抓取澳客篮球赛程: {date}")
    html = fetch_okooo(OKOOO_HUNHE_URL, referer=OKOOO_BASE + '/')
    if not html:
        log.warning("澳客篮球赛程为空(WAF/网络)")
        return []

    tables = re.findall(r'<table[^>]*>(.*?)</table>', html, re.S)
    if len(tables) < 2:
        log.warning(f"澳客篮球页未找到主表, tables={len(tables)}")
        return []

    main = max(tables, key=len)
    trs = re.findall(r'<tr([^>]*)>(.*?)</tr>', main, re.S)
    current_date = date
    matches = []

    for attrs, body in trs:
        date_m = re.search(r'(\d{4}-\d{2}-\d{2})', body)
        cls = re.search(r'class="([^"]*)"', attrs)
        cls_v = cls.group(1) if cls else ''

        if date_m and 'alltrObj' not in cls_v:
            current_date = date_m.group(1)
            continue

        if 'alltrObj' not in cls_v:
            continue

        try:
            match = _parse_match_row(body, current_date)
            if match:
                matches.append(match)
        except Exception as e:
            log.warning(f"解析澳客篮球行失败: {e}")

    # 以页面比分为准过滤完场；时间只用来标进行中，避免 WNBA 晨间场被误杀
    active = [m for m in matches if m.get('status') != 'finished']
    dated = [m for m in active if m.get('date') == date]
    if dated:
        live = dated
    elif active:
        # When today's card is already complete, 澳客 commonly shows the next
        # sale day's card on the same page. Return only that nearest date.
        nearest_date = min(m.get('date') or '9999-12-31' for m in active)
        live = [m for m in active if m.get('date') == nearest_date]
    else:
        live = []

    now = datetime.now()
    for m in live:
        try:
            if m.get('date') and m.get('time'):
                dt = datetime.strptime(f"{m['date']} {m['time']}", '%Y-%m-%d %H:%M')
                if dt <= now:
                    m['status'] = 'in_progress'
                else:
                    m['status'] = 'not_started'
        except ValueError:
            pass

    # 仅返回真正未开赛的场次。行内完整日期优先于分组表头，避免把次日比赛
    # 错标为今日已开赛并过滤掉。
    live = [m for m in live if m['status'] == 'not_started']
    log.info(
        f"澳客篮球获取到 {len(live)} 场未开赛比赛 (原始{len(matches)})"
    )
    return live


def blend_market_probs(jingcai_home: float, jingcai_away: float,
                       consensus: Optional[Dict], weight: float = 0.35
                       ) -> Tuple[float, float]:
    """竞彩 SP 与各家共识隐含概率融合。"""
    if not consensus or not consensus.get('available'):
        return jingcai_home, jingcai_away
    c_home = consensus.get('home_prob')
    c_away = consensus.get('away_prob')
    if c_home is None or c_away is None:
        return jingcai_home, jingcai_away
    w = max(0.0, min(0.6, weight))
    p_h = (1 - w) * jingcai_home + w * c_home
    p_a = (1 - w) * jingcai_away + w * c_away
    tot = p_h + p_a + 1e-9
    return p_h / tot, p_a / tot
