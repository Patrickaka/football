#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Australian Odds (okooo) JCZQ schedule enrichment.

500.com remains the market-analysis source.  This module only supplies the
China Sports Lottery offer: official integer handicap, opened play types and
display odds.  Network failure must never block football analysis.
"""

import re
import threading
import time
import urllib.error
import urllib.request
from html import unescape
from typing import Dict, List, Optional

from ..common.logger import setup_logger


log = setup_logger('football.okooo_lottery')
OKOOO_BASE = 'https://www.okooo.com'
OKOOO_JCZQ_URLS = (
    f'{OKOOO_BASE}/jingcai/',
    f'{OKOOO_BASE}/jingcai/hunhe/',
)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Referer': OKOOO_BASE + '/',
}
_CACHE_TTL = 300
_cache = {'at': 0.0, 'ttl': _CACHE_TTL, 'matches': []}
_cache_lock = threading.Lock()


def _text(fragment: str) -> str:
    value = re.sub(r'<[^>]+>', ' ', fragment or '')
    return re.sub(r'\s+', ' ', unescape(value)).strip()


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer_handicap(value):
    match = re.search(r'[-+]?\d+(?:\.\d+)?', str(value or ''))
    if not match:
        return None
    number = _number(match.group(0))
    if number is None or not number.is_integer() or abs(number) > 5:
        return None
    return int(number)


def parse_okooo_jczq_schedule(html: str) -> List[Dict]:
    """Parse both current table markup and lightweight row variants."""
    matches = []
    for row in re.findall(r'<tr\b[^>]*>(.*?)</tr>', html or '', re.I | re.S):
        match_id_match = re.search(r'/soccer/match/(\d+)', row, re.I)
        row_text = _text(row)
        num_match = re.search(r'(周[一二三四五六日]\s*\d{3})', row_text)
        if not num_match:
            xh_match = re.search(
                r'class=["\'][^"\']*xh[^"\']*["\'][^>]*>(.*?)</span>', row, re.I | re.S
            )
            xh_number = re.search(r'\d+', _text(xh_match.group(1))) if xh_match else None
            if xh_number:
                num_match = re.match(r'(\d+)', xh_number.group(0).zfill(3))
        home_match = re.search(
            r'class=["\'][^"\']*homenameobj[^"\']*["\'][^>]*(?:title=["\']([^"\']+)["\'])?[^>]*>([^<]+)',
            row, re.I,
        )
        away_match = re.search(
            r'class=["\'][^"\']*awaynameobj[^"\']*["\'][^>]*(?:title=["\']([^"\']+)["\'])?[^>]*>([^<]+)',
            row, re.I,
        )
        if not home_match or not away_match:
            continue

        handicap_match = re.search(
            r'class=["\'][^"\']*(?:handicapobj|rqhandicap|rangqiu)[^"\']*["\'][^>]*>([^<]+)',
            row, re.I,
        )
        handicap = _integer_handicap(handicap_match.group(1) if handicap_match else None)
        odds = [_number(value) for value in re.findall(r'<em\b[^>]*>([\d.]+)</em>', row, re.I)]
        odds = [value for value in odds if value is not None]
        time_match = re.search(r'(\d{2}-\d{2})?\s*(\d{2}:\d{2})', row_text)
        rq_available = handicap not in (None, 0)
        # 澳客有些场次只给一组三项赔率：有整数让球时，这组三项就是
        # 让球胜平负，并不代表同时开放了普通胜平负。
        spf_available = len(odds) >= 6 or (len(odds) >= 3 and not rq_available)
        if not spf_available and not rq_available and '胜平负' in row_text:
            spf_available = True
        spf_odds = None
        rqspf_odds = None
        if len(odds) >= 6:
            spf_odds = {'胜': odds[0], '平': odds[1], '负': odds[2]}
            rqspf_odds = {'让胜': odds[3], '让平': odds[4], '让负': odds[5]}
        elif len(odds) >= 3 and rq_available:
            rqspf_odds = {'让胜': odds[0], '让平': odds[1], '让负': odds[2]}
        elif len(odds) >= 3:
            spf_odds = {'胜': odds[0], '平': odds[1], '负': odds[2]}

        matches.append({
            'okooo_id': match_id_match.group(1) if match_id_match else None,
            'num': re.sub(r'\s+', '', num_match.group(1)) if num_match else '',
            'home': (home_match.group(2) or home_match.group(1) or '').strip(),
            'away': (away_match.group(2) or away_match.group(1) or '').strip(),
            'time': time_match.group(2) if time_match else '',
            'lottery_handicap': handicap,
            'spf_available': spf_available,
            'rqspf_available': rq_available,
            'available_markets': [
                market for market, available in (
                    ('spf', spf_available), ('rqspf', rq_available),
                    ('score', '比分' in row_text), ('goals', '进球' in row_text),
                    ('half_full', '半全场' in row_text),
                ) if available
            ],
            'spf_odds': spf_odds,
            'rqspf_odds': rqspf_odds,
            'source': 'okooo',
        })
    return matches


def fetch_okooo_jczq_schedule(force_refresh: bool = False) -> List[Dict]:
    now = time.time()
    with _cache_lock:
        if not force_refresh and now - _cache['at'] < _cache.get('ttl', _CACHE_TTL):
            return list(_cache['matches'])
    deadline = time.monotonic() + 6.0
    last_error = None
    for url in OKOOO_JCZQ_URLS:
        remaining = deadline - time.monotonic()
        if remaining <= 0.5:
            break
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=min(3.5, remaining)) as response:
                raw = response.read()
            page = None
            for encoding in ('gb18030', 'utf-8', 'gbk'):
                try:
                    page = raw.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            matches = parse_okooo_jczq_schedule(page or raw.decode('utf-8', errors='replace'))
            if matches:
                with _cache_lock:
                    _cache.update({'at': now, 'ttl': _CACHE_TTL, 'matches': matches})
                return matches
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            continue

    with _cache_lock:
        _cache.update({'at': now, 'ttl': 60, 'matches': []})
    if last_error:
        log.warning('澳客体彩赛程获取失败，保留500分析结果: %s', last_error)
    else:
        log.warning('澳客体彩赛程未解析到比赛，保留500分析结果')
    return []


def _clean_team(value: str) -> str:
    value = re.sub(r'\[[^\]]*\]|（[^）]*）|\([^)]*\)', '', str(value or ''))
    return re.sub(r'[^0-9A-Za-z\u4e00-\u9fff]', '', value).lower()


def _num(value: str) -> str:
    full = re.search(r'(周[一二三四五六日])\s*(\d{3})', str(value or '').strip())
    if full:
        return ''.join(full.groups())
    match = re.search(r'(\d{3})$', str(value or '').strip())
    return match.group(1) if match else ''


def _same_team(left: str, right: str) -> bool:
    left, right = _clean_team(left), _clean_team(right)
    return bool(left and right and (left == right or left in right or right in left))


def _match_score(base: Dict, lottery: Dict) -> int:
    score = 0
    base_num, lottery_num = _num(base.get('num')), _num(lottery.get('num'))
    if base_num and base_num == lottery_num:
        score += 8 if base_num.startswith('周') else 4
    if _same_team(base.get('home'), lottery.get('home')):
        score += 4
    if _same_team(base.get('away'), lottery.get('away')):
        score += 4
    if base.get('time') and lottery.get('time') and str(base['time'])[-5:] == str(lottery['time'])[-5:]:
        score += 2
    return score


def enrich_with_okooo_lottery(matches: List[Dict], lottery_matches: Optional[List[Dict]] = None) -> List[Dict]:
    offers = fetch_okooo_jczq_schedule() if lottery_matches is None else lottery_matches
    for match in matches:
        ranked = sorted(((_match_score(match, offer), offer) for offer in offers), key=lambda item: -item[0])
        if not ranked or ranked[0][0] < 8:
            match.update({
                'lottery_source': 'unavailable', 'lottery_offer_matched': False,
                'lottery_primary_market': None,
            })
            continue
        offer = ranked[0][1]
        handicap = offer.get('lottery_handicap')
        rq_available = bool(offer.get('rqspf_available') and handicap not in (None, 0))
        spf_available = bool(offer.get('spf_available'))
        primary = 'rqspf' if rq_available else 'spf' if spf_available else None
        match.update({
            'lottery_source': 'okooo',
            'lottery_offer_matched': True,
            'okooo_id': offer.get('okooo_id'),
            'lottery_handicap': handicap,
            'lottery_primary_market': primary,
            'lottery_available_markets': offer.get('available_markets') or [],
            'lottery_spf_available': spf_available,
            'lottery_rqspf_available': rq_available,
            'lottery_spf_odds': offer.get('spf_odds'),
            'lottery_rqspf_odds': offer.get('rqspf_odds'),
        })
    return matches
