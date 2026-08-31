#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""中国足彩网竞彩足球赛程增强。

500.com 仍负责球队基本面和盘口分析；本模块只读取 zgzcw.com 公开展示的
竞彩足球销售页，补齐官方让球、已开售玩法和展示 SP。抓取失败不能阻断主流程。
"""

import re
import threading
import time
import urllib.parse
import urllib.request
from html import unescape
from typing import Dict, List, Optional

from ..common.logger import setup_logger

try:
    import requests
except ImportError:  # pragma: no cover - 生产环境安装 requests
    requests = None


log = setup_logger('football.zgzcw_lottery')
ZGZCW_BASE = 'https://cp.zgzcw.com'
ZGZCW_JCZQ_URL = (
    f'{ZGZCW_BASE}/lottery/jchtplayvsForJsp.action?lotteryId=47&type=jcmini'
)
HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36'),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Referer': ZGZCW_BASE + '/',
}

_CACHE_TTL = 300
_cache: Dict[str, Dict] = {}
_cache_lock = threading.Lock()
_session = None
_last_status = {'reason': 'not_requested', 'detail': None}

_ROW = re.compile(r'<tr\b([^>]*)>(.*?)</tr>', re.I | re.S)
_ATTR = r'\b{name}=["\']([^"\']*)["\']'
_TAG = re.compile(r'<[^>]+>')


def _set_status(reason: str, detail=None):
    _last_status.update({'reason': reason,
                         'detail': str(detail)[:180] if detail else None})


def get_zgzcw_lottery_status() -> Dict:
    return dict(_last_status)


def _get_session():
    global _session
    if requests is None:
        return None
    if _session is None:
        _session = requests.Session()
        _session.headers.update(HEADERS)
    return _session


def _decode(raw: bytes) -> str:
    for encoding in ('utf-8', 'gb18030', 'gbk'):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', errors='replace')


def _fetch_page(url: str, timeout: float) -> str:
    session = _get_session()
    if session is not None:
        response = session.get(url, timeout=(min(3.0, timeout), timeout))
        response.raise_for_status()
        return _decode(response.content)
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return _decode(response.read())


def _is_block_page(page: str) -> bool:
    lowered = (page or '').lower()
    return any(marker in lowered for marker in (
        '访问验证', '安全验证', 'captcha', '系统维护中',
    ))


def _attr(fragment: str, name: str, default=None):
    found = re.search(_ATTR.format(name=re.escape(name)), fragment or '', re.I)
    return unescape(found.group(1)).strip() if found else default


def _text(fragment: str) -> str:
    return re.sub(r'\s+', ' ', unescape(_TAG.sub(' ', fragment or ''))).strip()


def _number(value):
    try:
        return float(str(value).replace('↑', '').replace('↓', '').strip())
    except (TypeError, ValueError):
        return None


def _cell(row: str, class_name: str) -> str:
    found = re.search(
        rf'<td\b[^>]*class=["\'][^"\']*\b{re.escape(class_name)}\b[^"\']*["\'][^>]*>'
        rf'(.*?)</td>', row, re.I | re.S)
    return found.group(1) if found else ''


def _team(row: str, class_name: str) -> str:
    cell = _cell(row, class_name)
    link = re.search(r'<a\b[^>]*?(?:title=["\']([^"\']*)["\'])?[^>]*>(.*?)</a>',
                     cell, re.I | re.S)
    if not link:
        return ''
    visible = _text(link.group(2))
    return visible or (link.group(1) or '').strip()


def _market(row: str, pid: str):
    found = re.search(
        rf'<div\b[^>]*\bpid=["\']{re.escape(pid)}["\'][^>]*>(.*?)</div>',
        row, re.I | re.S)
    if not found:
        return None, []
    block = found.group(1)
    line_found = re.search(r'<em\b[^>]*class=["\'][^"\']*(?:rq|total)[^"\']*["\'][^>]*>(.*?)</em>',
                           block, re.I | re.S)
    line = _number(_text(line_found.group(1))) if line_found else None
    odds = [_number(_text(value)) for value in re.findall(
        r'<a\b[^>]*\bid=["\']td_[^"\']+["\'][^>]*>(.*?)</a>',
        block, re.I | re.S)]
    return line, [odd for odd in odds if odd and odd > 1.0]


def parse_zgzcw_jczq_schedule(html: str) -> List[Dict]:
    """解析中国足彩网竞彩足球胜平负/让球销售页。"""
    matches = []
    for attrs, row in _ROW.findall(html or ''):
        row_id = _attr(attrs, 'id', '')
        if not row_id.startswith('tr_'):
            continue
        match_id = row_id[3:]
        num = _attr(attrs, 'mn', '')
        if not num:
            day = _text(re.search(r'<code\b[^>]*>(.*?)</code>', row,
                                  re.I | re.S).group(1)) if re.search(
                                      r'<code\b[^>]*>(.*?)</code>', row, re.I | re.S) else ''
            order = _text(re.search(r'<i\b[^>]*>(.*?)</i>', row,
                                    re.I | re.S).group(1)) if re.search(
                                        r'<i\b[^>]*>(.*?)</i>', row, re.I | re.S) else ''
            num = f'{day}{order}'
        home, away = _team(row, 'wh-4'), _team(row, 'wh-6')
        if not (match_id and num and home and away):
            continue

        kickoff = re.search(r'title=["\']比赛时间:([^"\']+)["\']', row, re.I)
        kickoff = kickoff.group(1).strip() if kickoff else ''
        date = kickoff[:10] if re.match(r'\d{4}-\d{2}-\d{2}', kickoff) else ''
        clock = re.search(r'(\d{2}:\d{2})', kickoff)
        league = _attr(re.search(r'<td\b[^>]*class=["\'][^"\']*\bwh-2\b[^"\']*["\'][^>]*>',
                                 row, re.I).group(0), 'title', '') if re.search(
                                     r'<td\b[^>]*class=["\'][^"\']*\bwh-2\b[^"\']*["\'][^>]*>',
                                     row, re.I) else _text(_cell(row, 'wh-2'))

        _, spf = _market(row, '49')
        handicap, rqspf = _market(row, '22')
        spf_odds = ({'胜': spf[0], '平': spf[1], '负': spf[2]}
                    if len(spf) >= 3 else None)
        rqspf_odds = ({'让胜': rqspf[0], '让平': rqspf[1], '让负': rqspf[2]}
                      if len(rqspf) >= 3 and handicap not in (None, 0) else None)
        analysis = re.search(r'\bnewplayid=["\'](\d+)["\']', row, re.I)
        matches.append({
            'zgzcw_id': match_id,
            'analysis_id': analysis.group(1) if analysis else None,
            'num': re.sub(r'\s+', '', num),
            'date': date,
            'time': clock.group(1) if clock else '',
            'league': league,
            'home': home,
            'away': away,
            'lottery_handicap': int(handicap) if handicap is not None and handicap.is_integer() else handicap,
            'spf_available': spf_odds is not None,
            'rqspf_available': rqspf_odds is not None,
            'available_markets': [name for name, value in (
                ('spf', spf_odds), ('rqspf', rqspf_odds)) if value is not None],
            'spf_odds': spf_odds,
            'rqspf_odds': rqspf_odds,
            'source': 'zgzcw',
        })
    return matches


def fetch_zgzcw_jczq_schedule(date: Optional[str] = None,
                               force_refresh: bool = False) -> List[Dict]:
    key = date or 'current'
    now = time.time()
    with _cache_lock:
        cached = _cache.get(key)
        if (cached and not force_refresh
                and now - cached['at'] < cached['ttl']):
            return list(cached['matches'])

    url = ZGZCW_JCZQ_URL
    if date:
        url += '&issue=' + urllib.parse.quote(date)
    try:
        page = _fetch_page(url, 8.0)
        if _is_block_page(page):
            raise IOError('verification_page')
        matches = parse_zgzcw_jczq_schedule(page)
        with _cache_lock:
            _cache[key] = {'at': now, 'ttl': _CACHE_TTL if matches else 60,
                           'matches': matches}
        _set_status('ok' if matches else 'empty_schedule', f'{len(matches)} matches')
        return matches
    except Exception as exc:
        with _cache_lock:
            _cache[key] = {'at': now, 'ttl': 60, 'matches': []}
        _set_status('network_error', exc)
        log.warning('中国足彩网竞彩足球赛程获取失败，保留500分析结果: %s', exc)
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
    score += 4 if _same_team(base.get('home'), lottery.get('home')) else 0
    score += 4 if _same_team(base.get('away'), lottery.get('away')) else 0
    if (base.get('time') and lottery.get('time')
            and str(base['time'])[-5:] == str(lottery['time'])[-5:]):
        score += 2
    return score


def enrich_with_zgzcw_lottery(matches: List[Dict],
                               lottery_matches: Optional[List[Dict]] = None) -> List[Dict]:
    offers = fetch_zgzcw_jczq_schedule() if lottery_matches is None else lottery_matches
    status = get_zgzcw_lottery_status()
    for match in matches:
        ranked = sorted(((_match_score(match, offer), offer) for offer in offers),
                        key=lambda item: -item[0])
        if not ranked or ranked[0][0] < 8:
            match.update({
                'lottery_source': 'unavailable', 'lottery_offer_matched': False,
                'lottery_primary_market': None,
                'lottery_unavailable_reason': (
                    status.get('reason') if not offers else 'cross_source_match_failed'),
            })
            continue
        offer = ranked[0][1]
        handicap = offer.get('lottery_handicap')
        rq_available = bool(offer.get('rqspf_available') and handicap not in (None, 0))
        spf_available = bool(offer.get('spf_available'))
        match.update({
            'lottery_source': 'zgzcw',
            'lottery_offer_matched': True,
            'lottery_unavailable_reason': None,
            'zgzcw_id': offer.get('zgzcw_id'),
            'analysis_id': offer.get('analysis_id'),
            'lottery_handicap': handicap,
            'lottery_primary_market': 'rqspf' if rq_available else 'spf' if spf_available else None,
            'lottery_available_markets': offer.get('available_markets') or [],
            'lottery_spf_available': spf_available,
            'lottery_rqspf_available': rq_available,
            'lottery_spf_odds': offer.get('spf_odds'),
            'lottery_rqspf_odds': offer.get('rqspf_odds'),
        })
    return matches
