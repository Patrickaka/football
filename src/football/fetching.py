# -*- coding: utf-8 -*-
"""足球数据抓取：HTTP缓存/限流/节流、比赛列表"""

import sys
import os
import math
import re
import time
import gzip
import json
import urllib.request
import urllib.error
import random
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple

from ..common.logger import setup_logger
from ..common.paths import data_path

log = setup_logger('football')
from . import config as _cfg

from .config import (
    HEADERS, _MATCH_LIST_STATUS,
)

FETCH_PAGE_TTL = float(os.getenv('FOOTBALL_PAGE_TTL', '90'))


FETCH_MAX_CONCURRENCY = max(1, int(os.getenv('FOOTBALL_FETCH_CONCURRENCY', '4')))


FETCH_RATE_LIMIT = float(os.getenv('FOOTBALL_FETCH_RATE', '3'))


FETCH_RETRY_ATTEMPTS = max(1, int(os.getenv('FOOTBALL_FETCH_RETRIES', '4')))


FETCH_THROTTLE_SECONDS = float(os.getenv('FOOTBALL_FETCH_THROTTLE', '1.0'))


FETCH_THROTTLE_CEILING = float(os.getenv('FOOTBALL_FETCH_THROTTLE_MAX', '4'))


RATE_LIMIT_STATUSES = frozenset({428, 429, 503})


_FETCH_CACHE_LIMIT = 256


_FETCH_URL_LOCK_LIMIT = 512


_fetch_cache = OrderedDict()


_fetch_url_locks = OrderedDict()


_fetch_cache_lock = threading.Lock()


_fetch_semaphore = threading.BoundedSemaphore(FETCH_MAX_CONCURRENCY)


_fetch_throttle_lock = threading.Lock()


_fetch_throttle_until = 0.0


_fetch_rate_lock = threading.Lock()


_fetch_next_slot = 0.0


def _fetch_cache_get(cache_key):
    if FETCH_PAGE_TTL <= 0:
        return None
    with _fetch_cache_lock:
        entry = _fetch_cache.get(cache_key)
        if entry is None:
            return None
        cached_at, value = entry
        if time.time() - cached_at >= FETCH_PAGE_TTL:
            _fetch_cache.pop(cache_key, None)
            return None
        _fetch_cache.move_to_end(cache_key)
        return value


def _fetch_cache_set(cache_key, value):
    if FETCH_PAGE_TTL <= 0:
        return
    with _fetch_cache_lock:
        _fetch_cache[cache_key] = (time.time(), value)
        _fetch_cache.move_to_end(cache_key)
        while len(_fetch_cache) > _FETCH_CACHE_LIMIT:
            _fetch_cache.popitem(last=False)


def _fetch_url_lock(cache_key):
    """取 URL 级锁：丢弃最老的锁只会削弱去重，不影响正确性。"""
    with _fetch_cache_lock:
        lock = _fetch_url_locks.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            _fetch_url_locks[cache_key] = lock
        _fetch_url_locks.move_to_end(cache_key)
        while len(_fetch_url_locks) > _FETCH_URL_LOCK_LIMIT:
            _fetch_url_locks.popitem(last=False)
        return lock


def clear_fetch_cache():
    """清空页面级抓取缓存（force_refresh / 手动清缓存时调用）。"""
    with _fetch_cache_lock:
        _fetch_cache.clear()


def _enter_fetch_throttle(seconds):
    """源站限流后设置全局冷却截止时间（只延后、不提前）。"""
    global _fetch_throttle_until
    with _fetch_throttle_lock:
        _fetch_throttle_until = max(_fetch_throttle_until, time.time() + seconds)


def _await_fetch_throttle():
    """冷却期内所有抓取一起等，避免重试风暴把限流窗口拉长。"""
    while True:
        with _fetch_throttle_lock:
            remaining = _fetch_throttle_until - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.5))


def _await_rate_slot():
    """发号器：给每个上游请求分配一个不早于「上一个号 + 间隔」的出发时刻。"""
    global _fetch_next_slot
    if FETCH_RATE_LIMIT <= 0:
        return
    interval = 1.0 / FETCH_RATE_LIMIT
    with _fetch_rate_lock:
        slot = max(time.time(), _fetch_next_slot)
        _fetch_next_slot = slot + interval
    delay = slot - time.time()
    if delay > 0:
        time.sleep(delay)


def fetch(url, encoding='gbk', referer=None):
    """抓取网页，自动处理 gzip 压缩和编码（带 TTL 复用与并发去重）"""
    cache_key = (url, encoding)
    cached = _fetch_cache_get(cache_key)
    if cached is not None:
        return cached
    with _fetch_url_lock(cache_key):
        cached = _fetch_cache_get(cache_key)
        if cached is not None:
            return cached
        with _fetch_semaphore:
            result = _fetch_raw(url, encoding, referer)
        _fetch_cache_set(cache_key, result)
        return result


def _fetch_raw(url, encoding='gbk', referer=None):
    """发起网络抓取（无缓存），对源站限流做全局退避重试"""
    last_error = None
    for attempt in range(FETCH_RETRY_ATTEMPTS):
        _await_fetch_throttle()
        _await_rate_slot()
        try:
            return _fetch_once(url, encoding, referer)
        except urllib.error.HTTPError as e:
            if e.code not in RATE_LIMIT_STATUSES:
                raise
            last_error = e
            backoff = min(
                FETCH_THROTTLE_SECONDS * (attempt + 1) + random.uniform(0, 0.3),
                FETCH_THROTTLE_CEILING,
            )
            _enter_fetch_throttle(backoff)
            log.warning('源站限流 HTTP %s，%.1fs 后重试: %s', e.code, backoff, url)
    raise last_error


def _fetch_once(url, encoding='gbk', referer=None):
    """真正发起一次网络抓取（无缓存、不重试）"""
    start = time.perf_counter()
    headers = {**HEADERS, 'Referer': referer} if referer else HEADERS
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
    except urllib.error.HTTPError:
        from http import cookiejar
        cj = cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
        with opener.open(req, timeout=20) as resp:
            raw = resp.read()

    # 自动解压 gzip
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)

    for enc in [encoding, 'gb2312', 'gb18030', 'utf-8']:
        try:
            result = raw.decode(enc)
            # 清理 surrogate 字符
            result = result.encode('utf-8', errors='replace').decode('utf-8')
            log.debug('fetch %s → %d bytes (%.3fs)', url, len(raw), time.perf_counter() - start)
            return result
        except (UnicodeDecodeError, LookupError, UnicodeEncodeError):
            continue
    log.debug('fetch %s → %d bytes (%.3fs)', url, len(raw), time.perf_counter() - start)
    return raw.decode('utf-8', errors='replace')


def fetch_json(url, referer=None):
    """抓取并解析 JSON 接口"""
    return json.loads(fetch(url, encoding='utf-8', referer=referer))


def _fetch_match_list_remote():
    """抓取今日比赛列表，返回 [{home, away, match_id, league, time}, ...]"""
    log.info('获取比赛列表')
    html = None
    upstream_errors = []
    for index_url in _cfg.INDEX_URLS:
        try:
            candidate = fetch(index_url)
            if 'shuju-' in candidate:
                html = candidate
                break
            upstream_errors.append(f'{index_url}: missing schedule markers')
        except Exception as exc:
            upstream_errors.append(f'{index_url}: {type(exc).__name__}: {str(exc)[:120]}')
    if html is None:
        raise OSError('; '.join(upstream_errors) or 'all schedule upstreams failed')

    matches = []

    # 方案A: shuju 链接在前、title 锚点紧随其后，故以 id 为锚向后取本场 title
    # 布局: ...shuju-<id>.shtml... title="主队名VS客队名数据分析"...
    title_pat = re.compile(
        r'shuju-(\d+)\.shtml.*?title="([^"]+?)VS([^"]+?)'
        r'(?:数据|盘口|百家|欧赔|亚赔|亚盘|指数|对比|分析)[^"]*"',
        re.DOTALL
    )
    for m in title_pat.finditer(html):
        match_id = m.group(1).strip()
        home_name = m.group(2).strip()
        away_name = m.group(3).strip()
        # 清理队名尾部残留后缀
        for suffix in ['百家', '欧赔', '亚赔', '亚盘', '数据', '盘口', '指数', '对比', '分析', '百家欧赔', '百家亚盘']:
            if home_name.endswith(suffix):
                home_name = home_name[:-len(suffix)].strip()
            if away_name.endswith(suffix):
                away_name = away_name[:-len(suffix)].strip()
        if home_name and away_name and match_id:
            matches.append({
                'home': home_name,
                'away': away_name,
                'match_id': match_id
            })

    # 如果 title 方案没找到，用方案B: 正则匹配 team 链接
    if not matches:
        row_pat = re.compile(
            r'<a[^>]*href="//liansai\.500\.com/team/\d+/"[^>]*>([^<]+)</a>'
            r'.*?VS.*?'
            r'<a[^>]*href="//liansai\.500\.com/team/\d+/"[^>]*>([^<]+)</a>'
            r'.*?shuju-(\d+)\.shtml',
            re.DOTALL
        )
        for m in row_pat.finditer(html):
            home_name = m.group(1).strip()
            away_name = m.group(2).strip()
            match_id = m.group(3)
            if home_name and away_name and match_id:
                matches.append({
                    'home': home_name,
                    'away': away_name,
                    'match_id': match_id
                })

    # 提取联赛和时间（通过match_id关联）
    # 创建 match_id -> time 的映射（基于表格行结构）
    match_time_map = {}
    
    # 基于表格行结构的匹配：<td>时间</td>...<a href="...shuju-ID.shtml">
    # 时间格式：<td rowspan="2">06-06 13:00</td>
    time_row_pat = re.compile(
        r'<td[^>]*?rowspan="2"[^>]*?>(\d{2}-\d{2}\s+\d{2}:\d{2})</td>.*?'
        r'shuju-(\d+)\.shtml',
        re.DOTALL
    )
    for m in time_row_pat.finditer(html):
        time_val = m.group(1)
        match_id = m.group(2)
        if match_id not in match_time_map:
            match_time_map[match_id] = time_val
    
    # 如果上面的模式没找到，尝试其他模式
    if not match_time_map:
        time_patterns = [
            r'shuju-(\d+)\.shtml.*?(\d{2}-\d{2}\s+\d{2}:\d{2})',
            r'(\d{2}-\d{2}\s+\d{2}:\d{2}).*?shuju-(\d+)\.shtml',
        ]
        for pattern in time_patterns:
            time_row_pat = re.compile(pattern, re.DOTALL)
            for m in time_row_pat.finditer(html):
                if m.group(1).isdigit():
                    match_id = m.group(1)
                    time_val = m.group(2)
                else:
                    match_id = m.group(2)
                    time_val = m.group(1)
                
                if match_id not in match_time_map:
                    match_time_map[match_id] = time_val
    
    # 创建 match_id -> 竞彩编号 的映射（如 周三201），编号星期前缀可用于按时间分组
    # 布局: <input ... value="<match_id>" />周三201
    match_num_map = dict(
        re.findall(r'value="(\d+)"\s*/>\s*(周[一二三四五六日]\d{3})', html)
    )

    # 创建 match_id -> league 的映射（基于联赛区块结构）
    match_league_map = {}
    
    # 查找所有联赛区块（联赛名称后面跟着该联赛的比赛）
    # 模式：联赛链接...比赛列表...下一个联赛链接
    league_blocks = re.split(r'<a[^>]*href="//liansai\.500\.com/zuqiu-\d+/"[^>]*>([^<]+)</a>', html)
    
    current_league = ''
    for i, block in enumerate(league_blocks):
        if i % 2 == 1:
            # 这是联赛名称
            current_league = block.strip()
        else:
            # 这是联赛区块内容，提取其中的比赛ID
            match_ids_in_block = re.findall(r'shuju-(\d+)\.shtml', block)
            for match_id in match_ids_in_block:
                if match_id not in match_league_map:
                    match_league_map[match_id] = current_league

    # 将时间和联赛添加到比赛信息中
    for match in matches:
        match_id = match['match_id']
        if match_id in match_time_map:
            match['time'] = match_time_map[match_id]
        if match_id in match_league_map:
            match['league'] = match_league_map[match_id].strip()
        if match_id in match_num_map:
            match['num'] = match_num_map[match_id]
        match['lottery_handicap'] = None
        match['lottery_primary_market'] = None

    # 如果通过行匹配没有找到时间，则回退到原来的方法
    if not match_time_map:
        league_pat = re.compile(r'<a[^>]*href="//liansai\.500\.com/zuqiu-\d+/"[^>]*>([^<]+)</a>')
        time_pat = re.compile(r'(\d{2}-\d{2}\s+\d{2}:\d{2})')

        leagues = league_pat.findall(html)
        times = time_pat.findall(html)

        for i, match in enumerate(matches):
            if 'league' not in match and i < len(leagues):
                match['league'] = leagues[i].strip()
            if 'time' not in match and i < len(times):
                match['time'] = times[i]

    # 500.com only supplies analysis markets. JCZQ offer/handicap comes from
    # 中国足彩网 and is matched onto the 500 schedule without blocking on failure.
    try:
        from .zgzcw_lottery import enrich_with_zgzcw_lottery
        matches = enrich_with_zgzcw_lottery(matches)
    except Exception as exc:
        log.warning('中国足彩网体彩玩法综合失败，返回纯500分析赛程: %s', exc)

    log.info('获取到 %d 场比赛', len(matches))
    return matches


def _save_match_list_cache(matches):
    """持久化最后一次成功赛程，供生产出网抖动时降级。"""
    if not matches:
        return
    payload = {
        'schema_version': 1,
        'saved_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'matches': matches,
    }
    temporary = _cfg.MATCH_LIST_CACHE_PATH + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, _cfg.MATCH_LIST_CACHE_PATH)


def _load_match_list_cache():
    try:
        with open(_cfg.MATCH_LIST_CACHE_PATH, encoding='utf-8') as handle:
            payload = json.load(handle)
        matches = payload.get('matches') if isinstance(payload, dict) else None
        return list(matches) if isinstance(matches, list) else []
    except (OSError, ValueError, TypeError):
        return []


def get_match_list_status():
    return dict(_MATCH_LIST_STATUS)


def _zgzcw_schedule_fallback(cached_matches=None):
    """将中国足彩网竞彩足球赛程转换为足球主列表结构。

    若上次 500 快照中存在同一竞彩编号，沿用其 match_id，
    这样比分分析仍可使用原有的 500 盘口链路。
    """
    from .zgzcw_lottery import fetch_zgzcw_jczq_schedule

    offers = fetch_zgzcw_jczq_schedule(force_refresh=True)
    if not offers:
        return []
    cached_matches = cached_matches or []
    cached_by_num = {
        str(item.get('num') or '').replace(' ', ''): item
        for item in cached_matches if item.get('num')
    }
    converted = []
    for offer in offers:
        num = str(offer.get('num') or '').replace(' ', '')
        previous = cached_by_num.get(num, {})
        handicap = offer.get('lottery_handicap')
        rq_available = bool(offer.get('rqspf_available') and handicap not in (None, 0))
        spf_available = bool(offer.get('spf_available'))
        converted.append({
            'home': offer.get('home') or previous.get('home') or '',
            'away': offer.get('away') or previous.get('away') or '',
            'match_id': str(previous.get('match_id') or offer.get('analysis_id')
                            or offer.get('zgzcw_id') or ''),
            'time': previous.get('time') or offer.get('time') or '',
            'league': previous.get('league') or offer.get('league') or '',
            'num': num,
            'schedule_source': 'zgzcw',
            'analysis_source_id_available': bool(previous.get('match_id')),
            'zgzcw_id': offer.get('zgzcw_id'),
            'analysis_id': offer.get('analysis_id'),
            'lottery_source': 'zgzcw',
            'lottery_offer_matched': True,
            'lottery_unavailable_reason': None,
            'lottery_handicap': handicap,
            'lottery_primary_market': 'rqspf' if rq_available else 'spf' if spf_available else None,
            'lottery_available_markets': offer.get('available_markets') or [],
            'lottery_spf_available': spf_available,
            'lottery_rqspf_available': rq_available,
            'lottery_spf_odds': offer.get('spf_odds'),
            'lottery_rqspf_odds': offer.get('rqspf_odds'),
        })
    return [item for item in converted if item['home'] and item['away'] and item['match_id']]


def fetch_match_list():
    """优先抓取实时赛程；源站不可用时回退最后成功快照。

    生产环境偶发 DNS/TLS/WAF 故障不应让整个足球页面变空。
    后续的 server 层仍会过滤已开赛项，因此陈旧快照不会展示过期比赛。
    """
    try:
        matches = _fetch_match_list_remote()
        if not matches:
            raise ValueError('upstream returned no parseable matches')
        try:
            _save_match_list_cache(matches)
        except OSError as exc:
            log.warning('比赛列表快照写入失败: %s', exc)
        _MATCH_LIST_STATUS.update({'source': 'live', 'stale': False, 'error': None})
        return matches
    except Exception as exc:
        cached = _load_match_list_cache()
        try:
            zgzcw_matches = _zgzcw_schedule_fallback(cached)
        except Exception as zgzcw_exc:
            zgzcw_matches = []
            log.warning('中国足彩网备用赛程也失败: %s', zgzcw_exc)
        if zgzcw_matches:
            _MATCH_LIST_STATUS.update({
                'source': 'zgzcw', 'stale': False,
                'error': f'500 upstream: {type(exc).__name__}: {str(exc)[:140]}',
            })
            log.warning('500赛程源失败，已切换中国足彩网赛程 %d 场', len(zgzcw_matches))
            return zgzcw_matches
        if cached:
            _MATCH_LIST_STATUS.update({
                'source': 'disk_cache', 'stale': True,
                'error': f'{type(exc).__name__}: {str(exc)[:180]}',
            })
            log.warning('实时比赛源失败，回退快照 %d 场: %s', len(cached), exc)
            return cached
        _MATCH_LIST_STATUS.update({
            'source': 'unavailable', 'stale': False,
            'error': f'{type(exc).__name__}: {str(exc)[:180]}',
        })
        raise


def search_match(matches, home_key, away_key):
    """搜索匹配的比赛，支持模糊匹配"""
    home_key = home_key.strip()
    away_key = away_key.strip()

    # 精确匹配
    exact = []
    partial = []
    for m in matches:
        home_match = home_key in m['home'] if home_key else True
        away_match = away_key in m['away'] if away_key else True
        if home_match and away_match:
            if m['home'] == home_key and m['away'] == away_key:
                exact.append(m)
            else:
                partial.append(m)
    return exact + partial


