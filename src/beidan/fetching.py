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
    DC_SCHEDULE_URL, HEADERS, OKOOO_DANCHANG_URL, OKOOO_HEADERS, OKOOO_MATCH_URL, SCHEDULE_URL, _is_okooo_waf_blocked, _mark_okooo_waf_blocked, _okooo_session, ensure_okooo_session,
)

# 北单与足球打的是同一个 odds.500.com。两边各跑一条预热线程，若各自独立发请求，
# 就会互相挤占源站配额、把对方推进 429。这里让 500.com 的请求共用足球那条限速
# 令牌流：请求前领号，撞限流则写进全局冷却，足球侧也会一起退避。
# okooo 走另一个域名和独立 session，不参与这条预算。
_SHARED_RATE_LIMIT_HOST = 'odds.500.com'


# ─── 领域层适配（页面解析）───
#
# 走势历史的解析在 `src/domain/sports/beidan/parsing.py`——迁移前这里是
# 三段几乎相同的一百行，差别只在表头关键字、列数、字段名和几条校验上。
# 留在这一层的只有：拼 URL、抓、记日志。

from src.domain.sports.beidan import parsing as _parsing


def _fetch_history(match_id, spec, what, unit, path_suffixes):
    """按顺序试几个页面，第一个解析出记录的就用它。

    **日志要分清是从表格还是从脚本刮出来的**：脚本那条路不走表格的
    时间长度、让球值长度、比分格式那几道校验，同样一条记录的可信度不同。
    """
    urls = [f'{OKOOO_MATCH_URL}{match_id}/{suffix}/' for suffix in path_suffixes]
    log.debug(f"抓取okooo{what}: match_id={match_id}")

    for url in urls:
        try:
            html = fetch_okooo(url, referer=OKOOO_DANCHANG_URL)
            # 空页面直接换下一个。改成 `is None` 在输出上等价（空串解析出来
            # 也是空记录），留短路是为了不在空串上白跑一遍正则。
            if not html:
                continue
            records, source = _parsing.parse_history(html, spec)
            if records:
                via = '通过脚本' if source == _parsing.FROM_SCRIPT else ''
                log.info(f"从 {url} {via}获取到 {len(records)} 条{unit}记录")
                return {'history': records}
        except Exception as e:
            log.warning(f"抓取okooo{what}失败({url}): {e}")
            continue

    log.debug(f"未获取到{what}数据")
    return {'history': []}


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

    # 预热 session（拿 cookie）。**迁移前这发生在 import 期**，于是任何
    # import beidan 的测试都在联网；现在推迟到真正要发请求的这一刻。
    # 幂等，只有第一次会真的握手。
    ensure_okooo_session()
    
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
    """okooo 的赛程，取不到就回退到 500.com。

    **三种回退的日志要分开**：页面为空（多半是 WAF）、页面结构不对
    （表格数不够）、解析出来一场未完结的都没有。§十一·3 那类
    「接口返回 200 加 0 场比赛」的故障里，最难查的正是分不清这三者。
    """
    if date is None:
        date = time.strftime('%Y-%m-%d')

    url = f'{OKOOO_DANCHANG_URL}?date={date}'
    log.info(f"抓取okooo北单赛程: {date}")

    try:
        html = fetch_okooo(url, referer=OKOOO_DANCHANG_URL)
        if not html:
            log.warning("okooo页面返回为空(WAF拦截或网络错误)，尝试500.com备用数据源")
            return _fallback_schedule(date, try_dc_first=True)

        log.debug(f"okooo页面HTML长度: {len(html)}")
        matches, table_count = _parsing.parse_okooo_schedule(html, date)
        log.debug(f"okooo页面找到 {table_count} 个table标签")
        if matches is None:
            log.warning(f"okooo页面未找到比赛表格，找到 {table_count} 个table标签，"
                        "尝试500.com备用数据源")
            return fetch_beidan_schedule(date)
        if not matches:
            log.warning("okooo未找到未完结比赛，尝试500.com备用数据源")
            return fetch_beidan_schedule(date)

        log.info(f"okooo获取到 {len(matches)} 场未完结北单比赛")
        return matches
    except Exception as e:
        log.error(f"抓取okooo北单赛程失败: {e}，尝试500.com备用数据源")
        return _fallback_schedule(date)


def _fallback_schedule(date, try_dc_first=False):
    """回退到 500.com，并把来源标记改掉——下游靠它判断数据从哪来。"""
    matches = fetch_beidan_schedule(date, source='dc') if try_dc_first else []
    if not matches:
        matches = fetch_beidan_schedule(date, source='jczq' if try_dc_first else 'jczq')
    for match in matches:
        match['source'] = '500.com'
    return matches


def fetch_okooo_asian_history(match_id):
    """亚盘水位变化。**四个候选路径依次试**，第一个解析出记录的就返回。"""
    return _fetch_history(
        match_id, _parsing.ASIAN, '亚盘赔率变化', '亚盘赔率',
        ('ah', 'hodds', 'odds', 'history'))


def fetch_okooo_goals_history(match_id):
    """大小球水位变化。只有一个路径——与亚盘的四个不同，原样保留。"""
    return _fetch_history(
        match_id, _parsing.GOALS, '总进球赔率变化', '总进球赔率', ('goals',))


def fetch_okooo_cs_history(match_id):
    """比分盘赔率变化。同样只有一个路径。"""
    return _fetch_history(
        match_id, _parsing.CORRECT_SCORE, '比分赔率变化', '比分赔率', ('cs',))


def fetch_beidan_schedule(date=None, source='jczq'):
    """500.com 的赛程。解析在领域层，**时钟由这一层注入**。"""
    if date is None:
        date = time.strftime('%Y-%m-%d')

    url = referer = DC_SCHEDULE_URL if source == 'dc' else SCHEDULE_URL
    log.info(f"抓取北单赛程({source}): {date}")

    try:
        html = fetch(url, referer=referer)
        if not html:
            return []
        matches = _parsing.parse_500_schedule(html, date, datetime.now())
        log.info(f"获取到 {len(matches)} 场未完结北单比赛")
        return matches
    except Exception as e:
        # **这里兜住的不只是网络错误**：解析层在拿不到开赛时刻时会抛
        # TypeError（见 `_match_status` 的说明），于是整份赛程变成空列表。
        # 迁移前就是这样，原样保留——修它会改变回退路径的返回值。
        log.error(f"抓取北单赛程失败: {e}")
        return []


def fetch_jczq_schedule(date=None):
    return fetch_beidan_schedule(date, source='jczq')


def fetch_zqdc_schedule(date=None):
    return fetch_beidan_schedule(date, source='dc')


