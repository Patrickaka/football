# -*- coding: utf-8 -*-
"""北单数据抓取：500.com 与中国足彩网页面、赛程。"""

import sys
import math
import re
from collections import defaultdict
import time
import json
import urllib.request
import urllib.error
import urllib.parse
import random
import requests
import threading

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
    DC_SCHEDULE_URL, HEADERS, ZGZCW_ANALYSIS_BASE, ZGZCW_DANCHANG_URL,
    ZGZCW_HEADERS, SCHEDULE_URL, _is_zgzcw_blocked, _mark_zgzcw_blocked,
    _zgzcw_session, ensure_zgzcw_session,
)

# 北单与足球打的是同一个 odds.500.com。两边各跑一条预热线程，若各自独立发请求，
# 就会互相挤占源站配额、把对方推进 429。这里让 500.com 的请求共用足球那条限速
# 令牌流：请求前领号，撞限流则写进全局冷却，足球侧也会一起退避。
# 中国足彩网走另一个域名和独立 session，不参与这条预算。
_SHARED_RATE_LIMIT_HOST = 'odds.500.com'
_ZGZCW_REQUEST_LOCK = threading.Lock()
_ZGZCW_MIN_INTERVAL = 0.5
_zgzcw_last_request = 0.0
ZGZCW_PRIMARY_COMPANY_ID = '2'
ZGZCW_PRIMARY_COMPANY = '36*'


# ─── 领域层适配（页面解析）───
#
# 走势历史的解析在 `src/domain/sports/beidan/parsing.py`——迁移前这里是
# 三段几乎相同的一百行，差别只在表头关键字、列数、字段名和几条校验上。
# 留在这一层的只有：拼 URL、抓、记日志。

from src.domain.sports.beidan import parsing as _parsing


def _fetch_history(match_id, detail_parser, summary_parser, what, unit, path,
                   company_id=ZGZCW_PRIMARY_COMPANY_ID,
                   company=ZGZCW_PRIMARY_COMPANY):
    """优先取公司明细页的完整序列，失败才退回初盘/即时盘两点摘要。

    默认选择 company_id=2 的 Bet365 完整序列；每个市场只请求一个明细页，
    不逐公司扫描十几页。调用方也可以显式指定其他公司。
    """
    company_id = str(company_id or ZGZCW_PRIMARY_COMPANY_ID)
    company = str(company or ZGZCW_PRIMARY_COMPANY)
    if not company_id.isdigit():
        company_id, company = ZGZCW_PRIMARY_COMPANY_ID, ZGZCW_PRIMARY_COMPANY
    summary_url = f'{ZGZCW_ANALYSIS_BASE}/{match_id}/{path}'
    detail_url = (
        f'{summary_url}/zhishu?company_id={company_id}&company='
        f'{urllib.parse.quote(company, safe="*")}'
    )
    log.debug(f"抓取中国足彩网{what}: match_id={match_id}")
    try:
        html = fetch_zgzcw(detail_url, referer=summary_url)
        records = detail_parser(html) if html else []
        if records:
            for record in records:
                record.update({'company_id': company_id, 'company': company})
            log.info(f"从 {detail_url} 获取到 {len(records)} 条{unit}记录")
            return {
                'history': records,
                'history_source': 'zgzcw_company_detail',
                'company_id': company_id,
                'company': company,
                'samples': len(records),
            }
    except Exception as e:
        log.warning(f"抓取中国足彩网{what}明细失败({detail_url}): {e}")

    try:
        html = fetch_zgzcw(summary_url, referer=ZGZCW_DANCHANG_URL)
        records = summary_parser(html) if html else []
        if records:
            log.info(f"从 {summary_url} 获取到 {len(records)} 条{unit}摘要")
            return {
                'history': records,
                'history_source': 'zgzcw_opening_current_fallback',
                'company_id': '0',
                'company': '平均*',
                'samples': len(records),
            }
    except Exception as e:
        log.warning(f"抓取中国足彩网{what}摘要失败({summary_url}): {e}")

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


def fetch_zgzcw(url, encoding='utf-8', referer=None, max_retries=2):
    global _zgzcw_session, _zgzcw_last_request
    
    if _is_zgzcw_blocked():
        log.debug(f"中国足彩网处于短时熔断，跳过请求: {url}")
        return None

    # 预热 session（拿 cookie）。**迁移前这发生在 import 期**，于是任何
    # import beidan 的测试都在联网；现在推迟到真正要发请求的这一刻。
    # 幂等，只有第一次会真的握手。
    ensure_zgzcw_session()
    
    for attempt in range(max_retries):
        try:
            headers = {}
            if referer:
                headers['Referer'] = referer
            
            # 同一会话串行且限制到每秒最多两次，避免并行走势抓取形成突发流量。
            with _ZGZCW_REQUEST_LOCK:
                wait = _ZGZCW_MIN_INTERVAL - (time.monotonic() - _zgzcw_last_request)
                if wait > 0:
                    time.sleep(wait)
                _zgzcw_last_request = time.monotonic()
                # 快速失败后走现有备用数据源，避免单个站点拖住整批预测。
                resp = _zgzcw_session.get(url, headers=headers, timeout=(5, 12))
            
            if resp.status_code == 403 or resp.status_code == 503:
                log.warning(f"中国足彩网返回 {resp.status_code}: {url}")
                _mark_zgzcw_blocked()
                _zgzcw_session = requests.Session()
                _zgzcw_session.headers.update(ZGZCW_HEADERS)
                return None
            
            if resp.status_code != 200:
                log.warning(f"HTTP Error {resp.status_code} for {url}")
                if resp.status_code >= 500 and attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
            
            try:
                result = resp.content.decode('utf-8')
            except:
                result = resp.content.decode('gb18030', errors='replace')
            
            if any(marker in result.lower() for marker in ('captcha', '访问验证', '安全验证')):
                log.warning(f"中国足彩网返回验证页: {url}")
                _mark_zgzcw_blocked()
                _zgzcw_session = requests.Session()
                _zgzcw_session.headers.update(ZGZCW_HEADERS)
                return None
            
            return result
            
        except requests.RequestException as e:
            log.warning(f"Request Error {e} for {url} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    log.error(f"Failed to fetch {url} after {max_retries} attempts")
    return None


def fetch_zgzcw_schedule(date=None):
    """中国足彩网北单赛程，取不到就回退到 500.com。

    **三种回退的日志要分开**：页面为空（多半是 WAF）、页面结构不对
    （表格数不够）、解析出来一场未完结的都没有。§十一·3 那类
    「接口返回 200 加 0 场比赛」的故障里，最难查的正是分不清这三者。
    """
    if date is None:
        date = time.strftime('%Y-%m-%d')

    url = ZGZCW_DANCHANG_URL
    log.info(f"抓取中国足彩网北单赛程: {date}")

    try:
        html = fetch_zgzcw(url, referer=ZGZCW_DANCHANG_URL)
        if not html:
            log.warning("中国足彩网页面返回为空，尝试500.com备用数据源")
            return _fallback_schedule(date, try_dc_first=True)

        log.debug(f"中国足彩网页面HTML长度: {len(html)}")
        matches = _parsing.parse_zgzcw_schedule(html, date)
        if not matches:
            log.warning("中国足彩网未找到指定日期的未完结比赛，尝试500.com备用数据源")
            return fetch_beidan_schedule(date)

        log.info(f"中国足彩网获取到 {len(matches)} 场未完结北单比赛")
        return matches
    except Exception as e:
        log.error(f"抓取中国足彩网北单赛程失败: {e}，尝试500.com备用数据源")
        return _fallback_schedule(date)


def _fallback_schedule(date, try_dc_first=False):
    """回退到 500.com，并把来源标记改掉——下游靠它判断数据从哪来。"""
    matches = fetch_beidan_schedule(date, source='dc') if try_dc_first else []
    if not matches:
        matches = fetch_beidan_schedule(date, source='jczq' if try_dc_first else 'jczq')
    for match in matches:
        match['source'] = '500.com'
    return matches


def fetch_zgzcw_asian_history(
        match_id, company_id=ZGZCW_PRIMARY_COMPANY_ID,
        company=ZGZCW_PRIMARY_COMPANY):
    """完整亚盘变化序列；默认使用 Bet365。"""
    return _fetch_history(
        match_id, _parsing.parse_zgzcw_asian_company_history,
        _parsing.parse_zgzcw_asian_history,
        '亚盘赔率变化', '亚盘赔率', 'ypdb', company_id, company)


def fetch_zgzcw_goals_history(
        match_id, company_id=ZGZCW_PRIMARY_COMPANY_ID,
        company=ZGZCW_PRIMARY_COMPANY):
    """完整大小球变化序列；默认使用 Bet365。"""
    return _fetch_history(
        match_id, _parsing.parse_zgzcw_goals_company_history,
        _parsing.parse_zgzcw_goals_history,
        '总进球赔率变化', '总进球赔率', 'dxdb', company_id, company)


def fetch_zgzcw_cs_history(match_id):
    """中国足彩网不公开比分盘历史；明确返回空，避免把欧亚盘误当比分。"""
    return {'history': []}


def fetch_beidan_schedule(date=None, source='jczq'):
    """500.com 的赛程。解析在领域层，**时钟由这一层注入**。"""
    if date is None:
        date = time.strftime('%Y-%m-%d')

    url = referer = DC_SCHEDULE_URL if source == 'dc' else SCHEDULE_URL
    log.info(f"抓取北单赛程({source}): {date}")

    try:
        html = fetch(url, referer=referer)
        # 改成 `is None` 输出等价（空串解析出来也是空列表），
        # 短路只是省一次空转。
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


