# -*- coding: utf-8 -*-
"""北单常量配置、请求头、okooo 会话与 WAF 状态"""

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
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from ..common.logger import setup_logger
from ..common.paths import data_path
from ..common import kv_store

log = setup_logger('beidan')

BEIDAN_VERSION = '2026-08-20-web-gated-picks-rqspf-audit-joint-matrix-v12'
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


# 预热做没做过。**只记「试过」，不记「成功」**——失败时也置位，
# 否则每次取数都会重来一遍，而 okooo 不可达时那是两个 10 秒超时。
_okooo_session_warmed = False


def ensure_okooo_session():
    """首次真正要用 okooo 时才预热 session。

    **迁移前这是模块级调用**（`_init_okooo_session()` 直接写在文件末尾），
    于是 `import src.beidan` 就发两次 HTTP 请求加一次 `sleep(0.5)`：
    任何 import 了 beidan 的测试都在联网，CI 随第三方站点波动，
    okooo 不可达时导入还要等两个 10 秒超时。

    初始化本身没问题，问题是它发生在 import 期——**那时候还不知道这次运行
    要不要用 okooo**。跑一个纯计算的单元测试也得先跟人家握两次手。
    """
    global _okooo_session_warmed
    if _okooo_session_warmed:
        return
    _okooo_session_warmed = True
    _init_okooo_session()

BET_TYPES = {
    'spf': {'name': '胜平负', 'description': '预测比赛胜负平结果'},
    'rqspf': {'name': '让球胜平负', 'description': '主队让球后的胜负平'},
    'bifen': {'name': '比分', 'description': '预测具体比分'},
    'zjq': {'name': '总进球', 'description': '预测总进球数'},
    'bqc': {'name': '半全场', 'description': '预测半场和全场结果'},
}

MAX_GOALS = 7

# 比分/总进球 λ 主客强度分配系数
# 0.45 经 2744 场离线回测最优：比分 Top1 11.84%→12.03%、Top3 32.43%→32.94%，
# 总进球 Top2 45.26%→45.37%，且比分/总进球 LogLoss 同步下降（split 过小则主客强度被压平）
SCORE_SPLIT = 0.45

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

