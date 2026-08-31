# -*- coding: utf-8 -*-
"""北单常量配置与中国足彩网会话。"""

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
import threading
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

ZGZCW_BASE = 'https://cp.zgzcw.com'
ZGZCW_DANCHANG_URL = f'{ZGZCW_BASE}/lottery/bdplayvsforJsp.action?lotteryId=210'
ZGZCW_ANALYSIS_BASE = 'https://fenxi.zgzcw.com'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Referer': SCHEDULE_URL,
}

ZGZCW_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
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
    'Referer': ZGZCW_BASE + '/',
}

_zgzcw_session = requests.Session()
_zgzcw_session.headers.update(ZGZCW_HEADERS)

_zgzcw_blocked = False
_zgzcw_blocked_time = 0

def _mark_zgzcw_blocked():
    global _zgzcw_blocked, _zgzcw_blocked_time
    _zgzcw_blocked = True
    _zgzcw_blocked_time = time.time()

def _is_zgzcw_blocked():
    global _zgzcw_blocked, _zgzcw_blocked_time
    if not _zgzcw_blocked:
        return False
    if time.time() - _zgzcw_blocked_time > 60:
        _zgzcw_blocked = False
        return False
    return True

# 会话初始化只在内存中完成，不额外请求首页；真正需要的数据页是唯一网络请求。
_zgzcw_session_warmed = False
_zgzcw_warm_lock = threading.Lock()


def ensure_zgzcw_session():
    """首次真正抓取中国足彩网时才预热 session。"""
    global _zgzcw_session_warmed
    if _zgzcw_session_warmed:
        return
    with _zgzcw_warm_lock:
        if _zgzcw_session_warmed:
            return
        _zgzcw_session_warmed = True

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

