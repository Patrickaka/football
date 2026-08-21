# -*- coding: utf-8 -*-
"""福彩3D历史数据抓取"""

import json
import math
import os
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from contextlib import contextmanager
from itertools import combinations, product

from ..common.logger import setup_logger
from ..common.data_cache import cached_fetch
from ..common import kv_store

log = setup_logger('lottery3d')

from .config import (
    URL,
)

def _fetch_data_internal(url=URL, retries=3, timeout=30):
    """内部数据抓取函数（带重试，应对上游瞬时超时）"""
    log.debug('fetch 3D data')
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            html = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "ignore")
            break
        except Exception as e:
            last_err = e
            log.warning('3D 抓取第 %d/%d 次失败: %s', attempt + 1, retries, e)
            time.sleep(2 * (attempt + 1))
    else:
        # 全部重试失败，向上抛出由 cached_fetch 兜底（旧缓存）或 run_prediction 处理。
        raise last_err
    compact = re.sub(r"\s+", " ", html)
    pattern = re.compile(
        r'<td>(\d{7})期</td>\s*<td>(\d{4}-\d{2}-\d{2})</td>\s*<td>'
        r'\s*<span\s+class="ball">(\d)</span>\s*'
        r'<span\s+class="ball">(\d)</span>\s*'
        r'<span\s+class="ball">(\d)</span>'
    )
    rows = pattern.findall(compact)
    data = [(pid, dt, (int(a), int(b), int(c))) for pid, dt, a, b, c in rows]
    data.reverse()
    return data


def fetch_data(url=URL, force_refresh=False):
    """获取历史开奖数据（带缓存，每天只抓取一次）"""
    return cached_fetch('lottery3d', lambda: _fetch_data_internal(url), force_refresh)


