# -*- coding: utf-8 -*-
"""北单玩法赔率抓取：比分/总进球/半全场"""

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

from .config import (
    BASE_URL, SCHEDULE_URL,
)
from .fetching import (
    fetch,
)

# ─── 领域层适配 ───
#
# 三张表的解析在 `src/domain/sports/beidan/parsing.py`。留在这里的只有
# 取数、拼 URL、记日志——**「哪一行没读懂」由领域层返回，怎么记由这里决定**。

from src.domain.sports.beidan import parsing as _parsing


def _fetch_table(date, game_type, label, parse):
    if date is None:
        date = time.strftime('%Y-%m-%d')

    url = f'{BASE_URL}/football/jc/data/ssq_match_info.jsp?date={date}&gameType={game_type}'
    log.info(f"抓取北单{label}数据: {date}")

    try:
        content = fetch(url, referer=SCHEDULE_URL)
        # 同上：改成 `is None` 输出等价，短路只是省一次空转。
        if not content:
            return {}
        result, failures = parse(content)
        for line, reason in failures:
            log.warning(f"解析{label}数据失败: {line} - {reason}")
        return result
    except Exception as e:
        log.error(f"抓取北单{label}数据失败: {e}")
        return {}


def fetch_beidan_bifen(date=None):
    """比分盘赔率。**坏价只丢那一对**，整行还留着——比分一场几十个选项。"""
    return _fetch_table(date, 'bifen', '比分',
                        lambda content: (_parsing.parse_pair_table(content), []))


def fetch_beidan_zjq(date=None):
    """总进球赔率。定长八档，**一个坏价带走整行**。"""
    return _fetch_table(date, 'zjq', '总进球', lambda content:
                        _parsing.parse_column_table(
                            content, _parsing.ZJQ_COLUMNS, _parsing.ZJQ_MINIMUM))


def fetch_beidan_bqc(date=None):
    """半全场赔率。定长九档，**一个坏价带走整行**。"""
    return _fetch_table(date, 'bqc', '半全场', lambda content:
                        _parsing.parse_column_table(
                            content, _parsing.BQC_COLUMNS, _parsing.BQC_MINIMUM))


