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

def fetch_beidan_bifen(date=None):
    if date is None:
        date = time.strftime('%Y-%m-%d')
    
    url = f'{BASE_URL}/football/jc/data/ssq_match_info.jsp?date={date}&gameType=bifen'
    log.info(f"抓取北单比分数据: {date}")
    
    try:
        content = fetch(url, referer=SCHEDULE_URL)
        if not content:
            return {}
        
        result = {}
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('|')
            if len(parts) < 2:
                continue
            
            match_id = parts[0]
            odds = {}
            
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    score = parts[i]
                    try:
                        odd = float(parts[i + 1])
                        odds[score] = odd
                    except ValueError:
                        pass
            
            if odds:
                result[match_id] = odds
        
        return result
    
    except Exception as e:
        log.error(f"抓取北单比分数据失败: {e}")
        return {}


def fetch_beidan_zjq(date=None):
    if date is None:
        date = time.strftime('%Y-%m-%d')
    
    url = f'{BASE_URL}/football/jc/data/ssq_match_info.jsp?date={date}&gameType=zjq'
    log.info(f"抓取北单总进球数据: {date}")
    
    try:
        content = fetch(url, referer=SCHEDULE_URL)
        if not content:
            return {}
        
        result = {}
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('|')
            if len(parts) < 8:
                continue
            
            match_id = parts[0]
            try:
                zjq_odds = {
                    '0': float(parts[1]) if parts[1] else None,
                    '1': float(parts[2]) if parts[2] else None,
                    '2': float(parts[3]) if parts[3] else None,
                    '3': float(parts[4]) if parts[4] else None,
                    '4': float(parts[5]) if parts[5] else None,
                    '5': float(parts[6]) if parts[6] else None,
                    '6': float(parts[7]) if parts[7] else None,
                    '7+': float(parts[8]) if len(parts) > 8 else None,
                }
                result[match_id] = zjq_odds
            except Exception as e:
                log.warning(f"解析总进球数据失败: {line} - {e}")
        
        return result
    
    except Exception as e:
        log.error(f"抓取北单总进球数据失败: {e}")
        return {}


def fetch_beidan_bqc(date=None):
    if date is None:
        date = time.strftime('%Y-%m-%d')
    
    url = f'{BASE_URL}/football/jc/data/ssq_match_info.jsp?date={date}&gameType=bqc'
    log.info(f"抓取北单半全场数据: {date}")
    
    try:
        content = fetch(url, referer=SCHEDULE_URL)
        if not content:
            return {}
        
        result = {}
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('|')
            if len(parts) < 10:
                continue
            
            match_id = parts[0]
            try:
                bqc_odds = {
                    '胜胜': float(parts[1]) if parts[1] else None,
                    '胜平': float(parts[2]) if parts[2] else None,
                    '胜负': float(parts[3]) if parts[3] else None,
                    '平胜': float(parts[4]) if parts[4] else None,
                    '平平': float(parts[5]) if parts[5] else None,
                    '平负': float(parts[6]) if parts[6] else None,
                    '负胜': float(parts[7]) if parts[7] else None,
                    '负平': float(parts[8]) if parts[8] else None,
                    '负负': float(parts[9]) if parts[9] else None,
                }
                result[match_id] = bqc_odds
            except Exception as e:
                log.warning(f"解析半全场数据失败: {line} - {e}")
        
        return result
    
    except Exception as e:
        log.error(f"抓取北单半全场数据失败: {e}")
        return {}


