#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""排列五磁盘缓存：预测缓存与回测缓存的读写"""

import json
import logging
import os
from datetime import datetime, date

from ..common.paths import data_path

logger = logging.getLogger(__name__)

# 磁盘缓存文件路径
_DATA_DIR = None  # 延迟初始化
_PREDICTION_CACHE_FILE = None
_BACKTEST_CACHE_FILE = None


def _init_cache_paths():
    global _DATA_DIR, _PREDICTION_CACHE_FILE, _BACKTEST_CACHE_FILE
    if _DATA_DIR is None:
        _DATA_DIR = data_path('')
        _PREDICTION_CACHE_FILE = data_path('pailie5_prediction_cache.json')
        _BACKTEST_CACHE_FILE = data_path('pailie5_backtest_cache.json')


def _is_today_cache(timestamp):
    """判断时间戳是否是今天的缓存"""
    if timestamp is None:
        return False
    try:
        return datetime.fromtimestamp(timestamp).date() == date.today()
    except:
        return False


def _save_prediction_cache(cache_data, cache_time):
    """保存预测缓存到磁盘"""
    _init_cache_paths()
    try:
        with open(_PREDICTION_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                'cache_time': cache_time,
                'data': cache_data
            }, f, ensure_ascii=False, default=str)
        logger.info("预测缓存已保存到磁盘")
    except Exception as e:
        logger.warning(f"保存预测缓存失败: {e}")


def _load_prediction_cache():
    """从磁盘加载预测缓存"""
    _init_cache_paths()
    try:
        if os.path.exists(_PREDICTION_CACHE_FILE):
            with open(_PREDICTION_CACHE_FILE, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            cache_time = cached.get('cache_time')
            if _is_today_cache(cache_time):
                logger.info("从磁盘加载预测缓存")
                return cached.get('data'), cache_time
    except Exception as e:
        logger.warning(f"加载预测缓存失败: {e}")
    return None, None


def _save_backtest_cache(result, cache_time, history_count):
    """保存回测缓存到磁盘"""
    _init_cache_paths()
    try:
        with open(_BACKTEST_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                'cache_time': cache_time,
                'data': result,
                'history_count': history_count
            }, f, ensure_ascii=False, default=str)
        logger.info("回测缓存已保存到磁盘")
    except Exception as e:
        logger.warning(f"保存回测缓存失败: {e}")


def _load_backtest_cache(current_count):
    """从磁盘加载回测缓存，检查历史数据条数是否变化"""
    _init_cache_paths()
    try:
        if os.path.exists(_BACKTEST_CACHE_FILE):
            with open(_BACKTEST_CACHE_FILE, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            cache_time = cached.get('cache_time')
            history_count = cached.get('history_count', 0)
            # 缓存有效条件：历史数据条数未变 且 是今日缓存
            if history_count == current_count and _is_today_cache(cache_time):
                logger.info("从磁盘加载回测缓存")
                return cached.get('data'), cache_time
            else:
                logger.info(f"回测缓存已过期（历史数据 {history_count}→{current_count}）")
    except Exception as e:
        logger.warning(f"加载回测缓存失败: {e}")
    return None, None
