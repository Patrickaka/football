# -*- coding: utf-8 -*-
"""福彩3D线上预测记录、结算与稳定性"""

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
    EXPLORATION_RATE, ONLINE_PREDICTION_FILE, PREDICTOR_VERSION, RECENT_RECOMMEND_WINDOW, ZU6_RECENT_DECAY, ZU6_RECENT_PENALTY, ZU6_RECENT_WINDOW,
)
from .features import (
    max_digit_overlap,
)
from .scoring import (
    refresh_persisted_window_weights,
)
from .fusion import (
    settle_strategy_records,
)

def load_recent_3d_recommendations():
    """加载最近推荐历史"""
    try:
        return kv_store.load('lottery3d_recent_recommend', [])
    except Exception as e:
        log.error(f"加载推荐历史失败: {e}")
        return []


def load_recent_zu6_four():
    """加载最近组六四码历史"""
    try:
        return kv_store.load('lottery3d_recent_zu6', [])
    except Exception as e:
        log.error(f"加载组六四码历史失败: {e}")
        return []


def load_online_predictions():
    """加载线上预测记录"""
    try:
        return kv_store.load('lottery3d_online_predictions', [])
    except Exception as e:
        log.error(f"加载线上预测记录失败: {e}")
        return []



# ─── 领域层适配 ───
#
# 结算、统计、稳定度、组六轮换降权都在 `domain/numeric/lottery3d/records.py`。
# **这里留着全部副作用**：kv 读写、时间戳、日志。领域层要能喂任意记录跑，
# 一旦它自己去 `kv_store.load()`，测一行判断都得先造一套存储。

from src.domain.numeric.lottery3d import records as _records
from src.domain.numeric.lottery3d.recommendations import max_digit_overlap as _overlap

RECENT_RECOMMEND_KEY = 'lottery3d_recent_recommend'
RECENT_ZU6_KEY = 'lottery3d_recent_zu6'
ONLINE_PREDICTIONS_KEY = 'lottery3d_online_predictions'

get_stability_level = _records.stability_level
recommendation_stability = _records.stability


def _now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def adjust_exploration_rate(stability):
    return _records.exploration_rate(stability, EXPLORATION_RATE)


def recent_zu6_digit_penalty(score, recent_zu6, base=ZU6_RECENT_PENALTY,
                             decay=ZU6_RECENT_DECAY):
    return _records.recent_zu6_penalty(score, recent_zu6, ZU6_RECENT_WINDOW,
                                       base, decay)


def settle_prediction(record, actual):
    return _records.settle(record, actual, _overlap)


def save_recent_zu6_four(period, digits):
    """保存组六四码历史（按期号去重，仅保留最近 N 期）。"""
    try:
        history = _records.upsert_by_period(
            load_recent_zu6_four(), period, {'digits': list(digits)},
            ZU6_RECENT_WINDOW)
        kv_store.save(RECENT_ZU6_KEY, history)
    except Exception as e:
        log.error(f"保存组六四码历史失败: {e}")


def save_recent_3d_recommendations(period, recommendations):
    """保存推荐历史（按期号去重）。

    同一期号多次调用只保留最后一次——**推荐历史以「期」为单位，
    不是以「页面被访问了几次」为单位**，否则稳定度立刻虚高。
    """
    try:
        history = load_recent_3d_recommendations()
        existing = (history and isinstance(history[-1], dict)
                    and history[-1].get('period') == period)
        stamp = ({'updated_at': _now()} if existing
                 else {'created_at': _now(), 'updated_at': _now()})
        history = _records.upsert_by_period(
            history, period, {'recommendations': recommendations},
            RECENT_RECOMMEND_WINDOW, stamp)
        kv_store.save(RECENT_RECOMMEND_KEY, history)
        log.info(f"推荐历史已保存（期号: {period}）")
    except Exception as e:
        log.error(f"保存推荐历史失败: {e}")


def save_online_prediction(period, last_draw, zhixuan_top3, zhixuan, danma, kill):
    """保存线上预测记录。

    同一期已结算的记录**不覆盖**：那是当时真的发出去的推荐，改了它等于
    篡改自己的成绩单。未结算的只补空字段，同样保留首次发布的号码。

    迁移前这里还有一句 `os.makedirs(os.path.dirname(ONLINE_PREDICTION_FILE))`
    ——为一个**从来不写的文件**建目录，记录实际进的是 kv。线上那个路径
    根本不存在。删了。
    """
    try:
        records = load_online_predictions()
        record = {
            'version': PREDICTOR_VERSION,
            'period': period,
            'last_draw': last_draw,
            'zhixuan_top3': [item['num'] for item in zhixuan_top3],
            'zhixuan': [item['num'] for item in zhixuan],
            'danma': danma,
            'kill': kill,
            'actual': None,
            'settled': False,
            'hit_top3': False,
            'hit_top30': False,
            'ge2_digit': False,
            'created_at': _now(),
        }

        index = next((i for i, r in enumerate(records) if r['period'] == period), None)
        if index is not None:
            existing = records[index]
            if existing.get('settled'):
                log.info(f"预测记录已结算，跳过更新: {period}")
                return
            for field in ('zhixuan_top3', 'zhixuan', 'danma', 'kill', 'created_at'):
                record[field] = existing[field]
            records[index] = record
        else:
            records.append(record)

        kv_store.save(ONLINE_PREDICTIONS_KEY, records)
        log.info(f"线上预测记录已保存: {period}")
    except Exception as e:
        log.error(f"保存线上预测记录失败: {e}")


def settle_pending_online_predictions(periods, numbers):
    """按最新开奖结算未回填的记录，并在有结算时刷新窗口权重。"""
    records = load_online_predictions()
    if not records:
        return 0

    settled_count, changed = _records.settle_all(records, periods, numbers, _overlap)
    if changed:
        try:
            kv_store.save(ONLINE_PREDICTIONS_KEY, records)
            log.info(f"线上预测已结算 {settled_count} 条")
        except Exception as e:
            log.error(f"保存线上预测结算结果失败: {e}")

    settle_strategy_records(periods, numbers)

    if settled_count > 0:
        try:
            refresh_persisted_window_weights(numbers, periods[-1] if periods else None)
        except Exception as e:
            log.warning(f"回填后刷新窗口权重失败: {e}")
    return settled_count


def calculate_online_stats():
    return _records.online_stats(load_online_predictions())
