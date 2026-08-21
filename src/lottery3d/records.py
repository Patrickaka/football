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


def recent_zu6_digit_penalty(score, recent_zu6, base=ZU6_RECENT_PENALTY, decay=ZU6_RECENT_DECAY):
    """对近期组六四码用过的数字按新近度降权，返回调整后的数字评分列表。

    3D 选哪些码无 edge（任意 4 互异码组六覆盖率恒为 4*6/1000），故轮换零命中代价。
    最近一期惩罚最重，越久远越轻；连续多期出现的数字累计惩罚最大，优先被换出。
    """
    base = min(float(base), 3.0)
    adj = list(score)
    if not recent_zu6:
        return adj
    for age, entry in enumerate(reversed(recent_zu6[-ZU6_RECENT_WINDOW:])):
        digits = entry.get("digits", []) if isinstance(entry, dict) else entry
        w = base * (decay ** age)
        for d in digits:
            if 0 <= d < len(adj):
                adj[d] -= w
    return adj


def load_recent_zu6_four():
    """加载最近组六四码历史"""
    try:
        return kv_store.load('lottery3d_recent_zu6', [])
    except Exception as e:
        log.error(f"加载组六四码历史失败: {e}")
        return []


def save_recent_zu6_four(period, digits):
    """保存组六四码历史（按期号去重，仅保留最近 N 期）"""
    try:
        history = load_recent_zu6_four()
        if history and isinstance(history[-1], dict) and history[-1].get("period") == period:
            history[-1]["digits"] = list(digits)
        else:
            history.append({"period": period, "digits": list(digits)})
        history = history[-ZU6_RECENT_WINDOW:]
        kv_store.save('lottery3d_recent_zu6', history)
    except Exception as e:
        log.error(f"保存组六四码历史失败: {e}")


def save_recent_3d_recommendations(period, recommendations):
    """保存推荐历史（按期号去重）
    
    参数：
        period: 目标期号
        recommendations: 推荐号码列表
    
    说明：同一期号多次调用时，只会保存最后一次的推荐，避免重复写入。
    推荐历史必须以"期"为单位，不要以"页面调用次数"为单位。
    """
    try:
        # 加载现有历史
        history = load_recent_3d_recommendations()

        # 按期号去重：如果已有相同期号，更新推荐；否则添加新记录
        if (
            history
            and isinstance(history[-1], dict)
            and history[-1].get("period") == period
        ):
            # 更新当前期的推荐（覆盖）
            history[-1]["recommendations"] = recommendations
            history[-1]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            # 添加新期记录
            history.append({
                "period": period,
                "recommendations": recommendations,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

        # 保持最近 N 期
        history = history[-RECENT_RECOMMEND_WINDOW:]

        kv_store.save('lottery3d_recent_recommend', history)
        log.info(f"推荐历史已保存（期号: {period}）")
    except Exception as e:
        log.error(f"保存推荐历史失败: {e}")


def load_online_predictions():
    """加载线上预测记录"""
    try:
        return kv_store.load('lottery3d_online_predictions', [])
    except Exception as e:
        log.error(f"加载线上预测记录失败: {e}")
        return []


def save_online_prediction(period, last_draw, zhixuan_top3, zhixuan, danma, kill):
    """保存线上预测记录"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(ONLINE_PREDICTION_FILE), exist_ok=True)
        
        # 加载现有记录
        records = load_online_predictions()
        
        # 创建新记录
        record = {
            "version": PREDICTOR_VERSION,
            "period": period,
            "last_draw": last_draw,
            "zhixuan_top3": [item["num"] for item in zhixuan_top3],
            "zhixuan": [item["num"] for item in zhixuan],
            "danma": danma,
            "kill": kill,
            "actual": None,
            "settled": False,
            "hit_top3": False,
            "hit_top30": False,
            "ge2_digit": False,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # 检查是否已存在相同期数的记录
        existing_index = None
        for i, r in enumerate(records):
            if r["period"] == period:
                existing_index = i
                break
        
        if existing_index is not None:
            old_record = records[existing_index]
            # 如果已结算，不覆盖（保留原始预测记录）
            if old_record.get("settled"):
                log.info(f"预测记录已结算，跳过更新: {period}")
                return
            # 如果未结算，只补充空字段，不覆盖 zhixuan / top3（保留首次发布）
            record["zhixuan_top3"] = old_record["zhixuan_top3"]
            record["zhixuan"] = old_record["zhixuan"]
            record["danma"] = old_record["danma"]
            record["kill"] = old_record["kill"]
            record["created_at"] = old_record["created_at"]  # 保留首次创建时间
            records[existing_index] = record
        else:
            records.append(record)
        
        kv_store.save('lottery3d_online_predictions', records)
        log.info(f"线上预测记录已保存: {period}")
    except Exception as e:
        log.error(f"保存线上预测记录失败: {e}")


def settle_prediction(record, actual):
    """结算预测记录（赛后回填）"""
    top3 = record["zhixuan_top3"]
    top30 = record["zhixuan"]

    actual_s = "".join(map(str, actual))

    record["actual"] = actual_s
    record["settled"] = True
    record["hit_top3"] = actual_s in top3
    record["hit_top30"] = actual_s in top30
    record["ge2_digit"] = max_digit_overlap(actual_s, top30) >= 2

    return record


def settle_pending_online_predictions(periods, numbers):
    """根据最新开奖数据结算未回填的线上预测记录"""
    records = load_online_predictions()
    if not records:
        return 0

    period_index = {p: i for i, p in enumerate(periods)}
    changed = False
    settled_count = 0

    for record in records:
        if record.get("settled"):
            continue
        base_period = record.get("period")
        idx = period_index.get(base_period)
        if idx is None or idx + 1 >= len(numbers):
            continue
        settle_prediction(record, numbers[idx + 1])
        record["draw_period"] = periods[idx + 1]
        changed = True
        settled_count += 1

    if changed:
        try:
            kv_store.save('lottery3d_online_predictions', records)
            log.info(f"线上预测已结算 {settled_count} 条")
        except Exception as e:
            log.error(f"保存线上预测结算结果失败: {e}")

    # 结算三路策略记录
    settle_strategy_records(periods, numbers)

    if settled_count > 0:
        try:
            refresh_persisted_window_weights(numbers, periods[-1] if periods else None)
        except Exception as e:
            log.warning(f"回填后刷新窗口权重失败: {e}")

    return settled_count


def calculate_online_stats():
    """计算线上实盘命中率统计"""
    records = load_online_predictions()
    
    settled = [r for r in records if r["settled"]]
    unsettled = [r for r in records if not r["settled"]]
    
    n = len(settled)
    if n == 0:
        return {
            "total_records": len(records),
            "settled_count": 0,
            "unsettled_count": len(unsettled),
            "hit_top3_rate": 0.0,
            "hit_top30_rate": 0.0,
            "ge2_digit_rate": 0.0,
            "by_version": {},
        }
    
    hit_top3 = sum(1 for r in settled if r["hit_top3"])
    hit_top30 = sum(1 for r in settled if r["hit_top30"])
    ge2_digit = sum(1 for r in settled if r["ge2_digit"])
    
    # 按版本统计
    by_version = {}
    for r in settled:
        version = r["version"]
        if version not in by_version:
            by_version[version] = {"count": 0, "hit_top3": 0, "hit_top30": 0}
        by_version[version]["count"] += 1
        if r["hit_top3"]:
            by_version[version]["hit_top3"] += 1
        if r["hit_top30"]:
            by_version[version]["hit_top30"] += 1
    
    for v in by_version:
        by_version[v]["hit_top3_rate"] = by_version[v]["hit_top3"] / by_version[v]["count"]
        by_version[v]["hit_top30_rate"] = by_version[v]["hit_top30"] / by_version[v]["count"]
    
    return {
        "total_records": len(records),
        "settled_count": n,
        "unsettled_count": len(unsettled),
        "hit_top3_count": hit_top3,
        "hit_top3_rate": hit_top3 / n,
        "hit_top30_count": hit_top30,
        "hit_top30_rate": hit_top30 / n,
        "ge2_digit_count": ge2_digit,
        "ge2_digit_rate": ge2_digit / n,
        "by_version": by_version,
    }


def recommendation_stability(current, history):
    """计算推荐稳定度（最近7次推荐的重叠率）
    
    参数：
        current: 当前推荐号码列表
        history: 历史推荐列表（新格式：[{"period": ..., "recommendations": [...]}, ...]）
    
    返回：
        stability: 稳定度分数 (0.0-1.0)
    """
    current_set = set(current)
    scores = []

    for old_entry in history[-7:]:
        # 兼容新格式（字典）和旧格式（列表）
        if isinstance(old_entry, dict):
            old = old_entry.get("recommendations", [])
        else:
            old = old_entry
        old_set = set(old)
        if not old_set:
            continue
        overlap = len(current_set & old_set) / len(current_set)
        scores.append(overlap)

    return sum(scores) / len(scores) if scores else 0.0


def get_stability_level(stability):
    """获取稳定度等级"""
    if stability > 0.8:
        return "high"  # 过度稳定
    elif stability < 0.3:
        return "low"   # 过度随机
    else:
        return "normal"  # 正常


def adjust_exploration_rate(stability):
    """根据稳定度调整探索率"""
    if stability > 0.8:
        return 0.25
    elif stability < 0.3:
        return 0.08
    else:
        return EXPLORATION_RATE


