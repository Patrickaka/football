# -*- coding: utf-8 -*-
"""快乐8统计工具：超几何分布、显著性校正、验证评分"""

import math
import json
import time
import hashlib
import uuid
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple
from itertools import combinations
from pathlib import Path

from src.common.paths import data_path
from src.common.repositories import doc_store
from src.common.logger import setup_logger

log = setup_logger('kl8')

from .config import (
    FUSHI_CONFIG, KL8_DRAW_COUNT, KL8_NUM_RANGE,
)

def hypergeom_pmf(pick_n: int, hits: int) -> float:
    """超几何分布PMF: 从80个号码中选pick_n个，开出20个，命中hits个的概率

    P(X=hits) = C(20,hits) * C(60,pick_n-hits) / C(80,pick_n)
    """
    from math import comb
    if hits < 0 or hits > min(pick_n, KL8_DRAW_COUNT):
        return 0.0
    if pick_n - hits > KL8_NUM_RANGE - KL8_DRAW_COUNT:
        return 0.0
    return comb(KL8_DRAW_COUNT, hits) * comb(KL8_NUM_RANGE - KL8_DRAW_COUNT, pick_n - hits) / comb(KL8_NUM_RANGE, pick_n)


def hypergeom_p_ge(pick_n: int, min_hits: int) -> float:
    """超几何分布 P(X >= min_hits)"""
    total = 0.0
    for k in range(min_hits, min(pick_n, KL8_DRAW_COUNT) + 1):
        total += hypergeom_pmf(pick_n, k)
    return total


def hypergeom_expected(pick_n: int) -> float:
    """超几何分布期望命中数 = pick_n * 20/80"""
    return pick_n * KL8_DRAW_COUNT / KL8_NUM_RANGE


def _parse_play_pick_n(play_type: str) -> Optional[int]:
    """解析玩法用于置换检验的选号数量。"""
    if play_type in FUSHI_CONFIG:
        return FUSHI_CONFIG[play_type]['pool_size']
    if play_type.startswith('select_'):
        try:
            return int(play_type.split('_')[1])
        except (ValueError, IndexError):
            return None
    return None


def _play_lift(result: Dict, play_type: str) -> float:
    """从回测结果中取出标准玩法或复式玩法的Lift。"""
    if play_type in FUSHI_CONFIG:
        fushi_result = result.get(play_type, {})
        pool_mean = fushi_result.get('pool_mean_hits', 0)
        expected = fushi_result.get(
            'pool_expected_random',
            hypergeom_expected(FUSHI_CONFIG[play_type]['pool_size']),
        )
        return (pool_mean - expected) / expected if expected > 0 else 0
    return result.get(play_type, {}).get('lift', 0)


def _prize_tier_thresholds(play_type: str) -> List[str]:
    if play_type in FUSHI_CONFIG:
        base = FUSHI_CONFIG[play_type]['base_pick']
        if base <= 4:
            return ['>=2', '>=3']
        return ['>=3'] if FUSHI_CONFIG[play_type]['pool_size'] <= 7 else ['>=4', '>=5']
    pick_n = _parse_play_pick_n(play_type) or 5
    if pick_n <= 4:
        return ['>=2', '>=3']
    if pick_n == 5:
        return ['>=4', '>=3']
    if pick_n == 6:
        return ['>=5', '>=4']
    if pick_n <= 7:
        return ['>=3', '>=4']
    return ['>=4', '>=5']


def _hit_rate_priority_thresholds(play_type: str) -> List[str]:
    """Thresholds used to rank candidates for practical hit-rate goals."""
    if play_type in FUSHI_CONFIG:
        base = FUSHI_CONFIG[play_type]['base_pick']
        if base <= 4:
            return ['>=2']
        return ['>=3', '>=4'] if FUSHI_CONFIG[play_type]['pool_size'] <= 7 else ['>=4', '>=5']
    pick_n = _parse_play_pick_n(play_type) or 5
    if pick_n <= 4:
        return ['>=2']
    if pick_n == 5:
        return ['>=4', '>=3']
    if pick_n == 6:
        return ['>=5', '>=4']
    if pick_n <= 7:
        return ['>=3', '>=4']
    return ['>=3', '>=4']


def _hit_rate_priority_score(metrics: Dict, play_type: str) -> Tuple[float, Dict[str, float]]:
    """Score candidates by key hit thresholds instead of only mean hits."""
    probabilities = metrics.get('probabilities') or {}
    theoretical = metrics.get('theoretical_probs') or {}
    thresholds = _hit_rate_priority_thresholds(play_type)
    weights = [0.70, 0.30] if len(thresholds) > 1 else [1.0]
    if play_type in {'select_5', 'select_6'}:
        weights = [0.86, 0.14]

    detail = {}
    score = 0.0
    for threshold, weight in zip(thresholds, weights):
        actual = float(probabilities.get(threshold, 0) or 0)
        baseline = float(theoretical.get(threshold, 0) or 0)
        lift = (actual - baseline) / baseline if baseline > 0 else 0.0
        detail[threshold] = round(lift, 6)
        score += weight * lift

    return score, detail


def _practical_validation_score(metrics: Dict, play_type: str, mean_lift: Optional[float] = None) -> Tuple[float, Dict]:
    """Rank candidates by prize-relevant hit rates before mean-hit lift."""
    hit_rate_score, hit_rate_detail = _hit_rate_priority_score(metrics, play_type)
    lift = _play_lift({play_type: metrics}, play_type) if mean_lift is None else float(mean_lift or 0)
    roi = float(metrics.get('profit_roi', 0) or 0)
    random_roi = float(metrics.get('random_profit_roi', 0) or 0)
    roi_delta = roi - random_roi
    return_multiple = float(metrics.get('return_multiple', 0) or 0)

    score = (
        hit_rate_score
        + lift * 0.20
        + max(0.0, roi_delta) * 0.08
        + max(0.0, return_multiple - 1.0) * 0.04
    )
    if roi < random_roi:
        score -= min(0.25, (random_roi - roi) * 0.05)

    return score, {
        'hit_rate_score': round(hit_rate_score, 6),
        'hit_rate_lifts': hit_rate_detail,
        'mean_lift': round(lift, 6),
        'roi_delta_vs_random': round(roi_delta, 6),
        'return_multiple': round(return_multiple, 6),
    }


def _play_accuracy_profile(
    play_type: str,
    pick_n: int,
    selected_mode: str,
    variants: Optional[Dict] = None,
    target_hits: Optional[int] = None,
) -> Dict:
    """Return the theoretical hit profile for a KL8 play."""
    expected_hits = hypergeom_expected(pick_n)
    probabilities = {
        f'>={k}': round(hypergeom_p_ge(pick_n, k), 6)
        for k in range(1, pick_n + 1)
    }
    exact = {
        str(k): round(hypergeom_pmf(pick_n, k), 6)
        for k in range(0, pick_n + 1)
    }
    if pick_n == 5:
        practical_goal = '选5中奖需中3个（理论概率约9.7%）；中2个不中奖属正常随机波动，请勿误判为预测失准'
    elif pick_n == 6:
        practical_goal = '选6中奖需中3个（理论概率约16%）；中2个不中奖属正常随机波动'
    else:
        practical_goal = '命中阈值按玩法奖级和回测优先级评估'

    return {
        'expected_hits_random': round(expected_hits, 4),
        'zero_hit_probability_random': exact.get('0', 0.0),
        'theoretical_probabilities': probabilities,
        'exact_hit_probabilities': exact,
        'key_thresholds': _hit_rate_priority_thresholds(play_type),
        'target_hits': target_hits,
        'target_probability_random': (
            round(hypergeom_p_ge(pick_n, target_hits), 6)
            if target_hits else None
        ),
        'selected_mode': selected_mode,
        'variant_count': len(variants or {}),
        'practical_goal': practical_goal,
        'disclaimer': '快乐8每个号码理论开出率为25%，短期推荐命中0-2个属于常见随机波动。',
    }


def benjamini_hochberg_fdr(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR校正

    对同一玩法下所有特征、窗口、权重的p值做FDR校正
    步骤:
    1. p值从小到大排序
    2. 每个p值校正为: p_adjusted = p * m / rank
       (m = 总检验次数, rank = 该p值在排序中的序位)
    3. 确保单调性: 从大到小遍历，取min(p_adjusted, 下一个校正值)

    参数:
        p_values: 原始p值列表

    返回:
        校正后的p值列表（与输入顺序对应）
    """
    if not p_values:
        return []

    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [0.0] * m
    prev_adjusted = 1.0

    # 从最大p值开始，确保单调性
    for i in range(m - 1, -1, -1):
        original_idx, p = indexed[i]
        rank = i + 1  # 序位从1开始
        bh_adjusted = min(1.0, p * m / rank)
        # 确保单调性: 不比后面(更小rank)的校正值更大
        adjusted[original_idx] = min(bh_adjusted, prev_adjusted)
        prev_adjusted = adjusted[original_idx]

    return adjusted


def bonferroni_correction(p_value: float, n_experiments: int) -> float:
    """Bonferroni校正（保守版多重检验校正）

    p_adjusted = min(1.0, p_value * number_of_experiments)

    参数:
        p_value: 原始p值
        n_experiments: 总检验次数

    返回:
        校正后的p值
    """
    return min(1.0, p_value * n_experiments)


