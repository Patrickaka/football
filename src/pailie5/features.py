#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""排列五特征分析：频率/遗漏/马尔可夫/多窗口评分/窗口权重与推荐历史持久化"""

import logging
import math
import time
from collections import Counter, defaultdict

from ..common import kv_store
from .config import (
    EXP_DECAY, FEATURE_FLAGS, MARKOV_LAPLACE_ALPHA, MARKOV_MAX_SCORE,
    RECENT_RECOMMEND_CONSECUTIVE_PENALTY, RECENT_RECOMMEND_KV_KEY,
    RECENT_RECOMMEND_PENALTY, RECENT_RECOMMEND_WINDOW, RECENT_WINDOW,
    RECENT_WINDOWS, W_HOT_GLOBAL, W_HOT_POS, W_LAST_APPEAR, W_MARKOV,
    W_MARKOV2, W_MISS_HIGH, W_MISS_MID, WINDOW_WEIGHTS_KV_KEY,
)

logger = logging.getLogger(__name__)


def exp_weighted_counts(series, decay=EXP_DECAY):
    """指数衰减频率统计"""
    cnt = Counter()
    w = 1.0
    for item in reversed(series):
        cnt[item] += w
        w *= decay
    return cnt


def gaussian_score(value, center, sigma):
    """高斯软约束评分"""
    if sigma <= 0:
        return 0.0
    z = (value - center) / sigma
    return math.exp(-0.5 * z * z)


def _recent_slice(series, window):
    return series[-window:] if len(series) > window else list(series)


def has_consecutive_digits_5(nums):
    """5位号码是否含相邻连号（差值为1，不含9-0进位）"""
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if abs(nums[i] - nums[j]) == 1:
                return True
    return False


def odd_even_key_5(nums):
    """5位号码奇偶比"""
    odds = sum(1 for n in nums if n % 2 == 1)
    return odds, 5 - odds


def big_small_key_5(nums):
    """5位号码大小比（0-4小，5-9大）"""
    big = sum(1 for n in nums if n >= 5)
    return big, 5 - big


# ==================== 马尔可夫 ====================

def build_markov_pos(numbers_list, position):
    """按位置构建一阶马尔可夫转移矩阵"""
    trans = defaultdict(Counter)
    for i in range(len(numbers_list) - 1):
        a = numbers_list[i][position]
        b = numbers_list[i + 1][position]
        trans[a][b] += 1
    return trans


def build_markov2_pos(numbers_list, position):
    """按位置构建二阶马尔可夫转移矩阵"""
    trans2 = defaultdict(Counter)
    for i in range(len(numbers_list) - 2):
        p2 = numbers_list[i][position]
        p1 = numbers_list[i + 1][position]
        nx = numbers_list[i + 2][position]
        trans2[(p2, p1)][nx] += 1
    return trans2


def markov_prob_smoothed(row, states, alpha=MARKOV_LAPLACE_ALPHA):
    """拉普拉斯平滑转移概率"""
    states = list(states)
    row_total = sum(row.values())
    denom = row_total + alpha * len(states)
    return {s: (row.get(s, 0) + alpha) / denom for s in states}


def miss_value_pos(numbers_list, digit, position=None):
    """计算数字遗漏值（当前距离上次出现的期数）"""
    for i in range(len(numbers_list) - 1, -1, -1):
        n = numbers_list[i]
        if position is None:
            if digit in n:
                return len(numbers_list) - 1 - i
        elif n[position] == digit:
            return len(numbers_list) - 1 - i
    return len(numbers_list)


def average_miss_cycle_5(numbers_list, digit, window=200):
    """计算排列五单个数字的平均遗漏周期（所有5个位置合并）"""
    if len(numbers_list) < 10:
        return 2.0  # 排列五5位，期望遗漏约2期（1/10*5 → ~2期）
    recent = numbers_list[-window:] if len(numbers_list) > window else numbers_list
    miss_periods = []
    current_miss = 0
    for n in recent:
        if digit in n:
            miss_periods.append(current_miss)
            current_miss = 0
        else:
            current_miss += 1
    return sum(miss_periods) / len(miss_periods) if miss_periods else 2.0


# ==================== 多窗口分析 ====================

def digit_scores_single_window(numbers_list, window=RECENT_WINDOW, dynamic=None):
    """单窗口数字评分"""
    recent = _recent_slice(numbers_list, window)
    if not recent:
        return [0.0] * 10
    last = numbers_list[-1]
    score = [0.0] * 10
    flags = FEATURE_FLAGS

    # 全局热号（指数衰减）
    all_digits = [d for n in recent for d in n]
    freq_all = exp_weighted_counts(all_digits)
    if flags.get("hot"):
        for d, _ in freq_all.most_common(4):
            score[d] += W_HOT_GLOBAL
        # 位置热号
        for pos in range(5):
            pos_freq = exp_weighted_counts([n[pos] for n in recent])
            for d, _ in pos_freq.most_common(3):
                score[d] += W_HOT_POS

    # 位置级马尔可夫（一阶+二阶）
    if flags.get("markov"):
        for pos in range(5):
            trans = build_markov_pos(numbers_list, pos)
            prev_d = last[pos]
            row = trans.get(prev_d, Counter())
            for d, p in markov_prob_smoothed(row, range(10)).items():
                sc = W_MARKOV * p
                score[d] += min(sc, MARKOV_MAX_SCORE)

            if len(numbers_list) >= 2:
                trans2 = build_markov2_pos(numbers_list, pos)
                prev2_d = numbers_list[-2][pos]
                row2 = trans2.get((prev2_d, prev_d), Counter())
                for d, p in markov_prob_smoothed(row2, range(10)).items():
                    sc2 = W_MARKOV2 * p
                    score[d] += min(sc2, MARKOV_MAX_SCORE)

    # 遗漏奖励
    if flags.get("miss"):
        for d in range(10):
            mv = miss_value_pos(numbers_list, d)
            avg_mv = average_miss_cycle_5(numbers_list, d)
            if avg_mv > 0:
                ratio = mv / avg_mv
                if ratio >= 2.0:
                    score[d] += W_MISS_HIGH * ratio
                elif ratio >= 1.5:
                    score[d] += W_MISS_MID

    # 上期同号奖励（出现在上期就奖励）
    if flags.get("lag1_repeat"):
        for d in set(last):
            score[d] += W_LAST_APPEAR

    return score


def ensemble_digit_scores_multi_window(numbers_list, window_weights):
    """多窗口集成数字评分"""
    combined = [0.0] * 10
    total_wt = sum(window_weights.values()) or 1.0
    for w, wt in window_weights.items():
        sc = digit_scores_single_window(numbers_list, window=w)
        for d in range(10):
            combined[d] += (wt / total_wt) * sc[d]
    return combined


def analyze_sum_span_5(numbers_list, window=RECENT_WINDOW):
    """计算5位号码的和值/跨度分布中心"""
    recent = _recent_slice(numbers_list, window)
    if not recent:
        return {"sum_center": 22.5, "span_center": 7.0}
    sums = [sum(n) for n in recent]
    spans = [max(n) - min(n) for n in recent]
    w_sums = exp_weighted_counts(sums)
    w_spans = exp_weighted_counts(spans)
    total_s = sum(w_sums.values()) or 1.0
    total_p = sum(w_spans.values()) or 1.0
    sum_center = sum(k * v for k, v in w_sums.items()) / total_s
    span_center = sum(k * v for k, v in w_spans.items()) / total_p
    return {"sum_center": sum_center, "span_center": span_center}


def ensemble_sum_span_5(numbers_list, window_weights):
    """多窗口集成和值/跨度"""
    sum_center = span_center = 0.0
    total_wt = sum(window_weights.values()) or 1.0
    for w, wt in window_weights.items():
        r = analyze_sum_span_5(numbers_list, window=w)
        sum_center += (wt / total_wt) * r["sum_center"]
        span_center += (wt / total_wt) * r["span_center"]
    return {"sum_center": sum_center, "span_center": span_center}


def analyze_ratio_pattern(numbers_list, window=60):
    """分析奇偶比/大小比的热门模式"""
    recent = _recent_slice(numbers_list, window)
    oe_cnt = Counter()
    bs_cnt = Counter()
    for n in recent:
        oe_cnt[odd_even_key_5(n)] += 1
        bs_cnt[big_small_key_5(n)] += 1
    return {
        "hot_oe": {k for k, _ in oe_cnt.most_common(3)},
        "hot_bs": {k for k, _ in bs_cnt.most_common(3)},
    }


def default_window_weights():
    n = len(RECENT_WINDOWS)
    return {w: 1.0 / n for w in RECENT_WINDOWS}


def load_window_weights():
    """读取持久化窗口权重（带降级）"""
    try:
        data = kv_store.load(WINDOW_WEIGHTS_KV_KEY)
        if not data or not isinstance(data.get("weights"), dict):
            return default_window_weights()
        return {int(k): float(v) for k, v in data["weights"].items()}
    except Exception:
        return default_window_weights()


def save_window_weights(weights, period=None):
    """持久化窗口权重"""
    try:
        kv_store.save(WINDOW_WEIGHTS_KV_KEY, {"weights": {str(k): v for k, v in weights.items()}, "period": period})
    except Exception as e:
        logger.warning(f"保存窗口权重失败: {e}")


# ==================== 推荐去重 ====================

def load_recent_recommend():
    """加载推荐历史"""
    try:
        return kv_store.load(RECENT_RECOMMEND_KV_KEY, [])
    except Exception:
        return []


def save_recent_recommend(period, recommendations):
    """按期号保存推荐历史（同期去重）"""
    try:
        history = load_recent_recommend()
        if (history and isinstance(history[-1], dict)
                and history[-1].get("period") == period):
            history[-1]["recommendations"] = recommendations
        else:
            history.append({
                "period": period,
                "recommendations": recommendations,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
        history = history[-RECENT_RECOMMEND_WINDOW:]
        kv_store.save(RECENT_RECOMMEND_KV_KEY, history)
    except Exception as e:
        logger.warning(f"保存推荐历史失败: {e}")


def apply_recent_recommend_penalty(pool, recent_history):
    """对连续推荐过的号码施加惩罚"""
    if not recent_history:
        return pool
    recent_set = set()
    consecutive_count = {}
    for entry in recent_history[-RECENT_RECOMMEND_WINDOW:]:
        recs = entry.get("recommendations", []) if isinstance(entry, dict) else entry
        for num_str in recs:
            recent_set.add(num_str)
            consecutive_count[num_str] = consecutive_count.get(num_str, 0) + 1
    result = []
    for w, num_str in pool:
        penalty = 0.0
        if num_str in recent_set:
            penalty += RECENT_RECOMMEND_PENALTY
        if consecutive_count.get(num_str, 0) >= 2:
            penalty += RECENT_RECOMMEND_CONSECUTIVE_PENALTY
        result.append((w - penalty, num_str))
    return result


# ==================== 胆码/杀码 ====================

def pick_dan_kill(score, top_dan=2, top_kill=2):
    """根据评分挑选胆码和杀码"""
    sorted_scores = sorted(enumerate(score), key=lambda x: -x[1])
    dan = [d for d, _ in sorted_scores[:top_dan]]
    kill = [d for d, _ in sorted_scores[-top_kill:]]
    return dan, kill
