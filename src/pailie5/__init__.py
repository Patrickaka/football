#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
排列五号码分析模块 V2.0（准确率优化版）

分析功能：
1. 历史数据抓取
2. 位置频率统计（指数衰减加权）
3. 热冷号分析（多窗口集成）
4. 当前遗漏分析
5. 平均遗漏分析
6. 和值/跨度软约束（高斯评分）
7. 位置级一阶/二阶马尔可夫（拉普拉斯平滑）
8. 多窗口集成评分
9. 推荐去重惩罚（连续推荐降权）
10. 回测驱动权重优化
"""

import math
import os
import re
import json
import random
import logging
import urllib.request
import urllib.error
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import time

from ..common.paths import data_path
from ..common.data_cache import cached_fetch
from ..common import repositories
from ..common import kv_store

logger = logging.getLogger(__name__)

# ==================== 配置 ====================
DATA_FILE = data_path('pailie5_history.json')
HISTORY_URL = 'https://www.8300.cn/kjhhis/5/200.html'
NUMBERS = list(range(0, 10))  # 0-9

# 多窗口配置（参考3D）
RECENT_WINDOWS = (30, 60, 90, 150)
RECENT_WINDOW = 150  # 展示用最大窗口

# 指数衰减系数（越近期权重越高）
EXP_DECAY = 0.97

# ==================== 评分权重 ====================
# 全局热号权重
W_HOT_GLOBAL = 2.5
# 位置热号权重
W_HOT_POS = 3.0
# 遗漏奖励（高遗漏号码）
W_MISS_HIGH = 1.5   # 遗漏超过平均2倍
W_MISS_MID = 0.8    # 遗漏超过平均1.5倍
# 马尔可夫权重
W_MARKOV = 4.5
W_MARKOV2 = 1.2
MARKOV_MAX_SCORE = 5.0   # 马尔可夫得分上限
MARKOV_LAPLACE_ALPHA = 1.0  # 拉普拉斯平滑系数
# 和值/跨度软约束
SUM_SOFT_SIGMA = 4.0
SPAN_SOFT_SIGMA = 1.8
# 上期同号奖励
W_LAST_APPEAR = 1.5
# 连号奖励
W_CONSECUTIVE = 1.2
# 位置级数字评分奖励（位置热号超出全局热号的部分）
W_POS_SPECIFIC = 1.5
# 重复数字惩罚（避免推荐池过度集中于单个数字）
W_REPEAT = 2.0
# 不重复数字奖励（鼓励数字多样性）
W_DISTINCT = 11.0
W_POS_REPEAT_PENALTY = 1.0
# 奇偶比/大小比匹配奖励
W_RATIO_MATCH = 1.5
# 杀码惩罚（软约束）
W_KILL_PENALTY = 5.0
# 胆码奖励
W_DANMA_HIT = 3.5

# ==================== 功能开关 ====================
FEATURE_FLAGS = {
    "hot": True,           # 热号得分
    "miss": True,          # 遗漏加分
    "markov": True,        # 马尔可夫转移
    "sum_span": True,      # 和值跨度
    "consecutive": True,   # 连号奖励
    "lag1_repeat": True,   # 上期同位重复
    "ratio": True,         # 奇偶比/大小比
}

# ==================== 推荐配置 ====================
RECOMMEND_GROUPS = 30      # 推荐注数

# 推荐去重配置
RECENT_RECOMMEND_WINDOW = 5          # 最近N期推荐历史
RECENT_RECOMMEND_PENALTY = 2.0       # 重复推荐惩罚
RECENT_RECOMMEND_CONSECUTIVE_PENALTY = 4.0  # 连续推荐额外惩罚

# 多样性配置
DIVERSITY_WEIGHT = 0.5
CORRELATION_THRESHOLD = 3   # 数字重合阈值（5位号码，重合3个才算高相关）
CORRELATION_PENALTY = 1.0
COVERAGE_WEIGHT = 3.0       # 覆盖未选数字的奖励权重

# 预测结果缓存
_prediction_cache = None
_cache_time = 0

# KV 存储键
RECENT_RECOMMEND_KV_KEY = 'pailie5_recent_recommend'
WINDOW_WEIGHTS_KV_KEY = 'pailie5_window_weights'


# ==================== 工具函数 ====================

def _is_today_cache(cache_timestamp):
    """检查缓存是否是今天的（按自然天判断）"""
    if cache_timestamp is None or cache_timestamp == 0:
        return False
    import datetime
    cache_date = datetime.date.fromtimestamp(cache_timestamp)
    today = datetime.date.today()
    return cache_date == today


def clear_cache():
    """清除缓存"""
    global _prediction_cache, _cache_time
    _prediction_cache = None
    _cache_time = 0
    logger.info("排列五模块缓存已清除")


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


# ==================== 推荐池生成 ====================

def triplet_score_5(nums, score, pos_scores, sum_center, span_center, ratio_info, dan, kill, last_nums):
    """计算5位号码组合的综合评分"""
    # 基础数字评分（全局热号等）
    base = sum(score[d] for d in nums)

    # 位置级数字评分奖励：使用在该位置特别热的数字更有优势
    for pos, d in enumerate(nums):
        base += (pos_scores[pos][d] - score[d]) * W_POS_SPECIFIC

    # 和值软约束
    if FEATURE_FLAGS.get("sum_span"):
        s = sum(nums)
        sp = max(nums) - min(nums)
        base += gaussian_score(s, sum_center, SUM_SOFT_SIGMA) * 3.0
        base += gaussian_score(sp, span_center, SPAN_SOFT_SIGMA) * 2.0

    # 连号奖励
    if FEATURE_FLAGS.get("consecutive") and has_consecutive_digits_5(nums):
        base += W_CONSECUTIVE

    # 奇偶比/大小比匹配
    if FEATURE_FLAGS.get("ratio") and ratio_info:
        oe = odd_even_key_5(nums)
        bs = big_small_key_5(nums)
        if oe in ratio_info.get("hot_oe", set()):
            base += W_RATIO_MATCH
        if bs in ratio_info.get("hot_bs", set()):
            base += W_RATIO_MATCH

    # 胆码命中奖励
    for d in dan:
        if d in nums:
            base += W_DANMA_HIT

    # 杀码软惩罚（软约束，不是硬排除）
    for d in kill:
        if d in nums:
            base -= W_KILL_PENALTY

    # 重复数字惩罚
    digit_counts = Counter(nums)
    for d, cnt in digit_counts.items():
        if cnt >= 2:
            base -= W_REPEAT * (cnt - 1)

    # 不重复数字奖励：鼓励号码数字更丰富
    base += W_DISTINCT * len(set(nums))

    return base


def _get_position_scores(numbers_list, score):
    """
    计算每个位置的独立数字评分（融合位置热号+位置马尔可夫+全局评分）
    返回: List[List[float]] - shape (5, 10)
    """
    if not numbers_list:
        return [[score[d] for d in range(10)] for _ in range(5)]
    last_nums = numbers_list[-1]
    prev2_nums = numbers_list[-2] if len(numbers_list) >= 2 else None
    recent = _recent_slice(numbers_list, 90)
    pos_scores = []
    for pos in range(5):
        ps = [score[d] for d in range(10)]  # 全局评分基础

        # 位置热号（指数衰减）
        pos_freq = exp_weighted_counts([n[pos] for n in recent])
        max_pf = max(pos_freq.values()) if pos_freq else 1.0
        for d, v in pos_freq.items():
            ps[d] += W_HOT_POS * (v / max_pf)

        # 位置一阶马尔可夫
        trans = build_markov_pos(numbers_list, pos)
        prev_d = last_nums[pos]
        row = trans.get(prev_d, Counter())
        for d, p in markov_prob_smoothed(row, range(10)).items():
            ps[d] += min(W_MARKOV * p, MARKOV_MAX_SCORE)

        # 位置二阶马尔可夫
        if prev2_nums is not None:
            trans2 = build_markov2_pos(numbers_list, pos)
            prev2_d = prev2_nums[pos]
            row2 = trans2.get((prev2_d, prev_d), Counter())
            for d, p in markov_prob_smoothed(row2, range(10)).items():
                ps[d] += min(W_MARKOV2 * p, MARKOV_MAX_SCORE)

        pos_scores.append(ps)
    return pos_scores


def generate_pool(numbers_list, window_weights, top_n=RECOMMEND_GROUPS, apply_dedup=True):
    """
    生成排列五推荐池（位置级独立评分 + 多样性采样策略）

    策略：
    1. 计算每个位置的独立数字评分
    2. 每个位置取Top7候选 → 7^5=16807种组合（可快速枚举）
    3. 对每个组合评分（融合和值/跨度软约束、奇偶比等）
    4. 补充全局高分随机探索
    5. 多样性选池去相关

    返回: [(score, num_str), ...]
    """
    if len(numbers_list) < 5:
        return []

    # 多窗口集成数字评分（全局）
    score = ensemble_digit_scores_multi_window(numbers_list, window_weights)

    # 和值/跨度中心
    ss = ensemble_sum_span_5(numbers_list, window_weights)
    sum_center = ss["sum_center"]
    span_center = ss["span_center"]

    # 奇偶/大小比模式
    ratio_info = analyze_ratio_pattern(numbers_list)

    # 胆码/杀码
    dan, kill = pick_dan_kill(score, top_dan=2, top_kill=2)

    last_nums = numbers_list[-1]

    from itertools import product as iproduct

    # ---- 策略1：位置Top9候选枚举（9^5 = 59049）----
    pos_scores = _get_position_scores(numbers_list, score)
    pos_top = []
    for pos in range(5):
        sorted_pos = sorted(range(10), key=lambda d: -pos_scores[pos][d])
        pos_top.append(sorted_pos[:9])  # 每个位置取Top9

    pool = []
    seen = set()
    for combo in iproduct(*pos_top):
        num_str = ''.join(map(str, combo))
        if num_str in seen:
            continue
        seen.add(num_str)
        s = triplet_score_5(list(combo), score, pos_scores, sum_center, span_center, ratio_info, dan, kill, last_nums)
        pool.append((s, num_str))

    # ---- 策略2：全局Top6全排列补充（6^5 = 7776）----
    global_top6 = sorted(range(10), key=lambda d: -score[d])[:6]
    for combo in iproduct(global_top6, repeat=5):
        num_str = ''.join(map(str, combo))
        if num_str in seen:
            continue
        seen.add(num_str)
        s = triplet_score_5(list(combo), score, pos_scores, sum_center, span_center, ratio_info, dan, kill, last_nums)
        pool.append((s, num_str))

    # ---- 策略3：胆码固定 + 其余位置Top5探索 ----
    # 对于每个胆码，固定在5个位置上分别出现，其余位置取全局Top5
    global_top5 = sorted(range(10), key=lambda d: -score[d])[:5]
    for dan_d in dan:
        for fixed_pos in range(5):
            free_pos = [p for p in range(5) if p != fixed_pos]
            # 固定胆码位置，其余位置取Top5枚举
            free_tops = [global_top5 for _ in free_pos]
            for free_combo in iproduct(*free_tops):
                combo = list(free_combo[:fixed_pos]) + [dan_d] + list(free_combo[fixed_pos:])
                num_str = ''.join(map(str, combo))
                if num_str in seen:
                    continue
                seen.add(num_str)
                s = triplet_score_5(combo, score, pos_scores, sum_center, span_center, ratio_info, dan, kill, last_nums)
                pool.append((s, num_str))

    # ---- 策略4：冷号覆盖探索（确保 0-9 每个数字都有出现机会）----
    cold_digits = [d for d in range(10) if d not in global_top6]
    hot_filler = global_top6[:5]
    for cold_d in cold_digits:
        for pos in range(5):
            for free_combo in iproduct(hot_filler, repeat=4):
                combo = list(free_combo[:pos]) + [cold_d] + list(free_combo[pos:])
                num_str = ''.join(map(str, combo))
                if num_str in seen:
                    continue
                seen.add(num_str)
                s = triplet_score_5(combo, score, pos_scores, sum_center, span_center, ratio_info, dan, kill, last_nums)
                pool.append((s, num_str))

    # 按评分排序
    pool.sort(key=lambda x: -x[0])

    # 推荐去重惩罚
    if apply_dedup:
        recent_history = load_recent_recommend()
        pool = apply_recent_recommend_penalty(pool, recent_history)
        pool.sort(key=lambda x: -x[0])

    # 多样性选池（贪心去相关）
    selected = _select_diverse_pool(pool, top_n=top_n)

    return selected


def _select_diverse_pool(pool, top_n=RECOMMEND_GROUPS, candidate_size=2000):
    """贪心多样性选池：高分优先，兼顾数字覆盖去相关"""
    candidates = sorted(pool, key=lambda x: -x[0])[:candidate_size]
    if not candidates:
        return []

    # 归一化候选分数到 [0,1]，使多样性奖励/惩罚具有可比权重
    max_w = max(c[0] for c in candidates)
    min_w = min(c[0] for c in candidates)
    range_w = max_w - min_w + 1e-9
    candidates = [((w - min_w) / range_w, num) for w, num in candidates]

    selected = []
    selected_sets = []
    all_digits = set('0123456789')
    covered_digits = set()

    while candidates and len(selected) < top_n:
        best_item = None
        best_score = -float("inf")
        union_digits = set().union(*selected_sets) if selected_sets else set()
        missing_digits = all_digits - covered_digits

        for w, num in candidates:
            digits = set(num)
            # 跳过完全重复的候选
            if digits in selected_sets:
                continue
            overlap_penalty = sum(
                CORRELATION_PENALTY
                for old_digits in selected_sets
                if len(digits & old_digits) >= CORRELATION_THRESHOLD
            )
            # 奖励覆盖尚未被选中数字
            new_cover = len(digits - union_digits) * DIVERSITY_WEIGHT
            # 额外奖励覆盖全局缺失数字（0-9）
            coverage_bonus = len(digits & missing_digits) * COVERAGE_WEIGHT
            final_score = w + new_cover + coverage_bonus - overlap_penalty
            if final_score > best_score:
                best_score = final_score
                best_item = (w, num)

        if best_item is None:
            break
        selected.append(best_item)
        best_digits = set(best_item[1])
        selected_sets.append(best_digits)
        covered_digits.update(best_digits)
        candidates.remove(best_item)

    # 覆盖补充：从完整候选池中为缺失数字补选最佳候选
    covered_digits = set().union(*selected_sets) if selected_sets else set()
    missing_digits = all_digits - covered_digits
    coverage_items = []
    if missing_digits:
        pool_sorted = sorted(pool, key=lambda x: -x[0])
        for d in missing_digits:
            for w, num in pool_sorted:
                num_set = set(num)
                if d in num and num_set not in selected_sets and num_set not in [set(x[1]) for x in coverage_items]:
                    coverage_items.append((w, num))
                    covered_digits.update(num)
                    break

    # 合并并按原始分数排序，但确保覆盖补充项不被截断
    all_selected = selected + coverage_items
    all_selected = sorted(all_selected, key=lambda x: -x[0])
    coverage_sets = {frozenset(x[1]) for x in coverage_items}
    if len(all_selected) > top_n:
        # 先保留所有覆盖补充项，再用非覆盖高分项填充剩余名额
        result = [x for x in all_selected if frozenset(x[1]) in coverage_sets]
        non_coverage = [x for x in all_selected if frozenset(x[1]) not in coverage_sets]
        result.extend(non_coverage[:top_n - len(result)])
        all_selected = result

    return sorted(all_selected, key=lambda x: -x[0])[:top_n]


# ==================== 窗口权重回测 ====================

def backtest_window_weights(numbers_list, trials=80, window_candidates=None):
    """
    通过滚动回测计算最优窗口权重。
    评估标准：推荐Top30中是否包含实际开奖号码对应的数字覆盖率
    """
    if window_candidates is None:
        window_candidates = list(RECENT_WINDOWS)
    if len(numbers_list) < trials + max(window_candidates) + 5:
        return default_window_weights()

    start = len(numbers_list) - trials
    window_scores = {w: 0.0 for w in window_candidates}
    window_prior = 5.0  # 先验分，避免权重极端化

    for i in range(start, len(numbers_list)):
        train = numbers_list[:i]
        actual = numbers_list[i]
        actual_str = ''.join(map(str, actual))
        actual_digits = set(actual)

        for w in window_candidates:
            sc = digit_scores_single_window(train, window=w)
            # 用该窗口评分生成Top30推荐（简化版，不做完整多样性选池）
            top_digits = sorted(range(10), key=lambda d: -sc[d])[:6]
            # 检查实际号码中的数字有多少在Top6中
            coverage = len(actual_digits & set(top_digits))
            window_scores[w] += coverage / 5.0

    # 加上先验分后归一化
    total = sum(window_scores[w] + window_prior for w in window_candidates)
    weights = {w: (window_scores[w] + window_prior) / total for w in window_candidates}
    return weights


# ==================== 主分析器 ====================

class Pailie5Analyzer:
    """
    排列五分析器 V2.0
    """

    def __init__(self):
        self.history: List[Dict] = []
        self._load_history()
        if not self.history:
            self.fetch_history_data(90)

    def _load_history(self):
        try:
            self.history = repositories.pailie5_load()
            logger.info(f"已加载 {len(self.history)} 期排列五历史数据")
        except Exception as e:
            logger.error(f"加载排列五历史数据失败: {e}")
            self.history = []

    def _save_history(self):
        try:
            repositories.pailie5_save(self.history)
        except Exception as e:
            logger.error(f"保存排列五历史数据失败: {e}")

    def _fetch_history_data_internal(self, days: int = 30):
        try:
            url = HISTORY_URL
            logger.info(f"正在抓取排列五历史数据: {url}")
            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(url, headers=headers)
            html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")

            issue_tags = re.findall(r'<td>[^<]*期[^<]*</td>', html)
            issues = []
            for tag in issue_tags:
                match = re.search(r'(\d{7})', tag)
                if match:
                    issues.append(match.group(1))

            dates = re.findall(r'<td>\s*(\d{4}-\d{2}-\d{2})\s*</td>', html)
            balls = re.findall(r'<span class="ball">(\d)</span>', html)

            if not issues or not balls:
                logger.error("无法解析排列五数据，未找到期号或数字球")
                return 0

            count = 0
            for i in range(min(len(issues), len(dates)) - 1, -1, -1):
                ball_start = i * 5
                ball_end = ball_start + 5
                if ball_end <= len(balls):
                    numbers = [int(b) for b in balls[ball_start:ball_end]]
                    date = dates[i] if i < len(dates) else None
                    if self.add_result(issues[i], numbers, date):
                        count += 1

            logger.info(f"成功抓取 {count} 期排列五数据")
            return count

        except Exception as e:
            logger.error(f"抓取排列五历史数据失败：{e}")
            return 0

    def fetch_history_data(self, days: int = 30, force_refresh: bool = False):
        try:
            if not force_refresh:
                from ..common.data_cache import is_cache_valid
                if is_cache_valid('pailie5'):
                    logger.info("使用缓存的排列五数据")
                    self._load_history()
                    return len(self.history)
            count = self._fetch_history_data_internal(days)
            if count > 0:
                self._save_history()
                from ..common.data_cache import save_cached_data
                save_cached_data('pailie5', self.history)
            return count
        except Exception as e:
            logger.warning(f"缓存/抓取失败，使用已有历史数据: {e}")
            self._load_history()
            return len(self.history)

    def _calculate_date_from_issue(self, issue: str) -> str:
        try:
            if len(issue) != 7:
                return datetime.now().strftime('%Y-%m-%d')
            year = int(issue[:4])
            day_of_year = int(issue[4:])
            date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
            return date.strftime('%Y-%m-%d')
        except Exception:
            return datetime.now().strftime('%Y-%m-%d')

    def add_result(self, issue: str, numbers: List[int], date: str = None):
        if len(numbers) != 5:
            return False
        for n in numbers:
            if n < 0 or n > 9:
                return False
        if not date:
            date = self._calculate_date_from_issue(issue)
        for r in self.history:
            if r['issue'] == issue:
                r['numbers'] = numbers
                r['date'] = date
                r['timestamp'] = datetime.now().isoformat()
                self._save_history()
                return True
        self.history.append({
            'issue': issue,
            'numbers': numbers,
            'date': date,
            'timestamp': datetime.now().isoformat()
        })
        self.history.sort(key=lambda x: x['issue'], reverse=True)
        self._save_history()
        return True

    def fetch_latest_results(self, count: int = 10, force_refresh: bool = False) -> Dict:
        try:
            fetched_count = self.fetch_history_data(days=1, force_refresh=force_refresh)
            recent = self.get_recent_results(count)
            return {
                'success': fetched_count > 0,
                'source': 'web' if fetched_count > 0 else 'local',
                'count': fetched_count if fetched_count > 0 else len(recent),
                'message': f'成功抓取 {fetched_count} 期数据' if fetched_count > 0 else '使用本地数据',
                'latest_issue': recent[0]['issue'] if recent else None,
                'results': recent
            }
        except Exception:
            logger.error("排列五抓取失败", exc_info=True)
            recent = self.get_recent_results(count)
            return {'success': False, 'source': 'local', 'count': len(recent),
                    'message': '抓取异常，使用本地数据',
                    'latest_issue': recent[0]['issue'] if recent else None, 'results': recent}

    def get_recent_results(self, count: int = 10) -> List[Dict]:
        return self.history[:count]

    # ==================== 统计分析 ====================

    def _get_numbers_list(self) -> List[List[int]]:
        """获取历史号码列表（按开奖时间从旧到新）"""
        return [r['numbers'] for r in reversed(self.history)]

    def analyze_position_frequency(self) -> List[Dict[int, int]]:
        position_freq = []
        for pos in range(5):
            freq = {n: 0 for n in NUMBERS}
            for result in self.history:
                freq[result['numbers'][pos]] += 1
            position_freq.append(freq)
        return position_freq

    def analyze_frequency(self) -> Dict[int, int]:
        freq = {n: 0 for n in NUMBERS}
        for result in self.history:
            for n in result['numbers']:
                freq[n] += 1
        return freq

    def get_hot_numbers(self, top_n: int = 5) -> List[Tuple[int, int]]:
        numbers_list = self._get_numbers_list()
        recent = _recent_slice(numbers_list, 90)
        freq = exp_weighted_counts([d for n in recent for d in n])
        return freq.most_common(top_n)

    def get_cold_numbers(self, top_n: int = 5) -> List[Tuple[int, int]]:
        numbers_list = self._get_numbers_list()
        recent = _recent_slice(numbers_list, 90)
        freq = exp_weighted_counts([d for n in recent for d in n])
        all_digits = [(d, freq.get(d, 0)) for d in range(10)]
        return sorted(all_digits, key=lambda x: x[1])[:top_n]

    def analyze_current_gaps(self) -> Dict[int, int]:
        numbers_list = self._get_numbers_list()
        gaps = {}
        for d in range(10):
            gaps[d] = miss_value_pos(numbers_list, d)
        return gaps

    def analyze_average_gaps(self) -> Dict[int, float]:
        numbers_list = self._get_numbers_list()
        avg_gaps = {}
        for d in range(10):
            avg_gaps[d] = round(average_miss_cycle_5(numbers_list, d), 2)
        return avg_gaps

    def analyze_sum(self) -> Dict:
        sums = [sum(r['numbers']) for r in self.history]
        if not sums:
            return {'min': 0, 'max': 0, 'avg': 0, 'most_common': []}
        sum_counts = Counter(sums)
        return {
            'min': min(sums), 'max': max(sums),
            'avg': round(sum(sums) / len(sums), 2),
            'most_common': sum_counts.most_common(5),
            'distribution': dict(sum_counts)
        }

    def analyze_span(self) -> Dict:
        spans = [max(r['numbers']) - min(r['numbers']) for r in self.history]
        if not spans:
            return {'min': 0, 'max': 0, 'avg': 0, 'most_common': []}
        span_counts = Counter(spans)
        return {
            'min': min(spans), 'max': max(spans),
            'avg': round(sum(spans) / len(spans), 2),
            'most_common': span_counts.most_common(5),
            'distribution': dict(span_counts)
        }

    def analyze_odd_even(self) -> Dict:
        odd_counts = [sum(1 for n in r['numbers'] if n % 2 == 1) for r in self.history]
        if not odd_counts:
            return {'distribution': {}, 'most_common': []}
        dist = Counter(odd_counts)
        return {'distribution': dict(dist), 'most_common': dist.most_common(3)}

    def analyze_size(self) -> Dict:
        small_counts = [sum(1 for n in r['numbers'] if n <= 4) for r in self.history]
        if not small_counts:
            return {'distribution': {}, 'most_common': []}
        dist = Counter(small_counts)
        return {'distribution': dict(dist), 'most_common': dist.most_common(3)}

    def analyze_road(self) -> Dict:
        road_counts = {0: 0, 1: 0, 2: 0}
        for result in self.history:
            for n in result['numbers']:
                road_counts[n % 3] += 1
        total = sum(road_counts.values()) or 1
        return {
            'distribution': road_counts,
            'most_common': sorted(road_counts.items(), key=lambda x: -x[1]),
            'road_numbers': {
                0: [n for n in NUMBERS if n % 3 == 0],
                1: [n for n in NUMBERS if n % 3 == 1],
                2: [n for n in NUMBERS if n % 3 == 2]
            }
        }

    def analyze_transition_matrix(self) -> List[List[int]]:
        matrix = [[0] * 10 for _ in range(10)]
        for i in range(1, len(self.history)):
            prev_nums = self.history[i - 1]['numbers']
            curr_nums = self.history[i]['numbers']
            for p in prev_nums:
                for c in curr_nums:
                    matrix[p][c] += 1
        return matrix

    def bayesian_score(self) -> Dict[int, float]:
        """增强版贝叶斯评分（整合多窗口评分）"""
        numbers_list = self._get_numbers_list()
        window_weights = load_window_weights()
        score = ensemble_digit_scores_multi_window(numbers_list, window_weights)
        min_s = min(score)
        max_s = max(score) - min_s + 1e-9
        return {d: round((score[d] - min_s) / max_s, 4) for d in range(10)}

    def multi_model_voting(self) -> List[int]:
        """多模型投票（使用增强版评分）"""
        numbers_list = self._get_numbers_list()
        window_weights = load_window_weights()
        score = ensemble_digit_scores_multi_window(numbers_list, window_weights)

        votes = Counter()

        # 模型1：全局评分 Top5
        top5_global = [d for d, _ in sorted(enumerate(score), key=lambda x: -x[1])[:5]]
        for d in top5_global:
            votes[d] += 2

        # 模型2：位置级马尔可夫 Top5（每个位置的Top预测）
        last = numbers_list[-1] if numbers_list else [0] * 5
        for pos in range(5):
            trans = build_markov_pos(numbers_list, pos)
            prev_d = last[pos]
            row = trans.get(prev_d, Counter())
            probs = markov_prob_smoothed(row, range(10))
            top_markov = [d for d, _ in sorted(probs.items(), key=lambda x: -x[1])[:3]]
            for d in top_markov:
                votes[d] += 1

        # 模型3：热号 Top5
        hot = [d for d, _ in self.get_hot_numbers(5)]
        for d in hot:
            votes[d] += 1

        # 返回得票最多的5个数字
        return [d for d, _ in votes.most_common(5)]

    def generate_recommendation(self, method: str = 'balanced') -> List[int]:
        """生成推荐号码（5个数字）"""
        numbers_list = self._get_numbers_list()
        window_weights = load_window_weights()
        score = ensemble_digit_scores_multi_window(numbers_list, window_weights)

        if method == 'random':
            return [random.randint(0, 9) for _ in range(5)]

        sorted_digits = [d for d, _ in sorted(enumerate(score), key=lambda x: -x[1])]
        hot = sorted_digits[:5]
        cold = sorted_digits[-5:]

        if method == 'hot':
            return random.sample(hot, 5)
        if method == 'cold':
            return random.sample(cold, 5)

        # balanced: 热冷混合
        hot_count = random.choice([2, 3])
        cold_count = 5 - hot_count
        result = random.sample(hot, hot_count) + random.sample(cold, cold_count)
        random.shuffle(result)
        return result

    def rank_model(self, top_n: int = 5) -> List[Tuple[int, float]]:
        """排名模型：综合多特征评分"""
        numbers_list = self._get_numbers_list()
        window_weights = load_window_weights()
        score = ensemble_digit_scores_multi_window(numbers_list, window_weights)
        return sorted(enumerate(score), key=lambda x: -x[1])[:top_n]

    def identify_cycles(self) -> Dict[str, List]:
        """识别冷热周期状态"""
        freq = self.analyze_frequency()
        gaps = self.analyze_current_gaps()
        avg_gaps = self.analyze_average_gaps()
        avg_freq = sum(freq.values()) / len(freq) if freq else 1
        cycles = {'hot': [], 'cold': [], 'warming': [], 'cooling': [], 'stable': []}
        for n in NUMBERS:
            freq_dev = freq[n] / avg_freq if avg_freq > 0 else 1
            if freq_dev > 1.10:
                cycles['hot'].append(n)
            elif freq_dev < 0.90:
                cycles['cold'].append(n)
            else:
                cycles['stable'].append(n)
            if avg_gaps[n] > 0 and gaps[n] < avg_gaps[n] * 0.7:
                cycles['warming'].append(n)
            elif avg_gaps[n] > 0 and gaps[n] > avg_gaps[n] * 1.5:
                cycles['cooling'].append(n)
        return cycles

    def ensemble_predict(self) -> Dict:
        """集成预测（使用增强评分系统）"""
        numbers_list = self._get_numbers_list()
        window_weights = load_window_weights()

        # 生成推荐池
        pool = generate_pool(numbers_list, window_weights, top_n=RECOMMEND_GROUPS, apply_dedup=True)
        top30_str = [num for _, num in pool]
        top5 = [list(map(int, num)) for num in top30_str[:5]]

        # 综合数字评分
        score = ensemble_digit_scores_multi_window(numbers_list, window_weights)
        dan, kill = pick_dan_kill(score, top_dan=2, top_kill=2)

        # 周期分析
        cycles = self.identify_cycles()

        return {
            'prediction': top5[0] if top5 else [0, 1, 2, 3, 4],
            'top5_combos': top5,
            'top30': top30_str,
            'dan': dan,
            'kill': kill,
            'cycles': cycles,
            'ranked_numbers': self.rank_model(10),
        }

    def rolling_backtest(self, trials: int = 50) -> Dict:
        """滚动回测（改进版：评估推荐池覆盖率）"""
        numbers_list = self._get_numbers_list()
        n = len(numbers_list)
        if n < 30:
            logger.warning(f"排列五回测数据不足，仅 {n} 期")
            return {
                'trials': 0, 'top30_hit': 0, 'top30_rate': 0,
                'ge2_digit_hit': 0, 'ge2_digit_rate': 0,
                'ge3_digit_hit': 0, 'ge3_digit_rate': 0,
                'avg_digit_coverage': 0.0, 'predictions': []
            }
        if n < trials + 10:
            trials = max(20, n - 10)

        start = n - trials
        window_weights = default_window_weights()

        top30_hit = ge2_hit = ge3_hit = 0
        total_coverage = 0.0
        predictions = []

        for i in range(start, n):
            train = numbers_list[:i]
            actual = numbers_list[i]
            actual_str = ''.join(map(str, actual))
            actual_set = set(actual)

            score = ensemble_digit_scores_multi_window(train, window_weights)
            ss = ensemble_sum_span_5(train, window_weights)
            ratio_info = analyze_ratio_pattern(train)
            dan, kill = pick_dan_kill(score, top_dan=2, top_kill=2)

            pool = generate_pool(train, window_weights, top_n=30, apply_dedup=False)
            pool_strs = [num for _, num in pool]

            if actual_str in pool_strs:
                top30_hit += 1

            coverage = 0
            for num_str in pool_strs:
                overlap = len({int(c) for c in num_str} & actual_set)
                coverage = max(coverage, overlap)

            total_coverage += coverage / 5.0

            # 至少覆盖2/3个数字
            best_overlap = max((len({int(c) for c in s} & actual_set) for s in pool_strs), default=0)
            if best_overlap >= 2:
                ge2_hit += 1
            if best_overlap >= 3:
                ge3_hit += 1

            predictions.append({
                'actual': actual_str,
                'top30_hit': actual_str in pool_strs[:30],
                'best_overlap': best_overlap,
            })

        return {
            'trials': trials,
            'top30_hit': top30_hit,
            'top30_rate': round(top30_hit / trials, 4),
            'ge2_digit_hit': ge2_hit,
            'ge2_digit_rate': round(ge2_hit / trials, 4),
            'ge3_digit_hit': ge3_hit,
            'ge3_digit_rate': round(ge3_hit / trials, 4),
            'avg_digit_coverage': round(total_coverage / trials, 4),
            'predictions': predictions[-10:],  # 最近10条
        }

    def optimize_window_weights(self, trials: int = 80):
        """通过回测优化窗口权重并持久化"""
        numbers_list = self._get_numbers_list()
        weights = backtest_window_weights(numbers_list, trials=trials)
        latest_issue = self.history[0]['issue'] if self.history else None
        save_window_weights(weights, period=latest_issue)
        logger.info(f"窗口权重已更新: {weights}")
        return weights

    def get_statistics(self) -> Dict:
        """获取完整统计信息"""
        numbers_list = self._get_numbers_list()
        window_weights = load_window_weights()
        score = ensemble_digit_scores_multi_window(numbers_list, window_weights)
        ss = ensemble_sum_span_5(numbers_list, window_weights)
        return {
            'total_issues': len(self.history),
            'frequency': self.analyze_frequency(),
            'position_frequency': self.analyze_position_frequency(),
            'hot_numbers': self.get_hot_numbers(5),
            'cold_numbers': self.get_cold_numbers(5),
            'current_gaps': self.analyze_current_gaps(),
            'average_gaps': self.analyze_average_gaps(),
            'sum_analysis': self.analyze_sum(),
            'span_analysis': self.analyze_span(),
            'odd_even_analysis': self.analyze_odd_even(),
            'size_analysis': self.analyze_size(),
            'road_analysis': self.analyze_road(),
            'bayesian_scores': self.bayesian_score(),
            'sum_center': round(ss['sum_center'], 2),
            'span_center': round(ss['span_center'], 2),
            'window_weights': window_weights,
        }


# ==================== 全局实例 ====================

_pailie5_analyzer = None


def get_pailie5_analyzer() -> Pailie5Analyzer:
    global _pailie5_analyzer
    if _pailie5_analyzer is None:
        _pailie5_analyzer = Pailie5Analyzer()
    return _pailie5_analyzer


def run_prediction(force_refresh=False):
    """运行排列五预测，返回 JSON 可序列化 dict。"""
    global _prediction_cache, _cache_time

    if not force_refresh and _prediction_cache is not None:
        if _is_today_cache(_cache_time):
            elapsed = time.time() - _cache_time
            logger.info(f"使用今日缓存数据（缓存时间：{elapsed:.1f}秒前）")
            return _prediction_cache
        else:
            logger.info("缓存已过期，重新计算")

    try:
        analyzer = get_pailie5_analyzer()

        # 抓取最新数据
        analyzer.fetch_history_data(days=1, force_refresh=force_refresh)

        # 获取统计数据
        stats = analyzer.get_statistics()
        recent = analyzer.get_recent_results(10)

        # 集成预测
        ensemble = analyzer.ensemble_predict()

        # 保存推荐历史（用于去重）
        latest_issue = analyzer.history[0]['issue'] if analyzer.history else 'unknown'
        top30 = ensemble.get('top30', [])[:30]
        save_recent_recommend(latest_issue, top30)

        # 多种方法推荐（各取3组）
        recommendations = {}
        for method in ['balanced', 'hot', 'cold']:
            recs = []
            for _ in range(3):
                nums = analyzer.generate_recommendation(method)
                recs.append(nums)
            recommendations[method] = recs

        # 滚动回测
        backtest = analyzer.rolling_backtest(trials=30)

        result = {
            'statistics': stats,
            'recent_results': recent,
            'ensemble': ensemble,
            'recommendations': recommendations,
            'backtest': backtest,
        }

        _prediction_cache = result
        _cache_time = time.time()
        logger.info("排列五预测结果已缓存")
        return result

    except Exception:
        logger.error('排列五预测失败', exc_info=True)
        return {'error': '排列五预测失败'}
