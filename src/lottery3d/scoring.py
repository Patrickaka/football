# -*- coding: utf-8 -*-
"""福彩3D评分与选号：窗口权重、数字评分、组三/组六、直选排名"""

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
    MARKOV_LAPLACE_ALPHA, PAIR_BONUS, W_SLOPE_MATCH,
    COLD_RATIO, CORRELATION_PENALTY, CORRELATION_THRESHOLD, DANMA_RANDOM_RATE, DANMA_TOP_POOL, DIVERSITY_WEIGHT, EXPLORATION_RATE, EXP_DECAY, FEATURE_FLAGS, HOT_RATIO, HOT_WINDOW, MARKOV_MAX_SCORE, RANDOM_DIGIT_REUSE, RANDOM_NOISE, RANDOM_POS_REPEAT, RECENT_SUM_SPAN_SHIFT, RECENT_WINDOW, RECENT_WINDOWS, SERVED_POOL_CANDIDATE_SIZE, SPAN_SOFT_SIGMA, SUM_INTERVAL_WINDOW, SUM_SOFT_SIGMA, SUM_TREND_ADJUST, SUM_TREND_WINDOW, WARM_RATIO, WINDOW_BACKTEST_TRIALS, WINDOW_WEIGHTS_KV_KEY, WINDOW_WEIGHT_PRIOR, W_CONSECUTIVE, W_DANMA_HIT, W_FORM_PRIOR, W_HOT_GLOBAL, W_HOT_POS, W_KILL_PENALTY, W_LAST_APPEAR, W_MARKOV, W_MARKOV2, W_MISS_HIGH, W_MISS_MID, W_NEIGHBOR, W_POS_REPEAT, W_RATIO_MATCH, W_ROAD_MATCH, W_TRIPLET_GLOBAL, W_TRIPLET_POS, W_ZU6_PAIR, ZHIXUAN_TOP3, ZHXUAN_POS_TOPK, ZU3_MIN_SAMPLES, ZU3_PAIRS_COUNT, ZU3_PRESENCE_WINDOWS, ZU3_TIER_SIZES, ZU6_FOUR_SIZE, ZU6_POOL_SIZE, ZU6_PRESENCE_WINDOWS, ZU6_USE_KILL,
)
from .features import (
    FORM_LABELS, THEORY_FORM_P, _form_recent_p, _recent_slice, analyze_slope_patterns, big_small_key, build_markov, build_markov2, calc_span, classify_digits_by_hot, classify_form, entropy_model, exp_weighted_counts, form_miss, form_switch_bonus, gaussian_score, has_consecutive_digits, high_freq_pairs, markov_prob_smoothed, max_digit_overlap, miss_cycle_bonus, miss_value, neighbor, odd_even_key, pair_bonus, rebound_model, recent_recommend_penalty, road, slope_triplet_bonus, sum_interval_bonus, sum_trend_model,
)

def backtest_dan_kill(numbers, trials=100):
    """胆码/杀码独立回测
    
    参数：
        numbers: 历史号码数据
        trials: 回测期数
    
    返回：
        result: 胆码和杀码的回测统计
    """
    dan_hit1 = 0
    dan_hit2 = 0
    kill_fail = 0

    start = len(numbers) - trials

    for i in range(start, len(numbers)):
        train = numbers[:i]
        actual = numbers[i]
        actual_set = set(actual)

        ww = default_window_weights()
        meta = build_ranking_meta(train, ww)
        sc, _ = ensemble_digit_scores(train, ww, dynamic=meta.get("dynamic"))
        dan, _, kill, _ = pick_dan_tuo_kill(sc, enable_danma_random=False)

        hit_count = len(set(dan) & actual_set)

        if hit_count >= 1:
            dan_hit1 += 1
        if hit_count >= 2:
            dan_hit2 += 1

        if set(kill) & actual_set:
            kill_fail += 1

    return {
        "trials": trials,
        "dan_hit1_rate": dan_hit1 / trials,
        "dan_hit2_rate": dan_hit2 / trials,
        "kill_fail_rate": kill_fail / trials,
    }


def backtest_form_prediction(numbers, trials=100):
    """形态预测命中率回测
    
    参数：
        numbers: 历史号码数据
        trials: 回测期数
    
    返回：
        result: 形态预测回测统计
    """
    hit = 0
    zu6_hit = 0
    zu6_total = 0
    zu3_hit = 0
    zu3_total = 0

    start = len(numbers) - trials

    for i in range(start, len(numbers)):
        train = numbers[:i]
        actual_form = classify_form(numbers[i])

        ww = default_window_weights()
        pred = analyze_form_probability(train, window_weights=ww)
        pred_form = max(pred["blend_p"].items(), key=lambda x: x[1])[0]

        if pred_form == actual_form:
            hit += 1

        if pred_form == "zu6":
            zu6_total += 1
            if actual_form == "zu6":
                zu6_hit += 1

        if pred_form == "zu3":
            zu3_total += 1
            if actual_form == "zu3":
                zu3_hit += 1

    return {
        "trials": trials,
        "form_top1_rate": hit / trials,
        "zu6_precision": zu6_hit / zu6_total if zu6_total else 0,
        "zu3_precision": zu3_hit / zu3_total if zu3_total else 0,
    }


def backtest_sum_span_interval(numbers, trials=100):
    """和值/跨度区间独立回测
    
    参数：
        numbers: 历史号码数据
        trials: 回测期数
    
    返回：
        result: 和值/跨度区间回测统计
    """
    sum_hit_2 = 0
    sum_hit_3 = 0
    sum_hit_4 = 0
    span_hit_1 = 0
    span_hit_2 = 0

    start = len(numbers) - trials

    for i in range(start, len(numbers)):
        train = numbers[:i]
        actual = numbers[i]
        actual_sum = sum(actual)
        actual_span = max(actual) - min(actual)

        ww = default_window_weights()
        sums = [sum(x) for x in train]
        spans = [calc_span(x) for x in train]
        meta = build_ranking_meta(train, ww, sums, spans)

        sum_center = meta["sum_center"]
        span_center = meta["span_center"]

        if abs(actual_sum - sum_center) <= 2:
            sum_hit_2 += 1
        if abs(actual_sum - sum_center) <= 3:
            sum_hit_3 += 1
        if abs(actual_sum - sum_center) <= 4:
            sum_hit_4 += 1

        if abs(actual_span - span_center) <= 1:
            span_hit_1 += 1
        if abs(actual_span - span_center) <= 2:
            span_hit_2 += 1

    return {
        "trials": trials,
        "sum_hit_2_rate": sum_hit_2 / trials,
        "sum_hit_3_rate": sum_hit_3 / trials,
        "sum_hit_4_rate": sum_hit_4 / trials,
        "span_hit_1_rate": span_hit_1 / trials,
        "span_hit_2_rate": span_hit_2 / trials,
    }


def select_diverse_pool(
    pool,
    top_n=30,
    candidate_size=SERVED_POOL_CANDIDATE_SIZE,
    use_diversity=True,
    use_correlation=True,
):
    """贪心选池：从更大候选集中兼顾原始分、数字覆盖与去相关
    
    参数：
        pool: 候选池 [(权重, 号码字符串), ...]
        top_n: 目标推荐数量
        candidate_size: 候选集大小
        use_diversity: 是否启用数字覆盖奖励
        use_correlation: 是否启用重合惩罚
    """
    candidates = sorted(pool, key=lambda x: -x[0])[:candidate_size]
    selected = []
    selected_sets = []

    while candidates and len(selected) < top_n:
        best_item = None
        best_score = -float("inf")
        union_digits = set().union(*selected_sets) if selected_sets else set()

        for w, num in candidates:
            digits = set(num)
            
            # 重合惩罚（可选）
            overlap_penalty = (
                sum(
                    CORRELATION_PENALTY
                    for old_digits in selected_sets
                    if len(digits & old_digits) >= CORRELATION_THRESHOLD
                )
                if use_correlation else 0.0
            )
            
            # 数字覆盖奖励（可选）
            new_cover = (
                len(digits - union_digits)
                if use_diversity and selected_sets
                else 0.0
            )
            
            final_score = w + new_cover * DIVERSITY_WEIGHT - overlap_penalty

            if final_score > best_score:
                best_score = final_score
                best_item = (w, num)

        if best_item is None:
            break
        selected.append(best_item)
        selected_sets.append(set(best_item[1]))
        candidates.remove(best_item)

    return selected


def position_repeat_count(triple, last_draw):
    """与上期同位置重复个数（直选复刻）"""
    return sum(1 for i in range(3) if triple[i] == last_draw[i])


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _empty_lag1():
    return {
        "pairs": 0,
        "pos_repeat_rate": [RANDOM_POS_REPEAT] * 3,
        "avg_pos_repeat": RANDOM_POS_REPEAT,
        "repeat_dist": {0: 1.0},
        "full_repeat_rate": 0.0,
        "same_set_rate": 0.0,
        "ge2_overlap_rate": 0.0,
        "digit_reuse_rate": RANDOM_DIGIT_REUSE,
    }


def analyze_lag1_dynamics(numbers, window=RECENT_WINDOW):
    """分析近窗「上期→本期」转移：同位复刻、重号、全同号等"""
    if len(numbers) < 2:
        return _empty_lag1()

    pairs = list(zip(numbers[:-1], numbers[1:]))
    recent_pairs = pairs[-window:] if len(pairs) > window else pairs

    pos_w = [0.0] * 3
    repeat_dist = Counter()
    full_w = same_set_w = ge2_w = digit_hit = digit_total = 0.0
    total_w = 0.0
    w = 1.0
    for prev, cur in reversed(recent_pairs):
        rep = position_repeat_count(cur, prev)
        repeat_dist[rep] += w
        for j in range(3):
            if prev[j] == cur[j]:
                pos_w[j] += w
        if prev == cur:
            full_w += w
        if set(prev) == set(cur):
            same_set_w += w
        if len(set(prev) & set(cur)) >= 2:
            ge2_w += w
        for d in set(prev):
            digit_total += w
            if d in cur:
                digit_hit += w
        total_w += w
        w *= EXP_DECAY

    total_w = total_w or 1.0
    return {
        "pairs": len(recent_pairs),
        "pos_repeat_rate": [pos_w[i] / total_w for i in range(3)],
        "avg_pos_repeat": sum(pos_w) / (3 * total_w),
        "repeat_dist": {k: v / total_w for k, v in sorted(repeat_dist.items())},
        "full_repeat_rate": full_w / total_w,
        "same_set_rate": same_set_w / total_w,
        "ge2_overlap_rate": ge2_w / total_w,
        "digit_reuse_rate": digit_hit / digit_total if digit_total else RANDOM_DIGIT_REUSE,
    }


def ensemble_lag1_dynamics(numbers, window_weights):
    """多窗口加权集成上期→本期转移统计"""
    acc = _empty_lag1()
    if len(numbers) < 2:
        return acc

    pos_rate = [0.0] * 3
    repeat_dist = Counter()
    full = same_set = ge2 = digit_hit = digit_total = avg_rep = 0.0
    pairs_n = 0

    for w, wt in window_weights.items():
        lag = analyze_lag1_dynamics(numbers, window=w)
        pairs_n = max(pairs_n, lag["pairs"])
        for i in range(3):
            pos_rate[i] += wt * lag["pos_repeat_rate"][i]
        for k, v in lag["repeat_dist"].items():
            repeat_dist[k] += wt * v
        full += wt * lag["full_repeat_rate"]
        same_set += wt * lag["same_set_rate"]
        ge2 += wt * lag["ge2_overlap_rate"]
        digit_hit += wt * lag["digit_reuse_rate"]
        digit_total += wt
        avg_rep += wt * lag["avg_pos_repeat"]

    return {
        "pairs": pairs_n,
        "pos_repeat_rate": pos_rate,
        "avg_pos_repeat": avg_rep,
        "repeat_dist": dict(repeat_dist),
        "full_repeat_rate": full,
        "same_set_rate": same_set,
        "ge2_overlap_rate": ge2,
        "digit_reuse_rate": digit_hit / digit_total if digit_total else RANDOM_DIGIT_REUSE,
    }


def derive_dynamic_weights(lag1, consec_rate):
    """根据历史转移统计动态缩放评分权重与惩罚项"""
    avg_rep = lag1["avg_pos_repeat"]
    w_pos = W_POS_REPEAT * _clamp(avg_rep / RANDOM_POS_REPEAT, 0.2, 1.6)
    pos_mult = [_clamp(r / RANDOM_POS_REPEAT, 0.3, 2.0) for r in lag1["pos_repeat_rate"]]
    w_last = W_LAST_APPEAR * _clamp(lag1["digit_reuse_rate"] / RANDOM_DIGIT_REUSE, 0.3, 1.4)
    consec_base = max(consec_rate, 0.15)
    w_consec = W_CONSECUTIVE * _clamp(consec_rate / consec_base, 0.6, 1.2)
    w_full_pen = _clamp(12.0 * (1.0 - lag1["full_repeat_rate"] * 80), 4.0, 15.0)
    w_perm_pen = _clamp(6.0 * (1.0 - lag1["same_set_rate"] * 40), 1.5, 8.0)
    return {
        "w_pos_repeat": w_pos,
        "pos_mult": pos_mult,
        "w_last_appear": w_last,
        "w_consecutive": w_consec,
        "w_full_repeat_penalty": w_full_pen,
        "w_same_set_penalty": w_perm_pen,
    }


def analyze_patterns(numbers, window=RECENT_WINDOW):
    """统计近窗连号占比、奇偶比/大小比频次"""
    recent = _recent_slice(numbers, window)
    oe_freq = Counter()
    bs_freq = Counter()
    consec_w = 0.0
    w = 1.0
    for n in reversed(recent):
        oe_freq[odd_even_key(n)] += w
        bs_freq[big_small_key(n)] += w
        if has_consecutive_digits(*n):
            consec_w += w
        w *= EXP_DECAY
    total = sum(oe_freq.values()) or 1.0
    return {
        "oe_freq": oe_freq,
        "bs_freq": bs_freq,
        "consec_rate": consec_w / total,
    }


def ensemble_patterns(numbers, window_weights):
    """多窗口加权集成形态模式统计"""
    oe_acc = Counter()
    bs_acc = Counter()
    consec_rate = 0.0
    for w, wt in window_weights.items():
        p = analyze_patterns(numbers, window=w)
        for k, v in p["oe_freq"].items():
            oe_acc[k] += wt * v
        for k, v in p["bs_freq"].items():
            bs_acc[k] += wt * v
        consec_rate += wt * p["consec_rate"]
    oe_total = sum(oe_acc.values()) or 1.0
    bs_total = sum(bs_acc.values()) or 1.0
    return {
        "oe_freq": oe_acc,
        "bs_freq": bs_acc,
        "oe_total": oe_total,
        "bs_total": bs_total,
        "hot_oe_set": {k for k, _ in oe_acc.most_common(3)},
        "hot_bs_set": {k for k, _ in bs_acc.most_common(3)},
        "consec_rate": consec_rate,
    }


def analyze_sum_span(sums, spans, window=RECENT_WINDOW):
    recent_s = _recent_slice(sums, window)
    recent_p = _recent_slice(spans, window)
    w_s = exp_weighted_counts(recent_s)
    w_p = exp_weighted_counts(recent_p)

    sum_center = sum(k * v for k, v in w_s.items()) / max(sum(w_s.values()), 1e-9)
    span_center = sum(k * v for k, v in w_p.items()) / max(sum(w_p.values()), 1e-9)

    # 近期趋势偏移（可配置开关）
    # 默认关闭，避免追涨杀跌，等消融回测证明有效再开启
    if RECENT_SUM_SPAN_SHIFT > 0 and len(recent_s) >= 5:
        recent5_s = recent_s[-5:]
        recent5_p = recent_p[-5:]
        avg5_s = sum(recent5_s) / 5
        avg5_p = sum(recent5_p) / 5
        sum_center = (
            sum_center * (1 - RECENT_SUM_SPAN_SHIFT)
            + avg5_s * RECENT_SUM_SPAN_SHIFT
        )
        span_center = (
            span_center * (1 - RECENT_SUM_SPAN_SHIFT)
            + avg5_p * RECENT_SUM_SPAN_SHIFT
        )

    return {
        "sum_center": sum_center,
        "span_center": span_center,
        "hot_sums": [x for x, _ in w_s.most_common(6)],
        "hot_spans": [x for x, _ in w_p.most_common(4)],
        "sum_tail_freq": Counter(s % 10 for s in recent_s),
    }


def ensemble_sum_span(sums, spans, window_weights):
    """多窗口加权集成和值/跨度中心与热号"""
    sum_center = span_center = 0.0
    hot_sums_vote = Counter()
    hot_spans_vote = Counter()
    tail_acc = Counter()
    for w, wt in window_weights.items():
        r = analyze_sum_span(sums, spans, window=w)
        sum_center += wt * r["sum_center"]
        span_center += wt * r["span_center"]
        for s in r["hot_sums"]:
            hot_sums_vote[s] += wt
        for s in r["hot_spans"]:
            hot_spans_vote[s] += wt
        for tail, cnt in r["sum_tail_freq"].items():
            tail_acc[tail] += wt * cnt
    return {
        # 和值/跨度都是整数统计量，中心必须取整：用整数容差(±k)去框一个分数中心会
        # 不对称地少框一个取值（如 |v-4.5|<=1 只含{4,5}，而 |v-5|<=1 含{4,5,6}）。
        # 实测取整后 跨度±1 命中 30.8%→45%、和值±2 28.8%→34.6%。四舍五入到最近整数
        # 同时贴近分布众数(和值13/14、跨度5)，对平滑高斯打分几乎无影响。
        "sum_center": float(round(sum_center)),
        "span_center": float(round(span_center)),
        "hot_sums": [x for x, _ in hot_sums_vote.most_common(6)],
        "hot_spans": [x for x, _ in hot_spans_vote.most_common(4)],
        "sum_tail_freq": tail_acc,
    }


def digit_scores(numbers, window=RECENT_WINDOW, dynamic=None):
    recent = _recent_slice(numbers, window)
    last = numbers[-1]
    score = [0.0] * 10
    dyn = dynamic or {}
    w_last = dyn.get("w_last_appear", W_LAST_APPEAR)
    flags = FEATURE_FLAGS

    freq_all = exp_weighted_counts([d for n in recent for d in n])

    if flags.get("hot", True):
        for d, _ in freq_all.most_common(4):
            score[d] += W_HOT_GLOBAL

        for pos in range(3):
            pos_freq = exp_weighted_counts([n[pos] for n in recent])
            for d, _ in pos_freq.most_common(3):
                score[d] += W_HOT_POS

    if flags.get("markov", True):
        for pos in range(3):
            trans = build_markov(numbers, pos)
            prev_d = last[pos]
            row = trans.get(prev_d, Counter())
            for d, p in markov_prob_smoothed(row, range(10)).items():
                markov_score = W_MARKOV * p
                score[d] += min(markov_score, MARKOV_MAX_SCORE)

            if len(numbers) >= 2:
                trans2 = build_markov2(numbers, pos)
                prev2 = numbers[-2][pos]
                prev1 = last[pos]
                row2 = trans2.get((prev2, prev1), Counter())
                for d, p in markov_prob_smoothed(row2, range(10)).items():
                    markov2_score = W_MARKOV2 * p
                    score[d] += min(markov2_score, MARKOV_MAX_SCORE)

    if flags.get("miss", True):
        for d in range(10):
            mv = miss_value(numbers, d)
            if mv >= 20:
                score[d] += W_MISS_HIGH * (1 + mv / 20)
            elif mv >= 12:
                score[d] += W_MISS_MID

        miss_cycle_bonus_scores = miss_cycle_bonus(numbers)
        for d in range(10):
            score[d] += miss_cycle_bonus_scores.get(d, 0.0)

        entropy_bonus = entropy_model(numbers)
        for d in range(10):
            score[d] += entropy_bonus.get(d, 0.0)

        rebound_bonus = rebound_model(numbers)
        for d in range(10):
            score[d] += rebound_bonus.get(d, 0.0)

    if flags.get("neighbor", True):
        for d in set(last):
            score[d] += w_last

        nb = set()
        for d in last:
            nb.update(neighbor(d))
        for d in nb:
            score[d] += W_NEIGHBOR

    if flags.get("road", True):
        last_roads = {road(d) for d in last}
        for d in range(10):
            if road(d) in last_roads:
                score[d] += W_ROAD_MATCH

    return score, freq_all


def ensemble_digit_scores(numbers, window_weights, dynamic=None):
    combined = [0.0] * 10
    freq_combined = Counter()
    for w, wt in window_weights.items():
        sc, freq = digit_scores(numbers, window=w, dynamic=dynamic)
        for d in range(10):
            combined[d] += wt * sc[d]
        for d, c in freq.items():
            freq_combined[d] += wt * c
    
    # 注意：熵值奖励和回补奖励已经在 digit_scores() 内添加过，
    # 这里不再重复添加，避免双重加权
    # entropy_model() 和 rebound_model() 的奖励已在 digit_scores() 中处理
    
    return combined, freq_combined


def zu6_digit_scores(numbers, window_weights=None, dynamic=None):
    """Return position-free digit inclusion scores for the zu6 pool.

    A repeated digit in one draw is counted once because a four-digit zu6 pool
    only cares whether a digit is present.  All draw forms are retained: they
    are valid observations of the next draw's digit marginals, while filtering
    to historical zu6 draws discards roughly a quarter of the sample.

    ``window_weights`` and ``dynamic`` stay in the signature for API
    compatibility; the dedicated pool model intentionally does not reuse the
    positional straight-selection model.
    """
    if not numbers:
        return [0.0] * 10

    score = [0.0] * 10
    windows = [w for w in ZU6_PRESENCE_WINDOWS if w > 0]
    for window in windows:
        recent = _recent_slice(numbers, window)
        denom = max(1, len(recent))
        presence = Counter(d for draw in recent for d in set(draw))
        for digit in range(10):
            score[digit] += presence[digit] / denom

    # Deterministic short-window tie-breaker; too small to alter non-ties.
    short_recent = _recent_slice(numbers, min(windows) if windows else len(numbers))
    short_presence = Counter(d for draw in short_recent for d in set(draw))
    for digit in range(10):
        score[digit] += short_presence[digit] * 1e-6 - digit * 1e-9
    return score


def position_digit_scores(numbers, position, window=RECENT_WINDOW, dynamic=None):
    """单码分位评分（百/十/个），与主模型共用 FEATURE_FLAGS"""
    recent = [n[position] for n in _recent_slice(numbers, window)]
    last_d = numbers[-1][position]
    sc = [0.0] * 10
    dyn = dynamic or {}
    w_last = dyn.get("w_last_appear", W_LAST_APPEAR)
    pos_mult = dyn.get("pos_mult", [1.0, 1.0, 1.0])
    flags = FEATURE_FLAGS

    if flags.get("hot", True):
        for d, _ in exp_weighted_counts(recent).most_common(4):
            sc[d] += W_HOT_POS + 1

    if flags.get("markov", True):
        trans = build_markov(numbers, position)
        row = trans.get(last_d, Counter())
        for d, p in markov_prob_smoothed(row, range(10)).items():
            markov_score = W_MARKOV * p
            sc[d] += min(markov_score, MARKOV_MAX_SCORE)
        if len(numbers) >= 2:
            trans2 = build_markov2(numbers, position)
            prev2_d = numbers[-2][position]
            row2 = trans2.get((prev2_d, last_d), Counter())
            for d, p in markov_prob_smoothed(row2, range(10)).items():
                markov2_score = W_MARKOV2 * p
                sc[d] += min(markov2_score, MARKOV_MAX_SCORE)

    if flags.get("miss", True):
        for d in range(10):
            miss_p = miss_value(numbers, d, position=position)
            if miss_p >= 20:
                sc[d] += W_MISS_HIGH * (1 + miss_p / 20)
            elif miss_p >= 12:
                sc[d] += W_MISS_MID

    if flags.get("neighbor", True):
        sc[last_d] += w_last * pos_mult[position]
        for d in neighbor(last_d):
            sc[d] += W_NEIGHBOR

    return sc


def ensemble_position_digit_scores(numbers, position, window_weights, dynamic=None):
    sc = [0.0] * 10
    for w, wt in window_weights.items():
        ps = position_digit_scores(numbers, position, window=w, dynamic=dynamic)
        for d in range(10):
            sc[d] += wt * ps[d]
    return sc


_window_weights_cache = None


_window_weights_cache_time = 0


_window_weights_cache_numbers_hash = None


def default_window_weights():
    n = len(RECENT_WINDOWS)
    return {w: 1.0 / n for w in RECENT_WINDOWS}


def load_persisted_window_weights():
    """读取持久化的动态窗口权重"""
    try:
        data = kv_store.load(WINDOW_WEIGHTS_KV_KEY)
        if not data or not isinstance(data.get("weights"), dict):
            return None
        weights = {int(k): float(v) for k, v in data["weights"].items()}
        scores = {int(k): float(v) for k, v in (data.get("scores") or {}).items()}
        return {"weights": weights, "scores": scores, "period": data.get("period")}
    except Exception as e:
        log.debug(f"读取窗口权重失败: {e}")
        return None


def save_persisted_window_weights(weights, scores, period=None):
    """持久化动态窗口权重"""
    try:
        payload = {
            "weights": {str(k): round(v, 6) for k, v in weights.items()},
            "scores": {str(k): round(v, 4) for k, v in (scores or {}).items()},
            "period": period,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        kv_store.save(WINDOW_WEIGHTS_KV_KEY, payload)
        log.info(f"窗口权重已持久化: period={period}")
    except Exception as e:
        log.warning(f"保存窗口权重失败: {e}")


def refresh_persisted_window_weights(numbers, period=None):
    """重新计算并持久化窗口权重（回填后或手动刷新时调用）"""
    weights, scores = compute_window_weights(numbers, enable_cache=False)
    save_persisted_window_weights(weights, scores, period)
    return weights, scores


def resolve_window_weights(numbers, compute_weights=False, period=None):
    """获取预测用窗口权重：优先持久化缓存，必要时重算"""
    if compute_weights:
        weights, scores = compute_window_weights(numbers, enable_cache=False)
        save_persisted_window_weights(weights, scores, period)
        return weights, scores

    persisted = load_persisted_window_weights()
    if persisted:
        return persisted["weights"], persisted.get("scores", {})

    if len(numbers) >= max(RECENT_WINDOWS) + 10:
        weights, scores = compute_window_weights(numbers, enable_cache=True)
        save_persisted_window_weights(weights, scores, period)
        return weights, scores

    return default_window_weights(), {}


def compute_window_weights(numbers, trials=WINDOW_BACKTEST_TRIALS, enable_cache=True):
    """回测各窗口 Top3 命中表现，拉普拉斯先验后归一化为集成权重
    
    参数：
        numbers: 历史号码数据
        trials: 回测次数
        enable_cache: 是否启用缓存（默认 True）
    
    返回：
        (weights, scores): 窗口权重字典和原始分数字典
    """
    global _window_weights_cache, _window_weights_cache_time, _window_weights_cache_numbers_hash
    
    max_w = max(RECENT_WINDOWS)
    if len(numbers) < max_w + 10:
        return default_window_weights(), {}
    
    # 检查缓存
    numbers_hash = hash(tuple(tuple(n) for n in numbers[-max_w-10:]))
    if enable_cache and _window_weights_cache is not None:
        elapsed = time.time() - _window_weights_cache_time
        if elapsed < 3600 and _window_weights_cache_numbers_hash == numbers_hash:
            log.debug("使用缓存的窗口权重")
            return _window_weights_cache
    
    trials = min(trials, len(numbers) - max_w - 5)
    trials = max(10, trials)
    raw = {w: 0.0 for w in RECENT_WINDOWS}
    start = len(numbers) - trials

    for i in range(start, len(numbers)):
        train = numbers[:i]
        actual = numbers[i]
        act_s = f"{actual[0]}{actual[1]}{actual[2]}"
        for w in RECENT_WINDOWS:
            if len(train) < w:
                continue
            sums = [sum(x) for x in train]
            spans = [calc_span(x) for x in train]
            meta = build_ranking_meta(train, {w: 1.0}, sums, spans, tail_top=4)
            sc, _ = digit_scores(train, window=w, dynamic=meta.get("dynamic"))
            dan, _, kill, _ = pick_dan_tuo_kill(sc, enable_danma_random=False)
            top = rank_triplets(
                sc, dan, kill, meta,
                top_n=ZHIXUAN_TOP3,
                enable_exploration=False,
                apply_noise=False,
                enable_cold_hot_balance=False,
                enable_diversity=False,
                enable_correlation=False,
                recent_recommendations=None,
            )
            top_nums = [t[1] for t in top]
            if act_s in top_nums:
                raw[w] += 1.0
            elif max_digit_overlap(act_s, top_nums) >= 2:
                raw[w] += 0.25

    prior = WINDOW_WEIGHT_PRIOR
    total = sum(raw[w] + prior for w in RECENT_WINDOWS)
    weights = {w: (raw[w] + prior) / total for w in RECENT_WINDOWS}
    
    # 更新缓存
    if enable_cache:
        _window_weights_cache = (weights, {w: round(raw[w], 1) for w in RECENT_WINDOWS})
        _window_weights_cache_time = time.time()
        _window_weights_cache_numbers_hash = numbers_hash
    
    return weights, {w: round(raw[w], 1) for w in RECENT_WINDOWS}


def analyze_form_probability(numbers, window_weights=None):
    """估算本期开出组六/组三/豹子的概率（多源融合）"""
    forms = [classify_form(n) for n in numbers]
    last_form = forms[-1]

    if window_weights:
        recent_p = {k: 0.0 for k in THEORY_FORM_P}
        for w, wt in window_weights.items():
            rp = _form_recent_p(forms, w)
            for k in THEORY_FORM_P:
                recent_p[k] += wt * rp[k]
    else:
        recent_p = _form_recent_p(forms, RECENT_WINDOW)

    hist_cnt = Counter(forms)
    hist_total = len(forms)
    hist_p = {k: hist_cnt.get(k, 0) / hist_total for k in THEORY_FORM_P}

    trans = defaultdict(Counter)
    for i in range(len(forms) - 1):
        trans[forms[i]][forms[i + 1]] += 1
    row = trans.get(last_form, Counter())
    row_total = sum(row.values())
    markov_p = markov_prob_smoothed(row, THEORY_FORM_P)

    blend = {}
    for k in THEORY_FORM_P:
        blend[k] = (
            0.40 * recent_p[k]
            + 0.35 * markov_p[k]
            + 0.15 * hist_p[k]
            + 0.10 * THEORY_FORM_P[k]
        )
    total = sum(blend.values()) or 1.0
    blend = {k: v / total for k, v in blend.items()}

    streak = 1
    for i in range(len(forms) - 2, -1, -1):
        if forms[i] == last_form:
            streak += 1
        else:
            break

    return {
        "last_form": last_form,
        "streak": streak,
        "miss_zu6": form_miss(forms, "zu6"),
        "miss_zu3": form_miss(forms, "zu3"),
        "recent_p": recent_p,
        "hist_p": hist_p,
        "markov_p": markov_p,
        "blend_p": blend,
        "markov_samples": row_total,
    }


def recommend_form_bet(form_prob, numbers):
    """动态形态主推：本期更可能是组六还是组三。

    v4.9 变更：原实现固定主推「组六」（注释称短期信号无法击败 base rate）。
    现改为动态——主推 = blend 融合概率最大者，并给出组三概率相对其基准(27%)的
    抬升/回落信号。诚实说明：组六基准概率 72% 远高于组三 27%，500期 walk-forward
    实测动态 max 选组六占 100%，即"动态判断"在大多数时候仍指向组六；其真正价值
    在于量化展示组三概率何时抬升（如连续组六后 markov 信号偏组三），供加注参考，
    而非声称能预测形态（形态无短期可预测性，追涨杀跌是赌徒谬误）。
    """
    blend = form_prob["blend_p"]
    primary = max(blend, key=blend.get)
    secondary = "zu3" if primary != "zu3" else "zu6"
    zu3_elevation = blend["zu3"] - THEORY_FORM_P["zu3"]
    forms = [classify_form(n) for n in numbers]
    n = len(forms) or 1
    hist_cnt = Counter(forms)
    emp = {k: hist_cnt.get(k, 0) / n for k in THEORY_FORM_P}
    return {
        "primary": primary,
        "primary_label": FORM_LABELS[primary],
        "primary_prob": round(blend[primary], 4),
        "secondary": secondary,
        "zu6_prob": round(blend["zu6"], 4),
        "zu3_prob": round(blend["zu3"], 4),
        "zu3_base_rate": THEORY_FORM_P["zu3"],
        "zu3_elevation": round(zu3_elevation, 4),
        "zu3_signal": (
            "elevated" if zu3_elevation > 0.03
            else "depressed" if zu3_elevation < -0.03
            else "normal"
        ),
        "expected_hit_rate": round(emp[primary], 4),  # 主推形态的历史 base rate
        "theory_hit_rate": THEORY_FORM_P[primary],
        "empirical_form_p": {k: round(v, 4) for k, v in emp.items()},
        "blend_p": {k: round(v, 4) for k, v in blend.items()},
        "note": (
            "主推=blend概率最大形态（组六以72%基准概率占绝对优势，500期实测动态选组六100%）；"
            "zu3_elevation>0 表示组三概率高于其27%基准，可作为加注组三的参考，"
            "但形态本身无短期可预测性，概率波动属噪声。"
        ),
    }


def zu3_digit_presence(numbers, window=None):
    """组三条件下的数字出现率：最近组三开奖去重后各数字出现的比例（每期只计一次）。

    与组六 presence 模型（ZU6_PRESENCE_WINDOWS）同思路：只预测"数字是否进入
    组三开奖号集合"，不预测位置/重复位。组三样本不足时自动扩大到 60 期；
    仍无组三数据则返回均匀 0.2（无信息先验）。
    """
    window = window if window is not None else ZU3_PRESENCE_WINDOWS[0]
    zu3 = [set(n) for n in numbers[-window:] if classify_form(n) == "zu3"]
    if len(zu3) < ZU3_MIN_SAMPLES:
        zu3 = [set(n) for n in numbers[-60:] if classify_form(n) == "zu3"]
    if not zu3:
        return {d: 0.2 for d in range(10)}
    cnt = Counter()
    for s in zu3:
        cnt.update(s)
    total = len(zu3)
    return {d: cnt.get(d, 0) / total for d in range(10)}


def zu3_pair_scores(presence):
    """45 个无序数对的组三条件概率（独立性假设）：P({a,b}|zu3) ∝ r_a·r_b，归一化。"""
    scored = []
    total = 0.0
    for a in range(10):
        for b in range(a + 1, 10):
            s = presence[a] * presence[b]
            scored.append(((a, b), s))
            total += s
    total = total or 1.0
    return [(pair, s / total) for pair, s in scored]


def zu3_combos_from_pair(pair):
    """组选三对子 {a,b} 覆盖的全部 6 注单选（aab/aba/baa/abb/bab/bba）。"""
    a, b = sorted(pair)
    combos = set()
    for rep, single in ((a, b), (b, a)):
        for p in {0, 1, 2}:
            slot = [rep] * 3
            slot[p] = single
            combos.add("".join(map(str, slot)))
    return sorted(combos)


def zu3_zu_notes_from_pair(pair):
    """对子 {a,b} 的组选三表达：2 注覆盖全部 6 种排列（4 元），与 6 注单选（12 元）等价。

    福彩3D 规则：组选3 一注 = 3 码含一重复位 → 3 种排列（如 225 → 225/252/522）。
    对子 {2,5} 有双号 2（225）与双号 5（552）两个方向，共 6 种排列 → 2 注组选三即可。
    命中概率与 6 注单选完全相同（EV 相同），成本仅 1/3。
    """
    a, b = sorted(pair)
    return sorted({f"{rep}{rep}{single}" for rep, single in ((a, b), (b, a))})


def pick_zu3_pairs(numbers, limit=ZU3_PAIRS_COUNT, presence=None):
    """组三推荐：取组三条件概率最高的 4 个对子（四组）。

    每组 = 一个组选三对子 {a,b}：组选三 2 注（4 元）覆盖 6 种排列（v4.10 高效口径，
    原 6 注单选 = 12 元仅作对比保留）。任取 K 组（不要求互异），给定开奖为组三的
    条件命中率 = K/C(10,2) = K/45 —— 与选哪些码无关，顶部对子的概率差异
    （0.17~0.24 的数字率）只带来 1% 量级的微小偏移，属噪声。
    """
    presence = presence if presence is not None else zu3_digit_presence(numbers)
    scored = zu3_pair_scores(presence)
    scored.sort(key=lambda x: -x[1])
    top = scored[:limit]
    pairs = []
    for (a, b), pr in top:
        combos = zu3_combos_from_pair((a, b))
        zu_notes = zu3_zu_notes_from_pair((a, b))
        pairs.append({
            "digits": [a, b],
            "digits_str": f"{a}{b}",
            "prob": round(pr, 4),
            "notes": len(zu_notes),               # 组选三注数 = 2
            "cost": len(zu_notes) * TICKET_PRICE,  # 组选三成本 = 4 元
            "zu_notes": zu_notes,                 # 2 注组选三（高效口径，主推）
            "combos": combos,                     # 6 注单选（直选口径，对比）
            "direct_notes": len(combos),
            "direct_cost": len(combos) * TICKET_PRICE,
        })
    cond_hit = sum(pr for _, pr in top)
    return {
        "method": "zu3_conditional_presence",
        "window": ZU3_PRESENCE_WINDOWS[0],
        "presence": {d: round(v, 4) for d, v in presence.items()},
        "pairs": pairs,
        # 模型内样本估计：top4 对子概率和（presence 噪声被取顶放大，系过拟合，
        # 500期回测实测 ≈ 随机基准，勿当作真实命中率）
        "conditional_hit_rate": round(cond_hit, 4),
        # 数学精确基准：任取 K 组对子条件命中 = K/45（与选哪些码无关），回测实测 ≈ 此值
        "random_conditional_hit_rate": round(limit / 45.0, 4),
        "notes_total": sum(p["notes"] for p in pairs),          # 组选三 8 注
        "total_cost": sum(p["cost"] for p in pairs),            # 组选三 16 元（v4.10 主口径）
        "direct_notes_total": sum(p["direct_notes"] for p in pairs),  # 单选 24 注
        "direct_total_cost": sum(p["direct_cost"] for p in pairs),    # 单选 48 元（v4.9 口径）
        "note": (
            f"组选三：每组={pairs[0]['digits_str'] if pairs else ''}式对子 = 2 注组选三/4 元"
            f"（覆盖 6 种排列，与 6 注单选 12 元等价，EV 相同）；"
            f"任取{limit}组条件命中率={limit}/45≈{limit/45:.1%}（与选哪些码基本无关，"
            "数字出现率0.17~0.24差异为噪声；conditional_hit_rate 为模型内样本估计，"
            "500期回测实测≈随机基准，属过拟合）。"
        ),
    }


def zu3_coverage_tiers(numbers, sizes=ZU3_TIER_SIZES, presence=None):
    """组三覆盖档位：K 组对子 → 组选三 2K 注/4K 元，条件命中率 K/45（线性）。

    与组六 build_zu6_coverage_tiers 对称：K 组对子 = 排序后 top-K 前缀（复用同一
    presence/评分），任取 K 组条件命中 = K/C(10,2)，与选哪些码无关（回测实测 ≈ K/45）。
    直选口径（6K 注/12K 元）一并给出作对比：同样的 K 组覆盖，组选三成本仅 1/3。
    """
    presence = presence if presence is not None else zu3_digit_presence(numbers)
    scored = zu3_pair_scores(presence)
    scored.sort(key=lambda x: -x[1])
    tiers = []
    for k in sizes:
        k = min(k, 45)
        top = scored[:k]
        pairs = [list(p) for p, _ in top]
        tiers.append({
            "size": k,
            "pairs": pairs,
            "pairs_str": " ".join(f"{a}{b}" for a, b in top),
            "notes": k * 2,                # 组选三注数
            "cost": k * 4,                 # 组选三成本（元）
            "conditional_hit_rate": round(k / 45.0, 4),
            "direct_notes": k * 6,         # 直选注数（对比）
            "direct_cost": k * 12,         # 直选成本（对比）
        })
    return tiers


def pick_dan_tuo_kill(score, enable_danma_random=True):
    """动态选择胆码、拖码和杀码
    
    参数：
        score: 各数字评分
        enable_danma_random: 是否启用胆码随机选择
    
    返回：
        (胆码，拖码，杀码，排名列表)
    """
    rank = sorted(enumerate(score), key=lambda x: x[1], reverse=True)
    # 动态胆码机制：70%选 Top2，30%从 Top6 中随机选 2 个
    danma = select_danma(rank, enable_random=enable_danma_random)
    tuoma = [x[0] for x in rank[2:6]]
    kill = [rank[-1][0]] if rank[-1][1] + 3 < rank[-2][1] else [x[0] for x in rank[-2:]]
    return danma, tuoma, kill, rank


def pick_zu6_four(score, kill=None, use_kill=ZU6_USE_KILL, numbers=None, pair_freq=None):
    """组六四码：在 Top 候选中组合优化选 4 码"""
    return pick_zu6_pool(
        score, kill, pool_size=ZU6_FOUR_SIZE,
        use_kill=use_kill, numbers=numbers, pair_freq=pair_freq,
    )


def zu6_notes_from_digits(digits):
    """N 码组六 → C(N,3) 注组六组合"""
    combos = [tuple(sorted(c)) for c in combinations(digits, 3)]
    return combos, ["".join(map(str, c)) for c in combos]


TICKET_PRICE = 2


def build_zu6_coverage_tiers(score, kill=None, sizes=(4, 5, 6, 7), numbers=None):
    """组六复式覆盖档位：N 码 → C(N,3) 注，给出注数/成本/理论命中率。

    3D 为公平均匀摇奖，选哪些码无 edge（实测评分选码≈随机选码），
    唯一的杠杆是覆盖多少注：持有 K 注互异组六，无条件命中率 = K*6/1000
    （命中需开奖为组六且三码全在所选码内）。本函数把各档位摊开，供按预算选择。
    """
    tiers = []
    for n in sizes:
        digits = pick_zu6_pool(score, kill, pool_size=n, numbers=numbers)
        combos, combo_strs = zu6_notes_from_digits(digits)
        notes = len(combos)
        tiers.append({
            "size": n,
            "digits_str": "".join(map(str, digits)),
            "notes": notes,
            "cost": notes * TICKET_PRICE,
            "hit_rate": round(notes * 6 / 1000.0, 4),  # 无条件命中率（含"开奖须为组六"）
            "conditional_hit_rate": round(notes / 120.0, 4),  # 给定开奖为组六时 = notes/C(10,3)
            "is_primary": n == ZU6_POOL_SIZE,
            "combos": combo_strs,
        })
    return tiers


def build_zu6_primary(score, kill=None, numbers=None, size=ZU6_POOL_SIZE):
    """组六主推池：默认 6 码 → C(6,3)=20 注组六。

    与 build_zu6_coverage_tiers 中同尺寸档位取号一致（同一 pick_zu6_pool），
    供前端 zu6_primary 直接渲染，避免退化回四码却仍标注五码。
    """
    digits = pick_zu6_pool(score, kill, pool_size=size, numbers=numbers)
    combos, combo_strs = zu6_notes_from_digits(digits)
    notes = len(combos)
    return {
        "size": size,
        "digits": digits,
        "digits_str": "".join(map(str, digits)),
        "notes": notes,
        "cost": notes * TICKET_PRICE,
        "hit_rate": round(notes * 6 / 1000.0, 4),
        "conditional_hit_rate": round(notes / 120.0, 4),
        "is_primary": True,
        "combos": combo_strs,
    }


def evaluate_zu6_pool_recent(numbers, sizes=(5, 6), trials=100):
    """最近 N 期逐期样本外检验号码池，专门衡量“中几个数字”。

    每一期只使用它之前的数据选码，避免把当期开奖泄漏进评分。完整命中只在
    组六期统计；ge2_rate 则回答用户最直观的“至少覆盖两个不同开奖号”频率。
    """
    sizes = tuple(sorted({int(s) for s in sizes if 3 <= int(s) <= 10}))
    minimum_train = max(ZU6_PRESENCE_WINDOWS) + 5
    if len(numbers) <= minimum_train or not sizes:
        return {"trials": 0, "zu6_draws": 0, "tiers": {}}
    start = max(minimum_train, len(numbers) - max(1, int(trials)))
    stats = {
        size: {"full_hit": 0, "ge2_hit": 0, "overlap_sum": 0}
        for size in sizes
    }
    zu6_draws = 0
    evaluated = 0
    for i in range(start, len(numbers)):
        train = numbers[:i]
        actual = numbers[i]
        actual_set = set(actual)
        is_zu6 = len(actual_set) == 3
        zu6_draws += int(is_zu6)
        evaluated += 1
        scores = zu6_digit_scores(train)
        for size in sizes:
            pool = set(pick_zu6_pool(scores, pool_size=size, numbers=train))
            overlap = len(actual_set & pool)
            stats[size]["overlap_sum"] += overlap
            stats[size]["ge2_hit"] += int(overlap >= 2)
            stats[size]["full_hit"] += int(is_zu6 and actual_set <= pool)

    tiers = {}
    for size, item in stats.items():
        notes = math.comb(size, 3)
        tiers[str(size)] = {
            "size": size,
            "trials": evaluated,
            "zu6_draws": zu6_draws,
            "full_hit": item["full_hit"],
            "conditional_full_rate": round(
                item["full_hit"] / zu6_draws, 4
            ) if zu6_draws else 0.0,
            "unconditional_full_rate": round(
                item["full_hit"] / evaluated, 4
            ) if evaluated else 0.0,
            "ge2_rate": round(item["ge2_hit"] / evaluated, 4) if evaluated else 0.0,
            "avg_unique_overlap": round(
                item["overlap_sum"] / evaluated, 3
            ) if evaluated else 0.0,
            "theoretical_conditional_rate": round(notes / 120.0, 4),
            "theoretical_unconditional_rate": round(notes * 6 / 1000.0, 4),
        }
    return {"trials": evaluated, "zu6_draws": zu6_draws, "tiers": tiers}


def _zu6_four_payload(label, digits):
    digits = sorted(int(d) for d in digits)
    combos, combo_strs = zu6_notes_from_digits(digits)
    return {
        "label": label,
        "digits": digits,
        "digits_str": "".join(map(str, digits)),
        "notes": len(combos),
        "cost": len(combos) * TICKET_PRICE,
        "hit_rate": round(len(combos) * 6 / 1000.0, 4),
        "combos": combo_strs,
    }


def _zu6_four_balance_score(combo, score, kill=None):
    digits = tuple(sorted(combo))
    kill_set = set(kill or [])
    base = sum(_effective_digit_score(score, d, kill) for d in digits)
    odd = sum(1 for d in digits if d % 2)
    big = sum(1 for d in digits if d >= 5)
    span = digits[-1] - digits[0]
    adjacent_pairs = sum(1 for a, b in zip(digits, digits[1:]) if b - a == 1)
    kill_count = sum(1 for d in digits if d in kill_set)
    return (
        base
        - abs(odd - 2) * 1.0
        - abs(big - 2) * 0.8
        + min(span, 8) * 0.15
        - adjacent_pairs * 0.35
        - kill_count * 1.2
    )


def build_zu6_four_variants(score, kill=None, limit=4, numbers=None):
    """Build several deterministic four-digit zu6 groups for coverage comparison."""
    kill_eff = kill if ZU6_USE_KILL else None
    rank = sorted(range(10), key=lambda d: -_effective_digit_score(score, d, kill_eff))
    candidate_pool = rank[:8]
    primary = tuple(pick_zu6_four(score, kill, numbers=numbers))
    variants = []
    seen = set()

    def add(label, digits):
        key = tuple(sorted(digits))
        if key in seen or len(key) != 4:
            return
        seen.add(key)
        variants.append(_zu6_four_payload(label, key))

    add("主推", primary)
    balanced = max(
        combinations(candidate_pool, 4),
        key=lambda c: _zu6_four_balance_score(c, score, kill),
    )
    add("均衡", balanced)

    kill_set = set(kill or [])
    no_kill_pool = [d for d in rank if d not in kill_set][:6]
    if len(no_kill_pool) >= 4:
        add("避杀", no_kill_pool[:4])

    wide = max(
        combinations(candidate_pool, 4),
        key=lambda c: _zu6_four_balance_score(c, score, kill) + (max(c) - min(c)) * 0.3,
    )
    add("扩散", wide)

    for combo in sorted(
        combinations(candidate_pool, 4),
        key=lambda c: _zu6_four_balance_score(c, score, kill),
        reverse=True,
    ):
        add("备选", combo)
        if len(variants) >= limit:
            break

    return variants[:limit]


def _effective_digit_score(score, digit, kill=None):
    """单码有效分：杀码降权而非排除"""
    kill_set = set(kill or [])
    return score[digit] - (W_KILL_PENALTY if digit in kill_set else 0.0)


def _zu6_combo_score(combo, score, kill=None, pair_freq=None):
    """组六 N 码组合得分：单码分 + 对内共现 + 奇偶大小均衡。"""
    digits = tuple(sorted(combo))
    val = sum(_effective_digit_score(score, d, kill) for d in digits)
    if pair_freq:
        for i in range(len(digits)):
            for j in range(i + 1, len(digits)):
                val += pair_freq.get((digits[i], digits[j]), 0.0) * W_ZU6_PAIR
    odd = sum(1 for d in digits if d % 2)
    big = sum(1 for d in digits if d >= 5)
    val -= abs(odd - len(digits) / 2) * 0.5
    val -= abs(big - len(digits) / 2) * 0.4
    return val


def pick_zu6_pool(
    score, kill=None, pool_size=ZU6_POOL_SIZE,
    use_kill=ZU6_USE_KILL, pair_freq=None, numbers=None,
):
    """组六复式选号：按组六专用分取 Top N（默认不用杀码）。"""
    kill_eff = kill if use_kill else None
    rank = sorted(range(10), key=lambda d: -_effective_digit_score(score, d, kill_eff))
    return sorted(rank[:pool_size])


def _blend_dan_score(score, meta):
    """胆码/杀码用分位融合评分，与直选分位排序一致。"""
    dan_score = list(score)
    if meta.get("pos_scores"):
        for d in range(10):
            pos_sum = sum(meta["pos_scores"][p][d] for p in range(3))
            dan_score[d] = score[d] * 0.45 + pos_sum * 0.55
    return dan_score


def _triplet_digit_base(a, b, c, score, meta):
    """直选三位基础分：分位评分为主，全局评分为辅。"""
    pos_scores = meta.get("pos_scores")
    if pos_scores and len(pos_scores) == 3:
        return (
            W_TRIPLET_POS * (pos_scores[0][a] + pos_scores[1][b] + pos_scores[2][c])
            + W_TRIPLET_GLOBAL * (score[a] + score[b] + score[c])
        )
    return score[a] + score[b] + score[c]


def triplet_weight(a, b, c, score, danma, kill, meta, features=None):
    """计算三位数组合的评分权重
    
    参数：
        a, b, c: 百位、十位、个位数字
        score: 各数字评分数组
        danma: 胆码列表
        kill: 杀码列表
        meta: 元数据
        features: 特征开关字典（可选，默认为全局 FEATURE_FLAGS）
    """
    kill_set = set(kill or [])
    dyn = meta.get("dynamic") or {}
    flags = features if features is not None else FEATURE_FLAGS
    numbers = meta.get("numbers", [])

    w = _triplet_digit_base(a, b, c, score, meta)
    for x in (a, b, c):
        if x in danma:
            w += W_DANMA_HIT
        if x in kill_set:
            w -= W_KILL_PENALTY

    s = a + b + c
    
    # 和值跨度特征
    if flags.get("sum_span", True):
        w += 8.0 * gaussian_score(s, meta["sum_center"], SUM_SOFT_SIGMA)

        span = max(a, b, c) - min(a, b, c)
        w += 5.0 * gaussian_score(span, meta["span_center"], SPAN_SOFT_SIGMA)

        if s in meta["hot_sum_set"]:
            w += 2.0
        if span in meta["hot_span_set"]:
            w += 1.5
        if (s % 10) in meta["sum_tail_top"]:
            w += 1.0

    # 连号奖励
    if flags.get("consecutive", True) and has_consecutive_digits(a, b, c):
        w += dyn.get("w_consecutive", W_CONSECUTIVE)

    # 上期同位重复、全重复、同集合惩罚
    if flags.get("lag1_repeat", True):
        last_draw = meta.get("last_draw")
        w_pos = dyn.get("w_pos_repeat", W_POS_REPEAT)
        pos_mult = dyn.get("pos_mult", [1.0, 1.0, 1.0])
        if last_draw:
            triple = (a, b, c)
            for i in range(3):
                if triple[i] == last_draw[i]:
                    w += w_pos * pos_mult[i]
            if triple == tuple(last_draw):
                w -= dyn.get("w_full_repeat_penalty", 0.0)
            elif set(triple) == set(last_draw):
                w -= dyn.get("w_same_set_penalty", 0.0)

    # 奇偶比、大小比奖励
    if flags.get("ratio", True):
        oe = odd_even_key((a, b, c))
        bs = big_small_key((a, b, c))
        oe_freq = meta.get("oe_freq")
        bs_freq = meta.get("bs_freq")
        if oe_freq:
            w += W_RATIO_MATCH * oe_freq.get(oe, 0) / meta.get("oe_total", 1)
        if bs_freq:
            w += W_RATIO_MATCH * bs_freq.get(bs, 0) / meta.get("bs_total", 1)

    if flags.get("slope", True):
        w += slope_triplet_bonus(a, b, c, meta)
    
    # 数字配对奖励：使用 meta 预计算的高频对子
    high_pairs = meta.get("high_pairs") or set()
    if flags.get("pair", True) and high_pairs:
        w += pair_bonus((a, b, c), high_pairs)
    
    # 组三组六切换奖励：连续同形式出现后增加切换概率
    if flags.get("form_switch", True) and len(numbers) >= 5:
        form_bonus = form_switch_bonus(numbers)
        if a == b or a == c or b == c:
            w += form_bonus.get("zu3", 0.0)
        else:
            w += form_bonus.get("zu6", 0.0)

    # 形态先验：按真实形态概率加分(组六0.72/组三0.27/豹子0.01)，使推荐池形态分布贴合真实开奖。
    # 选哪些具体号无 edge(直选恒3%)，此项只调整推荐"长得像不像真实开奖"的形态构成。
    nd = len({a, b, c})
    w += W_FORM_PRIOR * (THEORY_FORM_P["zu6"] if nd == 3
                         else THEORY_FORM_P["zu3"] if nd == 2
                         else THEORY_FORM_P["baozi"])
    
    # 和值区间回归奖励：区间内加分，极端区间降权
    if flags.get("sum_span", True) and len(numbers) >= SUM_INTERVAL_WINDOW:
        sum_interval_info = sum_interval_bonus(numbers)
        w += sum_interval_info["bonus"].get(s, 0.0)
    
    return w


def triplet_weight_detail(a, b, c, score, danma, kill, meta):
    """计算三位数组合的详细得分分解，用于解释推荐原因
    
    参数：
        a, b, c: 百位、十位、个位数字
        score: 各数字评分数组
        danma: 胆码列表
        kill: 杀码列表
        meta: 元数据
    
    返回：
        detail: 包含各特征得分的字典
    """
    detail = {
        "base_digit": _triplet_digit_base(a, b, c, score, meta),
        "danma": 0.0,
        "kill": 0.0,
        "sum_span": 0.0,
        "pattern": 0.0,
        "last_repeat": 0.0,
        "ratio_match": 0.0,
        "pair": 0.0,
        "slope": 0.0,
        "form_switch": 0.0,
        "sum_interval": 0.0,
        "total": 0.0,
    }

    kill_set = set(kill or [])
    dyn = meta.get("dynamic") or {}
    flags = FEATURE_FLAGS

    for x in (a, b, c):
        if x in danma:
            detail["danma"] += W_DANMA_HIT
        if x in kill_set:
            detail["kill"] -= W_KILL_PENALTY

    s = a + b + c
    if flags.get("sum_span", True):
        detail["sum_span"] += 8.0 * gaussian_score(s, meta["sum_center"], SUM_SOFT_SIGMA)

        span = max(a, b, c) - min(a, b, c)
        detail["sum_span"] += 5.0 * gaussian_score(span, meta["span_center"], SPAN_SOFT_SIGMA)

        if s in meta["hot_sum_set"]:
            detail["sum_span"] += 2.0
        if span in meta["hot_span_set"]:
            detail["sum_span"] += 1.5
        if (s % 10) in meta["sum_tail_top"]:
            detail["sum_span"] += 1.0

    # 连号奖励
    if flags.get("consecutive", True) and has_consecutive_digits(a, b, c):
        detail["pattern"] += dyn.get("w_consecutive", W_CONSECUTIVE)

    # 上期同位重复、全重复、同集合惩罚
    if flags.get("lag1_repeat", True):
        last_draw = meta.get("last_draw")
        w_pos = dyn.get("w_pos_repeat", W_POS_REPEAT)
        pos_mult = dyn.get("pos_mult", [1.0, 1.0, 1.0])
        if last_draw:
            triple = (a, b, c)
            for i in range(3):
                if triple[i] == last_draw[i]:
                    detail["last_repeat"] += w_pos * pos_mult[i]
            if triple == tuple(last_draw):
                detail["last_repeat"] -= dyn.get("w_full_repeat_penalty", 0.0)
            elif set(triple) == set(last_draw):
                detail["last_repeat"] -= dyn.get("w_same_set_penalty", 0.0)

    # 奇偶比、大小比奖励
    if flags.get("ratio", True):
        oe = odd_even_key((a, b, c))
        bs = big_small_key((a, b, c))
        oe_freq = meta.get("oe_freq")
        bs_freq = meta.get("bs_freq")
        if oe_freq:
            detail["ratio_match"] += W_RATIO_MATCH * oe_freq.get(oe, 0) / meta.get("oe_total", 1)
        if bs_freq:
            detail["ratio_match"] += W_RATIO_MATCH * bs_freq.get(bs, 0) / meta.get("bs_total", 1)

    high_pairs = meta.get("high_pairs") or set()
    if flags.get("pair", True) and high_pairs:
        detail["pair"] += pair_bonus((a, b, c), high_pairs)

    if flags.get("slope", True):
        detail["slope"] += slope_triplet_bonus(a, b, c, meta)

    numbers = meta.get("numbers", [])
    if flags.get("form_switch", True) and len(numbers) >= 5:
        form_bonus = form_switch_bonus(numbers)
        if a == b or a == c or b == c:
            detail["form_switch"] += form_bonus.get("zu3", 0.0)
        else:
            detail["form_switch"] += form_bonus.get("zu6", 0.0)

    if flags.get("sum_span", True) and len(numbers) >= SUM_INTERVAL_WINDOW:
        sum_interval_info = sum_interval_bonus(numbers)
        detail["sum_interval"] += sum_interval_info["bonus"].get(s, 0.0)

    detail["total"] = sum(detail.values())
    return detail


def build_detail_list(items, score, danma, kill, meta):
    """为推荐号码列表构建带得分拆解的详情"""
    result = []
    for w, num in items:
        a, b, c = map(int, num)
        detail = triplet_weight_detail(a, b, c, score, danma, kill, meta)
        result.append({
            "num": num,
            "score": round(w, 1),
            "detail": {
                "base_digit": round(detail["base_digit"], 1),
                "danma": round(detail["danma"], 1),
                "kill": round(detail["kill"], 1),
                "sum_span": round(detail["sum_span"], 1),
                "pattern": round(detail["pattern"], 1),
                "last_repeat": round(detail["last_repeat"], 1),
                "ratio_match": round(detail["ratio_match"], 1),
                "pair": round(detail["pair"], 1),
                "slope": round(detail["slope"], 1),
                "form_switch": round(detail["form_switch"], 1),
                "sum_interval": round(detail["sum_interval"], 1),
            }
        })
    return result


def select_danma(score_rank, enable_random=True):
    """动态选择胆码
    
    参数：
        score_rank: 按评分排序的数字列表 [(数字, 分数), ...]
        enable_random: 是否启用随机选择
    
    返回：
        胆码列表（2 个数字）
    """
    top6_digits = [digit for digit, score in score_rank[:DANMA_TOP_POOL]]
    
    if enable_random and random.random() < DANMA_RANDOM_RATE:
        # 30%概率：从 Top6 中随机选 2 个
        return random.sample(top6_digits, 2)
    else:
        # 70%概率：选择前 2 个
        return top6_digits[:2]


def _position_constrained_pool(score, danma, kill, meta, per_pos=ZHXUAN_POS_TOPK):
    """百/十/个分位 Top 码笛卡尔积，用于精炼 Top3/Top5。"""
    pos_scores = meta.get("pos_scores")
    if not pos_scores:
        return []
    tops = [
        sorted(range(10), key=lambda d: -pos_scores[i][d])[:per_pos]
        for i in range(3)
    ]
    pool = []
    for a, b, c in product(*tops):
        w = triplet_weight(a, b, c, score, danma, kill, meta)
        pool.append((w, f"{a}{b}{c}"))
    pool.sort(key=lambda x: -x[0])
    return pool


def _merge_rank_pools(*pools, top_n):
    seen = set()
    merged = []
    for pool in pools:
        for item in pool:
            if item[1] not in seen:
                seen.add(item[1])
                merged.append(item)
    merged.sort(key=lambda x: -x[0])
    return merged[:top_n]


def rank_triplets(score, danma, kill, meta, top_n=20, enable_exploration=True, apply_noise=True, enable_cold_hot_balance=True, recent_recommendations=None, enable_diversity=True, enable_correlation=False):
    """对三位数组合进行评分排序，支持探索机制、随机扰动和冷热平衡
    
    参数：
        score: 各数字评分数组
        danma: 胆码列表
        kill: 杀码列表
        meta: 元数据
        top_n: 返回前 N 个推荐
        enable_exploration: 是否启用探索机制
        apply_noise: 是否应用随机噪声扰动
        enable_cold_hot_balance: 是否启用冷热平衡
        recent_recommendations: 最近推荐历史列表，用于排除重复推荐
        enable_diversity: 是否启用多样性控制
        enable_correlation: 是否启用到相关惩罚
    
    返回：
        排序后的推荐列表 [(权重，号码), ...]
    """
    pool = []
    for a, b, c in product(range(10), repeat=3):
        w = triplet_weight(a, b, c, score, danma, kill, meta)
        pool.append((w, f"{a}{b}{c}"))
    
    # 先排序
    pool.sort(key=lambda x: -x[0])
    
    # Top50 随机扰动：避免同分号长期霸榜
    if apply_noise:
        top50 = pool[:50]
        rest = pool[50:]
        top50 = [
            (w + random.uniform(-RANDOM_NOISE, RANDOM_NOISE), num)
            for w, num in top50
        ]
        pool = sorted(top50 + rest, key=lambda x: -x[0])
    
    # 最近5期排除机制：对重复推荐进行惩罚
    if recent_recommendations:
        pool = recent_recommend_penalty(pool, recent_recommendations)
        # 重新排序（应用惩罚后）
        pool.sort(key=lambda x: -x[0])
    
    # 冷热平衡模型：确保推荐池包含 40% 热号、40% 温号、20% 冷号
    if enable_cold_hot_balance:
        numbers = meta.get("numbers", [])
        if len(numbers) >= HOT_WINDOW:
            hot_digits, warm_digits, cold_digits = classify_digits_by_hot(numbers, HOT_WINDOW)
            
            # 冷热平衡先保留较大的候选池，不直接砍到 top_n
            balance_keep = max(top_n * 4, 100)
            
            # 计算各类别需要的号码数量
            hot_needed = max(1, int(balance_keep * HOT_RATIO))
            warm_needed = max(1, int(balance_keep * WARM_RATIO))
            cold_needed = max(1, int(balance_keep * COLD_RATIO))
            
            # 从各类别中选取最佳组合
            hot_pool = []
            warm_pool = []
            cold_pool = []
            
            for w, num_str in pool:
                digits = set(int(c) for c in num_str)
                hot_count = len(digits & set(hot_digits))
                warm_count = len(digits & set(warm_digits))
                cold_count = len(digits & set(cold_digits))
                
                # 根据组合中冷热号的比例分类
                if hot_count >= 2:
                    hot_pool.append((w, num_str))
                elif cold_count >= 1 and warm_count >= 1:
                    cold_pool.append((w, num_str))
                else:
                    warm_pool.append((w, num_str))
            
            # 合并并重新排序
            balanced_pool = []
            balanced_pool.extend(sorted(hot_pool, key=lambda x: -x[0])[:hot_needed])
            balanced_pool.extend(sorted(warm_pool, key=lambda x: -x[0])[:warm_needed])
            balanced_pool.extend(sorted(cold_pool, key=lambda x: -x[0])[:cold_needed])
            
            # 如果平衡池不足，从原池补充
            if len(balanced_pool) < balance_keep:
                remaining = [item for item in pool if item not in balanced_pool]
                balanced_pool.extend(remaining[:balance_keep - len(balanced_pool)])
            
            pool = balanced_pool[:balance_keep]
    
    # 探索机制：15%概率从 Top50 中随机选择，85%概率选择最高分
    if enable_exploration and random.random() < EXPLORATION_RATE:
        # 探索模式：从 Top50 中随机抽取
        top_50 = pool[:50] if len(pool) >= 50 else pool
        # 确保至少返回 top_n 个
        if len(top_50) >= top_n:
            # 随机打乱后取前 top_n 个
            random.shuffle(top_50)
            return top_50[:top_n]
        else:
            # 如果候选不足，返回全部
            return top_50
    
    # 正常模式：贪心选池或纯排序
    if enable_diversity or enable_correlation:
        result = select_diverse_pool(
            pool,
            top_n=top_n,
            candidate_size=max(top_n * 5, SERVED_POOL_CANDIDATE_SIZE),
            use_diversity=enable_diversity,
            use_correlation=enable_correlation,
        )
    else:
        result = pool[:top_n]

    # Top3/Top5：合并分位候选池，避免全量排序漏掉「分位热号组合」
    if (
        top_n <= 5
        and meta.get("pos_scores")
        and not enable_exploration
        and not enable_diversity
        and not enable_correlation
    ):
        pos_pool = _position_constrained_pool(score, danma, kill, meta)
        if pos_pool:
            result = _merge_rank_pools(pos_pool, pool, top_n=top_n)

    return result


def _meta_from_raw(meta_raw, tail_top=5):
    return {
        **meta_raw,
        "hot_sum_set": set(meta_raw["hot_sums"]),
        "hot_span_set": set(meta_raw["hot_spans"]),
        "sum_tail_top": {t for t, _ in meta_raw["sum_tail_freq"].most_common(tail_top)},
    }


def build_ranking_meta(numbers, window_weights, sums=None, spans=None, tail_top=5):
    """和值/跨度 + 模式 + 上期→本期转移，供直选排序使用"""
    if sums is None:
        sums = [sum(x) for x in numbers]
    if spans is None:
        spans = [calc_span(x) for x in numbers]
    meta = _meta_from_raw(ensemble_sum_span(sums, spans, window_weights), tail_top=tail_top)
    pat = ensemble_patterns(numbers, window_weights)
    meta.update(pat)
    lag1 = ensemble_lag1_dynamics(numbers, window_weights)
    meta["lag1"] = lag1
    meta["dynamic"] = derive_dynamic_weights(lag1, pat["consec_rate"])
    meta["last_draw"] = numbers[-1]
    meta["numbers"] = numbers  # 用于冷热平衡模型
    meta["high_pairs"] = high_freq_pairs(numbers) if len(numbers) >= 50 else set()
    meta["pos_scores"] = [
        ensemble_position_digit_scores(
            numbers, pos, window_weights, dynamic=meta.get("dynamic")
        )
        for pos in range(3)
    ]
    meta["slope"] = analyze_slope_patterns(numbers)
    
    # 和值趋势模型：仅在开启调整时才融合，否则保留多窗口中心
    base_sum_center = meta["sum_center"]
    adjusted_sum_center, trend_direction = sum_trend_model(numbers, SUM_TREND_WINDOW)
    if SUM_TREND_ADJUST != 0:
        meta["sum_center"] = (
            base_sum_center * 0.85
            + adjusted_sum_center * 0.15
        )
    else:
        meta["sum_center"] = base_sum_center
    meta["sum_trend"] = trend_direction
    
    return meta


def evaluate_strategy_admission(
    served_last100_rate,
    raw_last100_rate,
    actual_rank_avg,
    random_baseline=None,
    significance=None,
):
    """策略准入检查：仅当多项指标同时达标才建议进入实盘融合
    
    参数：
        random_baseline: 随机基准命中率（可选，默认使用理论基准 3%）
    """
    # 使用固定理论基准 3%（30/1000），避免单次随机抽样波动
    baseline_rate = random_baseline if random_baseline is not None else 0.03
    
    checks = {
        "served_top30_last100_above_baseline": {
            "passed": served_last100_rate >= baseline_rate,
            "actual": round(served_last100_rate, 4),
            "required": round(baseline_rate, 4),
            "reason": f"近100期 served Top30 不低于理论基准({baseline_rate*100:.1f}%)",
        },
        "raw_top30_last100_above_baseline": {
            "passed": raw_last100_rate >= baseline_rate,
            "actual": round(raw_last100_rate, 4),
            "required": round(baseline_rate, 4),
            "reason": f"近100期 raw Top30 不低于理论基准({baseline_rate*100:.1f}%)",
        },
        "avg_rank_below_500": {
            "passed": actual_rank_avg < 500,
            "actual": actual_rank_avg,
            "required": 500,
            "reason": "平均真实号码排名 < 500",
        },
    }
    if significance is not None:
        checks["permutation_significant"] = {
            "passed": significance.get("pvalue", 1.0) < 0.10,
            "actual": significance.get("pvalue"),
            "required": 0.10,
            "reason": "置换检验 p 值 < 0.10",
        }

    eligible = all(item["passed"] for item in checks.values())
    return {"eligible": eligible, "checks": checks}




# ─── 领域层适配 ───
#
# 窗口集成、数字评分、直选排名三段已迁入 `src/domain/numeric/lottery3d/`。
# 下面这一层只做两件事：把配置常量装进 `DigitWeights` / `TripletWeights`，
# 以及保住旧的函数名与签名（`prediction.py` / `backtest.py` / `__init__.py`
# 都按旧名字导入）。

from src.domain.numeric.lottery3d import digit_scoring as _digits
from src.domain.numeric.lottery3d import ranking as _ranking
from src.domain.numeric.lottery3d import weights as _weights
from src.domain.numeric.lottery3d import windows as _windows

_DIGIT_WEIGHTS = _weights.DigitWeights(
    hot_global=W_HOT_GLOBAL, hot_position=W_HOT_POS,
    markov=W_MARKOV, markov2=W_MARKOV2, markov_max=MARKOV_MAX_SCORE,
    markov_alpha=MARKOV_LAPLACE_ALPHA,
    miss_high=W_MISS_HIGH, miss_mid=W_MISS_MID,
    last_appear=W_LAST_APPEAR, neighbor=W_NEIGHBOR, road_match=W_ROAD_MATCH,
    decay=EXP_DECAY,
)

_TRIPLET_WEIGHTS = _weights.TripletWeights(
    danma_hit=W_DANMA_HIT, kill_penalty=W_KILL_PENALTY,
    sum_sigma=SUM_SOFT_SIGMA, span_sigma=SPAN_SOFT_SIGMA,
    consecutive=W_CONSECUTIVE, position_repeat=W_POS_REPEAT,
    ratio_match=W_RATIO_MATCH, slope_match=W_SLOPE_MATCH, pair_bonus=PAIR_BONUS,
    form_prior=W_FORM_PRIOR,
    triplet_position=W_TRIPLET_POS, triplet_global=W_TRIPLET_GLOBAL,
    diversity=DIVERSITY_WEIGHT,
    correlation_penalty=CORRELATION_PENALTY,
    correlation_threshold=CORRELATION_THRESHOLD,
    noise=RANDOM_NOISE, exploration_rate=EXPLORATION_RATE,
    danma_top_pool=DANMA_TOP_POOL, danma_random_rate=DANMA_RANDOM_RATE,
)

_BASELINES = _weights.Baselines(
    position_repeat=RANDOM_POS_REPEAT, digit_reuse=RANDOM_DIGIT_REUSE,
)

# 动态权重缩放的基准。领域层不读全局配置，所以这几个静态值由这里给。
_DYNAMIC_BASE = {
    'position_repeat': W_POS_REPEAT,
    'last_appear': W_LAST_APPEAR,
    'consecutive': W_CONSECUTIVE,
}

_clamp = _windows.clamp
def _empty_lag1():
    return _windows.empty_lag1(_BASELINES)
position_repeat_count = _windows.position_repeat_count


def analyze_lag1_dynamics(numbers, window=RECENT_WINDOW):
    return _windows.analyze_lag1(numbers, window, EXP_DECAY, _BASELINES)


def ensemble_lag1_dynamics(numbers, window_weights):
    return _windows.ensemble_lag1(numbers, window_weights, EXP_DECAY, _BASELINES)


def derive_dynamic_weights(lag1, consec_rate):
    return _windows.derive_dynamic_weights(lag1, consec_rate, _DYNAMIC_BASE, _BASELINES)


def analyze_patterns(numbers, window=RECENT_WINDOW):
    return _windows.analyze_patterns(numbers, window, EXP_DECAY)


def ensemble_patterns(numbers, window_weights):
    return _windows.ensemble_patterns(numbers, window_weights, EXP_DECAY)


def analyze_sum_span(sums, spans, window=RECENT_WINDOW):
    return _windows.analyze_sum_span(sums, spans, window, EXP_DECAY,
                                     RECENT_SUM_SPAN_SHIFT)


def ensemble_sum_span(sums, spans, window_weights):
    return _windows.ensemble_sum_span(sums, spans, window_weights, EXP_DECAY,
                                      RECENT_SUM_SPAN_SHIFT)


def _meta_from_raw(meta_raw, tail_top=5):
    return _windows.with_hot_sets(meta_raw, tail_top)


def digit_scores(numbers, window=RECENT_WINDOW, dynamic=None):
    """三份弱先验（遗漏周期、回补、熵值）在这里算好再喂进去：
    它们各自有窗口与阈值，那是配置问题。"""
    return _digits.digit_scores(
        numbers, window, _DIGIT_WEIGHTS, FEATURE_FLAGS, dynamic,
        miss_cycle=miss_cycle_bonus(numbers) if FEATURE_FLAGS.get('miss', True) else None,
        rebound=rebound_model(numbers) if FEATURE_FLAGS.get('miss', True) else None,
        entropy=entropy_model(numbers) if FEATURE_FLAGS.get('miss', True) else None,
    )


def ensemble_digit_scores(numbers, window_weights, dynamic=None):
    return _digits.ensemble_digit_scores(
        numbers, window_weights, _DIGIT_WEIGHTS, FEATURE_FLAGS, dynamic,
        miss_cycle=miss_cycle_bonus(numbers) if FEATURE_FLAGS.get('miss', True) else None,
        rebound=rebound_model(numbers) if FEATURE_FLAGS.get('miss', True) else None,
        entropy=entropy_model(numbers) if FEATURE_FLAGS.get('miss', True) else None,
    )


def position_digit_scores(numbers, position, window=RECENT_WINDOW, dynamic=None):
    return _digits.position_digit_scores(numbers, position, window,
                                         _DIGIT_WEIGHTS, FEATURE_FLAGS, dynamic)


def ensemble_position_digit_scores(numbers, position, window_weights, dynamic=None):
    return _digits.ensemble_position_digit_scores(
        numbers, position, window_weights, _DIGIT_WEIGHTS, FEATURE_FLAGS, dynamic)


def zu6_digit_scores(numbers, window_weights=None, dynamic=None):
    """`window_weights` 与 `dynamic` 留在签名里只为兼容旧调用方：
    组六选池模型刻意不复用分位直选模型。"""
    return _digits.zu6_digit_scores(numbers, ZU6_PRESENCE_WINDOWS)


def _triplet_context(danma, kill, meta):
    """建一次上下文。`form_switch` 与 `sum_interval` 只依赖历史——迁移前它们
    是在一千注的循环里各算一遍的。"""
    numbers = meta.get('numbers', [])
    return _ranking.build_context(
        meta, _TRIPLET_WEIGHTS, FEATURE_FLAGS, danma, kill,
        form_switch=(form_switch_bonus(numbers)
                     if len(numbers) >= 5 else None),
        sum_interval=(sum_interval_bonus(numbers)
                      if len(numbers) >= SUM_INTERVAL_WINDOW else None),
    )


def _blend_dan_score(score, meta):
    return _ranking.blend_dan_score(score, meta)


def _triplet_digit_base(a, b, c, score, meta):
    return _ranking._term_base((a, b, c), score, meta,
                               _triplet_context([], [], meta))


def triplet_weight(a, b, c, score, danma, kill, meta, features=None):
    context = _ranking.build_context(
        meta, _TRIPLET_WEIGHTS, features if features is not None else FEATURE_FLAGS,
        danma, kill,
        form_switch=(form_switch_bonus(meta.get('numbers', []))
                     if len(meta.get('numbers', [])) >= 5 else None),
        sum_interval=(sum_interval_bonus(meta.get('numbers', []))
                      if len(meta.get('numbers', [])) >= SUM_INTERVAL_WINDOW else None),
    )
    return _ranking.weight((a, b, c), score, meta, context)


def triplet_weight_detail(a, b, c, score, danma, kill, meta):
    return _ranking.detail((a, b, c), score, meta, _triplet_context(danma, kill, meta))


def build_detail_list(items, score, danma, kill, meta):
    return _ranking.build_detail_list(items, score, meta,
                                      _triplet_context(danma, kill, meta))


def select_danma(score_rank, enable_random=True):
    return _ranking.select_danma(score_rank, DANMA_TOP_POOL, DANMA_RANDOM_RATE,
                                 enable_random)


def select_diverse_pool(pool, top_n=30, candidate_size=SERVED_POOL_CANDIDATE_SIZE,
                        use_diversity=True, use_correlation=True):
    return _ranking.select_diverse_pool(
        pool, top_n, candidate_size, DIVERSITY_WEIGHT,
        CORRELATION_PENALTY, CORRELATION_THRESHOLD, use_diversity, use_correlation)


def _position_constrained_pool(score, danma, kill, meta, per_pos=ZHXUAN_POS_TOPK):
    return _ranking.position_constrained_pool(
        score, meta, _triplet_context(danma, kill, meta), per_pos)


def _merge_rank_pools(*pools, top_n):
    return _ranking.merge_pools(*pools, top_n=top_n)


def rank_triplets(score, danma, kill, meta, top_n=20, enable_exploration=True,
                  apply_noise=True, enable_cold_hot_balance=True,
                  recent_recommendations=None, enable_diversity=True,
                  enable_correlation=False):
    numbers = meta.get('numbers', [])
    hot_cold = None
    if enable_cold_hot_balance and len(numbers) >= HOT_WINDOW:
        hot, warm, cold = classify_digits_by_hot(numbers, HOT_WINDOW)
        hot_cold = {'hot': hot, 'warm': warm, 'cold': cold,
                    'hot_share': HOT_RATIO, 'warm_share': WARM_RATIO,
                    'cold_share': COLD_RATIO}
    return _ranking.rank_triplets(
        score, meta, _triplet_context(danma, kill, meta), top_n,
        hot_cold=hot_cold,
        recent_recommendations=recent_recommendations,
        penalise_recent=recent_recommend_penalty,
        diversity={'candidate_size': SERVED_POOL_CANDIDATE_SIZE},
        enable_exploration=enable_exploration, apply_noise=apply_noise,
        enable_diversity=enable_diversity, enable_correlation=enable_correlation,
        position_top_k=ZHXUAN_POS_TOPK,
    )
