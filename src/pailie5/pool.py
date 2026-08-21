#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""排列五推荐池生成：组合评分、位置级评分、多样性选池、窗口权重回测"""

from collections import Counter
from itertools import product as iproduct

from .config import (
    CORRELATION_PENALTY, CORRELATION_THRESHOLD, COVERAGE_WEIGHT,
    DIVERSITY_WEIGHT, FEATURE_FLAGS, MARKOV_MAX_SCORE, RECENT_WINDOWS,
    RECOMMEND_GROUPS, SPAN_SOFT_SIGMA, SUM_SOFT_SIGMA, W_CONSECUTIVE,
    W_DANMA_HIT, W_DISTINCT, W_HOT_POS, W_KILL_PENALTY, W_MARKOV, W_MARKOV2,
    W_POS_SPECIFIC, W_RATIO_MATCH, W_REPEAT,
)
from .features import (
    _recent_slice, analyze_ratio_pattern, apply_recent_recommend_penalty,
    big_small_key_5, build_markov2_pos, build_markov_pos,
    default_window_weights, digit_scores_single_window,
    ensemble_digit_scores_multi_window, ensemble_sum_span_5,
    exp_weighted_counts, gaussian_score, has_consecutive_digits_5,
    load_recent_recommend, markov_prob_smoothed, odd_even_key_5,
    pick_dan_kill,
)


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
