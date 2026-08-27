# -*- coding: utf-8 -*-
"""福彩3D回测、随机基线、置换检验与权重搜索"""

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
    BACKTEST_TRIALS, FEATURE_FLAGS, PERMUTATION_SHUFFLES, RECENT_WINDOWS, RECOMMEND_GROUPS, TUNABLE_WEIGHTS, WEIGHT_SEARCH_RANGES, WINDOW_BACKTEST_TRIALS, ZHIXUAN_TOP3, ZU6_FOUR_SIZE, ZU6_POOL_SIZE, default_weights, patch_weights,
)
from .fetching import (
    fetch_data,
)
from .features import (
    calc_span, classify_form, max_digit_overlap,
)
from .scoring import (
    _blend_dan_score, build_ranking_meta, compute_window_weights, ensemble_digit_scores, evaluate_strategy_admission, pick_dan_tuo_kill, pick_zu6_four, pick_zu6_pool, rank_triplets, zu6_digit_scores,
)

def backtest(numbers, trials=BACKTEST_TRIALS, window_weights=None):
    """
    增强版滚动回测（稳定基础版）
    
    核心指标：
        - Top3 命中率
        - Top30 命中率
        - Top100 覆盖率
        - 平均真实号码排名
        - 真实号码中位排名
        - Top30 至少命中两个数字比例
    
    使用滚动窗口训练，与实盘逻辑保持一致。
    窗口权重每 10 期更新一次，更接近实际线上运行逻辑。
    """
    max_w = max(RECENT_WINDOWS)
    if len(numbers) < trials + max_w + 5:
        trials = max(20, len(numbers) - max_w - 5)

    hit_top = hit_top3 = hit_top100 = hit_ge2 = 0
    hit_raw_top30 = hit_served_top30 = 0
    raw_top30_hits = []
    served_top30_hits = []
    actual_ranks = []
    zu6_four_hit = zu6_pool_hit = zu6_draws = zu6_ge2_hit = 0
    random_zu6_four_hit = random_zu6_pool_hit = 0
    rng_zu6 = random.Random(42)
    start = len(numbers) - trials
    
    # 如果传入了固定窗口权重，使用它（用于参数搜索）
    # 否则动态计算（用于正常滚动回测）
    ww = dict(window_weights) if window_weights else None

    rank_kw_pure = dict(
        enable_exploration=False,
        apply_noise=False,
        enable_cold_hot_balance=False,
        enable_diversity=False,
        enable_correlation=False,
        recent_recommendations=None,
    )
    rank_kw_served = dict(
        enable_exploration=False,
        apply_noise=False,
        enable_cold_hot_balance=FEATURE_FLAGS.get("cold_hot_balance", False),
        enable_diversity=True,   # v4.6: 开启 diversity 提升 Top30
        enable_correlation=False, # v4.6: 保持关闭
        recent_recommendations=None,
    )

    for i in range(start, len(numbers)):
        # 使用滚动窗口：每次只用当前可用的数据
        train = numbers[:i]
        actual = numbers[i]
        
        # 每 10 期更新一次窗口权重（模拟实盘"回填后刷新权重"逻辑）
        # 只有当 window_weights 为 None 时才动态更新（正常滚动回测）
        # 如果传入了固定窗口权重（参数搜索），则保持不变
        if window_weights is None and (ww is None or (i - start) % 10 == 0):
            ww, _ = compute_window_weights(
                train,
                trials=WINDOW_BACKTEST_TRIALS,
                enable_cache=False,
            )
        
        sums = [sum(x) for x in train]
        spans = [calc_span(x) for x in train]
        meta = build_ranking_meta(train, ww, sums, spans, tail_top=4)
        sc, _ = ensemble_digit_scores(train, ww, dynamic=meta.get("dynamic"))
        dan, _, kill, _ = pick_dan_tuo_kill(_blend_dan_score(sc, meta), enable_danma_random=False)
        
        # 纯模型排名（1000 候选）
        all_ranked = rank_triplets(
            sc, dan, kill, meta,
            top_n=1000,
            **rank_kw_pure,
        )
        rank_map = {num: idx + 1 for idx, (_, num) in enumerate(all_ranked)}
        act_s = f"{actual[0]}{actual[1]}{actual[2]}"
        actual_rank = rank_map.get(act_s, 1001)
        actual_ranks.append(actual_rank)

        # 纯模型 Top3
        top3 = rank_triplets(sc, dan, kill, meta, top_n=ZHIXUAN_TOP3, **rank_kw_pure)
        top3_nums = [t[1] for t in top3]

        # raw Top30：纯排序能力
        raw_top30 = rank_triplets(
            sc, dan, kill, meta,
            top_n=RECOMMEND_GROUPS,
            **rank_kw_pure,
        )
        raw_top30_nums = [t[1] for t in raw_top30]

        # served Top30：模拟实盘推荐池（多样性+去相关，近期惩罚关闭）
        served_top30 = rank_triplets(
            sc, dan, kill, meta,
            top_n=RECOMMEND_GROUPS,
            **rank_kw_served,
        )
        served_top30_nums = [t[1] for t in served_top30]

        # Top100（用于覆盖率统计）
        top100_nums = [t[1] for t in all_ranked[:100]]

        raw_hit = act_s in raw_top30_nums
        served_hit = act_s in served_top30_nums
        raw_top30_hits.append(1 if raw_hit else 0)
        served_top30_hits.append(1 if served_hit else 0)
        if raw_hit:
            hit_raw_top30 += 1
        if served_hit:
            hit_served_top30 += 1
        if served_hit:
            hit_top += 1
        if act_s in top3_nums:
            hit_top3 += 1
        if act_s in top100_nums:
            hit_top100 += 1

        if max_digit_overlap(act_s, served_top30_nums) >= 2:
            hit_ge2 += 1

        if classify_form(actual) == "zu6":
            zu6_draws += 1
            actual_set = set(actual)
            z6_sc = zu6_digit_scores(train, ww, dynamic=meta.get("dynamic"))
            z4 = set(pick_zu6_four(z6_sc))
            z5 = set(pick_zu6_pool(z6_sc, pool_size=ZU6_POOL_SIZE))
            if actual_set <= z4:
                zu6_four_hit += 1
            if actual_set <= z5:
                zu6_pool_hit += 1
            if len(actual_set & z4) >= 2:
                zu6_ge2_hit += 1
            if actual_set <= set(rng_zu6.sample(range(10), ZU6_FOUR_SIZE)):
                random_zu6_four_hit += 1
            if actual_set <= set(rng_zu6.sample(range(10), ZU6_POOL_SIZE)):
                random_zu6_pool_hit += 1

    n = trials
    
    # 计算真实号码排名统计
    sorted_ranks = sorted(actual_ranks)
    actual_rank_avg = sum(actual_ranks) / len(actual_ranks) if actual_ranks else 0.0
    actual_rank_median = sorted_ranks[len(sorted_ranks) // 2] if sorted_ranks else 0
    actual_rank_top100_rate = sum(1 for r in actual_ranks if r <= 100) / n if n > 0 else 0.0
    actual_rank_top300_rate = sum(1 for r in actual_ranks if r <= 300) / n if n > 0 else 0.0
    
    # 计算随机基准
    random_result = random_baseline_backtest(numbers, trials=trials, top_n=RECOMMEND_GROUPS)

    last100 = min(100, n)
    raw_last100_rate = sum(raw_top30_hits[-last100:]) / last100 if last100 else 0.0
    served_last100_rate = sum(served_top30_hits[-last100:]) / last100 if last100 else 0.0
    random_baseline = round(random_result["random_rate"], 4)

    return {
        "trials": n,
        "strategy": "stable_baseline",
        # TopK 命中率
        "top3_hit": hit_top3,
        "top3_rate": round(hit_top3 / n, 4) if n > 0 else 0.0,
        "top3_rate_baseline": round(ZHIXUAN_TOP3 / 1000.0, 4),
        # raw / served Top30
        "raw_top30_hit": hit_raw_top30,
        "raw_top30_rate": round(hit_raw_top30 / n, 4) if n > 0 else 0.0,
        "served_top30_hit": hit_served_top30,
        "served_top30_rate": round(hit_served_top30 / n, 4) if n > 0 else 0.0,
        "raw_top30_last100_rate": round(raw_last100_rate, 4),
        "served_top30_last100_rate": round(served_last100_rate, 4),
        # 主展示指标 = 实盘推荐池
        "top30_hit": hit_served_top30,
        "top30_rate": round(hit_served_top30 / n, 4) if n > 0 else 0.0,
        "top30_rate_baseline": round(RECOMMEND_GROUPS / 1000.0, 4),
        # 兼容旧字段名
        "top_hit": hit_served_top30,
        "top_rate": round(hit_served_top30 / n, 4) if n > 0 else 0.0,
        "top_rate_baseline": round(RECOMMEND_GROUPS / 1000.0, 4),
        "top100_hit": hit_top100,
        "top100_rate": round(hit_top100 / n, 4) if n > 0 else 0.0,
        # 真实号码排名指标（核心）
        "actual_rank_avg": round(actual_rank_avg, 1),
        "actual_rank_median": actual_rank_median,
        "actual_rank_top100_rate": round(actual_rank_top100_rate, 4),
        "actual_rank_top300_rate": round(actual_rank_top300_rate, 4),
        # 数字命中比例
        "ge2_digit_rate": round(hit_ge2 / n, 4) if n > 0 else 0.0,
        # 组六四码/五码（仅组六开奖期统计）
        "zu6_draws": zu6_draws,
        "zu6_four_hit": zu6_four_hit,
        "zu6_four_rate": round(zu6_four_hit / zu6_draws, 4) if zu6_draws else 0.0,
        "zu6_pool_hit": zu6_pool_hit,
        "zu6_pool_rate": round(zu6_pool_hit / zu6_draws, 4) if zu6_draws else 0.0,
        "zu6_ge2_hit": zu6_ge2_hit,
        "zu6_ge2_rate": round(zu6_ge2_hit / zu6_draws, 4) if zu6_draws else 0.0,
        "zu6_random_four_rate": round(random_zu6_four_hit / zu6_draws, 4) if zu6_draws else 0.0,
        "zu6_random_pool_rate": round(random_zu6_pool_hit / zu6_draws, 4) if zu6_draws else 0.0,
        # 随机基准（仅用于页面展示，准入使用固定理论基准 3%）
        "random_rate": random_baseline,
        "random_hit": random_result["random_hit"],
        "admission": evaluate_strategy_admission(
            served_last100_rate,
            raw_last100_rate,
            actual_rank_avg,
            # 不传 random_baseline，使用默认理论基准 3%
        ),
    }


def random_baseline_backtest(numbers, trials=80, top_n=30, seed=42):
    """随机基准回测：作为模型效果的对照基准
    
    参数：
        numbers: 历史号码数据
        trials: 回测期数
        top_n: 推荐数量
        seed: 随机种子（固定以保证可重复）
    
    返回：
        result: 随机基准回测结果
    """
    rng = random.Random(seed)
    hit = 0

    start = len(numbers) - trials

    for i in range(start, len(numbers)):
        actual = numbers[i]
        act_s = f"{actual[0]}{actual[1]}{actual[2]}"

        pool = [f"{a}{b}{c}" for a in range(10) for b in range(10) for c in range(10)]
        picks = set(rng.sample(pool, top_n))

        if act_s in picks:
            hit += 1

    return {
        "trials": trials,
        "random_hit": hit,
        "random_rate": hit / trials if trials > 0 else 0.0,
    }


def permutation_test(numbers, observed_rate, trials=BACKTEST_TRIALS,
                     window_weights=None, shuffles=PERMUTATION_SHUFFLES, seed=20):
    """打乱历史顺序重跑回测，估计直选命中率优于随机的显著性。

    福彩3D 为独立均匀摇奖，期间无时序可学。若打乱顺序后命中率不降，
    说明模型未抓到真实信号；p 值为打乱样本命中率 >= 实际命中率的比例。
    """
    seq = [list(n) for n in numbers]
    rng = random.Random(seed)
    perm_rates = []
    for _ in range(shuffles):
        rng.shuffle(seq)
        perm_rates.append(backtest(seq, trials=trials, window_weights=window_weights)["top30_rate"])
    ge = sum(1 for r in perm_rates if r >= observed_rate)
    mean = sum(perm_rates) / len(perm_rates) if perm_rates else 0.0
    pvalue = (ge + 1) / (shuffles + 1)
    return {
        "shuffles": shuffles,
        "observed_rate": observed_rate,
        "shuffled_mean_rate": mean,
        "shuffled_max_rate": max(perm_rates) if perm_rates else 0.0,
        "baseline_rate": RECOMMEND_GROUPS / 1000.0,
        "pvalue": pvalue,
        "significant": pvalue < 0.05,
    }


def backtest_objective(bt, metric="top3_rate"):
    """从回测结果提取优化目标"""
    if metric == "top_rate":
        metric = "top30_rate"
    if metric == "composite":
        return (
            0.55 * bt["top3_rate"]
            + 0.30 * bt["top30_rate"]
            + 0.15 * bt["ge2_digit_rate"]
        )
    if metric not in bt:
        raise ValueError(f"未知 metric: {metric}")
    return bt[metric]


def evaluate_weights(
    numbers,
    weights,
    trials=60,
    window_weights=None,
    metric="top3_rate",
):
    """给定权重在历史数据上跑滚动回测，返回 (目标值, 回测详情)

    参数：
        window_weights: 固定窗口权重（用于参数搜索时公平比较）。
                       设为 None 时，由 backtest() 内部按时间滚动计算，
                       避免训练集内部前视。
    """
    with patch_weights(weights):
        bt = backtest(
            numbers,
            trials=trials,
            window_weights=window_weights,
        )
    return backtest_objective(bt, metric), bt


def _sample_random_weights(base, rng):
    """在默认权重附近随机采样一组候选参数"""
    candidate = {}
    for k in TUNABLE_WEIGHTS:
        lo, hi = WEIGHT_SEARCH_RANGES.get(k, (0.5, 2.0))
        if k.endswith("_SIGMA"):
            candidate[k] = rng.uniform(lo, hi)
        else:
            candidate[k] = base[k] * rng.uniform(lo, hi)
    return candidate


def _mutate_weights(weights, base, rng, scale=0.15):
    """在最优解附近做局部扰动"""
    candidate = dict(weights)
    k = rng.choice(TUNABLE_WEIGHTS)
    lo, hi = WEIGHT_SEARCH_RANGES.get(k, (0.5, 2.0))
    if k.endswith("_SIGMA"):
        delta = (hi - lo) * scale * rng.uniform(-1, 1)
        candidate[k] = max(lo, min(hi, candidate[k] + delta))
    else:
        candidate[k] = max(0.1, candidate[k] * (1 + scale * rng.uniform(-1, 1)))
    return candidate


def search_weights(
    numbers=None,
    iterations=80,
    backtest_trials=60,
    metric="top3_rate",
    seed=42,
    refine_rounds=30,
    verbose=True,
    test_ratio=0.15,  # 预留测试集比例，不参与搜索
):
    """
    随机搜索 + 局部 refine，最大化历史回测命中率。

    参数：
        test_ratio: 预留测试集比例，用于最终验收，不参与参数搜索（防止数据泄漏）
    
    metric: top3_rate | top_rate | ge2_digit_rate | composite
    返回 dict：baseline / best / improvement / history / test_result
    """
    if numbers is None:
        numbers = [x[2] for x in fetch_data()]
    if not numbers:
        return {"error": "未获取到数据"}

    # 时序切分：训练集用于参数搜索，测试集用于最终验收
    train_size = int(len(numbers) * (1 - test_ratio))
    train_numbers = numbers[:train_size]
    test_numbers = numbers[train_size:]
    
    if verbose:
        print(f"数据切分: 训练集 {len(train_numbers)} 期, 测试集 {len(test_numbers)} 期")
        print(f"参数搜索: {iterations} 次随机采样 + {refine_rounds} 次局部 refine")
        print(f"回测期数={backtest_trials}, 目标={metric}")

    rng = random.Random(seed)
    base = default_weights()
    
    # 不预先计算窗口权重，让 backtest() 内部按时间滚动计算
    # 这样训练集前面的预测不会看到训练集后段的开奖结果
    _, baseline_bt = evaluate_weights(
        train_numbers, base, trials=backtest_trials, window_weights=None, metric=metric
    )
    baseline_score = backtest_objective(baseline_bt, metric)
    best_weights = dict(base)
    best_score = baseline_score
    best_bt = baseline_bt
    history = []

    for i in range(iterations):
        candidate = _sample_random_weights(base, rng)
        # 不传固定窗口权重，让回测内部按时间滚动计算
        score, bt = evaluate_weights(
            train_numbers, candidate, trials=backtest_trials, window_weights=None, metric=metric
        )
        history.append({"phase": "random", "score": score, "weights": candidate})
        if score > best_score:
            best_score, best_weights, best_bt = score, candidate, bt
            if verbose:
                print(f"  [random {i + 1:3d}] 新最优 {score * 100:.2f}%  top3={bt['top3_rate'] * 100:.1f}%")

    for i in range(refine_rounds):
        candidate = _mutate_weights(best_weights, base, rng)
        # 不传固定窗口权重，避免训练集内未来信息泄漏
        score, bt = evaluate_weights(
            train_numbers, candidate, trials=backtest_trials, window_weights=None, metric=metric
        )
        history.append({"phase": "refine", "score": score, "weights": candidate})
        if score > best_score:
            best_score, best_weights, best_bt = score, candidate, bt
            if verbose:
                print(f"  [refine {i + 1:3d}] 新最优 {score * 100:.2f}%  top3={bt['top3_rate'] * 100:.1f}%")

    # 在测试集上验收最优参数（测试集从未参与搜索）
    # 注意：传入完整数据（训练集+测试集），但只统计测试段的最后 N 期
    # 这样测试期有真实的历史上下文，与线上逻辑一致
    test_result = None
    test_trials = min(len(test_numbers), backtest_trials)
    if test_trials >= 20:  # 至少20期才有统计意义
        _, test_result = evaluate_weights(
            numbers, best_weights, trials=test_trials, window_weights=None, metric=metric
        )
        if verbose:
            print(f"\n测试集验收（测试段 {test_trials} 期，使用完整历史上下文）:")
            print(f"  Top3 命中率: {test_result['top3_rate'] * 100:.2f}%")
            print(f"  Top30 命中率: {test_result['top30_rate'] * 100:.2f}%")
            print(f"  平均排名: {test_result['actual_rank_avg']}")

    return {
        "metric": metric,
        "backtest_trials": backtest_trials,
        "train_size": len(train_numbers),
        "test_size": len(test_numbers),
        "baseline": {"weights": base, "score": baseline_score, "backtest": baseline_bt},
        "best": {"weights": best_weights, "score": best_score, "backtest": best_bt},
        "improvement": best_score - baseline_score,
        "history_len": len(history),
        "test_result": test_result,
    }


def print_search_report(result):
    """打印权重搜索结果"""
    if result.get("error"):
        print(result["error"])
        return

    base_w = result["baseline"]["weights"]
    best_w = result["best"]["weights"]
    base_bt = result["baseline"]["backtest"]
    best_bt = result["best"]["backtest"]

    print("\n" + "=" * 70)
    print("【评分权重搜索】")
    print("=" * 70)
    print(f"  目标指标: {result['metric']}  |  回测期数: {result['backtest_trials']}")
    print(f"  基线 {result['baseline']['score'] * 100:.2f}%  →  最优 {result['best']['score'] * 100:.2f}%  "
          f"(+{result['improvement'] * 100:.2f}%)")

    print("\n  回测对比:")
    for label, bt in ("基线", base_bt), ("最优", best_bt):
        print(
            f"    {label}: Top3 {bt['top3_rate'] * 100:.1f}% ({bt['top3_hit']}/{bt['trials']})  "
            f"| Top{RECOMMEND_GROUPS} {bt['top30_rate'] * 100:.1f}%  "
            f"| ≥2码 {bt['ge2_digit_rate'] * 100:.1f}%"
        )

    print("\n  权重变化 (默认 → 最优):")
    for k in TUNABLE_WEIGHTS:
        b, n = base_w[k], best_w[k]
        delta = ((n / b - 1) * 100) if b else 0
        print(f"    {k:16s}  {b:6.2f}  →  {n:6.2f}  ({delta:+.0f}%)")

    print("\n  可复制到 lottery3d.py 顶部:")
    for k in TUNABLE_WEIGHTS:
        v = best_w[k]
        fmt = f"{v:.2f}" if isinstance(v, float) and not v.is_integer() else str(int(v) if v == int(v) else v)
        print(f"    {k} = {fmt}")


