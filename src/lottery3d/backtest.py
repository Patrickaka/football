# -*- coding: utf-8 -*-
"""福彩3D回测、随机基线、置换检验与权重搜索：旧名字对领域层的适配层。

算法本身在 `src/domain/numeric/lottery3d/` 的 `backtest`（滚动切片、命中判定、
统计聚合）与 `weight_search`（采样、扰动、留出测试集）两个模块里。这里留下
的只有三样东西：

1. **旧的函数名与签名**——`prediction.py` 与 `src/lottery3d/__init__.py`
   都按这些名字导入。
2. **把配置常量喂进去**。领域层不读全局配置：窗口、推荐注数、可调权重与
   它们的搜索范围一律由调用方传入。
3. **组装每期的预测**。怎么出号要走完评分与选号的整条链路，那是这一层的
   事；领域层只负责「真号落在哪些池子里」。
"""

import random

from ..common.logger import setup_logger

log = setup_logger('lottery3d')

from src.domain.numeric.lottery3d import backtest as _bt
from src.domain.numeric.lottery3d import weight_search as _search

from .config import (
    BACKTEST_TRIALS, FEATURE_FLAGS, PERMUTATION_SHUFFLES, RECENT_WINDOWS, RECOMMEND_GROUPS, TUNABLE_WEIGHTS, WEIGHT_SEARCH_RANGES, WINDOW_BACKTEST_TRIALS, ZHIXUAN_TOP3, ZU6_FOUR_SIZE, ZU6_POOL_SIZE, default_weights, patch_weights,
)
from .fetching import (
    fetch_data,
)
from .features import (
    calc_span,
)
from .scoring import (
    _blend_dan_score, build_ranking_meta, compute_window_weights, ensemble_digit_scores, evaluate_strategy_admission, pick_dan_tuo_kill, pick_zu6_four, pick_zu6_pool, rank_triplets, zu6_digit_scores,
)

# 全部 1000 注都要排出来：实际号码的名次是核心指标，只排前几十位的话
# 没进榜的期只能记成「很靠后」，平均排名就失真了。
FULL_RANK_SIZE = 1000
# 随机对照的种子。固定住是为了让两次回测之间的差异来自模型而不是运气。
RANDOM_BASELINE_SEED = 42
ZU6_RANDOM_SEED = 42
PERMUTATION_SEED = 20
# 窗口权重每这么多期重算一次，模拟实盘「回填后刷新权重」的节奏。
# 每期都算太慢，一次算到底又会让后半段用着过时的权重。
WEIGHT_REFRESH_EVERY = 10
# 搜索范围里按绝对值取值的那几个键（其余是相对基线的倍率）
ABSOLUTE_RANGE_KEYS = frozenset(
    key for key in WEIGHT_SEARCH_RANGES if key.endswith('_SIGMA'))

# 纯模型排名：把探索、噪声、多样性、去相关全关掉，量的是排序能力本身
RANK_KW_PURE = dict(
    enable_exploration=False,
    apply_noise=False,
    enable_cold_hot_balance=False,
    enable_diversity=False,
    enable_correlation=False,
    recent_recommendations=None,
)
# 实盘推荐池：与线上真正推出去的那一池保持一致（近期惩罚关闭，
# 因为回测里没有「最近推过什么」这回事）
RANK_KW_SERVED = dict(
    enable_exploration=False,
    apply_noise=False,
    enable_cold_hot_balance=FEATURE_FLAGS.get("cold_hot_balance", False),
    enable_diversity=True,
    enable_correlation=False,
    recent_recommendations=None,
)


def _rank_pools(train, window_weights):
    """一期的四个排名池：全排名、Top3、raw Top30、served Top30。"""
    sums = [sum(x) for x in train]
    spans = [calc_span(x) for x in train]
    meta = build_ranking_meta(train, window_weights, sums, spans, tail_top=4)
    score, _ = ensemble_digit_scores(train, window_weights, dynamic=meta.get("dynamic"))
    danma, _, kill, _ = pick_dan_tuo_kill(
        _blend_dan_score(score, meta), enable_danma_random=False)

    def ranked(top_n, **kwargs):
        return [item[1] for item in
                rank_triplets(score, danma, kill, meta, top_n=top_n, **kwargs)]

    return {
        'meta': meta,
        'score': score,
        'all': ranked(FULL_RANK_SIZE, **RANK_KW_PURE),
        'top3': ranked(ZHIXUAN_TOP3, **RANK_KW_PURE),
        'raw_top': ranked(RECOMMEND_GROUPS, **RANK_KW_PURE),
        'served_top': ranked(RECOMMEND_GROUPS, **RANK_KW_SERVED),
    }


def backtest(numbers, trials=BACKTEST_TRIALS, window_weights=None):
    """滚动回测。传了 `window_weights` 就固定用它（参数搜索时公平比较），
    传 None 则按实盘节奏每 10 期重算一次。
    """
    trials = _bt.resolve_trials(len(numbers), trials, max(RECENT_WINDOWS))
    accumulator = _bt.RankingBacktest(
        top3_size=ZHIXUAN_TOP3, recommend_size=RECOMMEND_GROUPS,
        zu6_four_size=ZU6_FOUR_SIZE, zu6_pool_size=ZU6_POOL_SIZE,
        rng=random.Random(ZU6_RANDOM_SEED))

    weights = dict(window_weights) if window_weights else None
    for index, (train, actual) in enumerate(_bt.rolling_slices(numbers, trials)):
        if window_weights is None and (weights is None or index % WEIGHT_REFRESH_EVERY == 0):
            weights, _ = compute_window_weights(
                train, trials=WINDOW_BACKTEST_TRIALS, enable_cache=False)

        pools = _rank_pools(train, weights)
        accumulator.observe(actual, pools['all'], pools['top3'],
                            pools['raw_top'], pools['served_top'])

        if accumulator.is_zu6(actual):
            zu6_score = zu6_digit_scores(train, weights, dynamic=pools['meta'].get("dynamic"))
            accumulator.observe_zu6(
                actual, pick_zu6_four(zu6_score),
                pick_zu6_pool(zu6_score, pool_size=ZU6_POOL_SIZE))

    random_result = random_baseline_backtest(
        numbers, trials=trials, top_n=RECOMMEND_GROUPS)
    raw_last, served_last = accumulator.recent_rates()
    return accumulator.summarise(
        random_rate=random_result["random_rate"],
        random_hit=random_result["random_hit"],
        # 不传 random_baseline，准入用固定的理论基准 3%——单次随机回测的
        # 波动能把门槛抬高或压低几个百分点
        admission=evaluate_strategy_admission(served_last, raw_last,
                                              accumulator.average_rank))


def random_baseline_backtest(numbers, trials=80, top_n=RECOMMEND_GROUPS,
                             seed=RANDOM_BASELINE_SEED):
    """随机基准回测：作为模型效果的对照基准。"""
    return _bt.random_baseline(numbers, trials, top_n, random.Random(seed))


def permutation_test(numbers, observed_rate, trials=BACKTEST_TRIALS,
                     window_weights=None, shuffles=PERMUTATION_SHUFFLES,
                     seed=PERMUTATION_SEED):
    """打乱历史顺序重跑回测，估计直选命中率优于随机的显著性。"""
    rng = random.Random(seed)
    rates = [backtest(seq, trials=trials, window_weights=window_weights)["top30_rate"]
             for seq in _bt.shuffled_series(numbers, shuffles, rng)]
    return _bt.permutation_summary(
        rates, observed_rate, RECOMMEND_GROUPS / FULL_RANK_SIZE)


def backtest_objective(bt, metric="top3_rate"):
    """从回测结果提取优化目标"""
    return _bt.objective(bt, metric)


def evaluate_weights(numbers, weights, trials=60, window_weights=None,
                     metric="top3_rate"):
    """给定权重在历史数据上跑滚动回测，返回 (目标值, 回测详情)。

    `patch_weights` 改的是模块级配置，是副作用，所以留在这一层——领域层的
    搜索只认 `evaluate(weights)` 这个接口，不知道权重是怎么生效的。
    """
    with patch_weights(weights):
        bt = backtest(numbers, trials=trials, window_weights=window_weights)
    return backtest_objective(bt, metric), bt


def _sample_random_weights(base, rng):
    """在默认权重附近随机采样一组候选参数"""
    return _search.sample_weights(
        base, TUNABLE_WEIGHTS, WEIGHT_SEARCH_RANGES, rng, ABSOLUTE_RANGE_KEYS)


def _mutate_weights(weights, rng, scale=_search.DEFAULT_MUTATION_SCALE):
    """在最优解附近做局部扰动。

    **签名比迁移前少了 `base`**：旧实现收着它，函数体里从没用过，
    而调用方一直在传。
    """
    return _search.mutate_weights(
        weights, TUNABLE_WEIGHTS, WEIGHT_SEARCH_RANGES, rng, scale,
        ABSOLUTE_RANGE_KEYS)


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
    """随机搜索 + 局部 refine，最大化历史回测命中率。

    metric: top3_rate | top_rate | ge2_digit_rate | composite
    返回 dict：baseline / best / improvement / history_len / test_result
    """
    if numbers is None:
        numbers = [x[2] for x in fetch_data()]
    if not numbers:
        return {"error": "未获取到数据"}

    train_numbers, test_numbers = _search.split_series(numbers, test_ratio)

    if verbose:
        print(f"数据切分: 训练集 {len(train_numbers)} 期, 测试集 {len(test_numbers)} 期")
        print(f"参数搜索: {iterations} 次随机采样 + {refine_rounds} 次局部 refine")
        print(f"回测期数={backtest_trials}, 目标={metric}")

    def evaluate(weights):
        # 不传固定窗口权重，让回测内部按时间滚动计算：训练集前面的预测
        # 不该看到训练集后段的开奖结果
        return evaluate_weights(train_numbers, weights, trials=backtest_trials,
                                window_weights=None, metric=metric)

    def report(phase, index, score, bt):
        if verbose:
            print(f"  [{phase} {index:3d}] 新最优 {score * 100:.2f}%  "
                  f"top3={bt['top3_rate'] * 100:.1f}%")

    outcome = _search.search(
        default_weights(), TUNABLE_WEIGHTS, WEIGHT_SEARCH_RANGES, evaluate,
        random.Random(seed), iterations, refine_rounds,
        absolute_keys=ABSOLUTE_RANGE_KEYS,
        on_improve=report if verbose else None)

    # 在测试集上验收最优参数（测试集从未参与搜索）。
    # 传入完整数据但只统计测试段的最后 N 期——测试期要有真实的历史上下文，
    # 与线上逻辑一致
    test_result = None
    test_trials = min(len(test_numbers), backtest_trials)
    if test_trials >= _bt.MIN_TRIALS:  # 少于这么多期，命中率没有统计意义
        _, test_result = evaluate_weights(
            numbers, outcome['best']['weights'], trials=test_trials,
            window_weights=None, metric=metric)
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
        "baseline": _as_report(outcome['baseline']),
        "best": _as_report(outcome['best']),
        "improvement": outcome['improvement'],
        "history_len": len(outcome['history']),
        "test_result": test_result,
    }


def _as_report(entry):
    """领域层管回测详情叫 detail，对外一直叫 backtest。"""
    return {"weights": entry['weights'], "score": entry['score'],
            "backtest": entry['detail']}


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
