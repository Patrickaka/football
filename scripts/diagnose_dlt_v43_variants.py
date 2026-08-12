#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大乐透特征变体实验 v4.3
=======================
目标：诊断当前模型"分布偏斜"问题，找出能让大底池覆盖更稳定、
      接近/超过随机基准的特征配置。

结论方向：
- zone 特征给整个区间无差别加分 → 推荐锁死两端区间 → 分布偏斜
- 测试去掉/中性化 zone/sum/road 等区间级偏置后的效果
"""
import sys
import logging
import random
import math
from collections import Counter

sys.path.insert(0, '.')
logging.disable(logging.WARNING)

from src.lottery import LotteryAnalyzer, FRONT_NUMBERS, BACK_NUMBERS, FEATURE_WEIGHTS
from src.lottery import RANDOM_BASELINE


def evaluate(analyzer, variant_weights, trials=300, top5_n=5, pool_n=15):
    """walk-forward 评估：返回 Top5 ge2、大底池 ge4/ge3、分布熵"""
    saved = list(analyzer.history_data)
    saved_stats = dict(analyzer.statistics)

    ge2 = ge3 = ge4 = 0
    pool_ge4 = pool_ge3 = 0
    n = 0
    pick_counter = Counter()
    for i in range(trials):
        if i >= len(saved) - 11:
            break
        analyzer.history_data = list(saved[i + 1:])
        if len(analyzer.history_data) < 30:
            continue
        analyzer.update_statistics()
        actual_f = set(saved[i]['front'])
        n += 1

        # 用变体权重计算排名
        ranking = _rank_with_weights(analyzer, variant_weights, top_n=pool_n)
        top5 = ranking[:top5_n]
        pool = ranking[:pool_n]
        for num in pool:
            pick_counter[num] += 1
        hits5 = len(actual_f & set(top5))
        hits_pool = len(actual_f & set(pool))
        if hits5 >= 2:
            ge2 += 1
        if hits_pool >= 4:
            pool_ge4 += 1
        if hits_pool >= 3:
            pool_ge3 += 1

    analyzer.history_data = saved
    analyzer.statistics = saved_stats

    # 分布熵 (均匀=log2(35)≈5.13)
    total = sum(pick_counter.values())
    probs = [c / total for c in pick_counter.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    # 覆盖度: 出现过的号码数/35
    coverage = len(pick_counter) / 35.0

    return {
        'n': n,
        'top5_ge2': ge2 / n,
        'pool_ge4': pool_ge4 / n,
        'pool_ge3': pool_ge3 / n,
        'entropy': round(entropy, 3),
        'coverage': round(coverage, 3),
    }


def _rank_with_weights(analyzer, weights, top_n=15):
    """按自定义权重计算前区排名"""
    scores = []
    for num in FRONT_NUMBERS:
        fs = analyzer._calculate_feature_score(num, True)
        if not fs:
            continue
        total = sum(fs.get(k, 0) * w for k, w in weights.items() if k in fs)
        scores.append({'number': num, 'score': total})
    scores.sort(key=lambda x: -x['score'])
    return [r['number'] for r in scores[:top_n]]


def main():
    a = LotteryAnalyzer()
    print(f"历史期数: {len(a.history_data)}")

    variants = {
        # 当前 v4.2
        'A.当前v4.2': dict(FEATURE_WEIGHTS),
        # zone 权重砍半
        'B.zone砍半(0.10)': {**FEATURE_WEIGHTS, 'zone': 0.10},
        # zone 归零
        'C.zone归零': {**FEATURE_WEIGHTS, 'zone': 0.0},
        # sum 归零
        'D.sum归零': {**FEATURE_WEIGHTS, 'sum': 0.0},
        # zone+sum 都归零
        'E.zone+sum归零': {**FEATURE_WEIGHTS, 'zone': 0.0, 'sum': 0.0},
        # zone 中性化：全部号码 zone 得分一样（0.5），相当于仅去掉区间偏置
        'F.zone均匀0.5': {**FEATURE_WEIGHTS, 'zone': 0.0},
        # 邻号+频率+遗漏 精简模型
        'G.精简(邻号+频率+gap)': {'frequency': 0.10, 'gap': 0.10, 'adjacent': 0.20},
    }

    print("\n" + "=" * 90)
    print("特征变体对比 (walk-forward 300期)")
    print("=" * 90)
    print(f"{'变体':<24}{'Top5 ge2':>10}{'池ge4':>8}{'池ge3':>8}{'熵':>8}{'覆盖':>8}")
    print("-" * 90)

    # 随机基准
    print(f"{'随机基准':<24}{RANDOM_BASELINE['front_ge2_rate']*100:>8.1f}%"
          f"{RANDOM_BASELINE['front_pool_ge4_rate']*100:>7.1f}%"
          f"{'~30':>8}{'~5.13':>8}{'1.00':>8}")

    results = {}
    for name, w in variants.items():
        try:
            r = evaluate(a, w, trials=300)
            results[name] = r
            print(f"{name:<24}{r['top5_ge2']*100:>9.1f}%{r['pool_ge4']*100:>7.1f}%"
                  f"{r['pool_ge3']*100:>7.1f}%{r['entropy']:>8}{r['coverage']:>8}")
        except Exception as e:
            print(f"{name:<24} ERROR: {e}")

    print("\n注: 熵越高=分布越均匀(理论最大5.13); 覆盖=出现过的号码/35")
    print("    池ge4随机基准=9.33%, 池ge3基准≈30%")

    # 保存结果
    import json
    json.dump(results, open('data/diagnose_dlt_v43_variants.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
