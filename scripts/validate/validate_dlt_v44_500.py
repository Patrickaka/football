#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大乐透 v4.4 候选权重 500期稳定性验证
====================================
候选（来自 diagnose_dlt_v44 单变量扫描, 方向一致的正向微调）:
- v4.4a: frequency 0.05→0.08 (Top5 ge2 14.4→15.4%), trend 0.03→0.0 (池ge4 12.0→12.2%)
- v4.4b: 在 a 基础上 sum 0.13→0.12 (avg 0.730)
- v4.4c: 在 a 基础上 road 0.12→0.14

判定标准：与 v4.3 相比, 全量500期 Top5 ge2 / 池ge4 不劣, 且至少3/5窗口同向。
"""
import sys
import math
import json
import logging
from collections import Counter

sys.path.insert(0, '.')
logging.disable(logging.WARNING)

from src.lottery import LotteryAnalyzer, FRONT_NUMBERS, FEATURE_WEIGHTS, RANDOM_BASELINE


def rank_with_weights(analyzer, weights, top_n=15):
    scores = []
    for num in FRONT_NUMBERS:
        fs = analyzer._calculate_feature_score(num, True)
        if not fs:
            continue
        total = sum(fs.get(k, 0) * w for k, w in weights.items() if k in fs)
        scores.append((num, total))
    scores.sort(key=lambda x: -x[1])
    return [n for n, _ in scores[:top_n]]


def evaluate_window(analyzer, weights, saved, start, end, top5_n=5, pool_n=15):
    ge2 = ge3 = pool_ge4 = pool_ge3 = 0
    n = 0
    for i in range(start, end):
        if i >= len(saved) - 11:
            break
        analyzer.history_data = list(saved[i + 1:])
        if len(analyzer.history_data) < 80:
            continue
        analyzer.update_statistics()
        actual_f = set(saved[i]['front'])
        n += 1
        ranking = rank_with_weights(analyzer, weights, top_n=pool_n)
        hits5 = len(actual_f & set(ranking[:top5_n]))
        hits_pool = len(actual_f & set(ranking))
        if hits5 >= 2:
            ge2 += 1
        if hits5 >= 3:
            ge3 += 1
        if hits_pool >= 4:
            pool_ge4 += 1
        if hits_pool >= 3:
            pool_ge3 += 1
    if n == 0:
        return None
    return {'n': n, 'top5_ge2': ge2 / n, 'top5_ge3': ge3 / n,
            'pool_ge4': pool_ge4 / n, 'pool_ge3': pool_ge3 / n}


def main():
    a = LotteryAnalyzer()
    saved = list(a.history_data)
    saved_stats = dict(a.statistics)
    print(f"历史期数: {len(saved)}", flush=True)

    base = dict(FEATURE_WEIGHTS)
    variants = {
        'v4.3(当前)': base,
        'v4.4a(freq.08+trend0)': {**base, 'frequency': 0.08, 'trend': 0.0},
        'v4.4b(a+sum.12)': {**base, 'frequency': 0.08, 'trend': 0.0, 'sum': 0.12},
        'v4.4c(a+road.14)': {**base, 'frequency': 0.08, 'trend': 0.0, 'road': 0.14},
    }

    N = 500
    WINDOW = 100
    print(f"\n{'变体':<20}{'Top5 ge2':>10}{'Top5 ge3':>10}{'池ge4':>8}{'池ge3':>8}")
    print(f"{'随机基准':<20}{RANDOM_BASELINE['front_ge2_rate']*100:>9.1f}%"
          f"{RANDOM_BASELINE['front_ge3_rate']*100:>9.2f}%"
          f"{RANDOM_BASELINE['front_pool_ge4_rate']*100:>7.1f}%{'~30':>8}")
    print("-" * 60)

    full = {}
    for name, w in variants.items():
        r = evaluate_window(a, w, saved, 0, N)
        full[name] = r
        print(f"{name:<20}{r['top5_ge2']*100:>9.1f}%{r['top5_ge3']*100:>9.2f}%"
              f"{r['pool_ge4']*100:>7.1f}%{r['pool_ge3']*100:>7.1f}%", flush=True)

    print("\n分窗口 Top5 ge2 / 池ge4（判定稳定性）")
    for name in variants:
        print(f"\n--- {name} ---")
        print(f"{'窗口':<8}{'Top5 ge2':>10}{'池ge4':>8}")
        for wi in range(0, N, WINDOW):
            r = evaluate_window(a, variants[name], saved, wi, wi + WINDOW)
            if r:
                print(f"窗口{wi//WINDOW+1:<6}{r['top5_ge2']*100:>9.1f}%{r['pool_ge4']*100:>7.1f}%")

    # 判定: v4.4 相对 v4.3
    v43 = full['v4.3(当前)']
    print("\n" + "=" * 60)
    print("相对 v4.3 判定")
    for name in variants:
        if name == 'v4.3(当前)':
            continue
        r = full[name]
        print(f"{name}: Top5 ge2 {r['top5_ge2']-v43['top5_ge2']:+.1%}, "
              f"池ge4 {r['pool_ge4']-v43['pool_ge4']:+.1%}, "
              f"池ge3 {r['pool_ge3']-v43['pool_ge3']:+.1%}")

    a.history_data = saved
    a.statistics = saved_stats
    json.dump({k: {kk: round(vv, 6) for kk, vv in v.items()}
               for k, v in full.items()},
              open('data/validate_dlt_v44.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print("已保存: data/validate_dlt_v44.json")


if __name__ == '__main__':
    main()
