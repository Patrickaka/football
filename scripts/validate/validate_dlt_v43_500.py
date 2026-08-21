#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大乐透 v4.3 权重优化 — 500期大样本稳定性验证
=============================================
目标：
1. 用最近500期 walk-forward 对比 v4.3(当前) vs v4.2(旧) 的核心指标
2. 将500期切成5个100期窗口，检验各窗口表现是否稳定（防小样本噪声）
3. 指标：Top5 ge2、大底池15码 ge4/ge3、分布熵、号码覆盖度

结论判定：
- v4.3 池ge4 若稳定 ≥ v4.2 且 > 随机基准9.33%，则优化有效
- 各窗口波动 < ±4% 视为稳定
"""
import sys
import math
import logging
from collections import Counter

sys.path.insert(0, '.')
logging.disable(logging.WARNING)

from src.lottery import LotteryAnalyzer, FRONT_NUMBERS, FEATURE_WEIGHTS, RANDOM_BASELINE

# v4.2 权重（v4.3 变更: gap 0.08→0.12, sum 0.15→0.13, zone 0.20→0.18）
V42_WEIGHTS = {**FEATURE_WEIGHTS, 'gap': 0.08, 'sum': 0.15, 'zone': 0.20}


def _rank_with_weights(analyzer, weights, top_n=15):
    """按自定义权重计算前区排名（与 diagnose_dlt_v43_variants 一致）"""
    scores = []
    for num in FRONT_NUMBERS:
        fs = analyzer._calculate_feature_score(num, True)
        if not fs:
            continue
        total = sum(fs.get(k, 0) * w for k, w in weights.items() if k in fs)
        scores.append({'number': num, 'score': total})
    scores.sort(key=lambda x: -x['score'])
    return [r['number'] for r in scores[:top_n]]


def evaluate_window(analyzer, weights, saved, start, end, top5_n=5, pool_n=15):
    """walk-forward 评估 [start, end) 区间（索引从0=最新开始）"""
    ge2 = ge3 = ge4 = 0
    pool_ge4 = pool_ge3 = 0
    n = 0
    pick_counter = Counter()
    for i in range(start, end):
        if i >= len(saved) - 11:
            break
        analyzer.history_data = list(saved[i + 1:])
        if len(analyzer.history_data) < 30:
            continue
        analyzer.update_statistics()
        actual_f = set(saved[i]['front'])
        n += 1

        ranking = _rank_with_weights(analyzer, weights, top_n=pool_n)
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

    if n == 0:
        return None

    total = sum(pick_counter.values())
    probs = [c / total for c in pick_counter.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    coverage = len(pick_counter) / 35.0

    return {
        'n': n,
        'top5_ge2': ge2 / n,
        'pool_ge4': pool_ge4 / n,
        'pool_ge3': pool_ge3 / n,
        'entropy': round(entropy, 3),
        'coverage': round(coverage, 3),
    }


def main():
    a = LotteryAnalyzer()
    saved = list(a.history_data)
    saved_stats = dict(a.statistics)
    print(f"历史期数: {len(saved)}  最新: {saved[0]['issue']}  最旧: {saved[-1]['issue']}")
    print(f"当前版本权重: {FEATURE_WEIGHTS}")
    print(f"v4.2对比权重: gap=0.08 sum=0.15 zone=0.20")

    N = 500
    WINDOW = 100

    variants = {
        'v4.2(旧)': V42_WEIGHTS,
        'v4.3(当前)': dict(FEATURE_WEIGHTS),
    }

    print("\n" + "=" * 100)
    print(f"全量500期 walk-forward 对比")
    print("=" * 100)
    header = f"{'变体':<12}{'Top5 ge2':>10}{'池ge4':>8}{'池ge3':>8}{'熵':>8}{'覆盖':>8}"
    print(header)
    print("-" * 100)
    print(f"{'随机基准':<12}{RANDOM_BASELINE['front_ge2_rate']*100:>9.1f}%"
          f"{RANDOM_BASELINE['front_pool_ge4_rate']*100:>7.1f}%"
          f"{'~30':>8}{'~5.13':>8}{'1.00':>8}")

    full_results = {}
    for name, w in variants.items():
        r = evaluate_window(a, w, saved, 0, N)
        full_results[name] = r
        print(f"{name:<12}{r['top5_ge2']*100:>9.1f}%{r['pool_ge4']*100:>7.1f}%"
              f"{r['pool_ge3']*100:>7.1f}%{r['entropy']:>8}{r['coverage']:>8}")

    print("\n" + "=" * 100)
    print(f"分窗口稳定性检验（每窗口 {WINDOW} 期，从最新往前数）")
    print("=" * 100)
    for name, w in variants.items():
        print(f"\n--- {name} ---")
        print(f"{'窗口':<12}{'期数范围':<22}{'Top5 ge2':>10}{'池ge4':>8}{'池ge3':>8}")
        for wi in range(0, N, WINDOW):
            r = evaluate_window(a, w, saved, wi, wi + WINDOW)
            if not r:
                continue
            start_issue = saved[wi]['issue']
            end_issue = saved[wi + r['n'] - 1]['issue']
            print(f"窗口{wi//WINDOW+1:<8}{end_issue+'~'+start_issue:<22}"
                  f"{r['top5_ge2']*100:>9.1f}%{r['pool_ge4']*100:>7.1f}%"
                  f"{r['pool_ge3']*100:>7.1f}%")

    # 恢复
    a.history_data = saved
    a.statistics = saved_stats

    # 判定
    v43 = full_results['v4.3(当前)']
    v42 = full_results['v4.2(旧)']
    print("\n" + "=" * 100)
    print("结论判定")
    print("=" * 100)
    print(f"v4.3 池ge4 = {v43['pool_ge4']*100:.1f}%  vs  v4.2 = {v42['pool_ge4']*100:.1f}%  vs  随机 = {RANDOM_BASELINE['front_pool_ge4_rate']*100:.1f}%")
    print(f"v4.3 Top5 ge2 = {v43['top5_ge2']*100:.1f}%  vs  v4.2 = {v42['top5_ge2']*100:.1f}%  vs  随机 = {RANDOM_BASELINE['front_ge2_rate']*100:.1f}%")
    print(f"v4.3 分布熵 = {v43['entropy']}  vs  v4.2 = {v42['entropy']}  (越高越均匀, 最大5.13)")
    print(f"v4.3 覆盖度 = {v43['coverage']}  vs  v4.2 = {v42['coverage']}  (越大越不锁死)")

    import json
    json.dump(full_results, open('data/dlt_v43_500_validation.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print("\n结果已保存: data/dlt_v43_500_validation.json")


if __name__ == '__main__':
    main()
