#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""大乐透 v3.3 特征消融实验 — 逐个关闭特征看命中率变化
目标: 找出哪些特征对≥2命中率有真正贡献, 哪些是噪声
"""

import sys, os, json, logging, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lottery import LotteryAnalyzer, FEATURE_WEIGHTS, BACK_FEATURE_WEIGHTS
from collections import defaultdict

logging.getLogger('lottery').setLevel(logging.WARNING)
logging.getLogger('lottery_ml').setLevel(logging.WARNING)

def run_ablation_backtest(analyzer, custom_weights, trials=50, is_front=True):
    """使用自定义权重的排名回测"""
    saved_data = list(analyzer.history_data)
    saved_stats = dict(analyzer.statistics) if analyzer.statistics else {}

    front_ge2 = front_ge3 = front_ge4 = 0
    back_ge1 = back_ge2 = 0
    evaluated = 0

    for i in range(trials):
        analyzer.history_data = list(saved_data[i + 1:])
        if len(analyzer.history_data) < 10:
            continue
        analyzer.update_statistics()

        actual = saved_data[i]
        evaluated += 1

        if is_front:
            # 前区排名 with custom weights
            front_scores = []
            from src.lottery import FRONT_NUMBERS
            for num in FRONT_NUMBERS:
                features = analyzer._calculate_feature_score(num, is_front=True)
                total = sum(features.get(k, 0) * custom_weights.get(k, 0) for k in custom_weights if k in features)
                front_scores.append((num, total))
            front_scores.sort(key=lambda x: -x[1])
            front_top5 = [num for num, _ in front_scores[:5]]

            front_common = len(set(actual['front']) & set(front_top5))
            if front_common >= 2: front_ge2 += 1
            if front_common >= 3: front_ge3 += 1
            if front_common >= 4: front_ge4 += 1
        else:
            # 后区排名 with custom weights
            back_scores = []
            from src.lottery import BACK_NUMBERS
            for num in BACK_NUMBERS:
                features = analyzer._calculate_back_feature_score(num)
                total = sum(features.get(k, 0) * custom_weights.get(k, 0) for k in custom_weights if k in features)
                back_scores.append((num, total))
            back_scores.sort(key=lambda x: -x[1])
            back_top3 = [num for num, _ in back_scores[:3]]

            back_common = len(set(actual['back']) & set(back_top3))
            if back_common >= 1: back_ge1 += 1
            if back_common >= 2: back_ge2 += 1

    analyzer.history_data = saved_data
    analyzer.statistics = saved_stats

    n = evaluated or 1
    if is_front:
        return {
            'trials': n,
            'front_ge2_rate': front_ge2/n,
            'front_ge3_rate': front_ge3/n,
        }
    else:
        return {
            'trials': n,
            'back_ge1_rate': back_ge1/n,
            'back_ge2_rate': back_ge2/n,
        }

def main():
    analyzer = LotteryAnalyzer()
    max_trials = min(50, len(analyzer.history_data) - 15)
    if max_trials < 10:
        print("数据不足")
        return

    print(f"\n{'='*70}")
    print(f"大乐透 v3.3 前区特征消融实验")
    print(f"{'='*70}")
    print(f"回测期数: {max_trials}")
    print(f"随机基线: 前区≥2=27.8%, ≥3=6.7%")

    # 1. 基准(v3.3全特征)
    print(f"\n--- 基准: v3.3全部特征 ---")
    base_result = run_ablation_backtest(analyzer, FEATURE_WEIGHTS, max_trials, is_front=True)
    print(f"  ≥2命中: {base_result['front_ge2_rate']*100:.1f}%  ≥3命中: {base_result['front_ge3_rate']*100:.1f}%")

    # 2. 消融: 逐个关闭每个特征
    features_to_test = list(FEATURE_WEIGHTS.keys())
    print(f"\n--- 特征消融(逐个关闭) ---")
    print(f"  特征        | ≥2命中 | ≥3命中 | ≥2变化 | ≥3变化")
    print(f"  {'─'*50}")

    results = {}
    for feat in features_to_test:
        test_weights = dict(FEATURE_WEIGHTS)
        test_weights[feat] = 0  # 关闭该特征
        # 重新归一化权重
        total_w = sum(test_weights.values())
        if total_w > 0:
            for k in test_weights:
                test_weights[k] = test_weights[k] / total_w

        result = run_ablation_backtest(analyzer, test_weights, max_trials, is_front=True)
        change_ge2 = result['front_ge2_rate'] - base_result['front_ge2_rate']
        change_ge3 = result['front_ge3_rate'] - base_result['front_ge3_rate']
        results[feat] = {
            'ge2': result['front_ge2_rate'],
            'ge3': result['front_ge3_rate'],
            'change_ge2': change_ge2,
            'change_ge3': change_ge3,
        }
        print(f"  {feat:12s} | {result['front_ge2_rate']*100:.1f}%  | {result['front_ge3_rate']*100:.1f}%  | {change_ge2*100:+.1f}% | {change_ge3*100:+.1f}%")

    # 3. 消融: 后区
    print(f"\n{'='*70}")
    print(f"大乐透 v3.3 后区特征消融实验")
    print(f"{'='*70}")
    print(f"随机基线: 后区≥1=41.7%")

    base_back = run_ablation_backtest(analyzer, BACK_FEATURE_WEIGHTS, max_trials, is_front=False)
    print(f"\n--- 基准: v3.3全部后区特征 ---")
    print(f"  ≥1命中: {base_back['back_ge1_rate']*100:.1f}%  ≥2命中: {base_back['back_ge2_rate']*100:.1f}%")

    print(f"\n--- 后区特征消融(逐个关闭) ---")
    print(f"  特征        | ≥1命中 | ≥2命中 | ≥1变化")
    print(f"  {'─'*50}")

    back_features = list(BACK_FEATURE_WEIGHTS.keys())
    for feat in back_features:
        test_weights = dict(BACK_FEATURE_WEIGHTS)
        test_weights[feat] = 0
        total_w = sum(test_weights.values())
        if total_w > 0:
            for k in test_weights:
                test_weights[k] = test_weights[k] / total_w

        result = run_ablation_backtest(analyzer, test_weights, max_trials, is_front=False)
        change_ge1 = result['back_ge1_rate'] - base_back['back_ge1_rate']
        print(f"  {feat:12s} | {result['back_ge1_rate']*100:.1f}%  | {result['back_ge2_rate']*100:.1f}%  | {change_ge1*100:+.1f}%")

    # 4. 最优组合实验: 只保留正信号特征
    print(f"\n{'='*70}")
    print(f"最优组合实验: 只保留正信号特征")
    print(f"{'='*70}")

    # 前区正信号: repeat(0.507), zone(0.505), road(0.502), sum(0.503)
    positive_front = {
        'repeat': 0.40,   # 最强正信号 → 最高权重
        'zone': 0.30,     # 第二正信号
        'sum': 0.20,      # 正信号
        'road': 0.10,     # 微正信号
    }
    result = run_ablation_backtest(analyzer, positive_front, max_trials, is_front=True)
    print(f"  前区正信号only: ≥2={result['front_ge2_rate']*100:.1f}% ≥3={result['front_ge3_rate']*100:.1f}%")
    print(f"  vs 基准: ≥2变化={(result['front_ge2_rate']-base_result['front_ge2_rate'])*100:+.1f}% ≥3变化={(result['front_ge3_rate']-base_result['front_ge3_rate'])*100:+.1f}%")

    # 后区正信号: adjacent(0.524), road(0.517), position(0.505)
    positive_back = {
        'adjacent': 0.45,  # 最强正信号
        'road': 0.25,      # 正信号
        'position': 0.20,  # 微正信号
        'sum': 0.10,       # 保留
    }
    result = run_ablation_backtest(analyzer, positive_back, max_trials, is_front=False)
    print(f"  后区正信号only: ≥1={result['back_ge1_rate']*100:.1f}% ≥2={result['back_ge2_rate']*100:.1f}%")
    print(f"  vs 基准: ≥1变化={(result['back_ge1_rate']-base_back['back_ge1_rate'])*100:+.1f}%")

if __name__ == '__main__':
    main()
