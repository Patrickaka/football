#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""大乐透精选一注 v7 回测诊断

对比：v7 组合优化 vs 各单一策略（主推/均衡/排名/热号/冷号）
指标：前区命中分布、后区命中分布、前区≥2/≥3 率、后区≥1/≥2 率
"""
import sys
import os
import logging
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.getLogger('lottery').setLevel(logging.WARNING)
logging.getLogger('lottery_ml').setLevel(logging.WARNING)

from src.lottery import LotteryAnalyzer


def main(trials=50):
    analyzer = LotteryAnalyzer()
    data = analyzer.history_data
    if len(data) < trials + 10:
        print(f"历史数据不足，需要至少 {trials + 10} 期")
        return

    saved_data = list(data)
    saved_stats = dict(analyzer.statistics) if analyzer.statistics else {}

    methods = ['primary_rank', 'balanced', 'rank', 'hot', 'cold', 'picked_v7', 'random']
    front_dist = {m: {i: 0 for i in range(6)} for m in methods}
    back_dist = {m: {i: 0 for i in range(3)} for m in methods}
    front_sum = {m: 0 for m in methods}
    back_sum = {m: 0 for m in methods}
    evaluated = 0

    for i in range(trials):
        analyzer.history_data = list(saved_data[i + 1:])
        if len(analyzer.history_data) < 30:
            continue
        analyzer.update_statistics()

        actual = saved_data[i]
        actual_front = set(actual['front'])
        actual_back = set(actual['back'])
        evaluated += 1

        # 真实随机基线：在同一历史切片上随机选号（公平对照）
        rand_front = set(random.sample(range(1, 36), 5))
        rand_back = set(random.sample(range(1, 13), 2))
        rfm = len(actual_front & rand_front)
        rbm = len(actual_back & rand_back)
        front_dist['random'][rfm] += 1
        back_dist['random'][rbm] += 1
        front_sum['random'] += rfm
        back_sum['random'] += rbm

        result = analyzer.generate_multi_strategy_recommendations()
        rec_map = {r.get('strategy', r.get('method')): r for r in result['recommendations']}

        for m in methods:
            rec = rec_map.get(m)
            if not rec:
                continue
            fm = len(actual_front & set(rec['front']))
            bm = len(actual_back & set(rec['back']))
            front_dist[m][fm] += 1
            back_dist[m][bm] += 1
            front_sum[m] += fm
            back_sum[m] += bm

    analyzer.history_data = saved_data
    analyzer.statistics = saved_stats

    n = evaluated or 1
    print(f"\n{'='*80}")
    print(f"大乐透精选一注 v7 回测（最近 {n} 期）")
    print(f"{'='*80}")
    print(f"\n{'方法':<14} | 前区≥2 | 前区≥3 | 后区≥1 | 后区≥2 | 前均 | 后均")
    print(f"{'-'*80}")

    for m in methods:
        fd = front_dist[m]
        bd = back_dist[m]
        fge2 = sum(fd[k] for k in range(2, 6)) / n
        fge3 = sum(fd[k] for k in range(3, 6)) / n
        bge1 = sum(bd[k] for k in range(1, 3)) / n
        bge2 = bd.get(2, 0) / n
        label = {
            'primary_rank': '主推',
            'balanced': '均衡',
            'rank': '排名',
            'hot': '热号',
            'cold': '冷号',
            'picked_v7': '精选一注v7',
            'random': '真实随机',
        }.get(m, m)
        print(f"{label:<14} | {fge2*100:5.1f}% | {fge3*100:5.1f}% | {bge1*100:5.1f}% | {bge2*100:5.1f}% | {front_sum[m]/n:.2f} | {back_sum[m]/n:.2f}")

    print(f"\n注：'真实随机'行为同一历史切片上的随机选号对照；")
    print(f"     其理论基线为 前区≥2≈13.9% / 前区≥3≈1.4% / 后区≥1≈31.8% / 后区=2≈1.5%（超几何分布），")
    print(f"     原脚本写死的 后区≥1=45.5% / 后区=2=4.5% 为错误数值，已废弃。")

    print(f"\n{'='*80}")
    print("前区命中分布")
    print(f"{'='*80}")
    header = "方法           | " + " | ".join(f"中{i}个" for i in range(6))
    print(header)
    print(f"{'-'*80}")
    for m in methods:
        label = {
            'primary_rank': '主推',
            'balanced': '均衡',
            'rank': '排名',
            'hot': '热号',
            'cold': '冷号',
            'picked_v7': '精选一注v7',
            'random': '真实随机',
        }.get(m, m)
        vals = " | ".join(f"{front_dist[m][i]:4d}" for i in range(6))
        print(f"{label:<14} | {vals}")

    print(f"\n{'='*80}")
    print("后区命中分布")
    print(f"{'='*80}")
    header = "方法           | " + " | ".join(f"中{i}个" for i in range(3))
    print(header)
    print(f"{'-'*80}")
    for m in methods:
        label = {
            'primary_rank': '主推',
            'balanced': '均衡',
            'rank': '排名',
            'hot': '热号',
            'cold': '冷号',
            'picked_v7': '精选一注v7',
            'random': '真实随机',
        }.get(m, m)
        vals = " | ".join(f"{back_dist[m][i]:4d}" for i in range(3))
        print(f"{label:<14} | {vals}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=50)
    args = parser.parse_args()
    main(args.trials)
