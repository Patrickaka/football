#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大乐透 v4.4 组合级诊断 — 5注推荐的联合命中率
============================================
用户感知的"准确率" = 5注里有没有一注中奖。模拟当前完整推荐管线
(generate_multi_strategy_recommendations, skip_ml) 的 walk-forward 表现。

指标（对比随机5注理论值）：
- front_any_ge2: 5注中任意一注前区≥2码  随机: 1-(1-0.1389)^5 = 52.3%
- front_any_ge3: 5注中任意一注前区≥3码  随机: 1-(1-0.0139)^5 = 6.8%
- back_any_ge1 : 5注后区至少一注中1码   随机(覆盖10码): 1-C(2,2)/66 = 98.5%
- 联合: 同一注 前区≥2 且 后区≥1         随机(独立): 1-(1-0.1389*0.318)^5 ≈ 20%
- 前区 union 号码数 / 平均每注前区命中
"""
import sys
import math
import json
import logging
from collections import Counter

sys.path.insert(0, '.')
logging.disable(logging.WARNING)

from src.lottery import LotteryAnalyzer, FRONT_NUMBERS

N = 200  # 组合模拟较慢，取200期


def simulate(analyzer, saved, trials=N):
    a = analyzer
    stats = {
        'front_any_ge2': 0, 'front_any_ge3': 0,
        'back_any_ge1': 0, 'back_any_ge2': 0,
        'joint_same_ticket': 0,           # 同一注 前ge2+后ge1
        'front_union': 0.0, 'front_avg': 0.0,
        'back_avg': 0.0,
        'per_ticket_front_ge2': 0, 'per_ticket_front_ge3': 0,
        'per_ticket_back_ge1': 0,
        'n': 0,
    }
    for i in range(trials):
        if i >= len(saved) - 11:
            break
        a.history_data = list(saved[i + 1:])
        if len(a.history_data) < 80:
            continue
        a.update_statistics()
        actual_f = set(saved[i]['front'])
        actual_b = set(saved[i]['back'])
        stats['n'] += 1

        multi = a.generate_multi_strategy_recommendations(
            voting_result=a.multi_model_voting(front_n=20, back_n=10, skip_ml=True)
        )
        recs = [r for r in multi['recommendations']
                if not r['strategy'].startswith('picked')]
        front_union = set()
        any_ge2 = any_ge3 = False
        back_any1 = back_any2 = False
        joint = False
        for r in recs:
            f = set(r['front'])
            b = set(r['back'])
            front_union |= f
            hf = len(actual_f & f)
            hb = len(actual_b & b)
            stats['front_avg'] += hf
            stats['back_avg'] += hb
            if hf >= 2:
                stats['per_ticket_front_ge2'] += 1
                any_ge2 = True
            if hf >= 3:
                stats['per_ticket_front_ge3'] += 1
                any_ge3 = True
            if hb >= 1:
                stats['per_ticket_back_ge1'] += 1
                back_any1 = True
            if hb >= 2:
                back_any2 = True
            if hf >= 2 and hb >= 1:
                joint = True
        if any_ge2:
            stats['front_any_ge2'] += 1
        if any_ge3:
            stats['front_any_ge3'] += 1
        if back_any1:
            stats['back_any_ge1'] += 1
        if back_any2:
            stats['back_any_ge2'] += 1
        if joint:
            stats['joint_same_ticket'] += 1
        stats['front_union'] += len(front_union)

    n = stats['n']
    res = {
        'n': n,
        'front_any_ge2': stats['front_any_ge2'] / n,
        'front_any_ge3': stats['front_any_ge3'] / n,
        'back_any_ge1': stats['back_any_ge1'] / n,
        'back_any_ge2': stats['back_any_ge2'] / n,
        'joint_same_ticket': stats['joint_same_ticket'] / n,
        'front_union_avg': stats['front_union'] / n,
        'front_avg_per_ticket': stats['front_avg'] / (n * 5),
        'back_avg_per_ticket': stats['back_avg'] / (n * 5),
        'per_ticket_front_ge2_rate': stats['per_ticket_front_ge2'] / (n * 5),
        'per_ticket_front_ge3_rate': stats['per_ticket_front_ge3'] / (n * 5),
        'per_ticket_back_ge1_rate': stats['per_ticket_back_ge1'] / (n * 5),
    }
    return res


def main():
    a = LotteryAnalyzer()
    saved = list(a.history_data)
    print(f"历史期数: {len(saved)}  组合模拟期数: {min(N, len(saved)-11)}", flush=True)

    r = simulate(a, saved)
    print("\n" + "=" * 90)
    print("当前 v4.3 完整推荐管线（5注）组合命中（200期 walk-forward）")
    print("=" * 90)
    print(f"{'指标':<26}{'模型':>10}{'随机理论':>10}")
    rows = [
        ('front_any_ge2(任1注前区≥2)', r['front_any_ge2'], 0.523),
        ('front_any_ge3(任1注前区≥3)', r['front_any_ge3'], 0.068),
        ('back_any_ge1(任1注后区≥1)', r['back_any_ge1'], 0.985),
        ('back_any_ge2(任1注后区≥2)', r['back_any_ge2'], 0.073),
        ('joint(同注前≥2且后≥1)', r['joint_same_ticket'], 0.20),
        ('前区union号码数(5注)', r['front_union_avg'], None),
        ('单注前区平均命中', r['front_avg_per_ticket'], 5*5/35),
        ('单注后区平均命中', r['back_avg_per_ticket'], 2*2/12),
        ('单注前区ge2率', r['per_ticket_front_ge2_rate'], 0.1389),
        ('单注前区ge3率', r['per_ticket_front_ge3_rate'], 0.0139),
        ('单注后区ge1率', r['per_ticket_back_ge1_rate'], 0.318),
    ]
    for name, m, b in rows:
        if b is None:
            print(f"{name:<26}{m:>10.1f}")
        else:
            diff = (m - b) * 100
            sign = '+' if diff >= 0 else ''
            print(f"{name:<26}{m:>9.1%}  {b:>9.1%}  ({sign}{diff:.1f}pp)")

    json.dump(r, open('data/diagnose_dlt_v44_portfolio.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print("\n已保存: data/diagnose_dlt_v44_portfolio.json")


if __name__ == '__main__':
    main()
