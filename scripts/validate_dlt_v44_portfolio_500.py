#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大乐透 v4.4 完整管线 500期组合级验证
====================================
对比 v4.4(组合覆盖) 与 v4.3(互斥锚点) 的5注组合命中指标 + 分窗口稳定性。
随机基准: front_any_ge2=52.3%, front_any_ge3=6.8%, back_any_ge1=98.5%(覆盖10码)
"""
import sys
import json
import logging
from collections import Counter

sys.path.insert(0, '.')
logging.disable(logging.WARNING)

from src.lottery import LotteryAnalyzer

N = 500
WINDOW = 100


def simulate(analyzer, saved, start, end):
    any_ge2 = any_ge3 = back_any1 = joint = 0
    per_ge2 = 0
    union_sum = 0
    n = 0
    for i in range(start, end):
        if i >= len(saved) - 11:
            break
        analyzer.history_data = list(saved[i + 1:])
        if len(analyzer.history_data) < 80:
            continue
        analyzer.update_statistics()
        actual_f = set(saved[i]['front'])
        actual_b = set(saved[i]['back'])
        n += 1

        multi = analyzer.generate_multi_strategy_recommendations(
            voting_result=analyzer.multi_model_voting(front_n=20, back_n=10, skip_ml=True)
        )
        recs = [x for x in multi['recommendations']
                if not x['strategy'].startswith('picked')]
        union = set()
        f2 = f3 = b1 = jt = False
        for r in recs:
            f = set(r['front'])
            b = set(r['back'])
            union |= f
            hf = len(actual_f & f)
            hb = len(actual_b & b)
            if hf >= 2:
                per_ge2 += 1
                f2 = True
            if hf >= 3:
                f3 = True
            if hb >= 1:
                b1 = True
            if hf >= 2 and hb >= 1:
                jt = True
        if f2:
            any_ge2 += 1
        if f3:
            any_ge3 += 1
        if b1:
            back_any1 += 1
        if jt:
            joint += 1
        union_sum += len(union)

    if n == 0:
        return None
    return {
        'n': n,
        'front_any_ge2': any_ge2 / n,
        'front_any_ge3': any_ge3 / n,
        'back_any_ge1': back_any1 / n,
        'joint': joint / n,
        'per_ticket_ge2': per_ge2 / (n * 5),
        'front_union_avg': union_sum / n,
    }


def main():
    a = LotteryAnalyzer()
    saved = list(a.history_data)
    saved_stats = dict(a.statistics)
    print(f"历史期数: {len(saved)}", flush=True)

    full = simulate(a, saved, 0, N)
    print("\n" + "=" * 90)
    print("v4.4 完整管线 500期 walk-forward 组合指标")
    print("=" * 90)
    print(f"{'指标':<28}{'v4.4模型':>10}{'随机理论':>10}")
    rows = [
        ('front_any_ge2(任1注前≥2)', full['front_any_ge2'], 0.523),
        ('front_any_ge3(任1注前≥3)', full['front_any_ge3'], 0.068),
        ('back_any_ge1(任1注后≥1)', full['back_any_ge1'], 0.985),
        ('joint(同注前≥2后≥1)', full['joint'], 0.20),
        ('单注前区ge2率', full['per_ticket_ge2'], 0.1389),
        ('前区union(5注)', full['front_union_avg'], 25.0),
    ]
    for name, m, b in rows:
        diff = (m - b) * 100
        print(f"{name:<28}{m:>9.1%}  {b:>9.1%}  ({'+' if diff>=0 else ''}{diff:.1f}pp)")

    print("\n分窗口 front_any_ge2 / back_any_ge1（稳定性）")
    print(f"{'窗口':<8}{'front_any_ge2':>16}{'back_any_ge1':>16}{'joint':>8}")
    windows = []
    for wi in range(0, N, WINDOW):
        r = simulate(a, saved, wi, wi + WINDOW)
        if r:
            windows.append(r)
            print(f"窗口{wi//WINDOW+1:<6}{r['front_any_ge2']:>15.1%}{r['back_any_ge1']:>15.1%}{r['joint']:>7.1%}")

    a.history_data = saved
    a.statistics = saved_stats

    out = {'full': full,
           'windows': [{'front_any_ge2': r['front_any_ge2'],
                        'back_any_ge1': r['back_any_ge1'],
                        'joint': r['joint'], 'n': r['n']} for r in windows]}
    json.dump(out, open('data/validate_dlt_v44_portfolio_500.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print("\n已保存: data/validate_dlt_v44_portfolio_500.json")


if __name__ == '__main__':
    main()
