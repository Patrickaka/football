#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3D v4.9 动态形态 + 组三推荐诊断 — 500期 walk-forward 诚实回测
============================================================
1. 动态形态判断（max blend）vs 永远押组六：准确率、组三被选中频率
2. 组三四对子条件命中率：滚动训练取 Top4 对子 vs 随机基准 4/45=8.9%
3. 组六四码条件命中率（对照，确认未破坏）
4. 双形态联合覆盖：每期同时持有组六四码+组三四对子 → 无条件总命中率
"""
import sys
import json
import logging

sys.path.insert(0, '.')
logging.disable(logging.WARNING)

from src.lottery3d import (fetch_data, classify_form, analyze_form_probability,
                           zu3_digit_presence, zu3_pair_scores, zu3_combos_from_pair,
                           pick_zu3_pairs, pick_zu6_four, zu6_digit_scores,
                           zu6_notes_from_digits)


def main():
    data = fetch_data()
    nums = [x[2] for x in data]
    forms = [classify_form(n) for n in nums]

    N = 500
    start = len(nums) - N

    # ---- 1. 动态形态判断 ----
    dyn_hit = always_hit = zu3_pick = 0
    n = 0
    for i in range(start, len(nums)):
        train = nums[:i]
        if len(train) < 300:
            continue
        fp = analyze_form_probability(train)
        pick = max(fp['blend_p'], key=fp['blend_p'].get)
        actual = forms[i]
        n += 1
        if pick == actual:
            dyn_hit += 1
        if actual == 'zu6':
            always_hit += 1
        if pick == 'zu3':
            zu3_pick += 1

    # ---- 2. 组三四对子条件命中（仅对组三开奖期评估） ----
    zu3_hit = zu3_total = 0
    zu6_hit = zu6_total = 0
    joint_hit = joint_total = 0
    per_period = 0  # 每期无条件：任一组六四码 或 组三四对子 命中
    per_period_total = 0
    rng_hit_est = 0.0  # 随机基准累计

    for i in range(start, len(nums)):
        train = nums[:i]
        if len(train) < 300:
            continue
        actual = nums[i]
        actual_form = forms[i]

        # 组三对子（仅组三期计条件命中）
        z3 = pick_zu3_pairs(train)
        z3_pairs = [tuple(p['digits']) for p in z3['pairs']]
        if actual_form == 'zu3':
            drawn_pair = tuple(sorted(set(actual)))
            zu3_total += 1
            if drawn_pair in z3_pairs:
                zu3_hit += 1
            rng_hit_est += 4 / 45.0

        # 组六四码（仅组六期计条件命中）
        z6_score = zu6_digit_scores(train)
        z6_four = pick_zu6_four(z6_score, numbers=train)
        _, z6_combos = zu6_notes_from_digits(z6_four)
        if actual_form == 'zu6':
            z6s = ''.join(map(str, sorted(actual)))
            zu6_total += 1
            if z6s in z6_combos:
                zu6_hit += 1

        # 双形态联合：每期至少持有一个形态的推荐
        per_period_total += 1
        hit = False
        if actual_form == 'zu6' and z6s in z6_combos:
            hit = True
        if actual_form == 'zu3' and drawn_pair in z3_pairs:
            hit = True
        if hit:
            per_period += 1

    print('=' * 70)
    print(f'3D v4.9 动态形态+组三推荐 回测（{n}期 walk-forward）')
    print('=' * 70)
    print(f'[1] 动态形态判断（max blend）')
    print(f'    动态准确率: {dyn_hit/n:.1%}   |   永远押组六: {always_hit/n:.1%}')
    print(f'    组三被选中频率: {zu3_pick/n:.1%}  （<3% 即证明形态不可短期预测）')
    print()
    print(f'[2] 组三 4 对子条件命中率（仅组三期）')
    print(f'    实测: {zu3_hit}/{zu3_total} = {zu3_hit/zu3_total:.1%}   |   随机基准 4/45 = 8.9%')
    print()
    print(f'[3] 组六四码条件命中率（对照，仅组六期）')
    print(f'    实测: {zu6_hit}/{zu6_total} = {zu6_hit/zu6_total:.1%}   |   理论 C(4,3)/C(10,3) = 3.3%')
    print()
    print(f'[4] 双形态联合覆盖（每期持有组六四码+组三四对子）')
    print(f'    无条件命中: {per_period}/{per_period_total} = {per_period/per_period_total:.1%}')
    print(f'    理论值 = 72%*3.33% + 27%*8.9% ≈ 4.8%')

    out = {
        'n': n,
        'form': {
            'dynamic_accuracy': round(dyn_hit / n, 4),
            'always_zu6_accuracy': round(always_hit / n, 4),
            'zu3_pick_rate': round(zu3_pick / n, 4),
        },
        'zu3_pairs': {
            'hit': zu3_hit,
            'total': zu3_total,
            'hit_rate': round(zu3_hit / zu3_total, 4) if zu3_total else 0.0,
            'random_baseline': round(4 / 45.0, 4),
        },
        'zu6_four': {
            'hit': zu6_hit,
            'total': zu6_total,
            'hit_rate': round(zu6_hit / zu6_total, 4) if zu6_total else 0.0,
            'theory_baseline': round(4 / 120.0, 4),
        },
        'joint': {
            'hit': per_period,
            'total': per_period_total,
            'hit_rate': round(per_period / per_period_total, 4),
            'theory': round(0.72 * 4 / 120 + 0.27 * 4 / 45, 4),
        },
    }
    with open('data/diagnose_3d_v49_form_zu3.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('结果已存 data/diagnose_3d_v49_form_zu3.json')


if __name__ == '__main__':
    main()
