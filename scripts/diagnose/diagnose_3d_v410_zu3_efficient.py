#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3D v4.10 组三高效覆盖诊断 — 500期 walk-forward 诚实回测
========================================================
1. 组三 K 对子（组选三口径）条件命中率：K=4/8/12/20 → 应 ≈ K/45（线性，纯数学覆盖）
2. 同预算对比：48元 → v4.9 直选 4 对子 vs v4.10 组选三 12 对子（条件命中 3 倍）
3. 联合档位：组六四码 + 组三 K 对子 → 无条件命中率 ≈ 理论（组六 72%×K6/120 + 组三 27%×K3/45）
"""
import sys
import json
import logging

sys.path.insert(0, '.')
logging.disable(logging.WARNING)

from src.lottery3d import (fetch_data, classify_form, analyze_form_probability,
                           pick_zu3_pairs, pick_zu6_four, zu6_digit_scores,
                           zu6_notes_from_digits)

TIERS = (4, 8, 12, 20)


def main():
    data = fetch_data()
    nums = [x[2] for x in data]
    forms = [classify_form(n) for n in nums]

    N = 500
    start = len(nums) - N

    # 各档位命中累计
    tier_hit = {k: 0 for k in TIERS}
    tier_total = 0
    # 联合（组六四码 + 组三 K 对子）
    joint_hit = {k: 0 for k in TIERS}
    joint_total = 0
    # 同预算对比（48元）：直选 4 对子 vs 组选三 12 对子
    v49_hit = v410_hit = 0
    zu3_periods = 0

    for i in range(start, len(nums)):
        train = nums[:i]
        if len(train) < 300:
            continue
        actual = nums[i]
        actual_form = forms[i]

        # 组三各档位（仅组三期计条件命中）
        base = pick_zu3_pairs(train, limit=max(TIERS))
        all_pairs = [tuple(p['digits']) for p in base['pairs']]
        if actual_form == 'zu3':
            drawn_pair = tuple(sorted(set(actual)))
            tier_total += 1
            zu3_periods += 1
            for k in TIERS:
                if drawn_pair in all_pairs[:k]:
                    tier_hit[k] += 1
            # 同预算对比
            if drawn_pair in all_pairs[:4]:
                v49_hit += 1
            if drawn_pair in all_pairs[:12]:
                v410_hit += 1

        # 联合：组六四码（8元）+ 组三 K 对子（4K 元）
        z6_score = zu6_digit_scores(train)
        z6_four = pick_zu6_four(z6_score, numbers=train)
        _, z6_combos = zu6_notes_from_digits(z6_four)
        joint_total += 1
        z6s = ''.join(map(str, sorted(actual)))
        for k in TIERS:
            if actual_form == 'zu6' and z6s in z6_combos:
                joint_hit[k] += 1
            elif actual_form == 'zu3' and drawn_pair in all_pairs[:k]:
                joint_hit[k] += 1

    print('=' * 70)
    print(f'3D v4.10 组三高效覆盖 回测（{joint_total}期 walk-forward，组三 {tier_total} 期）')
    print('=' * 70)
    print('[1] 组三 K 对子条件命中（组选三口径，与直选口径命中概率相同）')
    for k in TIERS:
        print(f'    K={k:<2} 实测 {tier_hit[k]}/{tier_total} = {tier_hit[k]/tier_total:.1%}'
              f'   |   理论上界 K/45 = {k/45:.1%}   |   成本 4K={k*4}元')
    print()
    print('[2] 同预算对比（48元组三预算）')
    print(f'    v4.9 直选 4 对子（24注48元）: {v49_hit}/{zu3_periods} = {v49_hit/zu3_periods:.1%}')
    print(f'    v4.10 组选三 12 对子（24注48元）: {v410_hit}/{zu3_periods} = {v410_hit/zu3_periods:.1%}'
          f'   (条件命中率理论 3 倍)')
    print()
    print('[3] 联合档位（组六四码 8元 + 组三 K 对子 4K元）无条件命中')
    for k in TIERS:
        theory = round(0.72 * 4 / 120 + 0.27 * k / 45, 4)
        print(f'    K={k:<2} 总成本 {8+k*4}元  实测 {joint_hit[k]}/{joint_total} = {joint_hit[k]/joint_total:.1%}'
              f'   |   理论 {theory:.1%}')

    out = {
        'n': joint_total,
        'zu3_periods': tier_total,
        'tiers': {
            str(k): {
                'hit': tier_hit[k],
                'total': tier_total,
                'hit_rate': round(tier_hit[k] / tier_total, 4),
                'theory': round(k / 45.0, 4),
                'cost': k * 4,
            } for k in TIERS
        },
        'same_budget_48': {
            'v49_direct_4pairs': {'hit': v49_hit, 'total': zu3_periods,
                                  'hit_rate': round(v49_hit / zu3_periods, 4)},
            'v410_zunotes_12pairs': {'hit': v410_hit, 'total': zu3_periods,
                                     'hit_rate': round(v410_hit / zu3_periods, 4)},
        },
        'joint': {
            str(k): {
                'hit': joint_hit[k],
                'total': joint_total,
                'hit_rate': round(joint_hit[k] / joint_total, 4),
                'theory': round(0.72 * 4 / 120 + 0.27 * k / 45, 4),
                'cost': 8 + k * 4,
            } for k in TIERS
        },
    }
    with open('data/diagnose_3d_v410_zu3_efficient.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('结果已存 data/diagnose_3d_v410_zu3_efficient.json')


if __name__ == '__main__':
    main()
