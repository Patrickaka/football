#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双色球 v3.2 红球蛇形覆盖诊断 — 2000期 walk-forward 基准回测
======================================================
对比旧实现（每注重叠加权采样）与 v3.2（排名池前30码蛇形分5组，union=30）：
1. 红球联合 ge2/ge3/ge4（核心收益：重叠 82% -> 覆盖 96%）
2. 单注 ge2/ge4（确认加权信号无损）
3. 5注红球 union 大小（旧~21 vs 新30，鸽笼前提）
4. 覆盖注合法性通过率 / 主推注回退率（v3.2 修复后验证）
5. 蓝球联合命中（确认 v3.1 去重覆盖不被破坏）
"""
import sys
import json
import random
import logging

sys.path.insert(0, '.')
logging.disable(logging.WARNING)

from src.ssq import (load_history, RED_RANGE, BLUE_RANGE, RED_COUNT, _analyze,
                     _is_valid_red, _weighted_sample, _next_period, _predict_sets)


def predict_old(train, analysis, n=5, seed=None):
    """复刻 v3.0/v3.1 红球逻辑：每注独立加权合法采样（允许重叠）+ 蓝球去重覆盖。"""
    rng = random.Random(seed)
    prev_red = set(train[-1]['red']) if train else set()
    prev_blue = train[-1]['blue'] if train else None
    red_w = [analysis['red_freq'][x] + 1.5 * analysis['red_recent'][x] + 0.5 for x in RED_RANGE]
    sets = []
    tries = 0
    while len(sets) < n and tries < 5000:
        tries += 1
        red = sorted(_weighted_sample(RED_RANGE, red_w, RED_COUNT, rng))
        if not _is_valid_red(red):
            continue
        if set(red) == prev_red:
            continue
        sets.append({'red': red})
    while len(sets) < n:
        sets.append({'red': sorted(rng.sample(RED_RANGE, RED_COUNT))})
    # 蓝球：与 v3.1 相同的去重覆盖
    avail_blues = [b for b in BLUE_RANGE if b != prev_blue]
    if len(avail_blues) < n:
        avail_blues = list(BLUE_RANGE)
    blues_pool = rng.sample(avail_blues, min(n, len(avail_blues)))
    while len(blues_pool) < n:
        blues_pool.append(rng.choice(BLUE_RANGE))
    for s, b in zip(sets, blues_pool):
        s['blue'] = b
    return sets


def main():
    history = load_history()
    history = sorted(history, key=lambda x: str(x['period']))
    N = 2000
    start = len(history) - N

    agg = {
        'old': {'red_ge2': 0, 'red_ge3': 0, 'red_ge4': 0,
                'red_single_ge2': 0, 'red_single_ge4': 0,
                'blue_any': 0, 'union': 0.0, 'valid_cover': 0, 'n': 0},
        'new': {'red_ge2': 0, 'red_ge3': 0, 'red_ge4': 0,
                'red_single_ge2': 0, 'red_single_ge4': 0,
                'blue_any': 0, 'union': 0.0, 'valid_cover': 0, 'n': 0},
    }
    fallback_cnt = 0
    n = 0

    for i in range(start, len(history)):
        train = history[:i]
        if len(train) < 200:
            continue
        analysis = _analyze(train)
        actual = history[i]
        actual_red = set(actual['red'])
        actual_blue = actual['blue']
        n += 1

        seed = int(actual['period'])
        sets_old = predict_old(train, analysis, n=5, seed=seed)
        sets_new = _predict_sets(train, analysis, n=5, seed=seed)

        for tag, sets in (('old', sets_old), ('new', sets_new)):
            reds = [set(s['red']) for s in sets]
            a = agg[tag]
            a['n'] += 1
            a['red_ge2'] += any(len(r & actual_red) >= 2 for r in reds)
            a['red_ge3'] += any(len(r & actual_red) >= 3 for r in reds)
            a['red_ge4'] += any(len(r & actual_red) >= 4 for r in reds)
            a['red_single_ge2'] += sum(1 for r in reds if len(r & actual_red) >= 2)
            a['red_single_ge4'] += sum(1 for r in reds if len(r & actual_red) >= 4)
            a['blue_any'] += actual_blue in {s['blue'] for s in sets}
            a['union'] += len(set().union(*reds))
            a['valid_cover'] += sum(1 for s in sets[1:] if _is_valid_red(s['red']))
        # 主推注输出合法率（回退后应≈100%；若回退失败则为0）
        main = sorted(set(sets_new[0]['red']))
        if not _is_valid_red(main):
            fallback_cnt += 1

    def pct(tag, key, denom=None):
        return f"{agg[tag][key] / (denom or n):.1%}"

    print('=' * 72)
    print(f'双色球红球 v3.2 蛇形覆盖 对比回测（{n}期 walk-forward，真实 _predict_sets）')
    print('=' * 72)
    print(f"{'指标':<24}{'旧(每注重叠)':>15}{'v3.2(蛇形覆盖)':>15}")
    print(f"{'红球 任1注>=2码':<20}{pct('old','red_ge2'):>15}{pct('new','red_ge2'):>15}  (蒙特卡洛理论: 重叠81.3% / 覆盖95.9%)")
    print(f"{'红球 任1注>=3码':<20}{pct('old','red_ge3'):>15}{pct('new','red_ge3'):>15}")
    print(f"{'红球 任1注>=4码':<20}{pct('old','red_ge4'):>15}{pct('new','red_ge4'):>15}")
    print(f"{'红球 单注>=2码':<20}{agg['old']['red_single_ge2']/(n*5):>15.1%}{agg['new']['red_single_ge2']/(n*5):>15.1%}")
    print(f"{'红球 单注>=4码':<20}{agg['old']['red_single_ge4']/(n*5):>15.1%}{agg['new']['red_single_ge4']/(n*5):>15.1%}")
    print(f"{'5注红球平均union':<20}{agg['old']['union']/n:>15.1f}{agg['new']['union']/n:>15.1f}  (理论: 33码随机~20.8 / 覆盖30)")
    print(f"{'覆盖注(2-5注)合法率':<18}{agg['old']['valid_cover']/(n*4):>15.1%}{agg['new']['valid_cover']/(n*4):>15.1%}")
    print(f"{'5注蓝球联合命中':<20}{pct('old','blue_any'):>15}{pct('new','blue_any'):>15}  (去重理论31.25%)")
    print(f"主推注输出合法率: {1 - fallback_cnt/n:.1%}")

    out = {
        'n': n,
        'primary_invalid_rate': round(fallback_cnt / n, 4),
        'old': {k: (round(v / (n * 5) if k.startswith('red_single') else v / n, 4))
                for k, v in agg['old'].items() if k != 'n'},
        'new': {k: (round(v / (n * 5) if k.startswith('red_single') else v / n, 4))
                for k, v in agg['new'].items() if k != 'n'},
    }
    with open('data/diagnose_ssq_v32_red.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('结果已存 data/diagnose_ssq_v32_red.json')


if __name__ == '__main__':
    main()
