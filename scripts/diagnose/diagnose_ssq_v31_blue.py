#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双色球 v3.1 蓝球诊断 — 500期 walk-forward 基准回测
==================================================
量化当前蓝球策略的真实表现，为优化提供依据：
1. 5注联合蓝球命中率（独立随机理论 1-(15/16)^5≈27.6%，去重理论 5/16=31.25%）
2. 5注蓝球重复比例（生日问题：独立随机约50%的期会有重复）
3. 红球联合 ge2/ge3/ge4（确认红球不受影响）
4. _next_period 期号推算准确率（跨年/前导零 bug 验证）
"""
import sys
import json
import random
import logging

sys.path.insert(0, '.')
logging.disable(logging.WARNING)

from src.ssq import (load_history, RED_RANGE, BLUE_RANGE, RED_COUNT, _analyze,
                     _is_valid_red, _weighted_sample, _next_period, _predict_sets)


def predict_current(train, analysis, n=5, seed=None):
    """复刻当前线上 _predict_sets 逻辑（v3.0：蓝球独立均匀随机）。"""
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
        blue = rng.choice(BLUE_RANGE)
        if blue == prev_blue:
            blue = rng.choice(BLUE_RANGE)
        sets.append({'red': red, 'blue': blue})
    while len(sets) < n:
        red = sorted(rng.sample(RED_RANGE, RED_COUNT))
        blue = rng.choice(BLUE_RANGE)
        sets.append({'red': red, 'blue': blue})
    return sets


def evaluate_sets(sets, actual_red, actual_blue):
    """给定5注与真实开奖，返回各项命中布尔。"""
    blues = [s['blue'] for s in sets]
    reds = [set(s['red']) for s in sets]
    return {
        'blue_any': actual_blue in blues,
        'blue_single_hit': sum(1 for b in blues if b == actual_blue),
        'blue_repeat': len(set(blues)) < len(blues),
        'red_ge2': any(len(r & actual_red) >= 2 for r in reds),
        'red_ge3': any(len(r & actual_red) >= 3 for r in reds),
        'red_ge4': any(len(r & actual_red) >= 4 for r in reds),
        'joint_ge3_b1': any(len(r & actual_red) >= 3 and s['blue'] == actual_blue for r, s in zip(reds, sets)),
        'joint_ge4_b1': any(len(r & actual_red) >= 4 and s['blue'] == actual_blue for r, s in zip(reds, sets)),
    }


def main():
    history = load_history()
    history = sorted(history, key=lambda x: str(x['period']))
    N = 2000
    start = len(history) - N

    agg = {
        'v30': {k: 0 for k in ['blue_any', 'blue_single_hit', 'blue_repeat', 'red_ge2', 'red_ge3', 'red_ge4', 'joint_ge3_b1', 'joint_ge4_b1']},
        'v31': {k: 0 for k in ['blue_any', 'blue_single_hit', 'blue_repeat', 'red_ge2', 'red_ge3', 'red_ge4', 'joint_ge3_b1', 'joint_ge4_b1']},
    }
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
        sets_v30 = predict_current(train, analysis, seed=seed)
        sets_v31 = _predict_sets(train, analysis, n=5, seed=seed)
        for tag, sets in (('v30', sets_v30), ('v31', sets_v31)):
            res = evaluate_sets(sets, actual_red, actual_blue)
            for k in agg[tag]:
                agg[tag][k] += res[k]

    def pct(tag, key, denom=None):
        return f"{agg[tag][key] / (denom or n):.1%}"

    print('=' * 66)
    print(f'双色球蓝球 v3.0 vs v3.1 对比回测（{n}期 walk-forward）')
    print('=' * 66)
    print(f"{'指标':<22}{'v3.0(独立随机)':>16}{'v3.1(去重覆盖)':>16}")
    print(f"{'5注蓝球联合命中':<18}{pct('v30','blue_any'):>16}{pct('v31','blue_any'):>16}  (理论: 独立27.6% / 去重31.25%)")
    blue_single_denom = n * 5
    print(f"{'蓝球单注命中率':<18}{agg['v30']['blue_single_hit']/blue_single_denom:>16.1%}{agg['v31']['blue_single_hit']/blue_single_denom:>16.1%}  (随机基准6.25%)")
    print(f"{'出现重复蓝球期占比':<16}{pct('v30','blue_repeat'):>16}{pct('v31','blue_repeat'):>16}")
    print(f"{'红球 任1注>=2码':<16}{pct('v30','red_ge2'):>16}{pct('v31','red_ge2'):>16}")
    print(f"{'红球 任1注>=3码':<16}{pct('v30','red_ge3'):>16}{pct('v31','red_ge3'):>16}")
    print(f"{'红球 任1注>=4码':<16}{pct('v30','red_ge4'):>16}{pct('v31','red_ge4'):>16}")
    print(f"{'同注红>=3且蓝中':<16}{pct('v30','joint_ge3_b1'):>16}{pct('v31','joint_ge3_b1'):>16}")
    print(f"{'同注红>=4且蓝中':<16}{pct('v30','joint_ge4_b1'):>16}{pct('v31','joint_ge4_b1'):>16}")

    # ---- _next_period 准确率验证 ----
    ok = fail = 0
    fail_examples = []
    for j in range(len(history) - 1):
        cur = str(history[j]['period'])
        real_next = str(history[j + 1]['period'])
        pred_next = _next_period(cur, history)
        if pred_next == real_next:
            ok += 1
        else:
            fail += 1
            if len(fail_examples) < 8:
                fail_examples.append(f'{cur} -> 推算{pred_next}, 实际{real_next}')

    print(f"[期号] _next_period 全历史({len(history)-1}个相邻对) 准确: {ok}  错误: {fail}")
    for ex in fail_examples:
        print(f'   错例: {ex}')

    out = {
        'n': n,
        'v30': {k: round(v / (n * 5 if k == 'blue_single_hit' else n), 4) for k, v in agg['v30'].items()},
        'v31': {k: round(v / (n * 5 if k == 'blue_single_hit' else n), 4) for k, v in agg['v31'].items()},
        'next_period_ok': ok,
        'next_period_fail': fail,
        'fail_examples': fail_examples,
    }
    with open('data/diagnose_ssq_v31_blue.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('结果已存 data/diagnose_ssq_v31_blue.json')


if __name__ == '__main__':
    main()
