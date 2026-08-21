#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双色球 v3.0 全量数据回测 — 大底池覆盖验证
==========================================
验证 web 页面展示的指标（用全量3489期数据 walk-forward）：
1. 红球 Top15 大底覆盖开奖≥4码概率（模型 vs 随机基准24.2%）
2. 蓝球 Top8 大底命中概率（模型 vs 随机基准50%）
3. 5注红球 ge2/ge3 命中、蓝球命中率（验证蓝球均匀随机的表现）
"""
import sys
import math
import logging
import random
from collections import Counter

sys.path.insert(0, '.')
logging.disable(logging.WARNING)

from src.ssq import load_history, RED_RANGE, BLUE_RANGE, _analyze, _is_valid_red, _weighted_sample, RED_COUNT


def evaluate(trials=600, seed=42):
    """walk-forward：用每期之前的数据预测当期（模拟在线流程）。"""
    history = load_history()
    history = sorted(history, key=lambda x: x['period'])
    print(f"全量历史: {len(history)}期 ({history[0]['period']} ~ {history[-1]['period']})")

    red_ge2 = red_ge3 = red_ge4 = 0
    blue_hit = 0
    pool_ge4 = pool_ge3 = 0
    blue_pool_hit = 0
    n = 0
    for i in range(len(history) - trials, len(history)):
        train = history[:i]  # 只用当期之前的数据
        if len(train) < 200:
            continue
        analysis = _analyze(train)
        actual_red = set(history[i]['red'])
        actual_blue = history[i]['blue']
        n += 1

        # 大底池（与 run_prediction 一致）
        red_pool = [item['number'] for item in analysis['red_ranking'][:15]]
        blue_pool = [item['number'] for item in analysis['blue_ranking'][:8]]
        hits_pool = len(actual_red & set(red_pool))
        if hits_pool >= 4:
            pool_ge4 += 1
        if hits_pool >= 3:
            pool_ge3 += 1
        if actual_blue in blue_pool:
            blue_pool_hit += 1

        # 5注单式（用期号作随机种子，与线上一致）
        rng = random.Random(int(history[i]['period']))
        red_w = [analysis['red_freq'][x] + 1.5 * analysis['red_recent'][x] + 0.5 for x in RED_RANGE]
        sets = []
        tries = 0
        while len(sets) < 5 and tries < 5000:
            tries += 1
            red = sorted(_weighted_sample(RED_RANGE, red_w, RED_COUNT, rng))
            if not _is_valid_red(red):
                continue
            if set(red) == actual_red:
                continue
            blue = rng.choice(BLUE_RANGE)
            sets.append({'red': red, 'blue': blue})
        while len(sets) < 5:
            red = sorted(rng.sample(RED_RANGE, RED_COUNT))
            blue = rng.choice(BLUE_RANGE)
            sets.append({'red': red, 'blue': blue})

        # 5注中最好的命中
        best_red = max(len(set(s['red']) & actual_red) for s in sets)
        blue_any = any(s['blue'] == actual_blue for s in sets)
        if best_red >= 2:
            red_ge2 += 1
        if best_red >= 3:
            red_ge3 += 1
        if best_red >= 4:
            red_ge4 += 1
        if blue_any:
            blue_hit += 1

    print(f"\n回测期数: {n}")
    print("=" * 70)
    print("【大底池】")
    print(f"  红球Top15 ge4 = {pool_ge4/n*100:.1f}%   (随机推15码理论: 24.2%)")
    print(f"  红球Top15 ge3 = {pool_ge3/n*100:.1f}%")
    print(f"  蓝球Top8 命中 = {blue_pool_hit/n*100:.1f}%   (随机推8码理论: 50%)")
    print("【5注单式】")
    print(f"  红球≥2码 = {red_ge2/n*100:.1f}%")
    print(f"  红球≥3码 = {red_ge3/n*100:.1f}%")
    print(f"  红球≥4码 = {red_ge4/n*100:.1f}%")
    print(f"  蓝球命中 = {blue_hit/n*100:.1f}%   (5注×16蓝=31.25%理论, 均匀随机≈32~34%)")

    return {
        'n': n,
        'pool_ge4': pool_ge4 / n,
        'pool_ge3': pool_ge3 / n,
        'blue_pool_hit': blue_pool_hit / n,
        'red_ge2': red_ge2 / n,
        'red_ge3': red_ge3 / n,
        'red_ge4': red_ge4 / n,
        'blue_hit': blue_hit / n,
    }


if __name__ == '__main__':
    evaluate(trials=600)
