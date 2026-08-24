#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DLT 新信号消融实验
===================
1. 特征消融：逐个关闭每个特征，测量对任1注2+1/任1注中奖的影响
2. 新信号：前区共现矩阵、后区马尔可夫、AC值约束
3. 权重变体：极端配置测试

判定：walk-forward 200 期，z > 2 才采纳
"""
import sys, json, math, logging
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, '.')
logging.disable(logging.WARNING)

from src.lottery.config import (
    FEATURE_WEIGHTS, BACK_FEATURE_WEIGHTS, FRONT_NUMBERS, BACK_NUMBERS,
    RANDOM_BASELINE,
)
from src.lottery.analyzer import get_lottery_analyzer
from src.lottery.records import dlt_prize_tier

TRIALS = 200


def sigma(p, n):
    return math.sqrt(p * (1 - p) / n) if n else 0.0


def run_dlt_walkforward(trials=TRIALS, front_weights=None, back_weights=None,
                        label='baseline'):
    """跑 DLT walk-forward 回测，返回指标 dict"""
    from src.lottery import config as dlt_config

    # 保存原始权重
    orig_fw = dict(dlt_config.FEATURE_WEIGHTS)
    orig_bw = dict(dlt_config.BACK_FEATURE_WEIGHTS)

    # 应用实验权重
    if front_weights is not None:
        dlt_config.FEATURE_WEIGHTS.clear()
        dlt_config.FEATURE_WEIGHTS.update(front_weights)
    if back_weights is not None:
        dlt_config.BACK_FEATURE_WEIGHTS.clear()
        dlt_config.BACK_FEATURE_WEIGHTS.update(back_weights)

    try:
        a = get_lottery_analyzer()
        saved = list(a.history_data)

        any_2plus1 = 0    # 任1注 2+1（九等奖）
        any_prize = 0    # 任1注中奖（任意奖级）
        front_any_ge2 = 0
        front_any_ge3 = 0
        back_any_ge1 = 0
        per_ticket_2plus1 = 0
        total_tickets = 0
        n = 0

        for i in range(trials):
            if i >= len(saved) - 81:
                break
            a.history_data = list(saved[i + 1:])
            a.update_statistics()

            multi = a.generate_multi_strategy_recommendations(
                voting_result=a.multi_model_voting(front_n=20, back_n=10, skip_ml=True))
            recs = [x for x in multi['recommendations']
                    if not x['strategy'].startswith('picked')]

            af = set(saved[i]['front'])
            ab = set(saved[i]['back'])
            n += 1
            f2 = f3 = b1 = prize = j21 = False
            for r in recs:
                hf = len(af & set(r['front']))
                hb = len(ab & set(r['back']))
                total_tickets += 1
                tier = dlt_prize_tier(hf, hb)
                if tier > 0:
                    prize = True
                if hf >= 2:
                    f2 = True
                if hf >= 3:
                    f3 = True
                if hb >= 1:
                    b1 = True
                if hf >= 2 and hb >= 1:
                    j21 = True
                    per_ticket_2plus1 += 1
            if f2: front_any_ge2 += 1
            if f3: front_any_ge3 += 1
            if b1: back_any_ge1 += 1
            if prize: any_prize += 1
            if j21: any_2plus1 += 1

        k = len(recs) if n else 0
        return {
            'label': label,
            'n': n,
            'tickets': k,
            'any_2plus1_rate': any_2plus1 / n if n else 0,
            'any_prize_rate': any_prize / n if n else 0,
            'front_any_ge2': front_any_ge2 / n if n else 0,
            'front_any_ge3': front_any_ge3 / n if n else 0,
            'back_any_ge1': back_any_ge1 / n if n else 0,
            'per_ticket_2plus1': per_ticket_2plus1 / (n * k) if n * k else 0,
        }
    finally:
        # 恢复原始权重
        dlt_config.FEATURE_WEIGHTS.clear()
        dlt_config.FEATURE_WEIGHTS.update(orig_fw)
        dlt_config.BACK_FEATURE_WEIGHTS.clear()
        dlt_config.BACK_FEATURE_WEIGHTS.update(orig_bw)


def run_experiment():
    print("[DLT] 加载分析器...")
    # 预加载
    a = get_lottery_analyzer()
    print(f"[DLT] 历史 {len(a.history_data)} 期")

    experiments = []

    # 1. 基线
    experiments.append(('baseline', None, None))

    # 2. 特征消融：逐个关闭
    for feat in list(FEATURE_WEIGHTS.keys()):
        if FEATURE_WEIGHTS[feat] == 0:
            continue  # 跳过已关闭的
        fw = dict(FEATURE_WEIGHTS)
        fw[feat] = 0.0
        # 重新归一化
        total = sum(fw.values())
        if total > 0:
            fw = {k: v / total * sum(FEATURE_WEIGHTS.values()) for k, v in fw.items()}
        experiments.append((f'no_{feat}', fw, None))

    # 3. 后区消融
    for feat in list(BACK_FEATURE_WEIGHTS.keys()):
        if BACK_FEATURE_WEIGHTS[feat] == 0:
            continue
        bw = dict(BACK_FEATURE_WEIGHTS)
        bw[feat] = 0.0
        total = sum(bw.values())
        if total > 0:
            bw = {k: v / total * sum(BACK_FEATURE_WEIGHTS.values()) for k, v in bw.items()}
        experiments.append((f'back_no_{feat}', None, bw))

    # 4. 极端权重变体
    # 4a. 全 gap（赌遗漏回补）
    fw_gap = {k: 0.0 for k in FEATURE_WEIGHTS}
    fw_gap['gap'] = 1.0
    experiments.append(('all_gap', fw_gap, None))

    # 4b. 全 road（最重要的特征）
    fw_road = {k: 0.0 for k in FEATURE_WEIGHTS}
    fw_road['road'] = 1.0
    experiments.append(('all_road', fw_road, None))

    # 4c. 全 adjacent（邻号真实信号）
    fw_adj = {k: 0.0 for k in FEATURE_WEIGHTS}
    fw_adj['adjacent'] = 1.0
    experiments.append(('all_adjacent', fw_adj, None))

    # 5. 后区全 gap
    bw_gap = {k: 0.0 for k in BACK_FEATURE_WEIGHTS}
    bw_gap['gap'] = 1.0
    experiments.append(('back_all_gap', None, bw_gap))

    results = []
    for name, fw, bw in experiments:
        print(f"\n运行 [{name}]...", end='', flush=True)
        r = run_dlt_walkforward(front_weights=fw, back_weights=bw, label=name)
        results.append(r)
        print(f" 2+1={r['any_2plus1_rate']:.2%} 中奖={r['any_prize_rate']:.2%} "
              f"前≥2={r['front_any_ge2']:.2%} 后≥1={r['back_any_ge1']:.2%}")

    # 对照表
    print("\n" + "=" * 90)
    base = results[0]
    s_2plus1 = sigma(base['any_2plus1_rate'], base['n'])
    s_prize = sigma(base['any_prize_rate'], base['n'])
    print(f"{'实验':<20} {'任1注2+1':>9} {'Δpp':>6} {'z':>5}  {'任1注中奖':>9} {'Δpp':>6} {'z':>5}")
    print("-" * 90)
    for r in results:
        d21 = (r['any_2plus1_rate'] - base['any_2plus1_rate']) * 100
        dp = (r['any_prize_rate'] - base['any_prize_rate']) * 100
        z21 = (r['any_2plus1_rate'] - base['any_2plus1_rate']) / s_2plus1 if s_2plus1 else 0
        zp = (r['any_prize_rate'] - base['any_prize_rate']) / s_prize if s_prize else 0
        flag = ""
        if r['label'] != 'baseline':
            if z21 > 2 or zp > 2:
                flag = " ★显著正"
            elif z21 < -2 or zp < -2:
                flag = " ✗显著负"
        print(f"{r['label']:<20} {r['any_2plus1_rate']:>8.1%} {d21:>+5.1f} {z21:>+5.2f}  "
              f"{r['any_prize_rate']:>8.1%} {dp:>+5.1f} {zp:>+5.2f}{flag}")

    out = {'date': '2026-08-24', 'trials': TRIALS, 'results': results}
    path = 'data/diagnose_dlt_signals_20260824.json'
    json.dump(out, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"\n已保存: {path}")
    return out


if __name__ == '__main__':
    run_experiment()
