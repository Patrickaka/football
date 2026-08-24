#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DLT 后区权重优化验证 (500 期)
================================
基于 200 期消融发现的显著信号：
  1. back_no_frequency: 去掉后区频率权重 → 中奖率 +7pp (z=2.15)
  2. back_no_gap: 去掉后区遗漏权重 → 2+1率 +6.5pp (z=2.30)
  3. back_no_position: 去掉后区位置 → 中奖率 -7.9pp (z=-2.43), 位置必须保留
  4. no_adjacent: 去掉前区邻号 → 2+1率 -5pp (z=-1.77), 邻号有真实信号

验证组合：
  A. back_no_freq_gap: 同时去掉后区频率+遗漏
  B. back_no_freq_gap_adj: 再去掉后区邻号（前区保留）
  C. front_no_gap: 前区也去掉遗漏（消融显示前区 no_gap z=+1.94）
  D. combined_best: 后区去freq+gap, 前区去gap, 保留adjacent
"""
import sys, json, math, logging

sys.path.insert(0, '.')
logging.disable(logging.WARNING)

from src.lottery.config import FEATURE_WEIGHTS, BACK_FEATURE_WEIGHTS
from src.lottery.analyzer import get_lottery_analyzer
from src.lottery.records import dlt_prize_tier

TRIALS = 500


def sigma(p, n):
    return math.sqrt(p * (1 - p) / n) if n else 0.0


def renormalize(weights, exclude_keys):
    """去掉指定 key 后重新归一化到原始总权重"""
    w = {k: (0.0 if k in exclude_keys else v) for k, v in weights.items()}
    total = sum(w.values())
    orig = sum(weights.values())
    if total > 0:
        w = {k: v / total * orig for k, v in w.items()}
    return w


def run_dlt_wf(trials, front_weights=None, back_weights=None, label='baseline'):
    from src.lottery import config as dlt_config
    orig_fw = dict(dlt_config.FEATURE_WEIGHTS)
    orig_bw = dict(dlt_config.BACK_FEATURE_WEIGHTS)
    if front_weights is not None:
        dlt_config.FEATURE_WEIGHTS.clear()
        dlt_config.FEATURE_WEIGHTS.update(front_weights)
    if back_weights is not None:
        dlt_config.BACK_FEATURE_WEIGHTS.clear()
        dlt_config.BACK_FEATURE_WEIGHTS.update(back_weights)
    try:
        a = get_lottery_analyzer()
        saved = list(a.history_data)
        any_2plus1 = any_prize = front_ge2 = back_ge1 = 0
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
            f2 = b1 = prize = j21 = False
            for r in recs:
                hf = len(af & set(r['front']))
                hb = len(ab & set(r['back']))
                if dlt_prize_tier(hf, hb) > 0: prize = True
                if hf >= 2: f2 = True
                if hb >= 1: b1 = True
                if hf >= 2 and hb >= 1: j21 = True
            if f2: front_ge2 += 1
            if b1: back_ge1 += 1
            if prize: any_prize += 1
            if j21: any_2plus1 += 1
        return {
            'label': label, 'n': n,
            'any_2plus1': any_2plus1 / n if n else 0,
            'any_prize': any_prize / n if n else 0,
            'front_ge2': front_ge2 / n if n else 0,
            'back_ge1': back_ge1 / n if n else 0,
        }
    finally:
        dlt_config.FEATURE_WEIGHTS.clear()
        dlt_config.FEATURE_WEIGHTS.update(orig_fw)
        dlt_config.BACK_FEATURE_WEIGHTS.clear()
        dlt_config.BACK_FEATURE_WEIGHTS.update(orig_bw)


def main():
    print("[DLT] 加载分析器...")
    a = get_lottery_analyzer()
    print(f"[DLT] 历史 {len(a.history_data)} 期, 跑 {TRIALS} 期验证\n")

    experiments = [
        ('baseline', None, None),
        ('back_no_freq', None, renormalize(BACK_FEATURE_WEIGHTS, {'frequency'})),
        ('back_no_gap', None, renormalize(BACK_FEATURE_WEIGHTS, {'gap'})),
        ('back_no_freq_gap', None, renormalize(BACK_FEATURE_WEIGHTS, {'frequency', 'gap'})),
        ('back_no_freq_gap_adj', None, renormalize(BACK_FEATURE_WEIGHTS, {'frequency', 'gap', 'adjacent'})),
        ('front_no_gap', renormalize(FEATURE_WEIGHTS, {'gap'}), None),
        ('combined_best', renormalize(FEATURE_WEIGHTS, {'gap'}),
         renormalize(BACK_FEATURE_WEIGHTS, {'frequency', 'gap'})),
        # 对照：前区也去掉 frequency（前区消融显示 -3pp 但不显著）
        ('combined_no_freq', renormalize(FEATURE_WEIGHTS, {'frequency', 'gap'}),
         renormalize(BACK_FEATURE_WEIGHTS, {'frequency', 'gap'})),
    ]

    results = []
    for name, fw, bw in experiments:
        print(f"运行 [{name}]...", end='', flush=True)
        r = run_dlt_wf(TRIALS, front_weights=fw, back_weights=bw, label=name)
        results.append(r)
        print(f" 2+1={r['any_2plus1']:.1%} 中奖={r['any_prize']:.1%} "
              f"前≥2={r['front_ge2']:.1%} 后≥1={r['back_ge1']:.1%}")

    print("\n" + "=" * 85)
    base = results[0]
    s21 = sigma(base['any_2plus1'], base['n'])
    sp = sigma(base['any_prize'], base['n'])
    print(f"{'实验':<24} {'任1注2+1':>8} {'Δpp':>6} {'z':>5}  {'任1注中奖':>8} {'Δpp':>6} {'z':>5}")
    print("-" * 85)
    for r in results:
        d21 = (r['any_2plus1'] - base['any_2plus1']) * 100
        dp = (r['any_prize'] - base['any_prize']) * 100
        z21 = (r['any_2plus1'] - base['any_2plus1']) / s21 if s21 else 0
        zp = (r['any_prize'] - base['any_prize']) / sp if sp else 0
        flag = ""
        if r['label'] != 'baseline':
            if z21 > 2 or zp > 2: flag = " ★显著正"
            elif z21 < -2 or zp < -2: flag = " ✗显著负"
        print(f"{r['label']:<24} {r['any_2plus1']:>7.1%} {d21:>+5.1f} {z21:>+5.2f}  "
              f"{r['any_prize']:>7.1%} {dp:>+5.1f} {zp:>+5.2f}{flag}")

    out = {'date': '2026-08-24', 'trials': TRIALS, 'results': results}
    path = 'data/diagnose_dlt_weights_500_20260824.json'
    json.dump(out, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"\n已保存: {path}")


if __name__ == '__main__':
    main()
