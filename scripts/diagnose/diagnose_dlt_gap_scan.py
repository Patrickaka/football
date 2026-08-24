#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DLT 前区 gap 权重扫描验证
=========================
front_no_gap 在 500 期中 z=2.64~3.24 (显著正)。
本脚本：
1. 扫描 gap 权重 [0, 0.03, 0.06, 0.09, 0.12(当前), 0.15] 确认单调递减
2. 验证 front_no_gap + 后区不变 的组合
3. 测试 front_no_gap + back_no_freq（200期z=2.15）
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


def set_gap_weight(new_gap):
    """返回 gap=new_gap, 其他等比缩放的新权重"""
    fw = dict(FEATURE_WEIGHTS)
    orig_total = sum(fw.values())  # 0.76
    orig_gap = fw['gap']  # 0.12
    # 去掉原 gap, 加上新 gap, 再等比缩放到原始总量
    without_gap = orig_total - orig_gap
    scale = (orig_total - new_gap) / without_gap if without_gap > 0 else 1.0
    result = {k: (v * scale if k != 'gap' else new_gap) for k, v in fw.items()}
    result['gap'] = new_gap
    return result


def renormalize_back(exclude):
    w = {k: (0.0 if k in exclude else v) for k, v in BACK_FEATURE_WEIGHTS.items()}
    total = sum(w.values())
    orig = sum(BACK_FEATURE_WEIGHTS.values())
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
    print(f"[DLT] 历史 {len(a.history_data)} 期, 跑 {TRIALS} 期\n")

    experiments = [
        # gap 权重扫描
        ('gap=0.12(当前)', None, None),
        ('gap=0.09', set_gap_weight(0.09), None),
        ('gap=0.06', set_gap_weight(0.06), None),
        ('gap=0.03', set_gap_weight(0.03), None),
        ('gap=0.00', set_gap_weight(0.00), None),
        ('gap=0.15', set_gap_weight(0.15), None),
        # 组合
        ('gap0+back_no_freq', set_gap_weight(0.00), renormalize_back({'frequency'})),
    ]

    results = []
    for name, fw, bw in experiments:
        print(f"运行 [{name}]...", end='', flush=True)
        r = run_dlt_wf(TRIALS, front_weights=fw, back_weights=bw, label=name)
        results.append(r)
        print(f" 2+1={r['any_2plus1']:.1%} 中奖={r['any_prize']:.1%}")

    print("\n" + "=" * 70)
    base = results[0]
    s21 = sigma(base['any_2plus1'], base['n'])
    sp = sigma(base['any_prize'], base['n'])
    print(f"{'实验':<22} {'任1注2+1':>8} {'Δpp':>6} {'z':>5}  {'任1注中奖':>8} {'Δpp':>6} {'z':>5}")
    print("-" * 70)
    for r in results:
        d21 = (r['any_2plus1'] - base['any_2plus1']) * 100
        dp = (r['any_prize'] - base['any_prize']) * 100
        z21 = (r['any_2plus1'] - base['any_2plus1']) / s21 if s21 else 0
        zp = (r['any_prize'] - base['any_prize']) / sp if sp else 0
        flag = ""
        if r['label'] != 'gap=0.12(当前)':
            if z21 > 2 or zp > 2: flag = " ★"
            elif z21 < -2 or zp < -2: flag = " ✗"
        print(f"{r['label']:<22} {r['any_2plus1']:>7.1%} {d21:>+5.1f} {z21:>+5.2f}  "
              f"{r['any_prize']:>7.1%} {dp:>+5.1f} {zp:>+5.2f}{flag}")

    out = {'date': '2026-08-24', 'trials': TRIALS, 'results': results}
    path = 'data/diagnose_dlt_gap_scan_500_20260824.json'
    json.dump(out, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"\n已保存: {path}")


if __name__ == '__main__':
    main()
