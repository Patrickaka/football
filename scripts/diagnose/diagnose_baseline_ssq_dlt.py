#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双色球 v3.2 / 大乐透 v4.4 当前版本 walk-forward 基线（改动前）
================================================================
原则（lottery-walkforward skill）：
- 必须调用真实预测函数（ssq._predict_sets / LotteryAnalyzer 完整管线）
- 逐期 train = 当期之前数据，防泄漏
- 与组合数学随机基准对照，差分判定（±1σ 内 = 噪声）

随机基准推导：
  双色球 5注覆盖结构（union=30, 5组×6码互不重叠）：
    任1注红球≥2: 1 - 全部6码落入不同组/池外的概率 ≈ 95.9%（鸽笼上界）
    单注红球≥2 = 1-C(27,6)/C(33,6)-6·C(27,5)/C(33,6) ≈ 0.4072
    单注红球≥3 ≈ 0.0789, 单注红球≥4 ≈ 0.00756
    蓝球5注互异联合命中 = 5/16 = 31.25%（理论最优）
  大乐透 5注组合（前区union=25, 后区覆盖10码）：
    任1注前区≥2 ≈ 52.3%, 任1注前区≥3 ≈ 6.8%
    后区任1注≥1 = 1 - C(2,2)/C(12,2) = 98.5%
    单注前区≥2 = 13.89%, 后区单注≥1 = 45.45%
"""
import sys
import json
import math
import random
import logging

sys.path.insert(0, '.')
logging.disable(logging.WARNING)

SSQ_TRIALS = 2000
DLT_TRIALS = 500


def sigma(p, n):
    return math.sqrt(p * (1 - p) / n) if n else 0.0


def judge(measured, baseline, n):
    """返回 (差值pp, σ倍数, 判定)"""
    s = sigma(baseline, n)
    z = (measured - baseline) / s if s > 0 else 0.0
    if abs(z) <= 1:
        v = '噪声(≤1σ)'
    elif z > 1:
        v = '高于基准 %.1fσ' % z
    else:
        v = '低于基准 %.1fσ' % abs(z)
    return round((measured - baseline) * 100, 2), round(z, 2), v


def ssq_baseline():
    from src.ssq import load_history, _analyze, _predict_sets
    history = sorted(load_history(), key=lambda x: x['period'])
    print(f"[SSQ] 全量历史 {len(history)} 期 ({history[0]['period']} ~ {history[-1]['period']})")

    any2 = any3 = any4 = 0
    per2 = per3 = per4 = 0
    blue_any = 0
    n = 0
    for i in range(len(history) - SSQ_TRIALS, len(history)):
        train = history[:i]
        if len(train) < 200:
            continue
        analysis = _analyze(train)
        sets = _predict_sets(train, analysis, n=5, seed=int(history[i]['period']))
        ar = set(history[i]['red'])
        ab = history[i]['blue']
        n += 1
        hits = [len(set(s['red']) & ar) for s in sets]
        if max(hits) >= 2: any2 += 1
        if max(hits) >= 3: any3 += 1
        if max(hits) >= 4: any4 += 1
        per2 += sum(1 for h in hits if h >= 2)
        per3 += sum(1 for h in hits if h >= 3)
        per4 += sum(1 for h in hits if h >= 4)
        if any(s['blue'] == ab for s in sets):
            blue_any += 1

    res = {
        'n': n,
        'any_red_ge2': any2 / n, 'any_red_ge3': any3 / n, 'any_red_ge4': any4 / n,
        'per_ticket_ge2': per2 / (n * 5), 'per_ticket_ge3': per3 / (n * 5),
        'per_ticket_ge4': per4 / (n * 5),
        'blue_any_hit': blue_any / n,
    }
    base = {
        # 精确组合数学基准（2026-08-21 复核）：
        # 单注6/33: ge2=0.29540, ge3=0.05772, ge4=0.00490
        # 5x6互斥覆盖(union=30): any_ge2 鸽笼上界≈95.9%, any_ge3/any_ge4 为结构实测参照
        'any_red_ge2': 0.959, 'any_red_ge3': 0.287, 'any_red_ge4': 0.024,
        'per_ticket_ge2': 0.2954, 'per_ticket_ge3': 0.0577, 'per_ticket_ge4': 0.0049,
        'blue_any_hit': 0.3125,
    }
    print(f"[SSQ] 回测 {n} 期")
    out = {'n': n, 'metrics': {}}
    for k in base:
        dpp, z, v = judge(res[k], base[k], n if not k.startswith('per_') else n * 5)
        out['metrics'][k] = {'measured': round(res[k], 4), 'baseline': base[k],
                             'diff_pp': dpp, 'z': z, 'verdict': v}
        print(f"  {k:<18} 实测 {res[k]:7.2%}  基准 {base[k]:7.2%}  {v}")
    return out


def dlt_baseline():
    from src.lottery import LotteryAnalyzer
    a = LotteryAnalyzer()
    saved = list(a.history_data)
    print(f"[DLT] 历史 {len(saved)} 期 (最新 {saved[0]['issue']})")

    any2 = any3 = back1 = joint = 0
    per2 = 0
    n = 0
    for i in range(DLT_TRIALS):
        if i >= len(saved) - 81:
            break
        a.history_data = list(saved[i + 1:])
        a.update_statistics()
        af = set(saved[i]['front'])
        ab = set(saved[i]['back'])
        multi = a.generate_multi_strategy_recommendations(
            voting_result=a.multi_model_voting(front_n=20, back_n=10, skip_ml=True))
        recs = [x for x in multi['recommendations']
                if not x['strategy'].startswith('picked')]
        n += 1
        f2 = f3 = b1 = jt = False
        for r in recs:
            hf = len(af & set(r['front']))
            hb = len(ab & set(r['back']))
            if hf >= 2: per2 += 1; f2 = True
            if hf >= 3: f3 = True
            if hb >= 1: b1 = True
            if hf >= 2 and hb >= 1: jt = True
        if f2: any2 += 1
        if f3: any3 += 1
        if b1: back1 += 1
        if jt: joint += 1

    k5 = len(recs)
    res = {
        'front_any_ge2': any2 / n, 'front_any_ge3': any3 / n,
        'back_any_ge1': back1 / n, 'joint_f2b1': joint / n,
        'per_ticket_front_ge2': per2 / (n * k5),
    }
    base = {
        # front_any_ge2: 5组x5码互斥(union=25,池外10码)结构的精确上界=0.6115
        # （独立5注对照=0.5257；旧脚本误用0.523作结构基准）
        'front_any_ge2': 0.6115, 'front_any_ge3': 0.068,
        'back_any_ge1': 0.985, 'joint_f2b1': 0.20,
        'per_ticket_front_ge2': 0.1389,
    }
    print(f"[DLT] 回测 {n} 期 x {k5} 注")
    out = {'n': n, 'tickets': k5, 'metrics': {}}
    for k in res:
        nn = n * k5 if k.startswith('per_') else n
        dpp, z, v = judge(res[k], base[k], nn)
        out['metrics'][k] = {'measured': round(res[k], 4), 'baseline': base[k],
                             'diff_pp': dpp, 'z': z, 'verdict': v}
        print(f"  {k:<22} 实测 {res[k]:7.2%}  基准 {base[k]:7.2%}  {v}")
    return out


if __name__ == '__main__':
    result = {
        'date': '2026-08-21',
        'ssq_version': 'ssq-v3.3-prize-stats',
        'dlt_version': 'dlt-v4.5-next-issue',
        'ssq': ssq_baseline(),
        'dlt': dlt_baseline(),
    }
    path = 'data/diagnose_baseline_ssq_dlt_20260821.json'
    json.dump(result, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"\n已保存: {path}")
