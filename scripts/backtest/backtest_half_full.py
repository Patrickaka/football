#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
半全场(HT/FT)离线回测 —— 用五大联赛 CSV 的真实半场(HTHG/HTAG/HTR)+全场结果验证。
从 1X2 欧赔 + 大小球反推 FT 比分矩阵，喂给生产 calculate_half_full_time_probs，
对比真实 9 种半全场结果(HH/HD/HA/DH/DD/DA/AH/AD/AA)。纯离线。
"""
import os, sys, csv, math
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import logging
logging.disable(logging.CRITICAL)
import src.football as fb
from src.beidan import euro_implied_lambdas, build_dixon_coles_matrix, implied_total_from_ou

CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
DIV = {'E0': '英超', 'D1': '德甲', 'SP1': '西甲', 'I1': '意甲', 'F1': '法甲'}


def sf(v, d=None):
    try:
        return float(v) if v not in (None, '') else d
    except (ValueError, TypeError):
        return d


def outcome(hg, ag):
    return 'H' if hg > ag else ('A' if hg < ag else 'D')


def load():
    rows = []
    for fn in sorted(os.listdir(CSV_DIR)):
        if not (fn.endswith('.csv') and fn[:2] in DIV):
            continue
        with open(os.path.join(CSV_DIR, fn), encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                fthg, ftag = sf(r.get('FTHG')), sf(r.get('FTAG'))
                hthg, htag = sf(r.get('HTHG')), sf(r.get('HTAG'))
                ah, ad, aa = sf(r.get('AvgH')), sf(r.get('AvgD')), sf(r.get('AvgA'))
                if None in (fthg, ftag, hthg, htag, ah, ad, aa):
                    continue
                rows.append({
                    'league': DIV[fn[:2]],
                    'ftr': outcome(fthg, ftag), 'htr': outcome(hthg, htag),
                    'oh': ah, 'od': ad, 'oa': aa,
                    'over': sf(r.get('Avg>2.5')), 'under': sf(r.get('Avg<2.5')),
                    'ahh': sf(r.get('AHh') or r.get('AHCh')),
                })
    return rows


def ft_candidates(m):
    ph, pd, pa = 1 / m['oh'], 1 / m['od'], 1 / m['oa']
    s = ph + pd + pa
    ph, pd, pa = ph / s, pd / s, pa / s
    prof = fb.resolve_league_profile(m['league'])
    avg = prof.get('avg_goal', 1.4) * 2 if prof else 2.6
    ou = implied_total_from_ou(m['over'], m['under']) if (m['over'] and m['under']) else None
    target = 0.6 * ou + 0.4 * avg if ou else avg
    target = max(1.8, min(3.6, target))
    lh, la = euro_implied_lambdas(ph, pd, pa, target)
    mat = build_dixon_coles_matrix(lh, la, rho=0.0)
    return [((h, a), p) for (h, a), p in mat.items()]


HF_KEYS = ['HH', 'HD', 'HA', 'DH', 'DD', 'DA', 'AH', 'AD', 'AA']


def evaluate(rows, patched_ratio=None):
    top1 = top3 = n = 0
    ll = 0.0
    pred_top1 = Counter()
    for m in rows:
        cand = ft_candidates(m)
        # football-data AHh 符号：负=主让，生产 handicap 正=主让 → 取反
        hcap = -m['ahh'] if m['ahh'] is not None else 0.0
        total = {'close_line': 2.5,
                 'close_prob': {'over': 1 / m['over'] / (1 / m['over'] + 1 / m['under'])
                                if (m['over'] and m['under']) else 0.5}}
        asian = {'handicap': hcap}
        res = fb.calculate_half_full_time_probs(cand, asian=asian, total=total, league=m['league'])
        pm = {}
        for item in (res.get('probs') or []):
            code = str(item.get('code', '')).upper()
            if code in HF_KEYS:
                pm[code] = float(item.get('probability', 0)) / 100.0
        if not pm:
            continue
        actual = m['htr'] + m['ftr']
        ranked = sorted(pm.items(), key=lambda x: -x[1])
        pred_top1[ranked[0][0]] += 1
        if ranked[0][0] == actual:
            top1 += 1
        if actual in [k for k, _ in ranked[:3]]:
            top3 += 1
        ll += -math.log(max(pm.get(actual, 0), 1e-9))
        n += 1
    return {'n': n, 'top1': 100 * top1 / n, 'top3': 100 * top3 / n, 'll': ll / n,
            'pred_top1': pred_top1}


def main():
    rows = load()
    print(f"半全场样本: {len(rows)} 场")
    real = Counter(m['htr'] + m['ftr'] for m in rows)
    print("真实半全场分布:", {k: f"{100*real[k]/len(rows):.1f}%" for k, _ in real.most_common()})
    r = evaluate(rows)
    print(f"\n当前模型: Top1={r['top1']:.2f}%  Top3={r['top3']:.2f}%  LogLoss={r['ll']:.4f}  (n={r['n']})")
    print("模型Top1预测分布:", {k: r['pred_top1'][k] for k in HF_KEYS if r['pred_top1'][k]})
    # 基准：永远猜最高频 H/H
    base = 100 * real['HH'] / len(rows)
    print(f"\n基准(永远猜HH): Top1={base:.2f}%")


if __name__ == '__main__':
    main()
