#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
足球 SPF 单选门限诊断（离线，五大联赛双赛季）
用与足球同源的 Dixon-Coles 构造 candidates，边际化得 1X2，
扫描 build_decision 的 min_single/min_margin，测量「单选」命中率与覆盖率。
"""
import os
import csv
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.local_match_analysis import build_decision
import src.beidan as beidan

CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
DIV_LEAGUE = {'E0': '英超', 'D1': '德甲', 'SP1': '西甲', 'I1': '意甲', 'F1': '法甲'}


def safe_float(v, default=None):
    try:
        return float(v) if v not in (None, '') else default
    except (ValueError, TypeError):
        return default


def load_matches():
    rows = []
    for fn in sorted(os.listdir(CSV_DIR)):
        div = fn.split('_', 1)[0]
        if not (fn.endswith('.csv') and div in DIV_LEAGUE):
            continue
        league = DIV_LEAGUE[div]
        with open(os.path.join(CSV_DIR, fn), encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                hg = safe_float(r.get('FTHG')); ag = safe_float(r.get('FTAG'))
                if hg is None or ag is None:
                    continue
                ah, ad, aa = safe_float(r.get('AvgH')), safe_float(r.get('AvgD')), safe_float(r.get('AvgA'))
                if not (ah and ad and aa):
                    continue
                rows.append({'league': league, 'hg': int(hg), 'ag': int(ag),
                             'h': ah, 'd': ad, 'a': aa})
    return rows


def candidates_from_odds(h, d, a, league):
    probs = {'胜': 1/h, '平': 1/d, '负': 1/a}
    tot = sum(probs.values()); probs = {k: v/tot for k, v in probs.items()}
    target = beidan.LEAGUE_PROFILES.get(league, {'avg_goals': 2.6})['avg_goals']
    lam_h, lam_a = beidan.euro_implied_lambdas(probs['胜'], probs['平'], probs['负'], target)
    dc = beidan.build_dixon_coles_matrix(lam_h, lam_a)
    return [(k, v) for k, v in dc.items()]


def marginalize(candidates):
    wdl = {'home': 0.0, 'draw': 0.0, 'away': 0.0}
    for (h, a), p in candidates:
        if h > a:
            wdl['home'] += p
        elif h < a:
            wdl['away'] += p
        else:
            wdl['draw'] += p
    return wdl


def actual_spf(m):
    if m['hg'] > m['ag']:
        return 'home'
    if m['hg'] < m['ag']:
        return 'away'
    return 'draw'


def main():
    matches = load_matches()
    recs = []
    for m in matches:
        cands = candidates_from_odds(m['h'], m['d'], m['a'], m['league'])
        wdl = marginalize(cands)
        recs.append({'wdl': wdl, 'actual': actual_spf(m)})
    print(f"样本: {len(recs)} 场\n")

    print(f"{'min_single':>11}{'min_margin':>11}{'single_acc':>12}{'single_cov':>12}{'playable_acc':>14}{'playable_cov':>14}")
    for ms in (0.52, 0.56, 0.60, 0.65, 0.70):
        for mm in (0.08, 0.10, 0.12):
            sh = sn = ph = pn = 0
            for rec in recs:
                dec = build_decision(rec['wdl'], confidence='high', min_single=ms, min_margin=mm)
                if dec['action'] == '单选':
                    sn += 1
                    if dec['primary'] == rec['actual']:
                        sh += 1
                if dec['playable']:
                    pn += 1
                    if dec['primary'] == rec['actual']:
                        ph += 1
            if ms == 0.65 and mm == 0.10:
                print(f"{ms:>11.2f}{mm:>11.2f}{100*sh/sn:>11.2f}%{100*sn/len(recs):>11.1f}%{100*ph/pn:>13.2f}%{100*pn/len(recs):>13.1f}%  <-- 当前线上")
            else:
                print(f"{ms:>11.2f}{mm:>11.2f}{100*sh/sn:>11.2f}%{100*sn/len(recs):>11.1f}%{100*ph/pn:>13.2f}%{100*pn/len(recs):>13.1f}%")
    print(f"\n当前线上 build_match_analysis 用 min_single=0.65, min_margin=0.10。")


if __name__ == '__main__':
    main()
