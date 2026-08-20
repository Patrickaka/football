#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
北单全管线离线回测 + 选门限诊断
- 用真实 analyze_spf / analyze_bifen / analyze_zjq 跑五大联赛双赛季历史
- 测量 SPF 全量/强推荐/中+ 命中率与覆盖率
- 扫描 assess_recommendation_quality 的 strong/medium 阈值，找经验最优
用法: python3 scripts/diagnose_beidan_full.py
"""
import os
import csv
import math
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.beidan as beidan

CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
DIV_LEAGUE = {'E0': '英超', 'D1': '德甲', 'SP1': '西甲', 'I1': '意甲', 'F1': '法甲'}


def safe_float(v, default=None):
    try:
        if v is None or v == '':
            return default
        return float(v)
    except (ValueError, TypeError):
        return default


def fake_ouzhi(match_id):
    return fake_ouzhi._store.get(match_id)


fake_ouzhi._store = {}


def load_matches():
    rows = []
    for fn in sorted(os.listdir(CSV_DIR)):
        div = fn.split('_', 1)[0]
        if not (fn.endswith('.csv') and div in DIV_LEAGUE):
            continue
        league = DIV_LEAGUE[div]
        with open(os.path.join(CSV_DIR, fn), encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                hg = safe_float(r.get('FTHG'))
                ag = safe_float(r.get('FTAG'))
                if hg is None or ag is None:
                    continue
                ah, ad, aa = safe_float(r.get('AvgH')), safe_float(r.get('AvgD')), safe_float(r.get('AvgA'))
                if not (ah and ad and aa):
                    continue
                mid = f"{div}_{r['Date']}_{r['HomeTeam']}_{r['AwayTeam']}"
                fake_ouzhi._store[mid] = {'home': ah, 'draw': ad, 'away': aa}
                rows.append({
                    'match_id': mid, 'league': league,
                    'home': r['HomeTeam'], 'away': r['AwayTeam'],
                    'hg': int(hg), 'ag': int(ag),
                    'over': safe_float(r.get('Avg>2.5')), 'under': safe_float(r.get('Avg<2.5')),
                })
    return rows


def actual_spf(m):
    if m['hg'] > m['ag']:
        return '胜'
    if m['hg'] < m['ag']:
        return '负'
    return '平'


def main():
    beidan.fetch_ouzhi_odds = fake_ouzhi
    matches = load_matches()
    print(f"样本: {len(matches)} 场")

    goal_keys = [str(i) for i in range(8)] + ['7+']
    score_keys = [(h, a) for h in range(8) for a in range(8)]

    # ===== SPF 全量收集（用真实 analyze_spf）=====
    spf_records = []
    z_ll = z_brier = z_top1 = z_top2 = z_n = 0.0
    s_ll = s_brier = s_top1 = s_top3 = s_n = 0.0

    for m in matches:
        mid = m['match_id']
        over, under = m['over'], m['under']
        goals_data = {'history': [{'over_odds': over, 'under_odds': under}]} if (over and under) else None

        # SPF
        sr = beidan.analyze_spf({'id': mid, 'league': m['league'], 'home': m['home'],
                                 'away': m['away'], 'num': '', 'time': '', 'handicap': 0})
        if 'probabilities' in sr:
            probs = sr['probabilities']
            pred = sr['prediction']
            spf_records.append({
                'probs': probs, 'pred': pred, 'actual': actual_spf(m),
                'quality': sr.get('quality', {}),
            })

        # 总进球
        zr = beidan.analyze_zjq({'id': mid, 'league': m['league'], 'home': m['home'], 'away': m['away'],
                                 'num': '', 'time': ''}, zjq_odds=None, asian_data=None, goals_data=goals_data)
        if 'probabilities' in zr:
            dist = zr['probabilities']
            at = m['hg'] + m['ag']
            ak = '7+' if at >= 7 else str(at)
            z_ll += -math.log(max(dist.get(ak, 0.0), 1e-9))
            z_brier += sum((dist.get(k, 0.0) - (1.0 if k == ak else 0.0)) ** 2 for k in goal_keys)
            ranked = sorted(dist.items(), key=lambda x: -x[1])
            if ak == ranked[0][0]:
                z_top1 += 1
            if ak in [k for k, _ in ranked[:2]]:
                z_top2 += 1
            z_n += 1

        # 比分
        br = beidan.analyze_bifen({'id': mid, 'league': m['league'], 'home': m['home'], 'away': m['away'],
                                   'num': '', 'time': '', 'handicap': 0}, bifen_odds=None, asian_data=None, goals_data=goals_data)
        if 'probabilities' in br:
            dist = br['probabilities']
            ak = (m['hg'], m['ag'])
            s_ll += -math.log(max(dist.get(ak, 0.0), 1e-9))
            s_brier += sum((dist.get(k, 0.0) - (1.0 if k == ak else 0.0)) ** 2 for k in score_keys)
            ranked = sorted(dist.items(), key=lambda x: -x[1])
            if ranked and ranked[0][0] == ak:
                s_top1 += 1
            if ak in [k for k, _ in ranked[:3]]:
                s_top3 += 1
            s_n += 1

    print("\n" + "=" * 70)
    print("北单全管线离线回测（五大联赛双赛季）")
    print("=" * 70)
    print(f"SPF  全量预测数: {len(spf_records)}")
    print(f"比分 bifen : LogLoss={s_ll/s_n:.4f}  Brier={s_brier/s_n:.4f}  Top1={100*s_top1/s_n:.2f}%  Top3={100*s_top3/s_n:.2f}%")
    print(f"总进球 zjq : LogLoss={z_ll/z_n:.4f}  Brier={z_brier/z_n:.4f}  Top1={100*z_top1/z_n:.2f}%  Top2={100*z_top2/z_n:.2f}%")

    # ===== SPF 分级命中率（当前阈值）=====
    def grade_stats(records, use_level=True, level_filter=None):
        hit = n = 0
        for rec in records:
            q = rec['quality']
            if use_level and level_filter and q.get('level') not in level_filter:
                continue
            n += 1
            if rec['pred'] == rec['actual']:
                hit += 1
        return hit, n

    full_h, full_n = grade_stats(spf_records, use_level=False)
    print(f"\n[SPF 当前分级]")
    print(f"  全量 Top1 命中率 : {100*full_h/full_n:.2f}%  (n={full_n})")
    for lvl, label in [({'strong'}, '强推荐'), ({'strong', 'medium'}, '中+')]:
        h, n = grade_stats(spf_records, level_filter=lvl)
        print(f"  {label:6s} 命中率 : {100*h/n:.2f}%  (n={n}, 覆盖 {100*n/full_n:.1f}%)")

    # ===== 扫描 strong/medium 阈值（直接复用模块评估，但用独立阈值）=====
    print("\n[SPF 选门限扫描 — 经验最优 strong 阈值]")
    print(f"{'strong_p':>8}{'strong_lead':>12}{'acc':>8}{'cov':>8}{'med_p':>8}{'med_lead':>10}{'m_acc':>8}{'m_cov':>8}")
    best = None
    for sp in (0.52, 0.56, 0.60, 0.65, 0.70):
        for slead in (0.08, 0.10, 0.12):
            for mp in (0.50, 0.52, 0.54):
                for mlead in (0.06, 0.08, 0.10):
                    sh = sn = mh = mn = 0
                    for rec in spf_records:
                        probs = rec['probs']
                        ranked = sorted(probs.items(), key=lambda x: -x[1])
                        top = ranked[0][1]
                        lead = top - ranked[1][1]
                        pred = ranked[0][0]
                        is_strong = top >= sp and lead >= slead
                        is_med = (top >= mp and lead >= mlead) or is_strong
                        if is_strong:
                            sn += 1
                            if pred == rec['actual']:
                                sh += 1
                        if is_med:
                            mn += 1
                            if pred == rec['actual']:
                                mh += 1
                    s_acc = 100*sh/sn if sn else 0
                    m_acc = 100*mh/mn if mn else 0
                    s_cov = 100*sn/len(spf_records)
                    m_cov = 100*mn/len(spf_records)
                    if best is None or (s_acc > best[0] and s_cov >= 20):
                        best = (s_acc, sp, slead, s_cov, m_acc, mp, mlead, m_cov)
                    if sp == 0.60 and slead == 0.10 and mp == 0.54 and mlead == 0.08:
                        print(f"{sp:>8.2f}{slead:>12.2f}{s_acc:>7.2f}%{s_cov:>7.1f}%{mp:>8.2f}{mlead:>10.3f}{m_acc:>7.2f}%{m_cov:>7.1f}%  (当前)")
                    if (sp, slead) in ((0.56, 0.10), (0.65, 0.10), (0.70, 0.10)) and mp == 0.54 and mlead == 0.08:
                        print(f"{sp:>8.2f}{slead:>12.2f}{s_acc:>7.2f}%{s_cov:>7.1f}%{mp:>8.2f}{mlead:>10.3f}{m_acc:>7.2f}%{m_cov:>7.1f}%")
    print(f"\n经验最优候选: strong(p>={best[1]}, lead>={best[2]}) acc={best[0]:.2f}% cov={best[3]:.1f}% | med acc={best[4]:.2f}% cov={best[7]:.1f}%")


if __name__ == '__main__':
    main()
