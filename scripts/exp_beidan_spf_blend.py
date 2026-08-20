#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验：北单 SPF 是否该把「泊松-Dixon-Coles 边缘化 1X2」融进推荐
对比不同融合权重 w（poisson 占比）下的全量/强推荐 命中率
"""
import os
import csv
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.beidan as beidan

CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
DIV_LEAGUE = {'E0': '英超', 'D1': '德甲', 'SP1': '西甲', 'I1': '意甲', 'F1': '法甲'}


def safe_float(v, default=None):
    try:
        return float(v) if v not in (None, '') else default
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
                hg = safe_float(r.get('FTHG')); ag = safe_float(r.get('FTAG'))
                if hg is None or ag is None:
                    continue
                ah, ad, aa = safe_float(r.get('AvgH')), safe_float(r.get('AvgD')), safe_float(r.get('AvgA'))
                if not (ah and ad and aa):
                    continue
                mid = f"{div}_{r['Date']}_{r['HomeTeam']}_{r['AwayTeam']}"
                fake_ouzhi._store[mid] = {'home': ah, 'draw': ad, 'away': aa}
                rows.append({'mid': mid, 'league': league, 'hg': int(hg), 'ag': int(ag)})
    return rows


def marginal_1x2(score_probs):
    d = {'胜': 0.0, '平': 0.0, '负': 0.0}
    for (h, a), p in score_probs.items():
        if h > a:
            d['胜'] += p
        elif h < a:
            d['负'] += p
        else:
            d['平'] += p
    return d


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

    # 预计算每场：市场 model_probs（analyze_spf 返回） + 泊松边缘 1X2
    recs = []
    for m in matches:
        sr = beidan.analyze_spf({'id': m['mid'], 'league': m['league'], 'home': '', 'away': '', 'num': '', 'time': '', 'handicap': 0})
        if 'probabilities' not in sr:
            continue
        market_probs = sr['probabilities']
        # 用同一组 1X2 概率驱动泊松（与 analyze_spf 内部一致）
        ph = market_probs.get('胜', 1/3); pd_ = market_probs.get('平', 1/3); pa = market_probs.get('负', 1/3)
        sp = beidan.predict_scores_by_poisson(ph, pd_, pa, league=m['league'], handicap=0,
                                              total_over_odds=None, total_under_odds=None, use_dc=True)
        poisson_1x2 = marginal_1x2(sp['score_probs'])
        recs.append({'market': market_probs, 'poisson': poisson_1x2, 'actual': actual_spf(m)})

    print(f"有效样本: {len(recs)}\n")
    print(f"{'w_poisson':>10}{'full_acc':>10}{'strong_acc':>12}{'strong_cov':>12}{'med+_acc':>11}{'med+_cov':>11}")
    for w in (0.0, 0.10, 0.15, 0.20, 0.25, 0.30):
        fh = fn_ = sh = sn = mh = mn = 0
        for rec in recs:
            mk = rec['market']; ps = rec['poisson']
            blended = {k: (1-w)*mk.get(k, 0) + w*ps.get(k, 0) for k in ('胜', '平', '负')}
            tot = sum(blended.values()) or 1
            blended = {k: v/tot for k, v in blended.items()}
            ranked = sorted(blended.items(), key=lambda x: -x[1])
            top, lead = ranked[0][1], ranked[0][1] - ranked[1][1]
            pred = ranked[0][0]
            fn_ += 1
            if pred == rec['actual']:
                fh += 1
            is_strong = top >= 0.50 and lead >= 0.08
            is_med = top >= 0.43 and lead >= 0.045
            if is_strong:
                sn += 1
                if pred == rec['actual']:
                    sh += 1
            if is_med:
                mn += 1
                if pred == rec['actual']:
                    mh += 1
        print(f"{w:>10.2f}{100*fh/fn_:>9.2f}%{100*sh/sn:>11.2f}%{100*sn/len(recs):>11.1f}%{100*mh/mn:>10.2f}%{100*mn/len(recs):>10.1f}%")
    print("\n当前线上 w=0.0（纯市场+校准）。若某 w>0 强推荐命中率明显更高且覆盖不降，则融合有效。")


if __name__ == '__main__':
    main()
