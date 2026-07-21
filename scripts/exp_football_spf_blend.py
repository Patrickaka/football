#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验：足球 SPF/RQSPF 融合权重 LOTTERY_OFFICIAL_ODDS_WEIGHT 是否该提高
市场赔率是有效信号，模型成分若弱于市场，提高市场权重应提升命中率。
扫描不同 market_weight 下的全量/强推荐 命中率与覆盖率。
"""
import os
import csv
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.football as fb
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
        if not (fn.endswith('.csv') and fn[:2] in DIV_LEAGUE):
            continue
        div = fn[:2]; league = DIV_LEAGUE[div]
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
    """用与足球同源的 Dixon-Coles 构造比分候选分布"""
    probs = {'胜': 1/h, '平': 1/d, '负': 1/a}
    tot = sum(probs.values())
    probs = {k: v/tot for k, v in probs.items()}
    target = beidan.LEAGUE_PROFILES.get(league, {'avg_goals': 2.6})['avg_goals']
    lam_h, lam_a = beidan.euro_implied_lambdas(probs['胜'], probs['平'], probs['负'], target)
    dc = beidan.build_dixon_coles_matrix(lam_h, lam_a)
    return [(k, v) for k, v in dc.items()]


def marginalize(candidates):
    spf = {'胜': 0.0, '平': 0.0, '负': 0.0}
    for (h, a), p in candidates:
        if h > a:
            spf['胜'] += p
        elif h < a:
            spf['负'] += p
        else:
            spf['平'] += p
    return spf


def actual_spf(m):
    if m['hg'] > m['ag']:
        return '胜'
    if m['hg'] < m['ag']:
        return '负'
    return '平'


def main():
    matches = load_matches()
    print(f"样本: {len(matches)} 场")

    recs = []
    for m in matches:
        market_spf = fb._lottery_odds_probabilities({'胜': m['h'], '平': m['d'], '负': m['a']}, ('胜', '平', '负'))
        if not market_spf:
            continue
        cands = candidates_from_odds(m['h'], m['d'], m['a'], m['league'])
        model_spf = marginalize(cands)
        recs.append({'market': market_spf, 'model': model_spf, 'actual': actual_spf(m)})

    print(f"有效样本: {len(recs)}\n")
    print(f"{'mkt_w':>7}{'full_acc':>10}{'strong_acc':>12}{'str_cov':>9}{'med+_acc':>11}{'med+_cov':>10}")
    for w in (0.0, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0):
        fh = fn_ = sh = sn = mh = mn = 0
        for rec in recs:
            blended = fb._blend_lottery_probabilities(rec['model'], rec['market'], market_weight=w)
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
        print(f"{w:>7.2f}{100*fh/fn_:>9.2f}%{100*sh/sn:>11.2f}%{100*sn/len(recs):>8.1f}%{100*mh/mn:>10.2f}%{100*mn/len(recs):>9.1f}%")
    print(f"\n当前线上 mkt_w=0.40。若更高 w 全量/强推荐命中率更高且不降覆盖，则提高权重有效。")


if __name__ == '__main__':
    main()
