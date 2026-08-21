#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
球队攻防信号是否提升准确率 —— 五大联赛CSV(真实进球)按时间序回测。
核心问题：把「近期攻防强度」融入 λ，能否在真实比分/进球上超过「纯市场(1X2+OU)」？
  A: 纯市场 λ (euro 1X2 + OU 隐含总进球)
  B: 纯球队攻防 λ (近6场 attack/defense)
  C: A×(1-w) + B×w 混合(扫 w)
指标：比分 Top1/Top3、进球 Top2、大小球方向、1X2、LogLoss。严格只用赛前历史。
"""
import os, sys, csv, math
from collections import defaultdict, deque, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.beidan import poisson_pmf, build_dixon_coles_matrix, implied_total_from_ou

CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
DIV = {'E0': '英超', 'D1': '德甲', 'SP1': '西甲', 'I1': '意甲', 'F1': '法甲'}
LEAGUE_AVG = {'英超': 2.8, '德甲': 3.1, '西甲': 2.7, '意甲': 2.5, '法甲': 2.6}
FORM_N = 6
MAXG = 7


def sf(v, d=None):
    try:
        return float(v) if v not in (None, '') else d
    except (ValueError, TypeError):
        return d


def parse_date(s):
    for fmt in ('%d/%m/%Y', '%d/%m/%y'):
        try:
            import datetime
            return datetime.datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    return None


def load_sorted():
    rows = []
    for fn in sorted(os.listdir(CSV_DIR)):
        if not (fn.endswith('.csv') and fn[:2] in DIV):
            continue
        league = DIV[fn[:2]]
        with open(os.path.join(CSV_DIR, fn), encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                fthg, ftag = sf(r.get('FTHG')), sf(r.get('FTAG'))
                ah, ad, aa = sf(r.get('AvgH')), sf(r.get('AvgD')), sf(r.get('AvgA'))
                if None in (fthg, ftag, ah, ad, aa):
                    continue
                dt = parse_date(r.get('Date'))
                rows.append({
                    'dt': dt, 'league': league, 'home': r.get('HomeTeam'), 'away': r.get('AwayTeam'),
                    'hg': int(fthg), 'ag': int(ftag), 'oh': ah, 'od': ad, 'oa': aa,
                    'over': sf(r.get('Avg>2.5')), 'under': sf(r.get('Avg<2.5')),
                })
    rows.sort(key=lambda x: (x['dt'] or __import__('datetime').datetime.min))
    return rows


def market_lambdas(m):
    ph, pd, pa = 1 / m['oh'], 1 / m['od'], 1 / m['oa']
    s = ph + pd + pa
    ph, pd, pa = ph / s, pd / s, pa / s
    avg = LEAGUE_AVG.get(m['league'], 2.7)
    ou = implied_total_from_ou(m['over'], m['under']) if (m['over'] and m['under']) else None
    target = 0.6 * ou + 0.4 * avg if ou else avg
    target = max(1.8, min(3.8, target))
    sup = (ph - pa) / (ph + pd + pa)
    lh = max(0.05, target * (0.5 + sup * 0.45))
    la = max(0.05, target * (0.5 - sup * 0.45))
    return lh, la, target


def team_lambdas(m, hist, target):
    """近6场：主队主场进攻×客队客场防守。归一到 target(与市场同总进球)。"""
    h, a = m['home'], m['away']
    hh = hist['home_gf'][h]; ha = hist['home_ga'][h]
    ag = hist['away_gf'][a]; aa = hist['away_ga'][a]
    if len(hh) < 3 or len(ag) < 3:
        return None
    lg_avg = LEAGUE_AVG.get(m['league'], 2.7) / 2.0
    atk_h = (sum(hh) / len(hh)) / lg_avg
    def_a = (sum(aa) / len(aa)) / lg_avg
    atk_a = (sum(ag) / len(ag)) / lg_avg
    def_h = (sum(ha) / len(ha)) / lg_avg
    lam_h = max(0.05, atk_h * def_a * lg_avg * 1.06)
    lam_a = max(0.05, atk_a * def_h * lg_avg)
    scale = target / max(lam_h + lam_a, 0.1)
    return lam_h * scale, lam_a * scale


def gkey(t):
    return '7+' if t >= 7 else str(t)


def evaluate(rows, w):
    """w=0 纯市场; w=1 纯球队; 0<w<1 混合。只在有球队历史时计入(公平对比)。"""
    hist = {'home_gf': defaultdict(lambda: deque(maxlen=FORM_N)),
            'home_ga': defaultdict(lambda: deque(maxlen=FORM_N)),
            'away_gf': defaultdict(lambda: deque(maxlen=FORM_N)),
            'away_ga': defaultdict(lambda: deque(maxlen=FORM_N))}
    s1 = s3 = g2 = ou_hit = x1 = n = 0
    sll = 0.0
    for m in rows:
        ml = market_lambdas(m)
        tl = team_lambdas(m, hist, ml[2])
        # 更新历史（赛后）——放在评估后
        if tl is not None:
            lh = ml[0] * (1 - w) + tl[0] * w
            la = ml[1] * (1 - w) + tl[1] * w
            # 混合后重新归一到市场 target，保证总进球一致（只比"分配"差异）
            sc = ml[2] / max(lh + la, 0.1)
            lh, la = lh * sc, la * sc
            mat = build_dixon_coles_matrix(lh, la, rho=0.0)
            ak = (m['hg'], m['ag'])
            ranked = sorted(mat.items(), key=lambda x: -x[1])
            top = [k for k, _ in ranked]
            if top[0] == ak:
                s1 += 1
            if ak in top[:3]:
                s3 += 1
            sll += -math.log(max(mat.get(ak, 0), 1e-9))
            g = defaultdict(float)
            for (h, a), p in mat.items():
                g[gkey(h + a)] += p
            gk = gkey(m['hg'] + m['ag'])
            gr = [k for k, _ in sorted(g.items(), key=lambda x: -x[1])]
            if gk in gr[:2]:
                g2 += 1
            p_over = sum(p for (h, a), p in mat.items() if h + a >= 3)
            if (p_over >= 0.5) == ((m['hg'] + m['ag']) >= 3):
                ou_hit += 1
            ph = sum(p for (h, a), p in mat.items() if h > a)
            pd = sum(p for (h, a), p in mat.items() if h == a)
            pa = sum(p for (h, a), p in mat.items() if h < a)
            pred = max((('H', ph), ('D', pd), ('A', pa)), key=lambda x: x[1])[0]
            act = 'H' if m['hg'] > m['ag'] else ('A' if m['hg'] < m['ag'] else 'D')
            if pred == act:
                x1 += 1
            n += 1
        # 赛后更新历史
        hist['home_gf'][m['home']].append(m['hg']); hist['home_ga'][m['home']].append(m['ag'])
        hist['away_gf'][m['away']].append(m['ag']); hist['away_ga'][m['away']].append(m['hg'])
    return dict(n=n, s1=100*s1/n, s3=100*s3/n, g2=100*g2/n, ou=100*ou_hit/n, x1=100*x1/n, sll=sll/n)


def main():
    rows = load_sorted()
    print(f"按时间序样本: {len(rows)} 场 (含日期 {sum(1 for r in rows if r['dt'])})\n")
    print(f"{'w(球队权重)':>10}{'样本':>6}{'比分T1':>8}{'比分T3':>8}{'进球T2':>8}{'大小球':>8}{'1X2':>8}{'比分LL':>9}")
    for w in [0.0, 0.15, 0.3, 0.5, 0.7, 1.0]:
        r = evaluate(rows, w)
        print(f"{w:>10.2f}{r['n']:>6}{r['s1']:>7.2f}%{r['s3']:>7.2f}%{r['g2']:>7.2f}%{r['ou']:>7.2f}%{r['x1']:>7.2f}%{r['sll']:>9.4f}")
    print("\nw=0 纯市场基线；若某 w>0 在多项指标稳健超过 w=0，则球队攻防有增益，值得重构融入。")


if __name__ == '__main__':
    main()
