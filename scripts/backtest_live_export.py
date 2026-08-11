#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用「真实线上导出」样本回测比分/进球预测参数
==========================================
数据来源：Downloads 下的 football-predictions-*.json 导出（已结算记录），
每条含 odds_snapshot.euro.close(1X2隐含概率) + total.implied_total(大小球隐含总进球)
+ asian.handicap + actual_score，覆盖世界杯/芬超/瑞典超/K1/挪超等真实投注联赛。

目的：忠实复现生产北单式管线，先复现基线，再扫描修复「1-1 过度集中 + 总进球系统性低估」。
纯离线、纯数学，无网络。
"""
import os, sys, json, math, argparse
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.beidan import poisson_pmf, LEAGUE_PROFILES

MAXG = 7


def load(path):
    d = json.load(open(path, encoding='utf-8'))
    recs = d['records'] if isinstance(d, dict) else d
    out = []
    for r in recs:
        if not (r.get('settled') and r.get('actual_score')):
            continue
        try:
            h, a = map(int, r['actual_score'].split('-'))
        except Exception:
            continue
        try:
            e = r['odds_snapshot']['euro']['close']
            ph, pd, pa = float(e['home']), float(e['draw']), float(e['away'])
            s = ph + pd + pa
            ph, pd, pa = ph / s, pd / s, pa / s
        except Exception:
            continue
        t = r['odds_snapshot'].get('total', {})
        imp = t.get('implied_total') or t.get('close_line')
        try:
            imp = float(imp) if imp else None
        except Exception:
            imp = None
        out.append({
            'lg': r.get('league'), 'ph': ph, 'pd': pd, 'pa': pa,
            'imp': imp, 'ah': h, 'aa': a, 'atot': h + a,
        })
    return out


def dc_matrix(lam_h, lam_a, rho, max_goals=MAXG):
    probs = {}
    ssum = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            base = poisson_pmf(h, lam_h) * poisson_pmf(a, lam_a)
            if h == 0 and a == 0:
                tau = 1 - lam_h * lam_a * rho
            elif h == 0 and a == 1:
                tau = 1 + lam_h * rho
            elif h == 1 and a == 0:
                tau = 1 + lam_a * rho
            elif h == 1 and a == 1:
                tau = 1 - rho
            else:
                tau = 1.0
            p = max(base * tau, 0.0)
            probs[(h, a)] = p
            ssum += p
    return {k: v / ssum for k, v in probs.items()} if ssum > 0 else probs


def predict(m, blend, split, rho, tmin, tmax, bias):
    """忠实复现北单式：target=blend*OU隐含+(1-blend)*联赛均值, 再*bias, clamp;
    supremacy 由 1X2 概率算, split 分配主客 λ; DC 修正矩阵。"""
    avg = LEAGUE_PROFILES.get(m['lg'], {'avg_goals': 2.6})['avg_goals']
    ou = m['imp']
    target = (blend * ou + (1 - blend) * avg) if ou else avg
    target *= bias
    target = max(tmin, min(tmax, target))
    sup = (m['ph'] - m['pa']) / (m['ph'] + m['pd'] + m['pa'] + 1e-9)
    lam_h = max(0.05, target * (0.5 + sup * split))
    lam_a = max(0.05, target * (0.5 - sup * split))
    return dc_matrix(lam_h, lam_a, rho)


def gkey(t):
    return '7+' if t >= 7 else str(t)


def evaluate(data, blend=0.6, split=0.45, rho=-0.08, tmin=1.8, tmax=3.6, bias=1.0):
    s1 = s3 = s5 = g1 = g2 = ou_hit = 0
    sll = gll = 0.0
    mean_pred = 0.0
    one_one = 0
    n = 0
    for m in data:
        mat = predict(m, blend, split, rho, tmin, tmax, bias)
        ranked = sorted(mat.items(), key=lambda x: -x[1])
        ak = (m['ah'], m['aa'])
        top = [k for k, _ in ranked]
        if top[0] == ak:
            s1 += 1
        if ak in top[:3]:
            s3 += 1
        if ak in top[:5]:
            s5 += 1
        sll += -math.log(max(mat.get(ak, 0), 1e-12))
        if top[0] == (1, 1):
            one_one += 1
        # goals
        g = defaultdict(float)
        for (h, a), p in mat.items():
            g[gkey(h + a)] += p
        gk = gkey(m['atot'])
        gr = [k for k, _ in sorted(g.items(), key=lambda x: -x[1])]
        if gr[0] == gk:
            g1 += 1
        if gk in gr[:2]:
            g2 += 1
        gll += -math.log(max(g.get(gk, 0), 1e-12))
        mean_pred += sum((h + a) * p for (h, a), p in mat.items())
        # over/under 2.5
        p_over = sum(p for (h, a), p in mat.items() if h + a >= 3)
        pred_over = p_over >= 0.5
        act_over = m['atot'] >= 3
        if pred_over == act_over:
            ou_hit += 1
        n += 1
    return dict(n=n, s1=100 * s1 / n, s3=100 * s3 / n, s5=100 * s5 / n,
               g1=100 * g1 / n, g2=100 * g2 / n, ou=100 * ou_hit / n,
               sll=sll / n, gll=gll / n, mean_pred=mean_pred / n,
               one_one=100 * one_one / n)


def show(tag, r):
    print(f"{tag:<26} 比分 T1={r['s1']:5.2f}% T3={r['s3']:5.2f}% T5={r['s5']:5.2f}% | "
          f"进球 T1={r['g1']:5.2f}% T2={r['g2']:5.2f}% | 大小球={r['ou']:5.2f}% | "
          f"均值={r['mean_pred']:.2f} 1-1占={r['one_one']:4.1f}% | sLL={r['sll']:.3f} gLL={r['gll']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    args = ap.parse_args()
    data = load(args.path)
    print(f"可用结算样本: {len(data)} 场")
    真实均值 = sum(m['atot'] for m in data) / len(data)
    print(f"真实平均总进球: {真实均值:.3f}\n")

    print("=" * 120)
    print("① 基线复现 (blend=0.6 split=0.45 rho=-0.08 clamp[1.8,3.6] bias=1.0)")
    print("=" * 120)
    base = evaluate(data)
    show("baseline", base)

    print("\n" + "=" * 120)
    print("② 总进球低估修正: 提高大小球权重 blend / 放宽上限 / bias")
    print("=" * 120)
    for blend in [0.6, 0.8, 1.0]:
        for bias in [1.0, 1.08, 1.15]:
            r = evaluate(data, blend=blend, bias=bias, tmax=4.2)
            show(f"blend={blend} bias={bias} tmax=4.2", r)

    print("\n" + "=" * 120)
    print("③ rho 扫描 (抑制/放开 1-1、0-0)")
    print("=" * 120)
    for rho in [-0.15, -0.08, 0.0, 0.05, 0.10, 0.15]:
        r = evaluate(data, rho=rho, blend=1.0, bias=1.08, tmax=4.2)
        show(f"rho={rho:+.2f}", r)

    print("\n" + "=" * 120)
    print("④ split 扫描 (拉开主客强度)")
    print("=" * 120)
    for split in [0.35, 0.45, 0.55, 0.65]:
        r = evaluate(data, split=split, blend=1.0, bias=1.08, tmax=4.2)
        show(f"split={split}", r)

    print("\n" + "=" * 120)
    print("⑤ 组合网格搜索 (最大化 比分T3 + 进球T2)")
    print("=" * 120)
    best = None
    for blend in [0.8, 1.0]:
        for bias in [1.0, 1.05, 1.1, 1.15]:
            for split in [0.45, 0.55, 0.65]:
                for rho in [-0.05, 0.0, 0.05, 0.10]:
                    for tmax in [3.6, 4.2]:
                        r = evaluate(data, blend=blend, split=split, rho=rho, bias=bias, tmax=tmax)
                        score = r['s3'] + r['g2']
                        if best is None or score > best[0]:
                            best = (score, dict(blend=blend, bias=bias, split=split, rho=rho, tmax=tmax), r)
    print(f"最优参数: {best[1]}")
    show("BEST", best[2])
    print(f"\n对比基线: 比分T3 {base['s3']:.2f}%→{best[2]['s3']:.2f}%  "
          f"进球T2 {base['g2']:.2f}%→{best[2]['g2']:.2f}%  "
          f"大小球 {base['ou']:.2f}%→{best[2]['ou']:.2f}%  "
          f"均值 {base['mean_pred']:.2f}→{best[2]['mean_pred']:.2f}")


if __name__ == '__main__':
    main()
