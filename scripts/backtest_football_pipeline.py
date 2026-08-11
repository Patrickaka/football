#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用真实足球管线(predict_scores)回测线上导出样本
=============================================
直接用导出记录里的 odds_snapshot(euro/asian/total 三档)喂给生产
predict_scores(negative_binomial + ensemble)，A/B 测试平局校准/去集中的效果。
无 team_strength(导出不含)，其余与生产一致。
"""
import os, sys, json, math, argparse
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.football as fb


def res(h, a):
    return 'H' if h > a else ('A' if h < a else 'D')


def load(path):
    d = json.load(open(path, encoding='utf-8'))
    recs = d['records'] if isinstance(d, dict) else d
    out = []
    for r in recs:
        if not (r.get('settled') and r.get('actual_score')):
            continue
        os_ = r.get('odds_snapshot') or {}
        if not (os_.get('euro') and os_.get('asian') and os_.get('total')):
            continue
        try:
            h, a = map(int, r['actual_score'].split('-'))
        except Exception:
            continue
        out.append({'lg': r.get('league'), 'os': os_, 'ah': h, 'aa': a})
    return out


def run(data, enable_draw_calibration=True):
    s1 = s3 = g1 = g2 = x1 = one_one = 0
    mean_pred = 0.0
    n = 0
    for m in data:
        os_ = m['os']
        prof = fb.resolve_league_profile(m['lg'])
        try:
            candidates, lam_h, lam_a, meta = fb.predict_scores(
                os_['asian'], os_['euro'], os_['total'],
                team_strength=None, league_profile=prof,
                model_type='negative_binomial',
                enable_draw_calibration=enable_draw_calibration,
                enable_calibration=True, calibration_method='platt',
                enable_ensemble=True, ensemble_size=2,
            )
        except Exception as e:
            continue
        mat = dict(candidates)
        ranked = sorted(mat.items(), key=lambda x: -x[1])
        ak = (m['ah'], m['aa'])
        top = [k for k, _ in ranked]
        if top[0] == ak:
            s1 += 1
        if ak in top[:3]:
            s3 += 1
        if top[0] == (1, 1):
            one_one += 1
        # goals
        g = defaultdict(float)
        for (hh, aa), p in mat.items():
            t = hh + aa
            g['7+' if t >= 7 else str(t)] += p
        gk = '7+' if (m['ah'] + m['aa']) >= 7 else str(m['ah'] + m['aa'])
        gr = [k for k, _ in sorted(g.items(), key=lambda x: -x[1])]
        if gr and gr[0] == gk:
            g1 += 1
        if gk in gr[:2]:
            g2 += 1
        # 1x2 from margins
        ph = sum(p for (hh, aa), p in mat.items() if hh > aa)
        pd = sum(p for (hh, aa), p in mat.items() if hh == aa)
        pa = sum(p for (hh, aa), p in mat.items() if hh < aa)
        pred = max((('H', ph), ('D', pd), ('A', pa)), key=lambda x: x[1])[0]
        if pred == res(m['ah'], m['aa']):
            x1 += 1
        mean_pred += sum((hh + aa) * p for (hh, aa), p in mat.items())
        n += 1
    return dict(n=n, s1=100 * s1 / n, s3=100 * s3 / n, g1=100 * g1 / n,
                g2=100 * g2 / n, x1=100 * x1 / n, one_one=100 * one_one / n,
                mean=mean_pred / n)


def show(tag, r):
    print(f"{tag:<20} n={r['n']:3d} | 比分T1={r['s1']:5.2f}% T3={r['s3']:5.2f}% | "
          f"进球T1={r['g1']:5.2f}% T2={r['g2']:5.2f}% | 1X2={r['x1']:5.2f}% | "
          f"1-1占={r['one_one']:4.1f}% 均值={r['mean']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    args = ap.parse_args()
    data = load(args.path)
    real_mean = sum(m['ah'] + m['aa'] for m in data) / len(data)
    real_x1 = 100 * sum(1 for m in data if True) and None
    from collections import Counter as C
    rx = C(res(m['ah'], m['aa']) for m in data)
    print(f"样本 {len(data)} 场 | 真实均值={real_mean:.3f} | 真实1X2分布={dict(rx)}\n")
    show("含平局校准(现网)", run(data, enable_draw_calibration=True))
    show("关闭平局校准", run(data, enable_draw_calibration=False))


if __name__ == '__main__':
    main()
