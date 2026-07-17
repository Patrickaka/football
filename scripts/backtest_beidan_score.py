#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
北单 比分/总进球 专项回测 + 爆冷(upset)识别评估
====================================================
用真实 CSV（五大联赛 2744 场，含 1X2 欧赔 + 大小球 2.5 盘口）离线回测：
  1) 比分 Top1/Top3、总进球 Top1/Top2、LogLoss、Brier 基线
  2) 参数扫描：Dixon-Coles rho、λ 强度分配 split、大小球融合权重 blend、目标总进球区间
  3) 爆冷识别：按"庄家热门强度"分桶统计真实爆冷率，评估"爆冷比分候选"命中价值

只依赖标准库 + src.beidan 的 poisson/OU 辅助函数，参数全部显式传入以便扫描。
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
MAXG = 7


def sf(v, d=None):
    try:
        return float(v) if v not in (None, '') else d
    except (ValueError, TypeError):
        return d


def load_matches():
    rows = []
    for fn in sorted(os.listdir(CSV_DIR)):
        if not (fn.endswith('.csv') and fn[:2] in DIV_LEAGUE):
            continue
        league = DIV_LEAGUE[fn[:2]]
        with open(os.path.join(CSV_DIR, fn), encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                hg, ag = sf(r.get('FTHG')), sf(r.get('FTAG'))
                if hg is None or ag is None:
                    continue
                ah, ad, aa = sf(r.get('AvgH')), sf(r.get('AvgD')), sf(r.get('AvgA'))
                if not (ah and ad and aa):
                    continue
                rows.append({
                    'league': league, 'hg': int(hg), 'ag': int(ag),
                    'oh': ah, 'od': ad, 'oa': aa,
                    'over': sf(r.get('Avg>2.5')), 'under': sf(r.get('Avg<2.5')),
                })
    return rows


def norm_1x2(oh, od, oa):
    ph, pd, pa = 1.0 / oh, 1.0 / od, 1.0 / oa
    s = ph + pd + pa
    return ph / s, pd / s, pa / s


def build_matrix(lam_h, lam_a, rho):
    probs = {}
    ssum = 0.0
    for h in range(MAXG + 1):
        for a in range(MAXG + 1):
            base = beidan.poisson_pmf(h, lam_h) * beidan.poisson_pmf(a, lam_a)
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


def model(m, rho=-0.15, split=0.35, blend=0.4, tmin=1.8, tmax=3.6,
          draw_boost=1.0):
    """返回比分矩阵 + 总进球分布 + 1X2 概率。参数全部显式便于扫描。"""
    ph, pd, pa = norm_1x2(m['oh'], m['od'], m['oa'])
    pd *= draw_boost
    s = ph + pd + pa
    ph, pd, pa = ph / s, pd / s, pa / s

    prof = beidan.LEAGUE_PROFILES.get(m['league'], {'avg_goals': 2.6})
    avg = prof['avg_goals']
    ou = beidan.implied_total_from_ou(m['over'], m['under']) if (m['over'] and m['under']) else None
    target = blend * ou + (1 - blend) * avg if ou else avg
    target = max(tmin, min(tmax, target))

    sup = (ph - pa) / (ph + pd + pa + 1e-9)
    lam_h = max(0.05, target * (0.5 + sup * split))
    lam_a = max(0.05, target * (0.5 - sup * split))

    mat = build_matrix(lam_h, lam_a, rho)
    goals = defaultdict(float)
    for (h, a), p in mat.items():
        t = h + a
        goals['7+' if t >= 7 else str(t)] += p
    return mat, dict(goals), (ph, pd, pa)


def result_of(h, a):
    return '胜' if h > a else ('负' if h < a else '平')


def evaluate(matches, **kw):
    s_top1 = s_top3 = z_top1 = z_top2 = n = 0
    s_ll = z_ll = 0.0
    for m in matches:
        mat, goals, _ = model(m, **kw)
        ak = (m['hg'], m['ag'])
        ap = max(mat.get(ak, 0.0), 1e-9)
        s_ll += -math.log(ap)
        ranked = sorted(mat.items(), key=lambda x: -x[1])
        if ranked[0][0] == ak:
            s_top1 += 1
        if ak in [k for k, _ in ranked[:3]]:
            s_top3 += 1
        at = m['hg'] + m['ag']
        gk = '7+' if at >= 7 else str(at)
        gp = max(goals.get(gk, 0.0), 1e-9)
        z_ll += -math.log(gp)
        gr = sorted(goals.items(), key=lambda x: -x[1])
        if gr[0][0] == gk:
            z_top1 += 1
        if gk in [k for k, _ in gr[:2]]:
            z_top2 += 1
        n += 1
    return {
        's_top1': 100 * s_top1 / n, 's_top3': 100 * s_top3 / n, 's_ll': s_ll / n,
        'z_top1': 100 * z_top1 / n, 'z_top2': 100 * z_top2 / n, 'z_ll': z_ll / n,
        'n': n,
    }


def upset_analysis(matches, **kw):
    """按庄家热门强度分桶，统计真实爆冷率，并评估爆冷比分候选命中。

    定义：
      favorite = 1X2 中概率最高的结果
      upset(爆冷) = 实际结果 != favorite
      爆冷比分候选 = 与 favorite 相反方向上概率最高的比分
        （若热门是主胜，则候选取最可能的"平/负"比分；反之亦然）
    """
    buckets = defaultdict(lambda: {'n': 0, 'upset': 0})
    # 爆冷比分候选：仅在被判定"高爆冷风险"时评估其命中率
    cand_n = cand_hit = 0          # 候选在爆冷场次的命中（候选比分==实际比分）
    cand_dir_n = cand_dir_hit = 0  # 候选方向命中（预测的相反结果==实际结果）
    alert_n = alert_upset = 0      # 预警场次里真实爆冷占比

    for m in matches:
        mat, goals, (ph, pd, pa) = model(m, **kw)
        probs = {'胜': ph, '平': pd, '负': pa}
        fav = max(probs, key=probs.get)
        fav_p = probs[fav]
        actual = result_of(m['hg'], m['ag'])
        is_upset = actual != fav

        # 分桶：热门强度
        if fav_p >= 0.6:
            b = '强热门>=0.60'
        elif fav_p >= 0.5:
            b = '中热门0.50-0.60'
        elif fav_p >= 0.42:
            b = '弱热门0.42-0.50'
        else:
            b = '混沌<0.42'
        buckets[b]['n'] += 1
        buckets[b]['upset'] += 1 if is_upset else 0

        # 爆冷风险预警：热门不强(<0.55) 且 次热门与热门接近，或双冷概率之和高
        second = sorted(probs.values(), reverse=True)[1]
        upset_mass = 1 - fav_p                      # 非热门总概率
        gap = fav_p - second
        alert = (fav_p < 0.52) and (upset_mass >= 0.52) and (gap <= 0.16)
        if alert:
            alert_n += 1
            alert_upset += 1 if is_upset else 0

            # 相反方向：热门是"胜"→候选取 平+负 中最可能比分；"负"→候选取 胜+平；"平"→取 胜/负中更可能一侧
            if fav == '胜':
                allow = {'平', '负'}
            elif fav == '负':
                allow = {'胜', '平'}
            else:
                allow = {'胜', '负'}
            cand = None
            for (h, a), p in sorted(mat.items(), key=lambda x: -x[1]):
                if result_of(h, a) in allow:
                    cand = (h, a)
                    break
            if cand and is_upset:
                cand_n += 1
                if cand == (m['hg'], m['ag']):
                    cand_hit += 1
                cand_dir_n += 1
                if result_of(*cand) == actual:
                    cand_dir_hit += 1

    print('  按热门强度分桶的真实爆冷率：')
    order = ['强热门>=0.60', '中热门0.50-0.60', '弱热门0.42-0.50', '混沌<0.42']
    for b in order:
        d = buckets.get(b)
        if not d or d['n'] == 0:
            continue
        print(f'    {b:16s}: 爆冷 {100*d["upset"]/d["n"]:.1f}%  ({d["upset"]}/{d["n"]})')
    if alert_n:
        print(f'  爆冷预警触发 {alert_n} 场，其中真实爆冷 {100*alert_upset/alert_n:.1f}% '
              f'(vs 全局 {100*sum(x["upset"] for x in buckets.values())/sum(x["n"] for x in buckets.values()):.1f}%)')
    if cand_dir_n:
        print(f'  爆冷方向候选命中率(预警且真爆冷时): {100*cand_dir_hit/cand_dir_n:.1f}% ({cand_dir_hit}/{cand_dir_n})')
    if cand_n:
        print(f'  爆冷比分精确命中率(预警且真爆冷时): {100*cand_hit/cand_n:.1f}% ({cand_hit}/{cand_n})')


def main():
    matches = load_matches()
    print(f'样本: {len(matches)} 场\n')

    print('=' * 72)
    print('当前基线参数 (rho=-0.15, split=0.35, blend=0.4)')
    print('=' * 72)
    base = evaluate(matches, rho=-0.15, split=0.35, blend=0.4)
    print(f"  比分 : Top1={base['s_top1']:.2f}%  Top3={base['s_top3']:.2f}%  LL={base['s_ll']:.4f}")
    print(f"  总进球: Top1={base['z_top1']:.2f}%  Top2={base['z_top2']:.2f}%  LL={base['z_ll']:.4f}\n")

    print('=' * 72)
    print('参数扫描 — split(强度分配) × rho(DC低分修正)')
    print('=' * 72)
    print(f"  {'split':>6} {'rho':>6} | {'比分T1':>7} {'比分T3':>7} {'球T1':>7} {'球T2':>7} {'比分LL':>8} {'球LL':>8}")
    best = None
    for split in [0.30, 0.35, 0.40, 0.45, 0.50]:
        for rho in [-0.08, -0.12, -0.15, -0.20]:
            r = evaluate(matches, rho=rho, split=split, blend=0.4)
            score = r['s_top3'] + r['z_top2']  # 综合可用性指标
            flag = ''
            if best is None or score > best[0]:
                best = (score, split, rho, r)
                flag = ' <--'
            print(f"  {split:>6.2f} {rho:>6.2f} | {r['s_top1']:>6.2f}% {r['s_top3']:>6.2f}% "
                  f"{r['z_top1']:>6.2f}% {r['z_top2']:>6.2f}% {r['s_ll']:>8.4f} {r['z_ll']:>8.4f}{flag}")
    print(f"\n  最优组合: split={best[1]} rho={best[2]} "
          f"(比分T3={best[3]['s_top3']:.2f}% 球T2={best[3]['z_top2']:.2f}%)")

    print('\n' + '=' * 72)
    print('参数扫描 — blend(大小球融合权重)')
    print('=' * 72)
    for blend in [0.3, 0.4, 0.5, 0.6]:
        r = evaluate(matches, rho=-0.15, split=0.35, blend=blend)
        print(f"  blend={blend:.1f} | 比分T1={r['s_top1']:.2f}% T3={r['s_top3']:.2f}% "
              f"球T1={r['z_top1']:.2f}% T2={r['z_top2']:.2f}% 球LL={r['z_ll']:.4f}")

    print('\n' + '=' * 72)
    print('爆冷(upset)识别评估 — 基线参数')
    print('=' * 72)
    upset_analysis(matches, rho=-0.15, split=0.35, blend=0.4)


if __name__ == '__main__':
    main()
