#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
进球数 / 比分 离线回测（北单 + 足球模块数学基线）

数据来源：data/ 下的历史 CSV（football-data.co.uk 格式，含 FTHG/FTAG 与多档赔率）
不使用任何网络请求，纯离线评估。

评估维度：
- 总进球数分布 {0..7+}：LogLoss / Brier / Top2 命中
- 精确比分分布 (h,a)：LogLoss / Brier / Top1 / Top3 命中

对比方案：
- zjq_current : 北单当前 analyze_zjq 数学（联赛均值 + 主客0.05分配）
- zjq_improved: 大小球隐含总进球 + 与比分一致的强度分配
- score_current: 北单当前 predict_scores_by_poisson（联赛均值 + 0.35分配）
- score_improved: 大小球隐含总进球 + 0.35分配
"""

import os
import csv
import json
import math
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.beidan import poisson_pmf, euro_implied_lambdas, LEAGUE_PROFILES

CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')

DIV_LEAGUE = {
    'E0': '英超', 'D1': '德甲', 'SP1': '西甲', 'I1': '意甲', 'F1': '法甲',
}

# 北单当前 zjq 的主客分配系数（待优化）
ZJQ_CURRENT_SPLIT = 0.05
# 与比分一致的强度分配系数
SCORE_SPLIT = 0.35

MAX_GOALS = 7


def safe_float(v, default=None):
    try:
        if v is None or v == '':
            return default
        return float(v)
    except (ValueError, TypeError):
        return default


def implied_probs_from_odds(h, d, a):
    """1/odds 归一化隐含概率"""
    if not h or not d or not a:
        return None
    ph, pd, pa = 1.0 / h, 1.0 / d, 1.0 / a
    tot = ph + pd + pa
    if tot <= 0:
        return None
    return ph / tot, pd / tot, pa / tot


def implied_total_from_ou(over_odds, under_odds):
    """由大小球赔率反推隐含总进球数 λ_total（Poisson 假设）"""
    if not over_odds or not under_odds:
        return None
    po = 1.0 / over_odds
    pu = 1.0 / under_odds
    tot = po + pu
    if tot <= 0:
        return None
    p_over = po / tot

    # 求解 λ 使得 P(Poisson(λ) > 2.5) = p_over
    lo, hi = 0.5, 9.0
    for _ in range(60):
        mid = (lo + hi) / 2
        # P(total <= 2) = e^-λ (1 + λ + λ^2/2)
        p_le2 = math.exp(-mid) * (1 + mid + mid * mid / 2.0)
        p_over_mid = 1.0 - p_le2
        if p_over_mid < p_over:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def zjq_distribution(avg_goals, home_prob_norm, away_prob_norm, split):
    """北单总进球卷积（与 analyze_zjq 一致），split 为主客分配系数"""
    mu1 = avg_goals * (0.5 + split * (home_prob_norm - away_prob_norm))
    mu2 = avg_goals * (0.5 - split * (home_prob_norm - away_prob_norm))
    mu1 = max(0.1, mu1)
    mu2 = max(0.1, mu2)
    zjq = {}
    for n in range(0, 8):
        if n == 0:
            prob = math.exp(-mu1 - mu2)
        elif n == 1:
            prob = (mu1 + mu2) * math.exp(-mu1 - mu2)
        elif n == 2:
            prob = (mu1 ** 2 + 2 * mu1 * mu2 + mu2 ** 2) * math.exp(-mu1 - mu2) / 2
        elif n == 3:
            prob = (mu1 ** 3 + 3 * mu1 ** 2 * mu2 + 3 * mu1 * mu2 ** 2 + mu2 ** 3) * math.exp(-mu1 - mu2) / 6
        elif n == 4:
            prob = (mu1 ** 4 + 4 * mu1 ** 3 * mu2 + 6 * mu1 ** 2 * mu2 ** 2 + 4 * mu1 * mu2 ** 3 + mu2 ** 4) * math.exp(-mu1 - mu2) / 24
        elif n == 5:
            prob = math.exp(-mu1 - mu2) * sum(mu1 ** (5 - i) * mu2 ** i / (math.factorial(5 - i) * math.factorial(i)) for i in range(6))
        elif n == 6:
            prob = math.exp(-mu1 - mu2) * sum(mu1 ** (6 - i) * mu2 ** i / (math.factorial(6 - i) * math.factorial(i)) for i in range(7))
        else:
            prob = max(0, 1 - sum(zjq.values()))
        zjq[str(n)] = prob if n < 7 else prob
    zjq['7+'] = max(0, 1 - sum(zjq.get(str(i), 0) for i in range(7)))
    total = sum(zjq.values())
    if total > 0:
        zjq = {k: v / total for k, v in zjq.items()}
    return zjq, mu1, mu2


def score_distribution(lam_home, lam_away):
    """独立泊松比分联合分布"""
    probs = {}
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            p = poisson_pmf(h, lam_home) * poisson_pmf(a, lam_away)
            if p > 1e-9:
                probs[(h, a)] = p
    total = sum(probs.values()) + 1e-9
    return {k: v / total for k, v in probs.items()}


def dc_tau(h, a, lam_h, lam_a, rho):
    """Dixon-Coles 低比分相关修正因子"""
    if h == 0 and a == 0:
        return 1 - lam_h * lam_a * rho
    if h == 0 and a == 1:
        return 1 + lam_h * rho
    if h == 1 and a == 0:
        return 1 + lam_a * rho
    if h == 1 and a == 1:
        return 1 - rho
    return 1.0


def score_distribution_dc(lam_home, lam_away, rho=-0.15):
    """Dixon-Coles 修正比分联合分布"""
    probs = {}
    raw_sum = 0.0
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            base = poisson_pmf(h, lam_home) * poisson_pmf(a, lam_away)
            tau = dc_tau(h, a, lam_home, lam_away, rho)
            p = base * tau
            probs[(h, a)] = p
            raw_sum += p
    if raw_sum <= 0:
        return score_distribution(lam_home, lam_away)
    return {k: v / raw_sum for k, v in probs.items()}


def aggregate_goal_from_score(score_dist):
    """从比分分布聚合总进球分布（桶 0..6, 7+）"""
    goal = defaultdict(float)
    for (h, a), p in score_dist.items():
        t = h + a
        key = '7+' if t >= 7 else str(t)
        goal[key] += p
    return dict(goal)


def logloss_cat(pred_dist, actual_key, all_keys):
    """多分类 LogLoss（对未出现类别做极小概率保护）"""
    p = max(pred_dist.get(actual_key, 0.0), 1e-9)
    return -math.log(p)


def brier_cat(pred_dist, actual_key, all_keys):
    return sum((pred_dist.get(k, 0.0) - (1.0 if k == actual_key else 0.0)) ** 2 for k in all_keys)


def load_matches():
    rows = []
    for fn in sorted(os.listdir(CSV_DIR)):
        if not (fn.endswith('.csv') and fn[:2] in DIV_LEAGUE):
            continue
        div = fn[:2]
        league = DIV_LEAGUE[div]
        path = os.path.join(CSV_DIR, fn)
        with open(path, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for r in reader:
                hg = safe_float(r.get('FTHG'))
                ag = safe_float(r.get('FTAG'))
                if hg is None or ag is None:
                    continue
                avg_h = safe_float(r.get('AvgH'))
                avg_d = safe_float(r.get('AvgD'))
                avg_a = safe_float(r.get('AvgA'))
                ou_over = safe_float(r.get('Avg>2.5'))
                ou_under = safe_float(r.get('Avg<2.5'))
                if not (avg_h and avg_d and avg_a):
                    continue
                rows.append({
                    'league': league,
                    'hg': int(hg), 'ag': int(ag),
                    'total': int(hg) + int(ag),
                    'avg_h': avg_h, 'avg_d': avg_d, 'avg_a': avg_a,
                    'ou_over': ou_over, 'ou_under': ou_under,
                })
    return rows


def main():
    matches = load_matches()
    print(f"加载比赛样本: {len(matches)} 场")

    goal_keys = [str(i) for i in range(8)] + ['7+']
    score_keys = [(h, a) for h in range(MAX_GOALS + 1) for a in range(MAX_GOALS + 1)]

    # 累加器
    agg = defaultdict(lambda: {'ll': 0.0, 'brier': 0.0, 'top2': 0, 'n': 0,
                               's_ll': 0.0, 's_brier': 0.0, 's_top1': 0, 's_top3': 0, 'sn': 0})

    for m in matches:
        league = m['league']
        prof = LEAGUE_PROFILES.get(league, {'avg_goals': 2.6})
        avg_goals = prof['avg_goals']

        probs = implied_probs_from_odds(m['avg_h'], m['avg_d'], m['avg_a'])
        if not probs:
            continue
        ph, pd, pa = probs
        home_norm = ph - pa  # 实力差信号

        # 大小球隐含总进球
        ou_total = implied_total_from_ou(m['ou_over'], m['ou_under'])

        actual_total = m['total']
        actual_total_key = str(min(actual_total, 7)) + ('+' if actual_total >= 7 else '')
        if actual_total >= 7:
            actual_total_key = '7+'
        else:
            actual_total_key = str(actual_total)
        actual_score = (m['hg'], m['ag'])

        # ---------- 总进球方案 ----------
        # current: 联赛均值 + 0.05 分配
        zjq_cur, mu1c, mu2c = zjq_distribution(avg_goals, home_norm, -(home_norm), ZJQ_CURRENT_SPLIT)
        # improved: 大小球隐含总进球（与联赛均值 7:3 融合） + 0.35 分配
        if ou_total:
            imp_total = 0.7 * ou_total + 0.3 * avg_goals
        else:
            imp_total = avg_goals
        zjq_imp, mu1i, mu2i = zjq_distribution(imp_total, home_norm, -(home_norm), SCORE_SPLIT)

        # ---------- 比分方案（先算，供 zjq_dc 聚合）----------
        # current: 联赛均值 + 0.35 分配
        lam_h_c, lam_a_c = euro_implied_lambdas(ph, pd, pa, avg_goals)
        sc_cur = score_distribution(lam_h_c, lam_a_c)
        # improved: 大小球隐含总进球 + 0.35 分配
        t_imp = imp_total  # 与 improved 总进球一致
        lam_h_i, lam_a_i = euro_implied_lambdas(ph, pd, pa, t_imp)
        sc_imp = score_distribution(lam_h_i, lam_a_i)
        # Dixon-Coles 修正（在改进总进球基础上加低比分相关修正）
        sc_dc = score_distribution_dc(lam_h_i, lam_a_i, rho=-0.15)
        zjq_dc = aggregate_goal_from_score(sc_dc)

        # 总进球累计
        for tag, dist in [('zjq_current', zjq_cur), ('zjq_improved', zjq_imp), ('zjq_dc', zjq_dc)]:
            a = agg[tag]
            a['ll'] += logloss_cat(dist, actual_total_key, goal_keys)
            a['brier'] += brier_cat(dist, actual_total_key, goal_keys)
            ranked = sorted(dist.items(), key=lambda x: -x[1])
            top2 = [k for k, _ in ranked[:2]]
            if actual_total_key in top2:
                a['top2'] += 1
            a['n'] += 1

        # 比分累计
        for tag, dist in [('score_current', sc_cur), ('score_improved', sc_imp), ('score_dc', sc_dc)]:
            a = agg[tag]
            a['s_ll'] += logloss_cat(dist, actual_score, score_keys)
            a['s_brier'] += brier_cat(dist, actual_score, score_keys)
            ranked = sorted(dist.items(), key=lambda x: -x[1])
            top1 = ranked[0][0]
            top3 = [k for k, _ in ranked[:3]]
            if actual_score == top1:
                a['s_top1'] += 1
            if actual_score in top3:
                a['s_top3'] += 1
            a['sn'] += 1

    # ---------- 输出 ----------
    print("\n" + "=" * 78)
    print("总进球数分布评估（越低 LogLoss/Brier 越好，Top2 越高越好）")
    print("=" * 78)
    print("\n" + "=" * 78)
    print("精确比分分布评估（越低 LogLoss/Brier 越好，Top1/Top3 越高越好）")
    print("=" * 78)
    print(f"{'方案':<16}{'样本':>6}{'LogLoss':>10}{'Brier':>10}{'Top1':>8}{'Top3':>8}")
    for tag in ['score_current', 'score_improved', 'score_dc']:
        a = agg[tag]
        print(f"{tag:<16}{a['sn']:>6}{a['s_ll']/a['sn']:>10.4f}{a['s_brier']/a['sn']:>10.4f}{100*a['s_top1']/a['sn']:>7.2f}%{100*a['s_top3']/a['sn']:>7.2f}%")

    print("\n" + "=" * 78)
    print("总进球分布评估（current / improved / Dixon-Coles聚合）")
    print("=" * 78)
    print(f"{'方案':<16}{'样本':>6}{'LogLoss':>10}{'Brier':>10}{'Top2命中':>10}")
    for tag in ['zjq_current', 'zjq_improved', 'zjq_dc']:
        a = agg[tag]
        print(f"{tag:<16}{a['n']:>6}{a['ll']/a['n']:>10.4f}{a['brier']/a['n']:>10.4f}{100*a['top2']/a['n']:>9.2f}%")

    # ---------- 按联赛细分（总进球）----------
    print("\n" + "=" * 78)
    print("总进球 LogLoss 按联赛（current / improved / dc）")
    print("=" * 78)
    by_league = defaultdict(lambda: defaultdict(lambda: {'cur': 0.0, 'imp': 0.0, 'dc': 0.0, 'n': 0}))
    for m in matches:
        league = m['league']
        prof = LEAGUE_PROFILES.get(league, {'avg_goals': 2.6})
        avg_goals = prof['avg_goals']
        probs = implied_probs_from_odds(m['avg_h'], m['avg_d'], m['avg_a'])
        if not probs:
            continue
        ph, pd, pa = probs
        home_norm = ph - pa
        ou_total = implied_total_from_ou(m['ou_over'], m['ou_under'])
        if m['total'] >= 7:
            actual_total_key = '7+'
        else:
            actual_total_key = str(m['total'])
        zjq_cur, _, _ = zjq_distribution(avg_goals, home_norm, -(home_norm), ZJQ_CURRENT_SPLIT)
        if ou_total:
            imp_total = 0.7 * ou_total + 0.3 * avg_goals
        else:
            imp_total = avg_goals
        zjq_imp, _, _ = zjq_distribution(imp_total, home_norm, -(home_norm), SCORE_SPLIT)
        lam_h_i, lam_a_i = euro_implied_lambdas(ph, pd, pa, imp_total)
        sc_dc = score_distribution_dc(lam_h_i, lam_a_i, rho=-0.15)
        zjq_dc = aggregate_goal_from_score(sc_dc)
        bl = by_league[league]
        bl['cur']['cur'] += logloss_cat(zjq_cur, actual_total_key, goal_keys)
        bl['imp']['imp'] += logloss_cat(zjq_imp, actual_total_key, goal_keys)
        bl['dc']['dc'] += logloss_cat(zjq_dc, actual_total_key, goal_keys)
        bl['cur']['n'] += 1
    for league in sorted(by_league):
        bl = by_league[league]
        n = bl['cur']['n']
        print(f"{league:<8} n={n:<5} current={bl['cur']['cur']/n:.4f}  improved={bl['imp']['imp']/n:.4f}  dc={bl['dc']['dc']/n:.4f}")


if __name__ == '__main__':
    main()
