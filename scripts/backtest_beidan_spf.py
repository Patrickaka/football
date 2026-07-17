#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
北单 SPF（胜平负）离线回测：用欧洲五大联赛历史数据（data/*.csv）量化命中率，
并对比不同"赔率->概率"方法与不同置信门限对命中率/覆盖率的真实影响。

背景：北单的预测本质上以市场欧赔隐含概率为骨架（calculate_implied_probability），
叠加亚盘修正与历史校准。本脚本在"无历史校准/无亚盘"的离线条件下，
测量基础模型的 Top1 命中率，并演示"只推高置信"能带来多少命中率提升。

用法：
    python3 scripts/backtest_beidan_spf.py
"""
import os
import csv
import math
import sys

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


def load_matches():
    rows = []
    for fn in sorted(os.listdir(CSV_DIR)):
        if not (fn.endswith('.csv') and fn[:2] in DIV_LEAGUE):
            continue
        div = fn[:2]
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
                rows.append({
                    'league': league,
                    'home': r['HomeTeam'], 'away': r['AwayTeam'],
                    'hg': int(hg), 'ag': int(ag),
                    'h': ah, 'd': ad, 'a': aa,
                })
    return rows


def shin_probabilities(odds):
    """Shin 法：相比基础归一化，减少长尾偏置，概率更贴近真实频率。"""
    h, d, a = odds['h'], odds['d'], odds['a']
    # 用数值法求 Shin 的 insider-probability z
    lo, hi = 0.0, 0.10
    z = 0.0

    def root(zz):
        # Shin: p_i = (1/(o_i*(1+z)) - z/(1-z)) / (1 - 2z/(1-z) * sum?) ... 直接用二分求两边等式
        # 标准 Shin 2001：p_i = ((1 - z)/(o_i) + z * b_i) / (n - z) ，这里用经典二分近似
        lhs = 1.0 / (h * (1 + zz)) + 1.0 / (d * (1 + zz)) + 1.0 / (a * (1 + zz))
        rhs = 1.0 + zz / (1.0 - zz)
        return lhs - rhs

    f_lo, f_hi = root(lo), root(hi)
    if f_lo * f_hi < 0:
        for _ in range(60):
            mid = (lo + hi) / 2
            fm = root(mid)
            if f_lo * fm <= 0:
                hi = mid
                f_hi = fm
            else:
                lo = mid
                f_lo = fm
        z = (lo + hi) / 2
    # 计算 Shin 概率
    ph = (1.0 / (h * (1 + z)) - z / (1 - z)) / 3.0
    pd = (1.0 / (d * (1 + z)) - z / (1 - z)) / 3.0
    pa = (1.0 / (a * (1 + z)) - z / (1 - z)) / 3.0
    s = ph + pd + pa
    if s <= 0:
        return None
    return {'胜': ph / s, '平': pd / s, '负': pa / s}


def main():
    matches = load_matches()
    print(f"样本: {len(matches)} 场\n")

    def actual(r):
        if r['hg'] > r['ag']:
            return '胜'
        if r['hg'] < r['ag']:
            return '负'
        return '平'

    # ---- 方法 A：用模块真实的 calculate_implied_probability + assess_recommendation_quality ----
    a_hit = a_n = 0
    a_strong_hit = a_strong_n = 0
    a_medium_hit = a_medium_n = 0
    for m in matches:
        # 与 analyze_spf 一致：欧赔 -> 隐含概率（基础去水）
        odds = {'胜': m['h'], '平': m['d'], '负': m['a']}
        probs = beidan.calculate_implied_probability(odds)
        if not probs:
            continue
        pred = max(probs, key=probs.get)
        top = probs[pred]
        sec = sorted(probs.values(), reverse=True)[1]
        lead = top - sec
        a_n += 1
        if pred == actual(m):
            a_hit += 1
        # 用模块真实分级（新阈值：strong>=0.50, medium>=0.43）
        q = beidan.assess_recommendation_quality(probs, pred, {})
        if q['level'] == 'strong':
            a_strong_n += 1
            if pred == actual(m):
                a_strong_hit += 1
        if q['level'] in ('strong', 'medium'):
            a_medium_n += 1
            if pred == actual(m):
                a_medium_hit += 1

    # ---- 方法 B：Shin 法 ----
    b_hit = b_n = 0
    for m in matches:
        probs = shin_probabilities(m)
        if not probs:
            continue
        pred = max(probs, key=probs.get)
        b_n += 1
        if pred == actual(m):
            b_hit += 1

    print("=" * 70)
    print("北单 SPF 胜平负 离线回测（欧洲五大联赛代理样本）")
    print("=" * 70)
    print(f"[模块真实分级(新阈值 strong>=0.50 / medium>=0.43)]")
    print(f"  全量 Top1 命中率 : {100*a_hit/a_n:.2f}%  (覆盖 {a_n})")
    print(f"  强推荐单推命中率 : {100*a_strong_hit/max(a_strong_n,1):.2f}%  (覆盖 {a_strong_n}, 占比 {100*a_strong_n/a_n:.1f}%)")
    print(f"  中+单推命中率    : {100*a_medium_hit/max(a_medium_n,1):.2f}%  (覆盖 {a_medium_n}, 占比 {100*a_medium_n/a_n:.1f}%)")
    print(f"[Shin 法 赔率->概率]")
    print(f"  全量 Top1 命中率 : {100*b_hit/b_n:.2f}%  (覆盖 {b_n})")

    # 与 zjq/bifen 基线对照
    print("\n对照（verify 脚本已测）：")
    print("  总进球 zjq Top2=45.26%  比分 bifen Top1=11.66%/Top3=32.11%")


if __name__ == '__main__':
    main()
