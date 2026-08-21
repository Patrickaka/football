#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大乐透 v4.4 优化诊断 — 定位「推荐准确率低」的环节
=================================================
针对用户反馈"目前预测准确率很低"，系统性诊断以下环节（全部 walk-forward）：

A. 主推生成方式（当前: 核心Top2+轮换3 vs 直接Top5 vs 全轮换）→ 是否稀释命中
B. 前区约束(奇偶/区间/连号) 对 Top5 命中的影响（约束是否强扭拉低命中）
C. 单变量权重扫描（v4.3 基准 ± 调整，找更高 Top5 ge2 / 池ge4 的配置）
D. 后区排名 Top2/Top5 命中 vs 随机
E. 5注组合整体表现（任意一注 ge2/ge3、后区 ge1、同一注 前ge2+后ge1）

随机基准：
  前区Top5: avg=0.714, ge1=56.1%, ge2=13.89%, ge3=1.39%
  后区Top2: ge1=45.45%, ge2=4.55%; 后区Top5池: ge1=68.18%
  前区15码池: ge4=9.33%, ge3≈30%
"""
import sys
import math
import json
import logging
from collections import Counter

sys.path.insert(0, '.')
logging.disable(logging.WARNING)

from src.lottery import (
    LotteryAnalyzer, FRONT_NUMBERS, BACK_NUMBERS, FEATURE_WEIGHTS,
    RANDOM_BASELINE,
)

# ---------- 工具 ----------

def rank_with_weights(analyzer, weights, top_n=35):
    """按权重算前区排名"""
    scores = []
    for num in FRONT_NUMBERS:
        fs = analyzer._calculate_feature_score(num, True)
        if not fs:
            continue
        total = sum(fs.get(k, 0) * w for k, w in weights.items() if k in fs)
        scores.append((num, total))
    scores.sort(key=lambda x: -x[1])
    return scores[:top_n]


def rank_back(analyzer, top_n=8):
    scores = []
    for num in BACK_NUMBERS:
        fs = analyzer._calculate_back_feature_score(num)
        if not fs:
            continue
        total = sum(fs.get(k, 0) * w for k, w in BACK_W_ITEMS.items() if k in fs)
        scores.append((num, total))
    scores.sort(key=lambda x: -x[1])
    return [n for n, _ in scores[:top_n]]


BACK_W_ITEMS = FEATURE_WEIGHTS  # 后区实际用 BACK_FEATURE_WEIGHTS，这里近似用默认


def current_primary(ranked, issue):
    """复刻 generate_multi_strategy_recommendations 的主推逻辑:
    核心=排名Top2, 从排名3-10 按 issue_seed 轮换3个, step=3"""
    try:
        issue_seed = int(issue)
    except (TypeError, ValueError):
        issue_seed = sum(ord(ch) for ch in str(issue))
    core = ranked[:2]
    pool = ranked[2:10]
    support = []
    if pool:
        offset = issue_seed % len(pool)
        step = 3 if len(pool) >= 8 else 1
        cursor = offset
        while len(support) < 3 and len(support) < len(pool):
            num = pool[cursor % len(pool)]
            if num not in support:
                support.append(num)
            cursor += step
    return core + support


def rotate_all(ranked, issue, n=5, pool_size=10):
    """全轮换: 从排名前 pool_size 按期号轮换 n 个"""
    try:
        issue_seed = int(issue)
    except (TypeError, ValueError):
        issue_seed = sum(ord(ch) for ch in str(issue))
    pool = ranked[:pool_size]
    sel = []
    cursor = issue_seed % len(pool)
    while len(sel) < n and len(sel) < len(pool):
        num = pool[cursor % len(pool)]
        if num not in sel:
            sel.append(num)
        cursor += 1
    return sel


def main():
    a = LotteryAnalyzer()
    saved = list(a.history_data)
    print(f"历史期数: {len(saved)}  最新: {saved[0]['issue']}", flush=True)

    N = 500
    # 每期统计: {变体名: {ge2, ge3, pool_ge4, pool_ge3, avg}}
    agg = {
        'A1_当前主推': Counter(), 'A2_直接Top5': Counter(), 'A3_全轮换Top5': Counter(),
        'B_约束后Top5': Counter(),
        'pool15_ge4': 0, 'pool15_ge3': 0,
    }
    back_top2_ge1 = back_top2_ge2 = back_top5_ge1 = 0
    n = 0

    # 单变量变体（基于 v4.3 基准）
    base = dict(FEATURE_WEIGHTS)
    variants = {
        'v4.3基准': base,
        'gap0.14': {**base, 'gap': 0.14},
        'gap0.16': {**base, 'gap': 0.16},
        'zone0.16': {**base, 'zone': 0.16},
        'zone0.14': {**base, 'zone': 0.14},
        'sum0.12': {**base, 'sum': 0.12},
        'sum0.10': {**base, 'sum': 0.10},
        'road0.14': {**base, 'road': 0.14},
        'road0.16': {**base, 'road': 0.16},
        'position0.08': {**base, 'position': 0.08},
        'adjacent0.10': {**base, 'adjacent': 0.10},
        'freq0.08': {**base, 'frequency': 0.08},
        'trend0.0': {**base, 'trend': 0.0},
        'gap0.14_zone0.16': {**base, 'gap': 0.14, 'zone': 0.16},
        'gap0.14_zone0.14': {**base, 'gap': 0.14, 'zone': 0.14},
        'sum0.12_zone0.16': {**base, 'sum': 0.12, 'zone': 0.16},
        'gap0.14_sum0.12': {**base, 'gap': 0.14, 'sum': 0.12},
        'gap0.16_zone0.16': {**base, 'gap': 0.16, 'zone': 0.16},
    }
    var_stats = {name: {'ge2': 0, 'ge3': 0, 'avg': 0.0, 'pool_ge4': 0, 'pool_ge3': 0}
                 for name in variants}

    for i in range(N):
        if i >= len(saved) - 11:
            break
        a.history_data = list(saved[i + 1:])
        if len(a.history_data) < 80:
            continue
        a.update_statistics()
        actual_f = set(saved[i]['front'])
        actual_b = set(saved[i]['back'])
        issue = saved[i]['issue']
        n += 1

        ranked = [num for num, _ in rank_with_weights(a, base, top_n=35)]

        # A1 当前主推逻辑
        a1 = current_primary(ranked, issue)
        hits = len(actual_f & set(a1))
        agg['A1_当前主推']['avg'] += hits
        if hits >= 2:
            agg['A1_当前主推']['ge2'] += 1
        if hits >= 3:
            agg['A1_当前主推']['ge3'] += 1

        # A2 直接Top5
        a2 = ranked[:5]
        hits = len(actual_f & set(a2))
        agg['A2_直接Top5']['avg'] += hits
        if hits >= 2:
            agg['A2_直接Top5']['ge2'] += 1
        if hits >= 3:
            agg['A2_直接Top5']['ge3'] += 1

        # A3 全轮换Top5（排名前10轮换）
        a3 = rotate_all(ranked, issue)
        hits = len(actual_f & set(a3))
        agg['A3_全轮换Top5']['avg'] += hits
        if hits >= 2:
            agg['A3_全轮换Top5']['ge2'] += 1
        if hits >= 3:
            agg['A3_全轮换Top5']['ge3'] += 1

        # B 约束后Top5（走 _score_based_select → _apply_front_constraints）
        try:
            b = a._score_based_select(ranked[:12], 5, is_front=True,
                                      fallback_pool=FRONT_NUMBERS)
            hits = len(actual_f & set(b))
            agg['B_约束后Top5']['avg'] += hits
            if hits >= 2:
                agg['B_约束后Top5']['ge2'] += 1
            if hits >= 3:
                agg['B_约束后Top5']['ge3'] += 1
        except Exception as e:
            pass

        # 15码池（v4.3 权重）
        pool = ranked[:15]
        hits_pool = len(actual_f & set(pool))
        if hits_pool >= 4:
            agg['pool15_ge4'] += 1
        if hits_pool >= 3:
            agg['pool15_ge3'] += 1

        # 后区
        bk = rank_back(a, 8)
        bt2 = set(bk[:2])
        if actual_b & bt2:
            back_top2_ge1 += 1
        if len(actual_b & bt2) >= 2:
            back_top2_ge2 += 1
        if actual_b & set(bk[:5]):
            back_top5_ge1 += 1

        # 单变量变体（复用已算出的特征分，仅重新加权，很快）
        for name, w in variants.items():
            vr = [num for num, _ in rank_with_weights(a, w, top_n=15)]
            hits_v = len(actual_f & set(vr[:5]))
            var_stats[name]['avg'] += hits_v
            if hits_v >= 2:
                var_stats[name]['ge2'] += 1
            if hits_v >= 3:
                var_stats[name]['ge3'] += 1
            hp = len(actual_f & set(vr[:15]))
            if hp >= 4:
                var_stats[name]['pool_ge4'] += 1
            if hp >= 3:
                var_stats[name]['pool_ge3'] += 1

    print(f"\n有效期数: {n}", flush=True)
    print("\n" + "=" * 100)
    print("A/B. 主推生成方式 & 约束影响（500期 walk-forward）")
    print("=" * 100)
    print(f"{'方式':<14}{'avg命中':>8}{'ge1':>8}{'ge2':>8}{'ge3':>8}")
    print(f"{'随机基准':<14}{0.714:>8.3f}{0.561:>8.1%}{0.1389:>8.1%}{0.0139:>8.2%}")
    for name in ('A1_当前主推', 'A2_直接Top5', 'A3_全轮换Top5', 'B_约束后Top5'):
        c = agg[name]
        avg = c['avg'] / n
        print(f"{name:<14}{avg:>8.3f}{c['ge2']/n:>8.1%}{c['ge3']/n:>8.2%}")

    print(f"\n15码池(基准权重): ge4={agg['pool15_ge4']/n:.1%} (随机9.33%), "
          f"ge3={agg['pool15_ge3']/n:.1%} (随机~30%)")

    print("\n" + "=" * 100)
    print("D. 后区命中（500期）")
    print("=" * 100)
    print(f"{'指标':<20}{'模型':>8}{'随机':>8}")
    print(f"{'后区Top2 ge1':<20}{back_top2_ge1/n:>8.1%}{0.4545:>8.1%}")
    print(f"{'后区Top2 ge2':<20}{back_top2_ge2/n:>8.1%}{0.0455:>8.1%}")
    print(f"{'后区Top5池 ge1':<20}{back_top5_ge1/n:>8.1%}{0.6818:>8.1%}")

    print("\n" + "=" * 100)
    print("C. 单变量权重扫描（Top5 ge2 / 池ge4，相对基准排序）")
    print("=" * 100)
    print(f"{'变体':<22}{'Top5 ge2':>10}{'Top5 ge3':>10}{'池ge4':>8}{'池ge3':>8}{'avg':>8}")
    rows = []
    for name, s in var_stats.items():
        rows.append((name, s['ge2'] / n, s['ge3'] / n, s['pool_ge4'] / n,
                     s['pool_ge3'] / n, s['avg'] / n))
    rows.sort(key=lambda x: -x[1])
    for name, ge2, ge3, pge4, pge3, avg in rows:
        mark = ' <==' if name == 'v4.3基准' else ''
        print(f"{name:<22}{ge2:>9.1%}{ge3:>9.2%}{pge4:>7.1%}{pge3:>7.1%}{avg:>8.3f}{mark}")

    a.history_data = saved

    # 保存结果
    out = {
        'n': n,
        'partA_B': {name: {'avg': agg[name]['avg'] / n, 'ge2': agg[name]['ge2'] / n,
                           'ge3': agg[name]['ge3'] / n} for name in
                    ('A1_当前主推', 'A2_直接Top5', 'A3_全轮换Top5', 'B_约束后Top5')},
        'pool15': {'ge4': agg['pool15_ge4'] / n, 'ge3': agg['pool15_ge3'] / n},
        'back': {'top2_ge1': back_top2_ge1 / n, 'top2_ge2': back_top2_ge2 / n,
                 'top5_ge1': back_top5_ge1 / n},
        'variants': {name: {'ge2': s['ge2'] / n, 'ge3': s['ge3'] / n,
                            'pool_ge4': s['pool_ge4'] / n,
                            'pool_ge3': s['pool_ge3'] / n, 'avg': s['avg'] / n}
                     for name, s in var_stats.items()},
    }
    json.dump(out, open('data/diagnose_dlt_v44.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print("\n已保存: data/diagnose_dlt_v44.json")


if __name__ == '__main__':
    main()
