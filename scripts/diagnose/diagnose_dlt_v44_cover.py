#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大乐透 v4.4 组合覆盖实验 — 5注前区分散度 vs 命中率
==================================================
对比三种5注组合策略（walk-forward 200期）:
- S1 当前: 主推(核心2+轮换3) + balanced/rank/hot/cold 互斥(保留2锚点)
- S2 分散: 主推不变, 第2-5注前区强制避开主推全部5码(union最大化)
- S3 全覆盖: 主推不变, 第2-5注在前区排名池中分散选(不重叠, 各注间尽量不重)

指标: front_any_ge2 / front_any_ge3 / 单注ge2 / 前区union / 后区any_ge1
随机基准: front_any_ge2=52.3%, front_any_ge3=6.8%
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
    BACK_FEATURE_WEIGHTS,
)

N = 200


def rank_front(analyzer, weights, top_n=35):
    scores = []
    for num in FRONT_NUMBERS:
        fs = analyzer._calculate_feature_score(num, True)
        if not fs:
            continue
        total = sum(fs.get(k, 0) * w for k, w in weights.items() if k in fs)
        scores.append((num, total))
    scores.sort(key=lambda x: -x[1])
    return scores[:top_n]


def rank_back_top(analyzer, top_n=10):
    scores = []
    for num in BACK_NUMBERS:
        fs = analyzer._calculate_back_feature_score(num)
        if not fs:
            continue
        total = sum(fs.get(k, 0) * w for k, w in BACK_FEATURE_WEIGHTS.items() if k in fs)
        scores.append((num, total))
    scores.sort(key=lambda x: -x[1])
    return [n for n, _ in scores[:top_n]]


def primary_ticket(ranked, issue):
    try:
        seed = int(issue)
    except (TypeError, ValueError):
        seed = sum(ord(ch) for ch in str(issue))
    core = ranked[:2]
    pool = ranked[2:10]
    support = []
    if pool:
        offset = seed % len(pool)
        step = 3 if len(pool) >= 8 else 1
        cursor = offset
        while len(support) < 3 and len(support) < len(pool):
            num = pool[cursor % len(pool)]
            if num not in support:
                support.append(num)
            cursor += step
    return core + support


def back_ticket(back_ranked, issue, used_back, exclude_all=False):
    """后区2码: 优先取未用号码中的最高排名2个（组合覆盖）"""
    pool = [n for n in back_ranked if n not in used_back]
    if len(pool) < 2:
        pool = [n for n in back_ranked]
    return pool[:2]


def pick_5(pool, used, count=5):
    """从pool中取count个不在used的号码"""
    avail = [n for n in pool if n not in used]
    if len(avail) < count:
        avail = list(dict.fromkeys(avail + pool))
    return avail[:count]


def simulate_strategy(analyzer, saved, strategy, trials=N):
    """strategy: 'S1_current' | 'S2_diverse' | 'S3_fullcover'"""
    any_ge2 = any_ge3 = back_any1 = joint = 0
    per_ge2 = per_ge3 = 0
    front_avg = 0.0
    union_sum = 0
    n = 0
    for i in range(trials):
        if i >= len(saved) - 11:
            break
        analyzer.history_data = list(saved[i + 1:])
        if len(analyzer.history_data) < 80:
            continue
        analyzer.update_statistics()
        actual_f = set(saved[i]['front'])
        actual_b = set(saved[i]['back'])
        issue = saved[i]['issue']
        n += 1

        fr = rank_front(analyzer, FEATURE_WEIGHTS, top_n=35)
        ranked = [x[0] for x in fr]
        br = rank_back_top(analyzer)

        p1 = primary_ticket(ranked, issue)
        b1 = br[:2]
        tickets = [{'front': p1, 'back': b1}]
        used_f = set(p1)
        used_b = set(b1)

        if strategy == 'S1_current':
            # 模拟当前: balanced/rank/hot/cold 各从不同池子选, 保留2锚点
            anchors = set(ranked[:2])
            pools = [
                ranked[0:15],                # balanced ≈ voting排名池
                ranked[0:15],                # rank
                [x[0] for x in analyzer.statistics.get('hot_front', [])[:20]],
                [x[0] for x in analyzer.statistics.get('cold_front', [])[:20]],
            ]
            for pool in pools:
                exclude_f = (used_f - anchors)
                avail = [x for x in pool if x not in exclude_f]
                if len(avail) < 5:
                    avail = [x for x in ranked if x not in exclude_f][:5]
                f5 = pick_5(avail, set(), 5)
                if len(f5) < 5:
                    f5 = pick_5(ranked, exclude_f, 5)
                bk = back_ticket(br, issue, used_b)
                tickets.append({'front': f5, 'back': bk})
                used_f |= set(f5)
                used_b |= set(bk)
        else:
            # S2/S3: 第2-5注 全部避开主推号码, 从排名池中取不重叠组合
            anchors = set(ranked[:2]) if strategy == 'S1_current' else set()
            pool = ranked[10:35] if strategy == 'S3_fullcover' else ranked[5:35]
            # 打散: 按步长取, 保证5注共25码尽量不重叠
            step = 5 if strategy == 'S3_fullcover' else 4
            cursor = 0
            for _ in range(4):
                f5 = []
                while len(f5) < 5 and cursor < len(pool):
                    num = pool[cursor]
                    cursor += step
                    if num not in used_f and num not in f5:
                        f5.append(num)
                if len(f5) < 5:
                    for num in ranked:
                        if num not in used_f and num not in f5:
                            f5.append(num)
                        if len(f5) >= 5:
                            break
                bk = back_ticket(br, issue, used_b)
                tickets.append({'front': f5, 'back': bk})
                used_f |= set(f5)
                used_b |= set(bk)

        # 统计
        f_any2 = f_any3 = False
        b_any1 = False
        jt = False
        union = set()
        for t in tickets:
            f = set(t['front'])
            b = set(t['back'])
            union |= f
            hf = len(actual_f & f)
            hb = len(actual_b & b)
            front_avg += hf
            if hf >= 2:
                per_ge2 += 1
                f_any2 = True
            if hf >= 3:
                per_ge3 += 1
                f_any3 = True
            if hb >= 1:
                b_any1 = True
            if hf >= 2 and hb >= 1:
                jt = True
        if f_any2:
            any_ge2 += 1
        if f_any3:
            any_ge3 += 1
        if b_any1:
            back_any1 += 1
        if jt:
            joint += 1
        union_sum += len(union)

    return {
        'n': n,
        'front_any_ge2': any_ge2 / n,
        'front_any_ge3': any_ge3 / n,
        'back_any_ge1': back_any1 / n,
        'joint': joint / n,
        'per_ticket_ge2': per_ge2 / (n * 5),
        'per_ticket_ge3': per_ge3 / (n * 5),
        'front_union_avg': union_sum / n,
        'front_avg': front_avg / (n * 5),
    }


def main():
    a = LotteryAnalyzer()
    saved = list(a.history_data)
    print(f"历史期数: {len(saved)}", flush=True)
    results = {}
    for s in ('S1_current', 'S2_diverse', 'S3_fullcover'):
        r = simulate_strategy(a, saved, s)
        results[s] = r
        print(f"\n--- {s} ---")
        print(f"  front_any_ge2: {r['front_any_ge2']:.1%} (随机52.3%)")
        print(f"  front_any_ge3: {r['front_any_ge3']:.1%} (随机6.8%)")
        print(f"  back_any_ge1 : {r['back_any_ge1']:.1%} (理论98.5%)")
        print(f"  joint(同注)   : {r['joint']:.1%}")
        print(f"  单注ge2率     : {r['per_ticket_ge2']:.1%} (随机13.9%)")
        print(f"  前区union     : {r['front_union_avg']:.1f} / 25")
        print(f"  单注前区平均  : {r['front_avg']:.3f} (随机0.714)")

    json.dump(results, open('data/diagnose_dlt_v44_cover.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print("\n已保存: data/diagnose_dlt_v44_cover.json")


if __name__ == '__main__':
    main()
