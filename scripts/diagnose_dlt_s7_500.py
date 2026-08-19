#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
7注全覆盖方案 500期真实回测
============================
对比:
  S5: 当前 v4.4 (5注25码覆盖)          -> 实测 62.2%/6.4%
  S7a: 7注全覆盖, 主推注=模型Top5, 其余30码覆盖6注
  S7b: 7注全覆盖, 7注均匀分散(无主推集中)
后区: 全覆盖12码 -> back_any_ge1 应=100%
目标指标: front_any_ge2 / front_any_ge3 / back_any_ge1 / joint(同注前≥3后≥1)
"""
import sys
import json
import logging

sys.path.insert(0, '.')
logging.disable(logging.WARNING)

from src.lottery import LotteryAnalyzer

N = 500
FRONT = list(range(1, 36))


def get_ranked_front(analyzer):
    """返回模型前区排名(降序)与后区排名, 用 rank_model 取全量"""
    front_ranked, back_ranked = analyzer.rank_model(top_n=35)
    ranked_f = [x[0] for x in front_ranked]
    ranked_b = [x[0] for x in back_ranked]
    return ranked_f, ranked_b


def build_s7a(ranked_f, ranked_b, issue_seed):
    """7注: 主推=Top5, 6注覆盖其余30码(步长5分散), 后区全覆盖12码"""
    primary = set(ranked_f[:5])
    pool = ranked_f[5:]  # 30码
    groups = [primary]
    covered = set(primary)
    step = 5
    cursor = issue_seed % step
    for i in range(6):
        g = set()
        while len(g) < 5:
            num = pool[cursor % len(pool)]
            if num not in covered:
                g.add(num)
                covered.add(num)
            cursor = (cursor + 1) % len(pool)
        groups.append(g)
    # 后区: 12码全覆盖, 7注14位置 -> 前5注2码, 后2注2码(含重复)
    backs = []
    b_cursor = 0
    for i in range(7):
        b = []
        for _ in range(2):
            b.append(ranked_b[b_cursor % len(ranked_b)])
            b_cursor += 1
        backs.append(b)
    return groups, backs


def build_s7b(ranked_f, ranked_b, issue_seed):
    """7注全覆盖均匀分散(无主推集中), 后区全覆盖12码"""
    groups = []
    covered = set()
    step = 7
    cursor = issue_seed % step
    pool = ranked_f
    for i in range(7):
        g = set()
        while len(g) < 5:
            num = pool[cursor % len(pool)]
            if num not in covered:
                g.add(num)
                covered.add(num)
            cursor = (cursor + 1) % len(pool)
        groups.append(g)
    backs = []
    b_cursor = 0
    for i in range(7):
        b = []
        for _ in range(2):
            b.append(ranked_b[b_cursor % len(ranked_b)])
            b_cursor += 1
        backs.append(b)
    return groups, backs


def build_s6(ranked_f, ranked_b, issue_seed):
    """6注: 覆盖30码(前30排名), 后区全覆盖12码(6注12位置)"""
    groups = []
    covered = set()
    step = 6
    cursor = issue_seed % step
    pool = ranked_f[:30]
    for i in range(6):
        g = set()
        while len(g) < 5:
            num = pool[cursor % len(pool)]
            if num not in covered:
                g.add(num)
                covered.add(num)
            cursor = (cursor + 1) % len(pool)
        groups.append(g)
    backs = []
    b_cursor = 0
    for i in range(6):
        b = []
        for _ in range(2):
            b.append(ranked_b[b_cursor % len(ranked_b)])
            b_cursor += 1
        backs.append(b)
    return groups, backs


def simulate(a, saved, start, end, builder, tag):
    any_ge2 = any_ge3 = back1 = joint = 0
    per_ge2 = per_ge3 = 0
    n = 0
    for i in range(start, end):
        if i >= len(saved) - 11:
            break
        a.history_data = list(saved[i + 1:])
        if len(a.history_data) < 80:
            continue
        a.update_statistics()
        rf, rb = get_ranked_front(a)
        actual_f = set(saved[i]['front'])
        actual_b = set(saved[i]['back'])
        n += 1
        groups, backs = builder(rf, rb, int(str(saved[i].get('issue', i))[-4:]))
        f2 = f3 = b1 = jt = False
        for g, b in zip(groups, backs):
            hf = len(actual_f & g)
            hb = len(actual_b & set(b))
            if hf >= 2:
                per_ge2 += 1
                f2 = True
            if hf >= 3:
                per_ge3 += 1
                f3 = True
            if hb >= 1:
                b1 = True
            if hf >= 3 and hb >= 1:
                jt = True
        if f2:
            any_ge2 += 1
        if f3:
            any_ge3 += 1
        if b1:
            back1 += 1
        if jt:
            joint += 1
    r = {
        'n': n,
        'front_any_ge2': any_ge2 / n,
        'front_any_ge3': any_ge3 / n,
        'back_any_ge1': back1 / n,
        'joint_ge3_ge1': joint / n,
        'per_ticket_ge2': per_ge2 / (n * len(groups)),
        'per_ticket_ge3': per_ge3 / (n * len(groups)),
    }
    print(f"\n--- {tag} ---")
    print(f"  front_any_ge2   = {r['front_any_ge2']:.1%}  (随机52.3% / 7注全覆盖理论~80%)")
    print(f"  front_any_ge3   = {r['front_any_ge3']:.1%}  (随机6.8% / 7注全覆盖理论~9.7%)")
    print(f"  back_any_ge1    = {r['back_any_ge1']:.1%}  (理论100%全覆盖)")
    print(f"  joint(前≥3后≥1) = {r['joint_ge3_ge1']:.1%}")
    print(f"  单注ge2率       = {r['per_ticket_ge2']:.1%}   单注ge3率 = {r['per_ticket_ge3']:.1%}")
    return r


def main():
    a = LotteryAnalyzer()
    saved = list(a.history_data)
    print(f"历史期数: {len(saved)}")
    results = {}
    results['S6'] = simulate(a, saved, 0, N, build_s6, 'S6: 6注覆盖30码(后区全覆盖)')
    results['S7a'] = simulate(a, saved, 0, N, build_s7a, 'S7a: 7注全覆盖(主推=Top5 + 6注覆盖30码)')
    results['S7b'] = simulate(a, saved, 0, N, build_s7b, 'S7b: 7注全覆盖(均匀分散)')
    json.dump(results, open('data/diagnose_dlt_s7_500.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print("\n已保存 data/diagnose_dlt_s7_500.json")


if __name__ == '__main__':
    main()
