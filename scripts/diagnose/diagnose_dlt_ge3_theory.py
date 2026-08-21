#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断: front_any_ge3 随机全覆盖理论值 vs 模型实测
================================================
蒙特卡洛模拟:
1) 随机选25码分5组(每组5码) -> front_any_ge2 / ge3 理论值
2) 随机选35码分7组 -> ge3 理论值(全覆盖)
3) 模型分层稀释假设验证: 若25码中15码"较好"+10码"较差"且中奖码偏向较好码,
   分组均匀时每注≤2 -> ge3 被压低
"""
import random
import sys

sys.path.insert(0, '.')
import logging
logging.disable(logging.WARNING)

from src.lottery import LotteryAnalyzer

FRONT = list(range(1, 36))
N_SIM = 20000
N_ISSUES = 500


def simulate_ge2_ge3(groups, n_sim=N_SIM):
    """给定分组(每组5个前区号码), 模拟开奖, 统计任一组≥2 / ≥3 的概率"""
    ge2 = ge3 = 0
    for _ in range(n_sim):
        draw = set(random.sample(FRONT, 5))
        f2 = f3 = False
        for g in groups:
            h = len(draw & g)
            if h >= 3:
                f3 = True
            if h >= 2:
                f2 = True
        if f2:
            ge2 += 1
        if f3:
            ge3 += 1
    return ge2 / n_sim, ge3 / n_sim


def random_split(n_groups):
    """把35个号码随机分成 n_groups 组, 每组5码, 返回组列表"""
    nums = list(FRONT)
    random.shuffle(nums)
    return [set(nums[i * 5:(i + 1) * 5]) for i in range(n_groups)]


def model_groups_from_ranks(analyzer, saved, start, end, top5_mode=False):
    """用模型排名构建5注分组: 返回每期的 groups 列表, 供精确模拟模型分布"""
    results = []
    for i in range(start, end):
        if i >= len(saved) - 11:
            break
        analyzer.history_data = list(saved[i + 1:])
        if len(analyzer.history_data) < 80:
            continue
        analyzer.update_statistics()
        multi = analyzer.generate_multi_strategy_recommendations(
            voting_result=analyzer.multi_model_voting(front_n=20, back_n=10, skip_ml=True)
        )
        recs = [x for x in multi['recommendations']
                if not x['strategy'].startswith('picked')]
        groups = [set(r['front']) for r in recs]
        results.append(groups)
    return results


def main():
    print("=" * 70)
    print("1) 随机25码均匀5组(理论覆盖上限)")
    print("=" * 70)
    for trial in range(3):
        groups = random_split(5)
        ge2, ge3 = simulate_ge2_ge3(groups)
        print(f"  trial{trial + 1}: ge2={ge2:.1%} ge3={ge3:.1%}")

    print()
    print("=" * 70)
    print("2) 随机35码7组(全覆盖7注理论)")
    print("=" * 70)
    for trial in range(3):
        groups = random_split(7)
        ge2, ge3 = simulate_ge2_ge3(groups)
        print(f"  trial{trial + 1}: ge2={ge2:.1%} ge3={ge3:.1%}")

    print()
    print("=" * 70)
    print("3) 随机10组(10注全覆盖35码) 参考")
    print("=" * 70)
    for trial in range(3):
        groups = random_split(10)
        ge2, ge3 = simulate_ge2_ge3(groups)
        print(f"  trial{trial + 1}: ge2={ge2:.1%} ge3={ge3:.1%}")

    print()
    print("=" * 70)
    print("4) 模型实际5注分组 500期 ge2/ge3 (对照实测 62.2%/6.4%)")
    print("=" * 70)
    a = LotteryAnalyzer()
    saved = list(a.history_data)
    print(f"  历史期数: {len(saved)}")
    groups_list = model_groups_from_ranks(a, saved, 0, 500)
    print(f"  有效期数: {len(groups_list)}")
    # 统计每期实测命中
    ge2 = ge3 = 0
    per_group_ge3 = 0
    total_groups = 0
    for idx, groups in enumerate(groups_list):
        actual_f = set(saved[idx]['front'])
        f2 = f3 = False
        for g in groups:
            h = len(actual_f & g)
            if h >= 3:
                f3 = True
                per_group_ge3 += 1
            if h >= 2:
                f2 = True
            total_groups += 1
        if f2:
            ge2 += 1
        if f3:
            ge3 += 1
    n = len(groups_list)
    print(f"  front_any_ge2 = {ge2 / n:.1%}  (随机52.3% / 全覆盖理论~66%)")
    print(f"  front_any_ge3 = {ge3 / n:.1%}  (随机6.8% / 全覆盖理论~11.3%)")
    print(f"  单注ge3率     = {per_group_ge3 / total_groups:.2%}")
    print(f"  平均注数      = {total_groups / n:.1f}")

    # 分层稀释检查: 主推注 vs 覆盖注 的命中分布
    print()
    print("=" * 70)
    print("5) 主推注 vs 覆盖注 分别的 ge2/ge3 (检查分层稀释)")
    print("=" * 70)
    primary_ge2 = primary_ge3 = cover_ge2 = cover_ge3 = 0
    for idx, groups in enumerate(groups_list):
        actual_f = set(saved[idx]['front'])
        for gi, g in enumerate(groups):
            h = len(actual_f & g)
            if gi == 0:
                if h >= 3:
                    primary_ge3 += 1
                if h >= 2:
                    primary_ge2 += 1
            else:
                if h >= 3:
                    cover_ge3 += 1
                if h >= 2:
                    cover_ge2 += 1
    n = len(groups_list)
    print(f"  主推注(第1注): ge2={primary_ge2 / n:.1%}  ge3={primary_ge3 / n:.1%}")
    print(f"  覆盖注(2-5注): ge2={cover_ge2 / (n * 4):.1%}  ge3={cover_ge3 / (n * 4):.1%}")


if __name__ == '__main__':
    main()
