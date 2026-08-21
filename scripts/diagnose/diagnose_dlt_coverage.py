#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大乐透命中率诊断：扫描不同前区/后区推荐数量下的命中率。

用户目标：前区命中≥4码，后区命中1-2码。
关键洞察：前区只推5码时命中≥4的随机概率仅0.046%，必须扩大推荐池才能达成目标。
本脚本量化"推荐多少码"能达到用户目标，并对比模型排名 vs 纯随机。
"""
import sys, os, json, math, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
logging.getLogger('lottery').setLevel(logging.WARNING)
logging.getLogger('lottery_ml').setLevel(logging.WARNING)

from src.lottery import LotteryAnalyzer

# 组合数 C(n,k)
def C(n, k):
    if k < 0 or k > n:
        return 0
    import math
    return math.comb(n, k)

# 前区35选5，推荐 m 码，开奖5码，命中≥4的纯随机概率
def front_ge4_random(m):
    # P(命中≥4) = [C(m,4)*C(35-m,1) + C(m,5)] / C(35,5)
    return (C(m, 4) * C(35 - m, 1) + C(m, 5)) / C(35, 5)

# 后区12选2，推荐 m 码，开奖2码
def back_ge1_random(m):
    # P(命中≥1) = 1 - C(12-m,2)/C(12,2)
    return 1 - C(12 - m, 2) / C(12, 2)

def back_ge2_random(m):
    # P(2码全中) = C(m,2)/C(12,2)  (从12选2，推荐m码含这2码的概率)
    return C(m, 2) / C(12, 2)


def rolling_eval(analyzer, front_n, back_n, trials=200, start_gap=50):
    """
    用模型排名回测：前区推 front_n 码，后区推 back_n 码。
    start_gap: 从倒数第 start_gap 期开始往前预测（保证训练窗口足够）。
    """
    saved_data = list(analyzer.history_data)
    saved_stats = dict(analyzer.statistics) if analyzer.statistics else {}

    front_ge4 = front_ge3 = front_ge2 = 0
    back_ge2 = back_ge1 = 0
    evaluated = 0

    # 用最近 trials 期做预测（从最新往前）
    max_i = min(len(saved_data) - 11, start_gap + trials - 1)
    min_i = start_gap  # 保证 history_data 足够大

    for i in range(min_i, max_i + 1):
        analyzer.history_data = list(saved_data[i + 1:])
        if len(analyzer.history_data) < 30:
            continue
        analyzer.update_statistics()

        actual = saved_data[i]
        actual_front = set(actual['front'])
        actual_back = set(actual['back'])
        evaluated += 1

        front_ranking = analyzer.get_ensemble_ranking(is_front=True)
        back_ranking = analyzer.get_ensemble_ranking(is_front=False)

        front_top = [r['number'] for r in front_ranking[:front_n]]
        back_top = [r['number'] for r in back_ranking[:back_n]]

        fc = len(actual_front & set(front_top))
        bc = len(actual_back & set(back_top))

        if fc >= 2: front_ge2 += 1
        if fc >= 3: front_ge3 += 1
        if fc >= 4: front_ge4 += 1
        if bc >= 1: back_ge1 += 1
        if bc >= 2: back_ge2 += 1

    analyzer.history_data = saved_data
    analyzer.statistics = saved_stats

    n = evaluated or 1
    return {
        'n': n,
        'front_ge2': front_ge2 / n,
        'front_ge3': front_ge3 / n,
        'front_ge4': front_ge4 / n,
        'back_ge1': back_ge1 / n,
        'back_ge2': back_ge2 / n,
    }


def main():
    print("=" * 70)
    print("大乐透命中率诊断 (真实数据 2898期)")
    print("=" * 70)
    analyzer = LotteryAnalyzer()
    print(f"历史期数: {len(analyzer.history_data)}")

    trials = 250
    start_gap = 50  # 从倒数第50期开始（数据充足）

    # ---------- 前区扫描 ----------
    print("\n" + "-" * 70)
    print(f"【前区扫描】模型排名命中≥4码概率 vs 纯随机 (trials={trials})")
    print("-" * 70)
    print(f"{'前区推码':>6} | {'模型ge4':>8} | {'随机ge4':>8} | {'模型ge3':>8} | {'模型ge2':>8}")
    front_results = {}
    for fn in [5, 8, 10, 12, 15, 18, 20]:
        r = rolling_eval(analyzer, fn, 3, trials=trials, start_gap=start_gap)
        rand = front_ge4_random(fn)
        front_results[fn] = r
        print(f"{fn:>6} | {r['front_ge4']*100:>7.2f}% | {rand*100:>7.2f}% | "
              f"{r['front_ge3']*100:>7.2f}% | {r['front_ge2']*100:>7.2f}%")

    # ---------- 后区扫描 ----------
    print("\n" + "-" * 70)
    print(f"【后区扫描】模型排名命中概率 vs 纯随机 (trials={trials})")
    print("-" * 70)
    print(f"{'后区推码':>6} | {'模型ge1':>8} | {'随机ge1':>8} | {'模型ge2':>8} | {'随机ge2':>8}")
    back_results = {}
    for bn in [3, 4, 5, 6, 8]:
        r = rolling_eval(analyzer, 5, bn, trials=trials, start_gap=start_gap)
        rand1 = back_ge1_random(bn)
        rand2 = back_ge2_random(bn)
        back_results[bn] = r
        print(f"{bn:>6} | {r['back_ge1']*100:>7.2f}% | {rand1*100:>7.2f}% | "
              f"{r['back_ge2']*100:>7.2f}% | {rand2*100:>7.2f}%")

    # ---------- 组合方案 ----------
    print("\n" + "=" * 70)
    print("【推荐方案组合】前区推N码 + 后区推M码")
    print("=" * 70)
    best_combos = []
    for fn in [10, 12, 15, 18]:
        for bn in [3, 4, 5]:
            r = rolling_eval(analyzer, fn, bn, trials=trials, start_gap=start_gap)
            best_combos.append((fn, bn, r))
            print(f"前区{fn:>2} + 后区{bn} → 前ge4={r['front_ge4']*100:>5.1f}% "
                  f"前ge3={r['front_ge3']*100:>5.1f}% | 后ge1={r['back_ge1']*100:>5.1f}% "
                  f"后ge2={r['back_ge2']*100:>5.1f}%")

    # 保存结果
    out = {
        'trials': trials,
        'start_gap': start_gap,
        'front_scan': {str(k): v for k, v in front_results.items()},
        'back_scan': {str(k): v for k, v in back_results.items()},
    }
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            'data', 'dlt_coverage_diagnostic.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == '__main__':
    main()
