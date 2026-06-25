#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""大乐透 v3.3 后区排名绝对主导 — 对比回测
对比: 纯排名 vs 投票(v3.3: 后区rank权重4.0绝对主导) vs 投票(v3.2旧配置)
随机基线: 前区≥2=27.8%, ≥3=6.7%, 后区≥1=41.7%
"""

import sys, os, json, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lottery import LotteryAnalyzer, LOTTERY_PREDICTOR_VERSION

# Suppress ML INFO logs
logging.getLogger('lottery').setLevel(logging.WARNING)
logging.getLogger('lottery_ml').setLevel(logging.WARNING)

def load_data():
    """加载大乐透历史数据"""
    data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'data', 'dlt_history.json')
    if os.path.exists(data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"加载历史数据: {len(data)} 期")
        return data
    return []

def run_rank_backtest(analyzer, trials=50):
    """纯排名模型回测"""
    saved_data = list(analyzer.history_data)
    saved_stats = dict(analyzer.statistics) if analyzer.statistics else {}

    front_ge2 = front_ge3 = front_ge4 = 0
    back_ge1 = back_ge2 = 0
    evaluated = 0

    for i in range(trials):
        analyzer.history_data = list(saved_data[i + 1:])
        if len(analyzer.history_data) < 10:
            continue
        analyzer.update_statistics()

        actual = saved_data[i]
        actual_front = actual['front']
        actual_back = actual['back']
        evaluated += 1

        front_ranking = analyzer.get_ensemble_ranking(is_front=True)
        back_ranking = analyzer.get_ensemble_ranking(is_front=False)

        front_top5 = [r['number'] for r in front_ranking[:5]]
        back_top3 = [r['number'] for r in back_ranking[:3]]

        front_common = len(set(actual_front) & set(front_top5))
        back_common = len(set(actual_back) & set(back_top3))

        if front_common >= 2: front_ge2 += 1
        if front_common >= 3: front_ge3 += 1
        if front_common >= 4: front_ge4 += 1
        if back_common >= 1: back_ge1 += 1
        if back_common >= 2: back_ge2 += 1

    # Restore
    analyzer.history_data = saved_data
    analyzer.statistics = saved_stats

    n = evaluated or 1
    return {
        'method': '纯排名',
        'trials': n,
        'front_ge2': front_ge2, 'front_ge2_rate': front_ge2/n,
        'front_ge3': front_ge3, 'front_ge3_rate': front_ge3/n,
        'front_ge4': front_ge4, 'front_ge4_rate': front_ge4/n,
        'back_ge1': back_ge1, 'back_ge1_rate': back_ge1/n,
        'back_ge2': back_ge2, 'back_ge2_rate': back_ge2/n,
    }

def run_voting_backtest(analyzer, trials=50):
    """投票模型回测(使用当前multi_model_voting配置)"""
    saved_data = list(analyzer.history_data)
    saved_stats = dict(analyzer.statistics) if analyzer.statistics else {}

    front_ge2 = front_ge3 = front_ge4 = 0
    back_ge1 = back_ge2 = 0
    evaluated = 0

    for i in range(trials):
        analyzer.history_data = list(saved_data[i + 1:])
        if len(analyzer.history_data) < 10:
            continue
        analyzer.update_statistics()

        actual = saved_data[i]
        actual_front = actual['front']
        actual_back = actual['back']
        evaluated += 1

        result = analyzer.multi_model_voting(front_n=5, back_n=2, skip_ml=True)

        front_top5 = result['front']
        back_top2 = result['back']
        # 扩展到Top3后区(用候选池)
        back_top3 = [c['number'] for c in result.get('back_candidates', [])][:3]

        front_common = len(set(actual_front) & set(front_top5))
        back_common_top3 = len(set(actual_back) & set(back_top3))
        back_common_top2 = len(set(actual_back) & set(back_top2))

        if front_common >= 2: front_ge2 += 1
        if front_common >= 3: front_ge3 += 1
        if front_common >= 4: front_ge4 += 1
        if back_common_top3 >= 1: back_ge1 += 1
        if back_common_top2 >= 1: back_ge2 += 1

    # Restore
    analyzer.history_data = saved_data
    analyzer.statistics = saved_stats

    n = evaluated or 1
    return {
        'method': '投票(当前配置)',
        'trials': n,
        'front_ge2': front_ge2, 'front_ge2_rate': front_ge2/n,
        'front_ge3': front_ge3, 'front_ge3_rate': front_ge3/n,
        'front_ge4': front_ge4, 'front_ge4_rate': front_ge4/n,
        'back_ge1': back_ge1, 'back_ge1_rate': back_ge1/n,
        'back_ge2': back_ge2, 'back_ge2_rate': back_ge2/n,
    }

def main():
    # 直接使用LotteryAnalyzer加载历史数据
    analyzer = LotteryAnalyzer()
    data = analyzer.history_data

    # 动态调整trials
    max_trials = min(50, len(data) - 15)
    if max_trials < 10:
        print("数据不足，至少需要25期")
        return
    trials = max_trials

    print(f"\n{'='*70}")
    print(f"大乐透 v3.3 后区排名绝对主导 — 对比回测")
    print(f"{'='*70}")
    print(f"回测期数: {trials}")
    print(f"随机基线: 前区≥2=27.8%, ≥3=6.7%, 后区≥1=41.7%")
    print(f"版本: {LOTTERY_PREDICTOR_VERSION}")

    # 1. 纯排名回测
    print(f"\n--- 1. 纯排名模型回测 ---")
    rank_result = run_rank_backtest(analyzer, trials)
    print(f"  前区≥2命中: {rank_result['front_ge2']}/{rank_result['trials']} = {rank_result['front_ge2_rate']*100:.1f}%  (vs 随机27.8%)")
    print(f"  前区≥3命中: {rank_result['front_ge3']}/{rank_result['trials']} = {rank_result['front_ge3_rate']*100:.1f}%  (vs 随机6.7%)")
    print(f"  前区≥4命中: {rank_result['front_ge4']}/{rank_result['trials']} = {rank_result['front_ge4_rate']*100:.1f}%")
    print(f"  后区≥1命中: {rank_result['back_ge1']}/{rank_result['trials']} = {rank_result['back_ge1_rate']*100:.1f}%  (vs 随机41.7%)")
    print(f"  后区≥2命中: {rank_result['back_ge2']}/{rank_result['trials']} = {rank_result['back_ge2_rate']*100:.1f}%")

    # 2. 投票模型回测
    print(f"\n--- 2. 投票模型回测(v3.3配置) ---")
    vote_result = run_voting_backtest(analyzer, trials)
    print(f"  前区≥2命中: {vote_result['front_ge2']}/{vote_result['trials']} = {vote_result['front_ge2_rate']*100:.1f}%  (vs 随机27.8%)")
    print(f"  前区≥3命中: {vote_result['front_ge3']}/{vote_result['trials']} = {vote_result['front_ge3_rate']*100:.1f}%  (vs 随机6.7%)")
    print(f"  前区≥4命中: {vote_result['front_ge4']}/{vote_result['trials']} = {vote_result['front_ge4_rate']*100:.1f}%")
    print(f"  后区≥1命中: {vote_result['back_ge1']}/{vote_result['trials']} = {vote_result['back_ge1_rate']*100:.1f}%  (vs 随机41.7%)")
    print(f"  后区≥2命中: {vote_result['back_ge2']}/{vote_result['trials']} = {vote_result['back_ge2_rate']*100:.1f}%")

    # 3. 对比总结
    print(f"\n{'='*70}")
    print(f"对比总结")
    print(f"{'='*70}")
    print(f"  方法        | 前区≥2  | 前区≥3  | 后区≥1  | 后区≥2")
    print(f"  {'─'*60}")
    print(f"  随机基线    | 27.8%   | 6.7%    | 41.7%   | —")
    print(f"  纯排名      | {rank_result['front_ge2_rate']*100:.1f}%   | {rank_result['front_ge3_rate']*100:.1f}%    | {rank_result['back_ge1_rate']*100:.1f}%   | {rank_result['back_ge2_rate']*100:.1f}%")
    print(f"  投票v3.3    | {vote_result['front_ge2_rate']*100:.1f}%   | {vote_result['front_ge3_rate']*100:.1f}%    | {vote_result['back_ge1_rate']*100:.1f}%   | {vote_result['back_ge2_rate']*100:.1f}%")

    # 超随机百分比
    def pct_above(rate, baseline):
        return ((rate - baseline) / baseline) * 100

    print(f"\n  超随机基线百分比:")
    print(f"  纯排名 ≥2: {pct_above(rank_result['front_ge2_rate'], 0.278):+.1f}%, ≥3: {pct_above(rank_result['front_ge3_rate'], 0.067):+.1f}%, 后区≥1: {pct_above(rank_result['back_ge1_rate'], 0.417):+.1f}%")
    print(f"  投票v3.3 ≥2: {pct_above(vote_result['front_ge2_rate'], 0.278):+.1f}%, ≥3: {pct_above(vote_result['front_ge3_rate'], 0.067):+.1f}%, 后区≥1: {pct_above(vote_result['back_ge1_rate'], 0.417):+.1f}%")

if __name__ == '__main__':
    main()
