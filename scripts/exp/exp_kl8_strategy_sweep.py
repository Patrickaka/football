"""快乐8 选5/选6 策略 walk-forward 扫描。

严格按时间前推（无未来数据泄漏）：用 raw[i+1:] 预测第 i 期。
对比多种策略配置的期望命中与中奖分布，找出 walk-forward 下真正优于随机基线的策略。
"""
import sys, os, json, math
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import logging
logging.disable(logging.CRITICAL)

import src.kl8 as kl8
from src.kl8 import (
    KL8Analyzer, _select_final_candidate_pool, _adaptive_repeat_cap,
)

random_seed = 20260716

raw = json.load(open('data/kl8_history.json', encoding='utf-8'))['results']
raw = sorted(raw, key=lambda r: r['issue'], reverse=True)  # 最新在前
print(f"总期数: {len(raw)}")

# ─── 待比较策略（仅改 feature_weights / model_weights / frequency_mode）───
STRATEGIES = {
    'current_cold_ref': {  # 当前线上回退的参考策略（冷号/均值回归）
        'feature_weights': {'frequency': 0.45, 'gap': 0.20, 'trend': 0.20,
                            'pair_cooccurrence': 0.10, 'position_residual': 0.05,
                            'position_residual_cross': 0.0, 'road_residual': 0.0,
                            'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'frequency_mode': 'mean_reversion',
        'window_size': 250,
    },
    'hot_momentum': {  # 热号动量（frequency 热模式）
        'feature_weights': {'frequency': 0.45, 'gap': 0.20, 'trend': 0.20,
                            'pair_cooccurrence': 0.10, 'position_residual': 0.05,
                            'position_residual_cross': 0.0, 'road_residual': 0.0,
                            'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'frequency_mode': 'hot',
        'window_size': 250,
    },
    'pure_freq_hot': {  # 纯频率热号
        'feature_weights': {'frequency': 1.0, 'position_residual': 0.0, 'road_residual': 0.0,
                            'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'frequency_mode': 'hot',
        'window_size': 250,
    },
    'pure_freq_cold': {  # 纯频率冷号（均值回归）
        'feature_weights': {'frequency': 1.0, 'position_residual': 0.0, 'road_residual': 0.0,
                            'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'frequency_mode': 'mean_reversion',
        'window_size': 250,
    },
    'bayesian_cold': {  # 内置贝叶斯后验（内部均值回归冷号）
        'feature_weights': {},
        'model_weights': {'rank': 0.0, 'bayesian': 1.0, 'markov': 0.0},
        'frequency_mode': 'mean_reversion',
        'window_size': 250,
    },
    'hot_trend_adj': {  # 热号 + 趋势 + 邻号
        'feature_weights': {'frequency': 0.5, 'trend': 0.3, 'adjacent': 0.2,
                            'position_residual': 0.0, 'road_residual': 0.0,
                            'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'frequency_mode': 'hot',
        'window_size': 250,
    },
}


def predict(analyzer, select_n, strat):
    pool_result = analyzer.build_pool_by_strategy(strat, pool_size=20)
    candidates = pool_result.get('candidates', [])[:20]
    if not candidates:
        return []
    repeat_cap = strat.get('pool_max_last_numbers') or _adaptive_repeat_cap(analyzer.history_data, select_n)
    final_pool, _ = _select_final_candidate_pool(
        candidates, select_n,
        analyzer.statistics.get('last_numbers', set()),
        max_last_numbers=repeat_cap,
        selection_mode=strat.get('final_selection_mode', 'balanced'),
    )
    return sorted(num for num, _ in final_pool)


def hypergeom_p_ge(pick, k):
    from math import comb
    N, K = 80, 20
    p = 0.0
    for j in range(k, pick + 1):
        p += comb(K, j) * comb(N - K, pick - j) / comb(N, pick)
    return p


def run(select_n):
    min_history = 120
    target_indices = range(len(raw) - min_history - 1, -1, -1)
    results = {name: {'hits': []} for name in STRATEGIES}
    for i in target_indices:
        target = raw[i]
        target_set = set(target['numbers'])
        history = raw[i + 1:]
        analyzer = KL8Analyzer(history_file=None)
        analyzer.history_data = history
        analyzer.using_simulated_data = False
        analyzer.update_statistics()
        for name, strat in STRATEGIES.items():
            nums = predict(analyzer, select_n, strat)
            if len(nums) != select_n:
                continue
            results[name]['hits'].append(len(set(nums) & target_set))

    print(f"\n{'='*100}\n选{select_n}  Walk-Forward 策略对比 (样本≈{len(results['current_cold_ref']['hits'])}期)\n{'='*100}")
    print(f"{'策略':<18}{'平均命中':>9}{'理论期望':>9}{'P(>=3)':>9}{'P(>=4)':>9}{'P(5全中)':>10}")
    print('-'*100)
    exp = select_n * 20 / 80
    for name in STRATEGIES:
        h = results[name]['hits']
        if not h:
            continue
        avg = sum(h) / len(h)
        p3 = sum(1 for x in h if x >= 3) / len(h)
        p4 = sum(1 for x in h if x >= 4) / len(h)
        p5 = sum(1 for x in h if x >= 5) / len(h)
        print(f"{name:<18}{avg:>9.3f}{exp:>9.3f}{p3*100:>8.2f}%{p4*100:>8.2f}%{p5*100:>9.2f}%")
    # 命中分布（仅对 best 打印）
    best = max(STRATEGIES, key=lambda n: sum(results[n]['hits']) / max(1, len(results[n]['hits'])))
    c = Counter(results[best]['hits'])
    dist = " ".join(f"{k}中:{c.get(k,0)}({100*c.get(k,0)/len(results[best]['hits']):.1f}%)" for k in range(0, select_n + 1))
    print(f"\n最高期望命中策略: {best}")
    print(f"命中分布: {dist}")


if __name__ == '__main__':
    run(5)
    run(6)
