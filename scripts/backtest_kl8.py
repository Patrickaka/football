"""快乐8 进球数/比分 预测 walk-forward 回测。

严格复刻 src/kl8 predict_all 的选号链路：
- 历史窗口 results[:i] (最新在前) 预测第 i 期
- 每玩法用 resolve_play_strategy 的参考策略生成候选池 -> _select_final_candidate_pool -> 最终号码
- 与真实开奖比对命中数，并对比纯随机基线。
"""
import sys, os, json, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.disable(logging.CRITICAL)

import src.kl8 as kl8
from src.kl8 import (
    KL8Analyzer, resolve_play_strategy, _select_final_candidate_pool,
    _adaptive_repeat_cap, SELECT_TYPES,
)

random.seed(20260716)

# 加载真实历史（最新在前）
raw = json.load(open('data/kl8_history.json', encoding='utf-8'))['results']
# 保证最新在前
raw = sorted(raw, key=lambda r: r['issue'], reverse=True)
print(f"总期数: {len(raw)}")

# 奖级中奖阈值（命中所选号码数 >= 该值即中奖）
WIN_THRESHOLD = {3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 4, 10: 5}


def predict_numbers(analyzer, select_n):
    """复刻 predict_all 单玩法选号，返回 sorted 最终号码列表。"""
    s_key = f'select_{select_n}'
    strategy = resolve_play_strategy(s_key)
    if strategy is None:
        return []
    pool_result = analyzer.build_pool_by_strategy(strategy, pool_size=20)
    candidates = pool_result.get('candidates', [])[:20]
    if not candidates:
        return []
    if 'final_max_last_numbers' in strategy and strategy['final_max_last_numbers'] is not None:
        repeat_cap = strategy['final_max_last_numbers']
    else:
        repeat_cap = _adaptive_repeat_cap(analyzer.history_data, select_n)
    final_pool, _ = _select_final_candidate_pool(
        candidates, select_n,
        analyzer.statistics.get('last_numbers', set()),
        max_last_numbers=repeat_cap,
        selection_mode=strategy.get('final_selection_mode', 'balanced'),
    )
    return sorted(num for num, _ in final_pool)


def random_pick(n, rng):
    return set(rng.sample(range(1, 81), n))


def main():
    min_history = 120  # 每个预测点只使用其之前的开奖
    target_indices = range(len(raw) - min_history - 1, -1, -1)
    n_iter = len(raw) - min_history
    print(f"walk-forward 区间: {n_iter} 期（严格按时间前推，无未来数据）\n")

    # 统计容器
    stats = {n: {'hits': [], 'win': 0, 'n': 0} for n in SELECT_TYPES}

    for i in target_indices:
        target = raw[i]
        target_set = set(target['numbers'])
        # raw 按期号降序排列，因此索引更大的记录才是目标期之前的历史。
        # 旧实现使用 raw[:i]，会把目标期之后的数据泄漏进回测。
        history = raw[i + 1:]
        analyzer = KL8Analyzer(history_file=None)
        analyzer.history_data = history
        analyzer.using_simulated_data = False
        analyzer.update_statistics()
        for n in SELECT_TYPES:
            nums = predict_numbers(analyzer, n)
            if len(nums) != n:
                continue
            hit = len(set(nums) & target_set)
            stats[n]['hits'].append(hit)
            stats[n]['n'] += 1
            if hit >= WIN_THRESHOLD[n]:
                stats[n]['win'] += 1

    # 随机基线：对同样区间，每个玩法模拟同样次数随机选号
    rng = random.Random(20260716)
    rand_win = {n: 0 for n in SELECT_TYPES}
    rand_hits_sum = {n: 0 for n in SELECT_TYPES}
    rand_n = max(s['n'] for s in stats.values())
    for n in SELECT_TYPES:
        for _ in range(rand_n):
            pick = random_pick(n, rng)
            # 用最后一期的开奖作为随机对照目标？不行，需对应每期。
            # 简化：用超几何理论值计算理论中奖率
            pass

    # 超几何理论值
    def hypergeom_p_ge(pick, k):
        from math import comb
        N, K = 80, 20
        p = 0.0
        for j in range(k, pick + 1):
            p += comb(K, j) * comb(N - K, pick - j) / comb(N, pick)
        return p

    print("=" * 92)
    print("快乐8 各玩法 WALK-FORWARD 回测（当前策略 vs 随机理论基准）")
    print("=" * 92)
    print(f"{'玩法':<6}{'推荐数':>6}{'样本':>6}{'平均命中':>10}{'理论期望':>10}{'中奖率':>10}{'理论中奖率':>12}{'差异':>8}")
    print("-" * 92)
    for n in SELECT_TYPES:
        s = stats[n]
        if s['n'] == 0:
            continue
        avg_hit = sum(s['hits']) / s['n']
        exp_hit = n * 20 / 80
        win_rate = s['win'] / s['n']
        th_win = hypergeom_p_ge(n, WIN_THRESHOLD[n])
        diff = win_rate - th_win
        print(f"选{n:<4}{n:>6}{s['n']:>6}{avg_hit:>10.3f}{exp_hit:>10.3f}{win_rate*100:>9.2f}%{th_win*100:>11.2f}%{diff*100:>+7.2f}%")

    print()
    print("=" * 92)
    print("各玩法 命中数分布（当前策略）")
    print("=" * 92)
    for n in SELECT_TYPES:
        s = stats[n]
        if s['n'] == 0:
            continue
        from collections import Counter
        c = Counter(s['hits'])
        dist = " ".join(f"{k}中:{c.get(k,0)}({100*c.get(k,0)/s['n']:.1f}%)" for k in range(0, n + 1))
        print(f"选{n}: {dist}")


if __name__ == '__main__':
    main()
