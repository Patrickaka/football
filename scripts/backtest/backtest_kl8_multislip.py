"""快乐8 多注覆盖方案 walk-forward 回测。

对比两种策略在 731 期历史上“命中4+/5+”的表现：
- 单注：取参考策略最终推荐的最优选6
- 多注：生成 N 组差异化选6，统计“至少一组命中≥4 / ≥5”的比例

用于诚实量化“多注覆盖”这一杠杆能提升多少组合层面命中率。
"""
import sys, os, json, logging
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
logging.disable(logging.CRITICAL)

import src.kl8 as kl8
from src.kl8 import (
    KL8Analyzer, resolve_play_strategy, generate_multi_slips,
    _select_final_candidate_pool, _adaptive_repeat_cap,
)

from math import comb
def hypergeom_p_ge(pick, k):
    N, K = 80, 20
    p = 0.0
    for j in range(k, pick + 1):
        p += comb(K, j) * comb(N - K, pick - j) / comb(N, pick)
    return p


raw = json.load(open('data/kl8_history.json', encoding='utf-8'))['results']
raw = sorted(raw, key=lambda r: r['issue'], reverse=True)

N_SLIPS = 8
sel_key = 'select_6'
sel_n = 6

single_hit4 = single_hit5 = 0
ms_any4 = ms_any5 = 0
best_hits_sum = 0
n = 0
best_dist = Counter()

for i in range(120, len(raw)):
    target = set(raw[i]['numbers'])
    history = raw[:i]
    a = KL8Analyzer(history_file=None)
    a.history_data = history
    a.using_simulated_data = False
    a.update_statistics()

    strat = resolve_play_strategy(sel_key, allow_reference=True)
    if strat is None:
        continue
    pool = a.build_pool_by_strategy(strat, pool_size=20)
    cands = pool.get('candidates', [])[:20]
    if len(cands) < sel_n:
        continue
    cap = strat.get('final_max_last_numbers', _adaptive_repeat_cap(a.history_data, sel_n))
    fp, _ = _select_final_candidate_pool(
        cands, sel_n, a.statistics.get('last_numbers', set()),
        max_last_numbers=cap, selection_mode=strat.get('final_selection_mode', 'balanced'),
    )
    single = sorted(num for num, _ in fp)
    sh = len(set(single) & target)
    if sh >= 4:
        single_hit4 += 1
    if sh >= 5:
        single_hit5 += 1

    slips = generate_multi_slips(a, sel_n, N_SLIPS)
    if not slips:
        continue
    best = 0
    any4 = any5 = False
    for s in slips:
        h = len(set(s) & target)
        best = max(best, h)
        if h >= 4:
            any4 = True
        if h >= 5:
            any5 = True
    best_dist[best] += 1
    best_hits_sum += best
    if any4:
        ms_any4 += 1
    if any5:
        ms_any5 += 1
    n += 1

print(f"样本: {n} 期, n_slips={N_SLIPS} (选6)")
print(f"理论单注 命中≥4: {hypergeom_p_ge(6,4)*100:.2f}%   命中≥5: {hypergeom_p_ge(6,5)*100:.2f}%")
print("-" * 70)
print(f"单注选6    命中≥4: {single_hit4/n*100:.2f}%   命中≥5: {single_hit5/n*100:.2f}%")
print(f"多注{N_SLIPS}组 至少1组命中≥4: {ms_any4/n*100:.2f}%   命中≥5: {ms_any5/n*100:.2f}%")
print(f"多注最优单组平均命中: {best_hits_sum/n:.3f} (单注理论期望1.5)")
print("-" * 70)
print("多注最优组命中分布:", " ".join(
    f"{k}中:{best_dist.get(k,0)}({100*best_dist.get(k,0)/n:.1f}%)" for k in range(0, 7)))
