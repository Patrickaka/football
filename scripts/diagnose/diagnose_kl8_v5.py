"""快乐8 覆盖杠杆性价比诊断 (walk-forward, step=2 加速)

验证三个优化方向的数据支撑:
1. 选五复式扩大: 7/8/9码 组合中5/中4 vs 成本(注数×2元)
2. 选7多注增强: 6/8/10/12组×7码 组合中5/中4
3. 选5多注增强: 8/10/12组×6码 组合中5/中4
4. 单注形态: top_ranked vs shape_balanced 单注中4命中率
"""
import sys, os, json, math, time
from collections import Counter
from itertools import combinations
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import logging
logging.disable(logging.CRITICAL)

import src.kl8 as kl8
from src.kl8 import (
    KL8Analyzer, resolve_play_strategy, _select_final_candidate_pool,
    _adaptive_repeat_cap, generate_multi_slips,
)
from math import comb

raw = json.load(open('data/kl8_history.json', encoding='utf-8'))['results']
raw = sorted(raw, key=lambda r: r['issue'], reverse=True)

def fair_strat(play, mode='shape_balanced'):
    s = resolve_play_strategy(play, allow_reference=True)
    s['final_selection_mode'] = mode
    return s

def gen_single(a, strat, pick_n):
    pool = a.build_pool_by_strategy(strat, pool_size=20)
    cands = pool.get('candidates', [])[:20]
    if len(cands) < pick_n:
        return []
    cap = strat.get('pool_max_last_numbers') or _adaptive_repeat_cap(a.history_data, pick_n)
    fp, _ = _select_final_candidate_pool(cands, pick_n, a.statistics.get('last_numbers', set()),
                                          max_last_numbers=cap, selection_mode=strat.get('final_selection_mode', 'balanced'))
    return sorted(num for num, _ in fp)

def gen_fu(a, strat, pz):
    pool = a.build_pool_by_strategy(strat, pool_size=pz)
    cands = pool.get('candidates', [])[:pz]
    if len(cands) < pz:
        return []
    cap = strat.get('pool_max_last_numbers') or _adaptive_repeat_cap(a.history_data, pz)
    fp, _ = _select_final_candidate_pool(cands, pz, a.statistics.get('last_numbers', set()),
                                          max_last_numbers=cap, selection_mode=strat.get('final_selection_mode', 'balanced'))
    core = sorted(num for num, _ in fp)
    return [sorted(c) for c in combinations(core, 5)]

# 统计
fu_stats = {pz: {'h5': 0, 'h4': 0, 'n': 0} for pz in [7, 8, 9, 10]}
ms7 = {g: {'h5': 0, 'h4': 0, 'n': 0} for g in [6, 8, 10, 12]}
ms5 = {g: {'h5': 0, 'h4': 0, 'n': 0} for g in [8, 10, 12]}
single_bal = {'h4': 0, 'h5': 0, 'n': 0}
single_top = {'h4': 0, 'h5': 0, 'n': 0}

t0 = time.time()
step = 2
n_iter = 0
for i in range(120, len(raw), step):
    target = set(raw[i]['numbers'])
    history = raw[i + 1:]
    a = KL8Analyzer(history_file=None)
    a.history_data = history
    a.using_simulated_data = False
    a.update_statistics()
    if len(a.history_data) == 0:
        continue
    n_iter += 1

    # 选五复式 7/8/9/10码
    s5 = fair_strat('select_5')
    for pz in [7, 8, 9, 10]:
        combos = gen_fu(a, s5, pz)
        if not combos:
            continue
        fu_stats[pz]['n'] += 1
        h5 = h4 = 0
        for c in combos:
            h = len(set(c) & target)
            if h >= 5: h5 += 1
            if h >= 4: h4 += 1
        if h5: fu_stats[pz]['h5'] += 1
        if h4: fu_stats[pz]['h4'] += 1

    # 选7多注 6/8/10/12组×7码
    for g in [6, 8, 10, 12]:
        try:
            slips = generate_multi_slips(a, 7, g, 7)
        except Exception:
            slips = []
        if not slips:
            continue
        ms7[g]['n'] += 1
        any5 = any4 = 0
        for sl in slips:
            h = len(set(sl) & target)
            if h >= 5: any5 += 1
            if h >= 4: any4 += 1
        if any5: ms7[g]['h5'] += 1
        if any4: ms7[g]['h4'] += 1

    # 选5多注 8/10/12组×6码
    for g in [8, 10, 12]:
        try:
            slips = generate_multi_slips(a, 5, g, 6)
        except Exception:
            slips = []
        if not slips:
            continue
        ms5[g]['n'] += 1
        any5 = any4 = 0
        for sl in slips:
            h = len(set(sl) & target)
            if h >= 5: any5 += 1
            if h >= 4: any4 += 1
        if any5: ms5[g]['h5'] += 1
        if any4: ms5[g]['h4'] += 1

    # 单注 top_ranked vs shape_balanced
    nums_bal = gen_single(a, fair_strat('select_5', 'shape_balanced'), 5)
    if len(nums_bal) == 5:
        single_bal['n'] += 1
        h = len(set(nums_bal) & target)
        if h >= 4: single_bal['h4'] += 1
        if h >= 5: single_bal['h5'] += 1
    nums_top = gen_single(a, fair_strat('select_5', 'top_ranked'), 5)
    if len(nums_top) == 5:
        single_top['n'] += 1
        h = len(set(nums_top) & target)
        if h >= 4: single_top['h4'] += 1
        if h >= 5: single_top['h5'] += 1

print(f"walk-forward: {n_iter} 期, 耗时 {time.time()-t0:.1f}s\n")

def pct(x, n):
    return f"{x/n*100:.2f}%" if n else "n/a"

print("─" * 70)
print("选五复式 (组合中5/中4 vs 成本=注数×2元):")
print(f"  {'码数':<6}{'注数':<8}{'组合中5':<12}{'组合中4':<12}{'成本/期':<10}{'中5/百元':<10}")
for pz in [7, 8, 9, 10]:
    n_combo = comb(pz, 5)
    cost = n_combo * 2
    s = fu_stats[pz]
    r5 = s['h5']/s['n']*100 if s['n'] else 0
    r4 = s['h4']/s['n']*100 if s['n'] else 0
    print(f"  {pz}码{'':<2}{n_combo:<8}{pct(s['h5'],s['n']):<12}{pct(s['h4'],s['n']):<12}{cost:<10}{r5/cost*100:.2f}%")

print("\n选7多注覆盖 (组数×7码):")
print(f"  {'组数':<6}{'组合中5':<12}{'组合中4':<12}")
for g in [6, 8, 10, 12]:
    s = ms7[g]
    print(f"  {g}组{'':<2}{pct(s['h5'],s['n']):<12}{pct(s['h4'],s['n']):<12}")

print("\n选5多注覆盖 (组数×6码):")
print(f"  {'组数':<6}{'组合中5':<12}{'组合中4':<12}")
for g in [8, 10, 12]:
    s = ms5[g]
    print(f"  {g}组{'':<2}{pct(s['h5'],s['n']):<12}{pct(s['h4'],s['n']):<12}")

print("\n单注形态 (选5单注中4命中率):")
print(f"  shape_balanced: 中5={pct(single_bal['h5'],single_bal['n'])} 中4={pct(single_bal['h4'],single_bal['n'])} (随机中4=1.27%)")
print(f"  top_ranked:     中5={pct(single_top['h5'],single_top['n'])} 中4={pct(single_top['h4'],single_top['n'])} (随机中4=1.27%)")
