"""快乐8 选5/6/7 & 选五复式 深度诊断 (walk-forward)

目标:
1. 量化当前 fair_coverage(确定性随机) 各玩法实际命中率 vs 超几何随机基线
2. 复测 reference 特征(频率+趋势+位置残差+邻号) 是否真有微弱但真实的信号
3. 测试选五复式扩大(7码->8/9码) 与 多注覆盖(选6/选7增强) 的性价比

严格按时间前推: history = raw[i+1:] (raw按issue降序)
"""
import sys, os, json, math, time
from collections import Counter
from itertools import combinations
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
print(f"总期数: {len(raw)}")

# ── 超几何随机基线 ──
def hg(pick, k):
    N, K = 80, 20
    return sum(comb(K, j) * comb(N - K, pick - j) / comb(N, pick) for j in range(k, pick + 1))

baselines = {
    'select_5': {'hit5': hg(5, 5), 'hit4': hg(5, 4)},
    'select_6': {'hit5': hg(6, 5), 'hit4': hg(6, 4)},
    'select_7': {'hit5': hg(7, 5), 'hit4': hg(7, 4)},
}

# ── 策略构造 ──
# fair: 真实 fair_coverage_v1 (确定性随机)
def fair_strategy(play):
    return resolve_play_strategy(play, allow_reference=True)

# ref: 在公平策略基础上覆盖特征权重为 reference 组合 (频率+趋势+位置残差+邻号)
REF_FW = {
    'frequency': 0.30, 'gap': 0.20, 'trend': 0.15, 'pair_cooccurrence': 0.05,
    'position_residual': 0.15, 'position_residual_cross': 0.10, 'road_residual': 0.05,
    'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0, 'seeded_random': 0.0,
}
def ref_strategy(play):
    s = resolve_play_strategy(play, allow_reference=True)
    s['feature_weights'] = dict(REF_FW)
    s['window_size'] = 100
    return s

FAIR_STRATS = {p: fair_strategy(p) for p in ['select_5', 'select_6', 'select_7']}
REF_STRATS = {p: ref_strategy(p) for p in ['select_5', 'select_6', 'select_7']}

# ── 单注生成 ──
def gen_single(analyzer, strat, pick_n):
    pool = analyzer.build_pool_by_strategy(strat, pool_size=20)
    cands = pool.get('candidates', [])[:20]
    if len(cands) < pick_n:
        return []
    cap = strat.get('pool_max_last_numbers') or _adaptive_repeat_cap(analyzer.history_data, pick_n)
    fp, _ = _select_final_candidate_pool(
        cands, pick_n, analyzer.statistics.get('last_numbers', set()),
        max_last_numbers=cap, selection_mode=strat.get('final_selection_mode', 'balanced'))
    return sorted(num for num, _ in fp)

# ── 选五复式 (fu_shi: pool_size码->C(pool_size,5)注) ──
def gen_fu_shi(analyzer, strat, pool_size):
    pool = analyzer.build_pool_by_strategy(strat, pool_size=pool_size)
    cands = pool.get('candidates', [])[:pool_size]
    if len(cands) < pool_size:
        return []
    cap = strat.get('pool_max_last_numbers') or _adaptive_repeat_cap(analyzer.history_data, pool_size)
    fp, _ = _select_final_candidate_pool(
        cands, pool_size, analyzer.statistics.get('last_numbers', set()),
        max_last_numbers=cap, selection_mode=strat.get('final_selection_mode', 'balanced'))
    core = sorted(num for num, _ in fp)
    return [sorted(c) for c in combinations(core, 5)]

# ── 统计容器 ──
stats = {}
for mode in ['fair', 'ref']:
    stats[mode] = {}
    for play in ['select_5', 'select_6', 'select_7']:
        stats[mode][play] = {'hit4': 0, 'hit5': 0, 'n': 0}
    stats[mode]['fu7_7'] = {'hit5': 0, 'hit4': 0, 'n': 0}
    stats[mode]['fu7_8'] = {'hit5': 0, 'hit4': 0, 'n': 0}
    stats[mode]['fu7_9'] = {'hit5': 0, 'hit4': 0, 'n': 0}

ms_stats = {
    'select_5_8x6': {'hit5': 0, 'hit4': 0, 'n': 0, 'any5': 0, 'any4': 0},
    'select_6_8x6': {'hit5': 0, 'hit4': 0, 'n': 0, 'any5': 0, 'any4': 0},
    'select_7_6x7': {'hit5': 0, 'hit4': 0, 'n': 0, 'any5': 0, 'any4': 0},
}

start = 120
t0 = time.time()
count = 0
for i in range(start, len(raw)):
    target = set(raw[i]['numbers'])
    history = raw[i + 1:]
    a = KL8Analyzer(history_file=None)
    a.history_data = history
    a.using_simulated_data = False
    a.update_statistics()
    if len(a.history_data) == 0:
        print(f"[WARN] i={i} history_len={len(history)} 跳过")
        continue

    for mode, strat_map in [('fair', FAIR_STRATS), ('ref', REF_STRATS)]:
        for play in ['select_5', 'select_6', 'select_7']:
            nums = gen_single(a, strat_map[play], int(play.split('_')[1]))
            if len(nums) != int(play.split('_')[1]):
                continue
            s = stats[mode][play]
            s['n'] += 1
            h = len(set(nums) & target)
            if h >= 5:
                s['hit5'] += 1
            if h >= 4:
                s['hit4'] += 1
        for pz, key in [(7, 'fu7_7'), (8, 'fu7_8'), (9, 'fu7_9')]:
            combos = gen_fu_shi(a, strat_map['select_5'], pz)
            if not combos:
                continue
            s = stats[mode][key]
            s['n'] += 1
            h5 = h4 = 0
            for c in combos:
                h = len(set(c) & target)
                if h >= 5:
                    h5 += 1
                if h >= 4:
                    h4 += 1
            if h5:
                s['hit5'] += 1
            if h4:
                s['hit4'] += 1

    for key, (play, nslip, psize) in [('select_5_8x6', (5, 8, 6)),
                                       ('select_6_8x6', (6, 8, 6)),
                                       ('select_7_6x7', (7, 6, None))]:
        try:
            slips = generate_multi_slips(a, play, nslip, psize)
        except Exception:
            slips = []
        if not slips:
            continue
        s = ms_stats[key]
        s['n'] += 1
        any5 = any4 = 0
        for sl in slips:
            h = len(set(sl) & target)
            if h >= 5:
                any5 += 1
            if h >= 4:
                any4 += 1
        if any5:
            s['hit5'] += 1
            s['any5'] += 1
        if any4:
            s['hit4'] += 1
            s['any4'] += 1
    count += 1

print(f"\nwalk-forward 完成: {count} 期, 耗时 {time.time()-t0:.1f}s\n")

def pct(x, n):
    return f"{x/n*100:.2f}%" if n else "n/a"

print("=" * 100)
print(f"{'玩法/方案':<20}{'模式':<6}{'中5(单注)':<13}{'中4(单注)':<13}{'中5基线':<11}{'中4基线':<11}")
print("=" * 100)
for play in ['select_5', 'select_6', 'select_7']:
    pb = baselines[play]
    for mode in ['fair', 'ref']:
        s = stats[mode][play]
        print(f"{play:<20}{mode:<6}{pct(s['hit5'], s['n']):<13}{pct(s['hit4'], s['n']):<13}{pb['hit5']*100:.2f}%{'':<6}{pb['hit4']*100:.2f}%")
    print()

print("选五复式 (组合中5/中4 比例):")
for mode in ['fair', 'ref']:
    for key, pz in [('fu7_7', '7码'), ('fu7_8', '8码'), ('fu7_9', '9码')]:
        s = stats[mode][key]
        print(f"  {mode:<5} 选五复式{pz:<4}: 组合中5={pct(s['hit5'], s['n'])}  组合中4={pct(s['hit4'], s['n'])}")
print()

print("多注覆盖 (当前实现, fair确定性随机扰动):")
for key in ['select_5_8x6', 'select_6_8x6', 'select_7_6x7']:
    s = ms_stats[key]
    print(f"  {key:<16}: 至少1组中5={pct(s['any5'], s['n'])}  至少1组中4={pct(s['any4'], s['n'])}")
print()

print("随机基线(超几何):")
for play in ['select_5', 'select_6', 'select_7']:
    pb = baselines[play]
    print(f"  {play}: 中5={pb['hit5']*100:.2f}%  中4={pb['hit4']*100:.2f}%")
