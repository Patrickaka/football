"""
双色球诊断脚本
==============
1. 当前评分系统排名区分度（开奖号能否排到 Top K）
2. 当前5注采样模式的真实命中率（基线）
3. 不同大底池尺寸的覆盖效果（优化目标参考）
"""
import json
import sys
from collections import Counter

sys.path.insert(0, '.')
from src.ssq import _analyze, _predict_sets, RED_RANGE, BLUE_RANGE, RECENT_WINDOW

DATA = json.load(open('data/ssq_history_full.json', encoding='utf-8'))
DATA.sort(key=lambda x: x['period'])
print(f'数据: {len(DATA)} 期 ({DATA[0]["period"]} ~ {DATA[-1]["period"]})')

# ---- 随机基准（纯组合概率）----
from math import comb
C33_6 = comb(33, 6)
def red_ge_k_prob(k, pool):
    """推 pool 个红球，开奖6个中命中>=k 的组合概率"""
    s = 0
    for j in range(k, min(pool, 6) + 1):
        s += comb(pool, j) * comb(33 - pool, 6 - j)
    return s / C33_6
C16_1 = 16
def blue_ge1_prob(pool):
    return 1 - comb(16 - pool, 1) / comb(16, 1)

# ---- walk-forward 诊断 ----
START = 100  # warmup
END = len(DATA) - 1

# 统计容器
red_topk_ge = {k: 0 for k in [10, 12, 15, 18, 20, 25]}
red_topk_ge5 = {k: 0 for k in [10, 12, 15, 18, 20, 25]}
red_topk_ge6 = {k: 0 for k in [10, 12, 15, 18, 20, 25]}
blue_topk_ge1 = {k: 0 for k in [3, 5, 8, 10]}

# 当前5注模式
five_red_ge = Counter()   # 5注中最佳红球命中数分布
five_red_ge4 = 0
five_red_ge5 = 0
five_red_ge6 = 0
five_blue_ge1 = 0         # 5注中任一注蓝球命中
five_combo = 0            # 红ge4 + 蓝ge1

n_eval = 0
for i in range(START, END):
    train = DATA[:i]
    actual = DATA[i]
    actual_red = set(actual['red'])
    actual_blue = actual['blue']
    n_eval += 1

    analysis = _analyze(train)

    # 评分权重（复刻 _predict_sets）
    red_w = [analysis['red_freq'][x] + 1.5 * analysis['red_recent'][x] + 0.5 for x in RED_RANGE]
    blue_w = [analysis['blue_freq'][x] + 1.5 * analysis['blue_recent'][x] + 0.3 for x in BLUE_RANGE]

    # 排名（按权重降序）
    red_rank = sorted(RED_RANGE, key=lambda x: -red_w[RED_RANGE.index(x)])
    blue_rank = sorted(BLUE_RANGE, key=lambda x: -blue_w[BLUE_RANGE.index(x)])

    for k in red_topk_ge:
        topk_red = set(red_rank[:k])
        hit = len(actual_red & topk_red)
        if hit >= 4:
            red_topk_ge[k] += 1
        if hit >= 5:
            red_topk_ge5[k] += 1
        if hit >= 6:
            red_topk_ge6[k] += 1
    for k in blue_topk_ge1:
        topk_blue = set(blue_rank[:k])
        if actual_blue in topk_blue:
            blue_topk_ge1[k] += 1

    # 当前5注模式（用原模块函数，seed=训练集最后一期期号）
    sets = _predict_sets(train, analysis, n=5, seed=int(train[-1]['period']))
    best_red = 0
    blue_hit = False
    for s in sets:
        rh = len(set(s['red']) & actual_red)
        best_red = max(best_red, rh)
        if s['blue'] == actual_blue:
            blue_hit = True
    five_red_ge[best_red] += 1
    if best_red >= 4:
        five_red_ge4 += 1
    if best_red >= 5:
        five_red_ge5 += 1
    if best_red >= 6:
        five_red_ge6 += 1
    if blue_hit:
        five_blue_ge1 += 1
    if best_red >= 4 and blue_hit:
        five_combo += 1

print()
print('=' * 64)
print('【诊断1】评分系统排名区分度（开奖红球落入 Top K 的比例）')
print('=' * 64)
print(f'{"TopK":>5} | {"红ge4":>7} | {"红ge5":>7} | {"红ge6":>7} | 随机红ge4')
print('-' * 50)
for k in [10, 12, 15, 18, 20, 25]:
    r4 = red_topk_ge[k] / n_eval * 100
    r5 = red_topk_ge5[k] / n_eval * 100
    r6 = red_topk_ge6[k] / n_eval * 100
    base = red_ge_k_prob(4, k) * 100
    print(f'{k:>5} | {r4:>6.1f}% | {r5:>6.1f}% | {r6:>6.1f}% | {base:>6.1f}%')

print()
print('蓝球 Top K 覆盖开奖（ge1）:')
for k in [3, 5, 8, 10]:
    b1 = blue_topk_ge1[k] / n_eval * 100
    base = blue_ge1_prob(k) * 100
    print(f'  Top{k:>2}: {b1:>6.1f}% (随机 {base:>5.1f}%)')

print()
print('=' * 64)
print('【诊断2】当前5注采样模式真实表现（基线）')
print('=' * 64)
print(f'评估期数: {n_eval}')
print(f'5注中最佳红球命中分布: {dict(sorted(five_red_ge.items()))}')
print(f'红球ge4命中率: {five_red_ge4/n_eval*100:.2f}%  (随机5注≈ {red_ge_k_prob(4,33)*5*100:.3f}% 不可比)')
print(f'红球ge5命中率: {five_red_ge5/n_eval*100:.2f}%')
print(f'红球ge6命中率: {five_red_ge6/n_eval*100:.2f}%')
print(f'蓝球ge1命中率: {five_blue_ge1/n_eval*100:.2f}%  (随机5注≈ {blue_ge1_prob(16)*100:.1f}%)')
print(f'红ge4+蓝ge1 综合: {five_combo/n_eval*100:.2f}%')

print()
print('=' * 64)
print('【诊断3】大底池覆盖目标（若输出排名Top K大底）')
print('=' * 64)
print(f'{"方案":>12} | {"红ge4":>7} | {"红ge5":>7} | {"蓝ge1":>7}')
print('-' * 50)
for rk, bk in [(15, 5), (18, 5), (20, 8), (25, 8)]:
    r4 = red_topk_ge[rk] / n_eval * 100
    r5 = red_topk_ge5[rk] / n_eval * 100
    b1 = blue_topk_ge1[bk] / n_eval * 100
    print(f'红{rk:>2}/蓝{bk:>2} | {r4:>6.1f}% | {r5:>6.1f}% | {b1:>6.1f}%')

# 保存结果
result = {
    'n_eval': n_eval,
    'red_topk_ge4': {k: red_topk_ge[k] / n_eval for k in red_topk_ge},
    'red_topk_ge5': {k: red_topk_ge5[k] / n_eval for k in red_topk_ge5},
    'red_topk_ge6': {k: red_topk_ge6[k] / n_eval for k in red_topk_ge6},
    'blue_topk_ge1': {k: blue_topk_ge1[k] / n_eval for k in blue_topk_ge1},
    'five_red_ge4': five_red_ge4 / n_eval,
    'five_red_ge5': five_red_ge5 / n_eval,
    'five_red_ge6': five_red_ge6 / n_eval,
    'five_blue_ge1': five_blue_ge1 / n_eval,
    'five_combo': five_combo / n_eval,
    'five_red_dist': dict(five_red_ge),
}
json.dump(result, open('data/diagnose_ssq_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('\n已保存 data/diagnose_ssq_result.json')
