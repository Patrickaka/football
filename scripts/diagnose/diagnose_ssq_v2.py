"""
双色球消融诊断 v2
=================
测试两个优化方向的真实增益：
A. 评分加「遗漏值」维度是否能提升排名区分度（Top15红ge4）
B. 单式5注改为「大底池内采样」是否能提升红ge4命中率
"""
import json
import sys
import random
from math import comb

sys.path.insert(0, '.')
from src.ssq import _analyze, _weighted_sample, _is_valid_red, RED_RANGE, BLUE_RANGE

DATA = json.load(open('data/ssq_history_full.json', encoding='utf-8'))
DATA.sort(key=lambda x: x['period'])
print(f'数据: {len(DATA)} 期')

def omission_norm(history, span):
    last_seen = {}
    for idx, r in enumerate(history):
        for n in r['red']:
            last_seen[n] = idx
    cur = len(history)
    omit = {n: (cur - last_seen.get(n, -1000)) for n in span}
    mx = max(omit.values()) or 1
    return {n: omit[n] / mx for n in span}

def predict_from_pool(train, analysis, pool, n=5, seed=None, a_omit=0.0):
    """从指定候选池内加权采样 n 注（大底池采样模式）"""
    rng = random.Random(seed)
    prev_red = set(train[-1]['red']) if train else set()
    prev_blue = train[-1]['blue'] if train else None
    omit = omission_norm(train, RED_RANGE) if a_omit > 0 else None
    red_w_full = []
    for x in RED_RANGE:
        w = analysis['red_freq'][x] + 1.5 * analysis['red_recent'][x] + 0.5
        if omit:
            w += a_omit * omit[x] * (analysis['red_freq'][x] + 1.5 * analysis['red_recent'][x] + 0.5)
        red_w_full.append(w)
    blue_w = [analysis['blue_freq'][x] + 1.5 * analysis['blue_recent'][x] + 0.3 for x in BLUE_RANGE]

    # 候选池索引
    pool_idx = [RED_RANGE.index(x) for x in pool]
    pool_w = [red_w_full[i] for i in pool_idx]

    sets = []
    tries = 0
    while len(sets) < n and tries < 5000:
        tries += 1
        red = sorted(_weighted_sample(pool_idx, pool_w, 6, rng))
        red_nums = [RED_RANGE[i] for i in red]
        if not _is_valid_red(red_nums):
            continue
        if set(red_nums) == prev_red:
            continue
        blue = _weighted_sample(BLUE_RANGE, blue_w, 1, rng)[0]
        if blue == prev_blue:
            blue = _weighted_sample(BLUE_RANGE, blue_w, 1, rng)[0]
        sets.append({'red': red_nums, 'blue': blue})
    while len(sets) < n:
        red_nums = sorted(rng.sample(pool, 6))
        blue = rng.choice(BLUE_RANGE)
        sets.append({'red': red_nums, 'blue': blue})
    return sets

START, END, STEP = 100, len(DATA) - 1, 2

# 方案统计
stats = {
    'base_top15_ge4': 0,          # 基线权重 Top15 红ge4 覆盖
    'omit05_top15_ge4': 0,        # +遗漏0.5 Top15 红ge4
    'omit10_top15_ge4': 0,        # +遗漏1.0 Top15 红ge4
    'base5_red_ge4': 0,           # 基线5注(全33采样) 红ge4
    'pool5_red_ge4': 0,           # 大底池15内采样5注 红ge4
    'pool5_red_ge5': 0,           # 大底池15内采样5注 红ge5
    'pool5_blue_ge1': 0,          # 大底池5注 蓝ge1
    'pool8_red_ge4': 0,           # 大底池15内采样8注 红ge4
}
n = 0
for i in range(START, END, STEP):
    train = DATA[:i]
    actual = DATA[i]
    ared, ablue = set(actual['red']), actual['blue']
    n += 1
    analysis = _analyze(train)
    red_w = [analysis['red_freq'][x] + 1.5 * analysis['red_recent'][x] + 0.5 for x in RED_RANGE]
    red_rank = sorted(RED_RANGE, key=lambda x: -red_w[RED_RANGE.index(x)])
    top15 = red_rank[:15]

    # 覆盖类
    if len(ared & set(top15)) >= 4:
        stats['base_top15_ge4'] += 1
    omit = omission_norm(train, RED_RANGE)
    red_w_o05 = [red_w[j] + 0.5 * omit[RED_RANGE[j]] * red_w[j] for j in range(33)]
    red_w_o10 = [red_w[j] + 1.0 * omit[RED_RANGE[j]] * red_w[j] for j in range(33)]
    rk05 = sorted(RED_RANGE, key=lambda x: -red_w_o05[RED_RANGE.index(x)])
    rk10 = sorted(RED_RANGE, key=lambda x: -red_w_o10[RED_RANGE.index(x)])
    if len(ared & set(rk05[:15])) >= 4:
        stats['omit05_top15_ge4'] += 1
    if len(ared & set(rk10[:15])) >= 4:
        stats['omit10_top15_ge4'] += 1

    # 5注类（用原模块 predict 逻辑需 import，这里复刻基线）
    from src.ssq import _predict_sets
    base_sets = _predict_sets(train, analysis, n=5, seed=int(train[-1]['period']))
    best = max(len(ared & set(s['red'])) for s in base_sets)
    if best >= 4:
        stats['base5_red_ge4'] += 1

    # 大底池内采样
    p5 = predict_from_pool(train, analysis, top15, n=5, seed=int(train[-1]['period']), a_omit=0.0)
    best5 = max(len(ared & set(s['red'])) for s in p5)
    if best5 >= 4:
        stats['pool5_red_ge4'] += 1
    if best5 >= 5:
        stats['pool5_red_ge5'] += 1
    if any(s['blue'] == ablue for s in p5):
        stats['pool5_blue_ge1'] += 1

    p8 = predict_from_pool(train, analysis, top15, n=8, seed=int(train[-1]['period']), a_omit=0.0)
    best8 = max(len(ared & set(s['red'])) for s in p8)
    if best8 >= 4:
        stats['pool8_red_ge4'] += 1

print(f'评估期数: {n}')
print()
print('=' * 60)
print('【A. 评分加遗漏对 Top15红ge4 覆盖的影响】')
print('=' * 60)
print(f'基线(频率+1.5近期):     {stats["base_top15_ge4"]/n*100:.1f}%')
print(f'+遗漏0.5:              {stats["omit05_top15_ge4"]/n*100:.1f}%')
print(f'+遗漏1.0:              {stats["omit10_top15_ge4"]/n*100:.1f}%')

print()
print('=' * 60)
print('【B. 单式生成模式对比（红ge4命中率）】')
print('=' * 60)
print(f'当前5注(全33采样):      {stats["base5_red_ge4"]/n*100:.2f}%  ← 基线')
print(f'大底池15内采样5注:      {stats["pool5_red_ge4"]/n*100:.2f}%  (红ge5={stats["pool5_red_ge5"]/n*100:.2f}%)')
print(f'大底池15内采样8注:      {stats["pool8_red_ge4"]/n*100:.2f}%')
print(f'大底池5注蓝ge1:         {stats["pool5_blue_ge1"]/n*100:.2f}%')

res = {k: v / n for k, v in stats.items()}
res['n'] = n
json.dump(res, open('data/diagnose_ssq_v2_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('\n已保存 data/diagnose_ssq_v2_result.json')
