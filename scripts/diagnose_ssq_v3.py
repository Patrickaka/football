"""
双色球消融诊断 v3：近期频率系数扫描 + 去重叠测试
=================================================
目标：确认近期频率系数是否可提升 5注红ge4，以及去重叠是否有用。
"""
import json
import sys
import random

sys.path.insert(0, '.')
from src.ssq import _analyze, _weighted_sample, _is_valid_red, RED_RANGE, BLUE_RANGE

DATA = json.load(open('data/ssq_history_full.json', encoding='utf-8'))
DATA.sort(key=lambda x: x['period'])

def predict_variant(train, analysis, n=5, seed=None, recent_coef=1.5, dedupe=False):
    rng = random.Random(seed)
    prev_red = set(train[-1]['red']) if train else set()
    prev_blue = train[-1]['blue'] if train else None
    red_w = [analysis['red_freq'][x] + recent_coef * analysis['red_recent'][x] + 0.5 for x in RED_RANGE]
    blue_w = [analysis['blue_freq'][x] + 1.5 * analysis['blue_recent'][x] + 0.3 for x in BLUE_RANGE]
    sets = []
    tries = 0
    used_reds = []
    while len(sets) < n and tries < 8000:
        tries += 1
        red = sorted(_weighted_sample(RED_RANGE, red_w, 6, rng))
        if not _is_valid_red(red):
            continue
        if set(red) == prev_red:
            continue
        if dedupe:
            # 要求与已选注重叠 < 4 个红球（避免高度雷同）
            if any(len(set(red) & s) >= 4 for s in used_reds):
                continue
        blue = _weighted_sample(BLUE_RANGE, blue_w, 1, rng)[0]
        if blue == prev_blue:
            blue = _weighted_sample(BLUE_RANGE, blue_w, 1, rng)[0]
        sets.append(red)
        used_reds.append(set(red))
    while len(sets) < n:
        red = sorted(rng.sample(RED_RANGE, 6))
        blue = rng.choice(BLUE_RANGE)
        sets.append(red)
    return sets

START, END, STEP = 100, len(DATA) - 1, 1  # 全量

# 近期系数扫描
coefs = [1.0, 1.5, 2.0, 2.5, 3.0]
stats = {f'c{c}': 0 for c in coefs}
stats['dedupe'] = 0
stats['base'] = 0
n = 0
for i in range(START, END, STEP):
    train = DATA[:i]
    actual = DATA[i]
    ared = set(actual['red'])
    n += 1
    analysis = _analyze(train)
    # 基线(调原模块)
    from src.ssq import _predict_sets
    base_sets = _predict_sets(train, analysis, n=5, seed=int(train[-1]['period']))
    if max(len(ared & set(s['red'])) for s in base_sets) >= 4:
        stats['base'] += 1
    for c in coefs:
        ps = predict_variant(train, analysis, n=5, seed=int(train[-1]['period']), recent_coef=c)
        if max(len(ared & set(s)) for s in ps) >= 4:
            stats[f'c{c}'] += 1
    pd = predict_variant(train, analysis, n=5, seed=int(train[-1]['period']), dedupe=True)
    if max(len(ared & set(s)) for s in pd) >= 4:
        stats['dedupe'] += 1

print(f'评估期数: {n}')
print()
print('=' * 55)
print('【近期频率系数扫描】5注红ge4命中率')
print('=' * 55)
print(f'基线(系数1.5, 原模块): {stats["base"]/n*100:.2f}%')
for c in coefs:
    print(f'系数{c}: {"":>8} {stats[f"c{c}"]/n*100:.2f}%')
print()
print(f'去重叠5注:            {stats["dedupe"]/n*100:.2f}%')

random_baseline = 5 * (comb6 := 1)  # 占位
# 随机5注红ge4概率
from math import comb
rb = 5 * sum(comb(6, j) * comb(27, 6 - j) for j in range(4, 7)) / comb(33, 6)
print(f'(随机5注红ge4基准: {rb*100:.2f}%)')

res = {k: v / n for k, v in stats.items()}
res['n'] = n
res['random_5_red_ge4'] = rb
json.dump(res, open('data/diagnose_ssq_v3_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('\n已保存 data/diagnose_ssq_v3_result.json')
