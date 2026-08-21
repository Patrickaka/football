"""
专项验证用户的"热号"思路：
  规则A（用户原话）：若某号在最近 N 期内出现次数占比 > 60%，则预测它下期出现。
  规则B（相对阈值）：取最近 N 期内出现次数最多的 Top-K 个号，预测下期出现。
  对照：公平摇奖下每个号下期出现概率恒为 20/80 = 0.25。

目的：用真实历史量化"热号下期是否更可能出"。
"""
import json
from collections import Counter

results = json.load(open('data/kl8_history.json', encoding='utf-8'))['results']
results = sorted(results, key=lambda r: r['issue'])  # 按时间升序
draws = [set(r['numbers']) for r in results]
TOTAL = 80


def run_user_rule(N, threshold=0.60):
    """规则A：占比 > threshold 的号预测下期出现。"""
    hot_total = 0      # 累计被预测(热)的号次数
    hot_hit = 0        # 其中下期真出的次数
    draws_with_hot = 0 # 有多少期存在热号
    all_rates = []     # 收集所有号的占比，看分布
    for t in range(N, len(draws) - 1):
        window = draws[t - N:t]
        cnt = Counter()
        for s in window:
            cnt.update(s)
        hot = [i for i in range(1, TOTAL + 1) if cnt[i] / N > threshold]
        all_rates.extend([cnt[i] / N for i in range(1, TOTAL + 1)])
        if hot:
            draws_with_hot += 1
            nxt = draws[t]
            hot_total += len(hot)
            hot_hit += len(set(hot) & nxt)
    rate = hot_hit / hot_total if hot_total else float('nan')
    # 占比 > 60% 的号有多少（整体频率）
    over = sum(1 for x in all_rates if x > threshold) / len(all_rates)
    return {
        'hot_total': hot_total,
        'hot_hit': hot_hit,
        'empirical_rate': rate,
        'baseline': 0.25,
        'draws_with_hot': draws_with_hot,
        'draws_total': len(draws) - 1 - N,
        'frac_numbers_over60pct': over,
    }


def run_topk_rule(N, K):
    """规则B：最近 N 期出现最多的 Top-K 号，预测下期出现。"""
    pred_total = 0
    pred_hit = 0
    for t in range(N, len(draws) - 1):
        window = draws[t - N:t]
        cnt = Counter()
        for s in window:
            cnt.update(s)
        topk = [i for i, _ in cnt.most_common(K)]
        nxt = draws[t]
        pred_total += len(topk)
        pred_hit += len(set(topk) & nxt)
    return {
        'pred_total': pred_total,
        'pred_hit': pred_hit,
        'empirical_rate': pred_hit / pred_total if pred_total else float('nan'),
        'baseline': 0.25,
        'K': K,
    }


print('=' * 70)
print('规则A：占比 > 60% 预测下期（用户原话）')
print('=' * 70)
for N in (50, 100):
    r = run_user_rule(N, 0.60)
    print(f'\n窗口 N={N}:')
    print(f'  有热号(>60%)的期数: {r["draws_with_hot"]} / {r["draws_total"]} '
          f'({100*r["draws_with_hot"]/r["draws_total"]:.2f}%)')
    print(f'  整体号码中占比>60%的比例: {100*r["frac_numbers_over60pct"]:.4f}% '
          f'(说明60%阈值有多罕见)')
    if r['hot_total']:
        print(f'  热号累计预测 {r["hot_total"]} 次，下期真出 {r["hot_hit"]} 次')
        print(f'  实测下期出现率 = {100*r["empirical_rate"]:.2f}%  | 随机基线 = 25.00%')
    else:
        print('  没有任何号码达到 >60% 阈值，规则从未触发。')

print('\n' + '=' * 70)
print('补充：60% 阈值下各档占比分布（说明为何几乎不触发）')
print('=' * 70)
N = 50
cnt_all = Counter()
all_rates = []
for t in range(N, len(draws)):
    window = draws[t - N:t]
    c = Counter()
    for s in window:
        c.update(s)
    all_rates.extend([c[i] / N for i in range(1, TOTAL + 1)])
buckets = {'<20%': 0, '20-30%': 0, '30-40%': 0, '40-50%': 0, '50-60%': 0, '>=60%': 0}
for x in all_rates:
    if x < 0.20: buckets['<20%'] += 1
    elif x < 0.30: buckets['20-30%'] += 1
    elif x < 0.40: buckets['30-40%'] += 1
    elif x < 0.50: buckets['40-50%'] += 1
    elif x < 0.60: buckets['50-60%'] += 1
    else: buckets['>=60%'] += 1
for k, v in buckets.items():
    print(f'  占比 {k}: {v} 个号码窗口 ({100*v/len(all_rates):.1f}%)')

print('\n' + '=' * 70)
print('规则B：相对阈值——Top-K 热号预测下期（这才是可能有效的形态）')
print('=' * 70)
for N in (50, 100):
    for K in (10, 20, 30):
        r = run_topk_rule(N, K)
        edge = r['empirical_rate'] - r['baseline']
        print(f'  N={N}, Top-{K}: 下期出现率 = {100*r["empirical_rate"]:.2f}% '
              f'| 基线 25.00% | 差值 {100*edge:+.2f}pp')
