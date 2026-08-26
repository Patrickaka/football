"""多注组合：把一份排名切成若干注，再估这组注的整体命中率。

**这是覆盖率杠杆，不改变任何单号的中奖概率。** 公平摇奖下单号没有可预测的
优势，所以这里不再假装用「特征扰动」制造不同视角，而是直接按排名切成互不
重叠的块——12 注 × 6 码能盖住 72 个号码，比 12 注全挤在同一批 top40 里的
「至少一注中 3 个」概率明显高，靠的纯粹是覆盖面。
"""
import random

from src.domain.numeric.kl8.space import DRAW_COUNT, SPACE


def coverage_slips(ranked_numbers, n_slips, pick_size):
    """按排名顺序切成 n_slips 注，每注 pick_size 个号。

    第 0 注就是排名最前的那几个，与主推号码一致——否则用户会看到「推荐号」
    和「第一注」是两组号。号不够铺满时从头循环复用排名靠前的。
    """
    if n_slips <= 0 or pick_size <= 0 or len(ranked_numbers) < pick_size:
        return []

    total_slots = n_slips * pick_size
    coverage = list(ranked_numbers[:total_slots])
    while len(coverage) < total_slots:
        coverage.append(ranked_numbers[(len(coverage) - len(ranked_numbers))
                                       % len(ranked_numbers)])
    return [sorted(coverage[i * pick_size:(i + 1) * pick_size]) for i in range(n_slips)]


def simulate_coverage(slips, simulations=12000, seed_key=''):
    """按这组注的**实际重叠**估命中率。

    不用 `1-(1-p)^n`：那个公式假设各注互相独立，而这里的注是从同一份排名切
    出来的，独立假设会把命中率算高。改为抽样公平的 80 选 20，种子固定，
    所以同一份预测每次渲染出的数字都一样。
    """
    tickets = [set(int(n) for n in slip) for slip in (slips or []) if slip]
    if not tickets or simulations <= 0:
        return {}

    rng = random.Random(f'kl8_coverage_v1_{seed_key}')
    tiers = {3: 0, 4: 0, 5: 0, 6: 0}
    total_best = 0
    for _ in range(simulations):
        draw = set(rng.sample(range(SPACE.low, SPACE.high + 1), DRAW_COUNT))
        best = max(len(ticket & draw) for ticket in tickets)
        total_best += best
        for threshold in tiers:
            tiers[threshold] += int(best >= threshold)

    overlaps = [len(tickets[i] & tickets[j])
                for i in range(len(tickets))
                for j in range(i + 1, len(tickets))]
    return {
        'method': 'deterministic_monte_carlo_actual_overlap',
        'simulations': simulations,
        'at_least_one_ge3': round(tiers[3] / simulations, 6),
        'at_least_one_ge4': round(tiers[4] / simulations, 6),
        'at_least_one_ge5': round(tiers[5] / simulations, 6),
        'at_least_one_ge6': round(tiers[6] / simulations, 6),
        'average_best_hits': round(total_best / simulations, 4),
        'unique_number_count': len(set().union(*tickets)),
        # 注与注之间重叠得越多，这组注实际盖住的号码就越少
        'max_pair_overlap': max(overlaps, default=0),
        'average_pair_overlap': round(sum(overlaps) / len(overlaps), 3) if overlaps else 0.0,
    }
