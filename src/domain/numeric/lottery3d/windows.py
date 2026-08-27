"""多窗口集成：同一个统计量在几个窗口上各算一遍，再加权合成一个。

**为什么不只用一个窗口。** 短窗口跟得紧但噪声大，长窗口稳但迟钝，而「哪个
窗口更准」本身是随时间变的。多窗口加权是承认这件事：不去赌一个窗口，
而是让几个窗口各出一份，权重由回测定（`resolve_window_weights`）。

三类统计各自独立：
- **lag1**：上期→本期的转移（同位复刻、重号、全同号），最贴近「下期会不会
  带出上期的号」这个问题
- **patterns**：连号占比、奇偶比与大小比的分布
- **sum_span**：和值与跨度的中心和热区

`derive_dynamic_weights` 把 lag1 的观测折算成评分权重的缩放系数——历史上
同位复刻得多，就把「与上期同位相同」这一项调重些。**这是随数据自适应，
不是预测**：它调的是「这个特征在近期有多少区分度」，不是「下期会怎样」。
"""
from collections import Counter

from src.domain.numeric.lottery3d import draw as draw_props
from src.domain.numeric.lottery3d import history
from src.domain.numeric.lottery3d.space import POSITIONS

def empty_lag1(baselines):
    """没有足够历史时的中性观测：一律取理论基线，而不是 0。

    0 会被下游当成「从不复刻」，那是个凭空造出来的强结论。
    """
    return {
        'pairs': 0,
        'pos_repeat_rate': [baselines.position_repeat] * POSITIONS,
        'avg_pos_repeat': baselines.position_repeat,
        'repeat_dist': {0: 1.0},
        'full_repeat_rate': 0.0,
        'same_set_rate': 0.0,
        'ge2_overlap_rate': 0.0,
        'digit_reuse_rate': baselines.digit_reuse,
    }


def position_repeat_count(triple, last_draw):
    """与上期**同一位**相同的个数。位置不同不算——那是重号，不是复刻。"""
    return sum(1 for i in range(POSITIONS) if triple[i] == last_draw[i])


def analyze_lag1(numbers, window, decay, baselines):
    """近窗内「上期→本期」的各种重复率，越近的一对权重越大。"""
    if len(numbers) < 2:
        return empty_lag1(baselines)

    pairs = list(zip(numbers[:-1], numbers[1:]))
    recent = pairs[-window:] if len(pairs) > window else pairs

    position_weight = [0.0] * POSITIONS
    repeat_dist = Counter()
    full = same_set = ge2 = digit_hit = digit_total = total = 0.0
    weight = 1.0
    for previous, current in reversed(recent):
        repeat_dist[position_repeat_count(current, previous)] += weight
        for index in range(POSITIONS):
            if previous[index] == current[index]:
                position_weight[index] += weight
        if previous == current:
            full += weight
        if set(previous) == set(current):
            same_set += weight
        if len(set(previous) & set(current)) >= 2:
            ge2 += weight
        for digit in set(previous):
            digit_total += weight
            if digit in current:
                digit_hit += weight
        total += weight
        weight *= decay

    total = total or 1.0
    return {
        'pairs': len(recent),
        'pos_repeat_rate': [position_weight[i] / total for i in range(POSITIONS)],
        'avg_pos_repeat': sum(position_weight) / (POSITIONS * total),
        'repeat_dist': {k: v / total for k, v in sorted(repeat_dist.items())},
        'full_repeat_rate': full / total,
        'same_set_rate': same_set / total,
        'ge2_overlap_rate': ge2 / total,
        'digit_reuse_rate': digit_hit / digit_total if digit_total else baselines.digit_reuse,
    }


def ensemble_lag1(numbers, window_weights, decay, baselines):
    """多窗口加权合成 lag1 观测。"""
    if len(numbers) < 2:
        return empty_lag1(baselines)

    position_rate = [0.0] * POSITIONS
    repeat_dist = Counter()
    full = same_set = ge2 = reuse = reuse_total = avg_repeat = 0.0
    pairs = 0

    for window, weight in window_weights.items():
        lag = analyze_lag1(numbers, window, decay, baselines)
        # 期数取最大而非加权：它描述「有多少样本」，加权平均没有意义
        pairs = max(pairs, lag['pairs'])
        for index in range(POSITIONS):
            position_rate[index] += weight * lag['pos_repeat_rate'][index]
        for key, value in lag['repeat_dist'].items():
            repeat_dist[key] += weight * value
        full += weight * lag['full_repeat_rate']
        same_set += weight * lag['same_set_rate']
        ge2 += weight * lag['ge2_overlap_rate']
        reuse += weight * lag['digit_reuse_rate']
        reuse_total += weight
        avg_repeat += weight * lag['avg_pos_repeat']

    return {
        'pairs': pairs,
        'pos_repeat_rate': position_rate,
        'avg_pos_repeat': avg_repeat,
        'repeat_dist': dict(repeat_dist),
        'full_repeat_rate': full,
        'same_set_rate': same_set,
        'ge2_overlap_rate': ge2,
        'digit_reuse_rate': reuse / reuse_total if reuse_total else baselines.digit_reuse,
    }


def clamp(value, low, high):
    return max(low, min(high, value))


# 各项自适应缩放的上下界。**界限本身就是这套机制的安全带**：没有它们，
# 一段偶然的高复刻率会把权重推到极端，让推荐整段照抄上期。
POSITION_REPEAT_BOUNDS = (0.2, 1.6)
POSITION_MULTIPLIER_BOUNDS = (0.3, 2.0)
DIGIT_REUSE_BOUNDS = (0.3, 1.4)
CONSECUTIVE_BOUNDS = (0.6, 1.2)
FULL_REPEAT_PENALTY_BOUNDS = (4.0, 15.0)
SAME_SET_PENALTY_BOUNDS = (1.5, 8.0)

# 全同号与同集合的惩罚基准。乘的那个大系数是把极小的发生率放大到可比区间：
# 全同号理论上每 1000 期才一次，不放大的话惩罚永远顶格。
FULL_REPEAT_BASE = 12.0
FULL_REPEAT_SCALE = 80
SAME_SET_BASE = 6.0
SAME_SET_SCALE = 40
# 连号率的下限。近窗完全没有连号时，除以 0 会让缩放炸掉。
MIN_CONSECUTIVE_RATE = 0.15


def derive_dynamic_weights(lag1, consecutive_rate, base, baselines):
    """把 lag1 观测折算成评分权重的缩放系数。

    每一项都是「实测 ÷ 理论基线」再夹进上下界。高于基线说明这个特征近期
    有区分度，调重；低于基线调轻。`base` 是静态权重（`DigitWeights` /
    `TripletWeights` 里那几个），这里只做缩放，不定绝对值。
    """
    position_scale = clamp(lag1['avg_pos_repeat'] / baselines.position_repeat,
                           *POSITION_REPEAT_BOUNDS)
    reuse_scale = clamp(lag1['digit_reuse_rate'] / baselines.digit_reuse,
                        *DIGIT_REUSE_BOUNDS)
    consecutive_scale = clamp(
        consecutive_rate / max(consecutive_rate, MIN_CONSECUTIVE_RATE),
        *CONSECUTIVE_BOUNDS)
    return {
        'w_pos_repeat': base['position_repeat'] * position_scale,
        'pos_mult': [clamp(rate / baselines.position_repeat, *POSITION_MULTIPLIER_BOUNDS)
                     for rate in lag1['pos_repeat_rate']],
        'w_last_appear': base['last_appear'] * reuse_scale,
        'w_consecutive': base['consecutive'] * consecutive_scale,
        # 越少发生，罚得越重：罕见事件出现在推荐里更像是模型跑偏
        'w_full_repeat_penalty': clamp(
            FULL_REPEAT_BASE * (1.0 - lag1['full_repeat_rate'] * FULL_REPEAT_SCALE),
            *FULL_REPEAT_PENALTY_BOUNDS),
        'w_same_set_penalty': clamp(
            SAME_SET_BASE * (1.0 - lag1['same_set_rate'] * SAME_SET_SCALE),
            *SAME_SET_PENALTY_BOUNDS),
    }


# ─── 形态模式 ───

HOT_RATIO_KEYS = 3   # 奇偶比/大小比各取前几名算「热门」


def analyze_patterns(numbers, window, decay):
    """近窗的连号占比与奇偶比、大小比分布。"""
    recent = history._recent(numbers, window)
    odd_even, big_small = Counter(), Counter()
    consecutive = 0.0
    weight = 1.0
    for current in reversed(recent):
        odd_even[draw_props.odd_even_key(current)] += weight
        big_small[draw_props.big_small_key(current)] += weight
        if draw_props.has_consecutive_digits(*current):
            consecutive += weight
        weight *= decay
    total = sum(odd_even.values()) or 1.0
    return {'oe_freq': odd_even, 'bs_freq': big_small, 'consec_rate': consecutive / total}


def ensemble_patterns(numbers, window_weights, decay):
    odd_even, big_small = Counter(), Counter()
    consecutive = 0.0
    for window, weight in window_weights.items():
        found = analyze_patterns(numbers, window, decay)
        for key, value in found['oe_freq'].items():
            odd_even[key] += weight * value
        for key, value in found['bs_freq'].items():
            big_small[key] += weight * value
        consecutive += weight * found['consec_rate']
    return {
        'oe_freq': odd_even,
        'bs_freq': big_small,
        # 总量随权重一起带出去：下游要用它做分母，各自再算一遍就可能对不上
        'oe_total': sum(odd_even.values()) or 1.0,
        'bs_total': sum(big_small.values()) or 1.0,
        'hot_oe_set': {key for key, _ in odd_even.most_common(HOT_RATIO_KEYS)},
        'hot_bs_set': {key for key, _ in big_small.most_common(HOT_RATIO_KEYS)},
        'consec_rate': consecutive,
    }


# ─── 和值与跨度 ───

HOT_SUM_COUNT = 6
HOT_SPAN_COUNT = 4
RECENT_SHIFT_WINDOW = 5


def analyze_sum_span(sums, spans, window, decay, recent_shift):
    """和值与跨度的指数加权中心，以及各自的热区。

    `recent_shift` 大于 0 时把中心往最近 5 期拉。**默认是 0**：追涨杀跌需要
    先有消融回测证明，否则只是让中心跟着噪声跑。
    """
    recent_sums = history._recent(sums, window)
    recent_spans = history._recent(spans, window)
    weighted_sums = history.exp_weighted_counts(recent_sums, decay)
    weighted_spans = history.exp_weighted_counts(recent_spans, decay)

    sum_center = _weighted_mean(weighted_sums)
    span_center = _weighted_mean(weighted_spans)
    if recent_shift > 0 and len(recent_sums) >= RECENT_SHIFT_WINDOW:
        sum_center = _shift(sum_center, recent_sums, recent_shift)
        span_center = _shift(span_center, recent_spans, recent_shift)

    return {
        'sum_center': sum_center,
        'span_center': span_center,
        'hot_sums': [value for value, _ in weighted_sums.most_common(HOT_SUM_COUNT)],
        'hot_spans': [value for value, _ in weighted_spans.most_common(HOT_SPAN_COUNT)],
        'sum_tail_freq': Counter(total % 10 for total in recent_sums),
    }


def _weighted_mean(counts):
    return (sum(value * weight for value, weight in counts.items())
            / max(sum(counts.values()), 1e-9))


def _shift(center, recent, ratio):
    tail = recent[-RECENT_SHIFT_WINDOW:]
    return center * (1 - ratio) + (sum(tail) / RECENT_SHIFT_WINDOW) * ratio


def ensemble_sum_span(sums, spans, window_weights, decay, recent_shift):
    """多窗口合成和值/跨度中心。

    **中心必须取整。** 和值与跨度都是整数统计量，用一个分数中心去框整数容差
    会不对称地少框一个取值：`|v-4.5| <= 1` 只含 {4,5}，而 `|v-5| <= 1` 含
    {4,5,6}。实测取整后跨度 ±1 命中 30.8%→45%、和值 ±2 命中 28.8%→34.6%。
    四舍五入同时贴近分布众数（和值 13/14、跨度 5），对高斯打分几乎无影响。
    """
    sum_center = span_center = 0.0
    hot_sums, hot_spans, tails = Counter(), Counter(), Counter()
    for window, weight in window_weights.items():
        found = analyze_sum_span(sums, spans, window, decay, recent_shift)
        sum_center += weight * found['sum_center']
        span_center += weight * found['span_center']
        for value in found['hot_sums']:
            hot_sums[value] += weight
        for value in found['hot_spans']:
            hot_spans[value] += weight
        for tail, count in found['sum_tail_freq'].items():
            tails[tail] += weight * count
    return {
        'sum_center': float(round(sum_center)),
        'span_center': float(round(span_center)),
        'hot_sums': [value for value, _ in hot_sums.most_common(HOT_SUM_COUNT)],
        'hot_spans': [value for value, _ in hot_spans.most_common(HOT_SPAN_COUNT)],
        'sum_tail_freq': tails,
    }


def with_hot_sets(raw, tail_top):
    """把热区列表转成集合，供逐注打分时做 O(1) 判断。"""
    return {
        **raw,
        'hot_sum_set': set(raw['hot_sums']),
        'hot_span_set': set(raw['hot_spans']),
        'sum_tail_top': {tail for tail, _ in raw['sum_tail_freq'].most_common(tail_top)},
    }
