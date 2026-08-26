"""kl8 的特征评分与集成排名。

13 个活跃特征各给号码打一个 [0,1] 区间的分，加权求和决定排名，排名决定最终
选出哪些号码。**任何一个特征的曲线变了，选号就变了，而不会有任何报错**——
所以这里的每个常数都是有来历的，改之前先想清楚改的是什么。

分组成员（5 码区、8 列 × 奇偶交叉、012 路）在模块加载时算一次。迁移前是
每算一个号码就重新遍历 1..80 拼一遍，80 个号码就是 6400 次——实测排名一次
只要 0.9ms，所以这不是为了快，而是为了让「这个号码属于哪些组」这件事
只有一处定义。
"""
import hashlib
import math

from src.domain.numeric.statistics import NumberSpace

SPACE = NumberSpace(low=1, high=80)
ZONE_SIZE = 5        # 16 个 5 码区，与 position_residual 粒度一致
CROSS_ROW_SIZE = 8   # 8 列 × 奇偶 = 每组 4 个号码
BIG_SMALL_THRESHOLD = 40
ROAD_MODULUS = 3

# 中性分。所有特征都以 0.5 为「没有意见」，加权和才有可比性。
NEUTRAL = 0.50
# 跨期带出的理论中性概率：80 选 20，任一号码开出的概率是 0.25
TRANSITION_BASELINE = 0.25


def _group_members():
    """预先算好每个号码所属的三种分组成员集合。"""
    zones, crosses, roads = {}, {}, {}
    for number in SPACE.numbers():
        zone = (number - SPACE.low) // ZONE_SIZE + 1
        zones.setdefault(zone, []).append(number)
        row = (number - SPACE.low) // CROSS_ROW_SIZE + 1
        crosses.setdefault((row, number % 2), []).append(number)
        roads.setdefault(number % ROAD_MODULUS, []).append(number)
    return zones, crosses, roads


_ZONES, _CROSSES, _ROADS = _group_members()


def _zone_of(number):
    return (number - SPACE.low) // ZONE_SIZE + 1


def _cross_of(number):
    return ((number - SPACE.low) // CROSS_ROW_SIZE + 1, number % 2)


def feature_scores(number, statistics, based_on_issue='',
                   repeat_direction='neutral', repeat_avoid_score=0.10,
                   repeat_non_avoid_score=0.85, repeat_follow_score=0.90,
                   repeat_non_follow_score=0.50,
                   frequency_mode='mean_reversion'):
    """给一个号码打出全部特征分。

    `sum` 与 `zone` 两个特征已停用，但**仍然出现在输出里**，恒为中性分。
    悄悄删掉会让带权重的旧策略少两项加权和——策略试验表里存着 23564 条历史
    记录，它们的权重字典是按当年的特征集写的。
    """
    freq = statistics['frequency']
    expected_freq = statistics['expected_freq']
    last_numbers = statistics['last_numbers']
    num_freq = freq.get(number, 0)

    scores = {
        'frequency': _frequency_score(num_freq, expected_freq, frequency_mode),
        'gap': _gap_score(statistics['gap'].get(number, 0),
                          statistics['expected_gap']),
        'position_residual': _group_residual_score(
            num_freq, _ZONES[_zone_of(number)], freq, expected_freq, amplitude=0.25),
        'position_residual_cross': _group_residual_score(
            num_freq, _CROSSES[_cross_of(number)], freq, expected_freq,
            amplitude=0.20),
        'road_residual': _group_residual_score(
            num_freq, _ROADS[number % ROAD_MODULUS], freq, expected_freq,
            amplitude=0.20),
        'trend': _trend_score(statistics.get('trend', {}).get(number, 0),
                              statistics['total_periods']),
        'pair_cooccurrence': _ratio_score(
            statistics.get('avg_cooccurrence', {}), number),
        'next_transition': _transition_score(
            statistics.get('next_transition_probability', {}).get(
                number, TRANSITION_BASELINE)),
        # 停用，保留在输出里以免旧策略的权重字典对不上
        'sum': NEUTRAL,
        'zone': NEUTRAL,
        'repeat': _repeat_score(number, last_numbers, repeat_direction,
                                repeat_avoid_score, repeat_non_avoid_score,
                                repeat_follow_score, repeat_non_follow_score),
        'adjacent': _ratio_score(statistics.get('adjacent_freq', {}), number),
        'odd_even': _share_score(statistics.get('freq_by_odd_even', {}),
                                 'odd' if number % 2 == 1 else 'even'),
        'big_small': _share_score(
            statistics.get('freq_by_big_small', {}),
            'big' if number > BIG_SMALL_THRESHOLD else 'small'),
        'seeded_random': _seeded_random(number, based_on_issue),
    }
    return scores


def _frequency_score(actual, expected, mode):
    """频率偏离度。三种取向互为反面，接反了不会报错，只会让选号系统性走偏。"""
    if mode == 'neutral':
        return NEUTRAL
    ratio = actual / max(expected, 0.01)
    # 追热：开得多的加分；均值回归：开得少的加分。两者只差一个方向
    distance = (ratio - 1.0) if mode == 'hot' else (1.0 - ratio)
    if distance >= 0:
        return 0.55 + 0.15 * distance
    return max(0.15, 0.55 * math.exp(1.8 * distance))


def _gap_score(actual, expected):
    """遗漏偏离度。久未开出的加分，但增速随遗漏拉长而收敛。"""
    ratio = actual / max(expected, 0.01)
    if ratio <= 1.0:
        return 0.25 + 0.60 * (ratio ** 0.7)
    return 0.85 - 0.45 * (1.0 - math.exp(-(ratio - 1.0) * 0.8))


def _group_residual_score(num_freq, group, freq, expected_freq, amplitude):
    """组内残差：该号频率相对**所在组均值**的偏离，再用全局均值归一。

    先减组均值再除全局均值，是为了剔掉「这一组整体偏热」的影响，只留下
    「这个号在组里是否突出」。低于组均值的加分（还没轮到它）。
    """
    if expected_freq <= 0:
        return NEUTRAL
    group_avg = sum(freq.get(n, 0) for n in group) / len(group)
    ratio = (num_freq - group_avg) / max(expected_freq, 0.01)
    if ratio <= 0:
        return 0.55 + amplitude * min(1.0, abs(ratio))
    return max(0.15, 0.55 * math.exp(-1.5 * ratio))


def _trend_score(trend, total_periods):
    """冷热趋势。以窗口的四分之一作为满幅参照。"""
    scale = max(1, total_periods // 4)
    ratio = trend / max(scale, 1)
    if ratio >= 0:
        return 0.55 + 0.30 * min(1.0, ratio)
    return max(0.20, 0.55 * math.exp(-2.0 * abs(ratio)))


def _ratio_score(values, number):
    """相对全场最大值的比例分。共现与邻号都用它。"""
    value = values.get(number, 0)
    ceiling = max(values.values(), default=1)
    return NEUTRAL + 0.30 * (value / max(ceiling, 0.01))


def _transition_score(probability):
    """跨期带出。夹在 [0.15, 0.85] 之间，防止稀疏关联压倒其余特征。"""
    lift = (probability - TRANSITION_BASELINE) / TRANSITION_BASELINE
    return max(0.15, min(0.85, NEUTRAL + 0.35 * lift))


def _repeat_score(number, last_numbers, direction, avoid, non_avoid,
                  follow, non_follow):
    """重号方向。这是策略之间差别最大的一维。"""
    if direction == 'avoid':
        return avoid if number in last_numbers else non_avoid
    if direction == 'follow':
        return follow if number in last_numbers else non_follow
    return NEUTRAL


def _share_score(group_frequency, key):
    """所在组占全部开出次数的比例，以 50% 为中性，线性放大到 ±0.30。"""
    total = sum(group_frequency.values()) or 1
    ratio = group_frequency.get(key, 0) / max(total, 0.01)
    return NEUTRAL + 0.30 * (ratio - 0.5) * 2


def _seeded_random(number, based_on_issue):
    """同一期内稳定、跨期变化的伪随机分，用来给并列打破僵局。

    用哈希而不是 random：并列的打破方式必须可复现，否则同一份输入在两次
    运行里会给出不同的选号，回测结论也就无从对照。
    """
    seed = f'kl8_seeded_random_v1_{based_on_issue}_{number}'
    return int(hashlib.sha256(seed.encode()).hexdigest()[:12], 16) / float(0xFFFFFFFFFFFF)


def ensemble_ranking(statistics, weights, top_n=20, based_on_issue='', **options):
    """按加权和给号码排名。

    **没有任何有效权重时返回空，而不是前 20 个号码。** 后者看起来像个结果，
    实际是「按号码大小排序」——一个没有任何信号的输出被当成了预测。
    """
    if not any(w > 0 for w in (weights or {}).values()):
        return []

    ranking = []
    for number in SPACE.numbers():
        scores = feature_scores(number, statistics, based_on_issue=based_on_issue,
                                **options)
        ranking.append({
            'num': number,
            'ranking_score': sum(scores.get(k, 0) * weights.get(k, 0)
                                 for k in scores),
            'score_type': 'heuristic_rank',
            # 排名分不是概率。标成概率会让下游拿它当胜率用。
            'is_probability': False,
            'scores': scores,
        })
    # 并列按号码升序：打破方式必须可复现，否则同一份输入两次运行结果不同
    ranking.sort(key=lambda item: (-item['ranking_score'], item['num']))
    return ranking[:top_n]
