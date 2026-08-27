"""把算好的东西整理成给人看的形状。

这一层不做任何计算，只做**取舍与四舍五入**：哪些字段值得展示、保留几位小数、
排前几名。分出来的理由是它变得最频繁（页面改版就要动），而它一动就容易把
下面几层的数据结构一起改了。

**四舍五入的位数是有讲究的**：概率保留四位（0.0001 的差别在千分之一量级的
问题上仍有意义），分数保留一位（评分本身就没有第二位小数的精度），
百分比保留一位。位数写成常量而不是散在各处的字面量——同一个量在两处保留
不同位数，页面上就会出现两个「不一样」的同一个数。
"""

PROBABILITY_DIGITS = 4
SCORE_DIGITS = 1
PERCENT_DIGITS = 1
RATIO_DIGITS = 2
WEIGHT_DIGITS = 3


def transition_view(lag1, dynamic, position_names, position_baseline):
    """上期→本期转移统计的接口视图。

    `vs_random` 是这里唯一真正新增的信息：**光看「同位复刻率 12%」说明不了
    什么，除以理论基线 10% 得到 1.2 才说明它偏高**。
    """
    return {
        'pairs_analyzed': lag1['pairs'],
        'pos_repeat_rate': [
            {'name': position_names[index],
             'rate': round(lag1['pos_repeat_rate'][index], PROBABILITY_DIGITS),
             'vs_random': round(lag1['pos_repeat_rate'][index] / position_baseline,
                                RATIO_DIGITS)}
            for index in range(len(position_names))
        ],
        # 键写成「N 位同」：这是直接显示在页面上的标签，不是内部字段名
        'repeat_dist': {f'{key}位同': round(value * 100, PERCENT_DIGITS)
                        for key, value in lag1['repeat_dist'].items()},
        'digit_reuse_rate': round(lag1['digit_reuse_rate'], PROBABILITY_DIGITS),
        'full_repeat_rate': round(lag1['full_repeat_rate'], PROBABILITY_DIGITS),
        'same_set_rate': round(lag1['same_set_rate'], PROBABILITY_DIGITS),
        'ge2_overlap_rate': round(lag1['ge2_overlap_rate'], PROBABILITY_DIGITS),
        'dynamic': _rounded(dynamic, WEIGHT_DIGITS),
    }


def _rounded(values, digits):
    """字典里的数字统一取整。列表逐项取，其余原样——**不递归**：
    再深一层的结构说明它不该直接进接口。"""
    return {key: ([round(item, digits) for item in value] if isinstance(value, list)
                  else round(value, digits))
            for key, value in values.items()}


def scored_digits(pairs, digits=SCORE_DIGITS):
    """`[(数字, 分), ...]` → 接口用的列表。"""
    return [{'digit': digit, 'score': round(score, digits)} for digit, score in pairs]


def position_top(position_scores, position_names, size):
    """每一位的前若干名。**按位分开**——百位的 7 和个位的 7 是两个量。"""
    result = []
    for index, name in enumerate(position_names):
        ranked = sorted(enumerate(position_scores[index]),
                        key=lambda item: -item[1])[:size]
        result.append({'name': name, 'digits': scored_digits(ranked)})
    return result


def long_miss_digits(miss_by_digit, threshold):
    """遗漏超过门槛的数字，按遗漏从多到少。

    **只列超过门槛的**：十个数字全列出来，那张表就没有信息量了——
    遗漏三五期是常态，看的人需要的是「哪几个明显偏久」。
    """
    return sorted(({'digit': digit, 'miss': miss}
                   for digit, miss in miss_by_digit.items() if miss >= threshold),
                  key=lambda item: -item['miss'])


def position_miss_top(miss_by_position, position_names, size):
    """每一位遗漏最久的前若干个。"""
    return [{'name': name,
             'digits': [{'digit': digit, 'miss': miss_by_position[index][digit]}
                        for digit in sorted(miss_by_position[index],
                                            key=lambda d: -miss_by_position[index][d])[:size]]}
            for index, name in enumerate(position_names)]


def weighted_digits(counter, size, digits=SCORE_DIGITS):
    """指数加权频次的前若干名。"""
    return [{'digit': digit, 'weight': round(weight, digits)}
            for digit, weight in counter.most_common(size)]


def sum_tails(counter, size, digits=RATIO_DIGITS):
    return [{'tail': tail, 'count': round(count, digits)}
            for tail, count in counter.most_common(size)]


def sum_span_view(meta, digits=SCORE_DIGITS):
    return {'sum_center': round(meta['sum_center'], digits),
            'hot_sums': meta['hot_sums'],
            'span_center': round(meta['span_center'], digits),
            'hot_spans': meta['hot_spans']}


def stability_view(score, level, exploration_rate, digits=RATIO_DIGITS):
    return {'score': round(score, digits),
            'level': level,
            'adjusted_exploration_rate': round(exploration_rate, digits)}
