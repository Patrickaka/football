"""给每个数字（0~9）打一个分：这一位上它值不值得进推荐。

三套评分，用途不同：
- `digit_scores`：**不分位**的全局分，一个数字在三位上的表现合起来看
- `position_digit_scores`：**分位**的分，百位的 7 与个位的 7 各算各的
- `zu6_digit_scores`：只问「这个数字会不会出现」，不问在哪一位

前两套共用同一批特征（热号、马尔可夫、遗漏、邻号、路数），由 `flags` 逐项
开关。**开关关掉的那一项在生产里就是死分支，但它们是调参旋钮，不是废弃
代码**——线上现在关着 `miss`/`neighbor`/`road`，随时可能打开。

**没有一项能提高中奖概率。** 直选恒定 1/1000。这些分决定的是推荐池里放哪
一千分之几，以及它们的结构像不像一次真实开奖。
"""
from collections import Counter

from src.domain.numeric.lottery3d import draw as draw_props
from src.domain.numeric.lottery3d import history
from src.domain.numeric.lottery3d.space import DIGIT_SPACE, POSITIONS

# 遗漏加分的两道门槛（期数）。低于 MID 不给分——正常波动里遗漏十期以内
# 太常见，给了分等于给几乎所有数字都加了分。
MISS_HIGH_THRESHOLD = 20
MISS_MID_THRESHOLD = 12
# 长期遗漏的加分随遗漏期数线性放大，除数与门槛一致
MISS_HIGH_SCALE = 20

HOT_GLOBAL_TOP = 4     # 全局热号取前几名
HOT_POSITION_TOP = 3   # 分位热号取前几名
# 分位评分里热号的额外加成。分位样本只有全局的三分之一，热度更不稳，
# 所以这里用的是「加一个固定值」而不是再乘一个系数。
POSITION_HOT_EXTRA = 1


def _digits():
    return DIGIT_SPACE.numbers()


def _zeros():
    return [0.0] * DIGIT_SPACE.size


def _markov_contribution(numbers, position, key, weights, order):
    """某一位上的转移分。

    `key` 是查转移表用的键，**由调用方按阶数构造**：一阶是上一期这一位的
    数字，二阶是前两期这一位的数字对。做成显式参数而不是从整注里推——
    「上一期」在一阶那里指一个数字、在二阶那里指一对，混起来取错了不会报错，
    只会拿另一位的转移表来打分。
    """
    if order == 1:
        table = history.build_markov(numbers, position)
        row = table.get(key, Counter())
        weight = weights.markov
    else:
        table = history.build_markov2(numbers, position)
        row = table.get(key, Counter())
        weight = weights.markov2
    probabilities = history.markov_prob_smoothed(row, _digits(), weights.markov_alpha)
    # 封顶：样本极少的转移经平滑后仍可能给出极高的分，那不是信号强、是分母小
    return {digit: min(weight * p, weights.markov_max)
            for digit, p in probabilities.items()}


def _miss_score(miss, weights):
    if miss >= MISS_HIGH_THRESHOLD:
        return weights.miss_high * (1 + miss / MISS_HIGH_SCALE)
    if miss >= MISS_MID_THRESHOLD:
        return weights.miss_mid
    return 0.0


def digit_scores(numbers, window, weights, flags, dynamic=None,
                 miss_cycle=None, rebound=None, entropy=None):
    """不分位的数字评分，返回 (各数字得分, 指数加权频次)。

    `miss_cycle` / `rebound` / `entropy` 是三份外部算好的弱先验加分。做成参数
    而不是在这里算，是因为它们各自有窗口与阈值——那是配置问题，不是评分问题。
    """
    recent = history._recent(numbers, window)
    last = numbers[-1]
    score = _zeros()
    dynamic = dynamic or {}
    last_appear = dynamic.get('w_last_appear', weights.last_appear)

    frequency = history.exp_weighted_counts(
        [digit for draw in recent for digit in draw], weights.decay)

    if flags.get('hot', True):
        for digit, _ in frequency.most_common(HOT_GLOBAL_TOP):
            score[digit] += weights.hot_global
        for position in range(POSITIONS):
            column = history.exp_weighted_counts(
                [draw[position] for draw in recent], weights.decay)
            for digit, _ in column.most_common(HOT_POSITION_TOP):
                score[digit] += weights.hot_position

    if flags.get('markov', True):
        for position in range(POSITIONS):
            for digit, value in _markov_contribution(
                    numbers, position, last[position], weights, order=1).items():
                score[digit] += value
            if len(numbers) >= 2:
                pair = (numbers[-2][position], last[position])
                for digit, value in _markov_contribution(
                        numbers, position, pair, weights, order=2).items():
                    score[digit] += value

    if flags.get('miss', True):
        for digit in _digits():
            score[digit] += _miss_score(history.miss_value(numbers, digit), weights)
            score[digit] += (miss_cycle or {}).get(digit, 0.0)
            score[digit] += (entropy or {}).get(digit, 0.0)
            score[digit] += (rebound or {}).get(digit, 0.0)

    if flags.get('neighbor', True):
        for digit in set(last):
            score[digit] += last_appear
        neighbours = set()
        for digit in last:
            neighbours.update(draw_props.neighbor(digit))
        for digit in neighbours:
            score[digit] += weights.neighbor

    if flags.get('road', True):
        last_roads = {draw_props.road(digit) for digit in last}
        for digit in _digits():
            if draw_props.road(digit) in last_roads:
                score[digit] += weights.road_match

    return score, frequency


def ensemble_digit_scores(numbers, window_weights, weights, flags, dynamic=None,
                          miss_cycle=None, rebound=None, entropy=None):
    """多窗口加权合成。

    三份弱先验**不在这里再加一遍**——它们已经在每个窗口的 `digit_scores` 里
    加过了。再加等于按窗口个数重复计权，那一项就会压过其余所有特征。
    """
    combined = _zeros()
    frequency = Counter()
    for window, weight in window_weights.items():
        scores, counts = digit_scores(numbers, window, weights, flags, dynamic,
                                      miss_cycle, rebound, entropy)
        for digit in _digits():
            combined[digit] += weight * scores[digit]
        for digit, count in counts.items():
            frequency[digit] += weight * count
    return combined, frequency


def position_digit_scores(numbers, position, window, weights, flags, dynamic=None):
    """某一位上的数字评分。与全局评分共用同一批开关。"""
    recent = [draw[position] for draw in history._recent(numbers, window)]
    last = numbers[-1][position]
    score = _zeros()
    dynamic = dynamic or {}
    last_appear = dynamic.get('w_last_appear', weights.last_appear)
    multiplier = dynamic.get('pos_mult', [1.0] * POSITIONS)

    if flags.get('hot', True):
        counts = history.exp_weighted_counts(recent, weights.decay)
        for digit, _ in counts.most_common(HOT_GLOBAL_TOP):
            score[digit] += weights.hot_position + POSITION_HOT_EXTRA

    if flags.get('markov', True):
        for digit, value in _markov_contribution(
                numbers, position, last, weights, order=1).items():
            score[digit] += value
        if len(numbers) >= 2:
            pair = (numbers[-2][position], last)
            for digit, value in _markov_contribution(
                    numbers, position, pair, weights, order=2).items():
                score[digit] += value

    if flags.get('miss', True):
        for digit in _digits():
            score[digit] += _miss_score(
                history.miss_value(numbers, digit, position=position), weights)

    if flags.get('neighbor', True):
        score[last] += last_appear * multiplier[position]
        for digit in draw_props.neighbor(last):
            score[digit] += weights.neighbor

    return score


def ensemble_position_digit_scores(numbers, position, window_weights, weights,
                                   flags, dynamic=None):
    score = _zeros()
    for window, weight in window_weights.items():
        column = position_digit_scores(numbers, position, window, weights, flags, dynamic)
        for digit in _digits():
            score[digit] += weight * column[digit]
    return score


# 短窗口打破并列用的极小量。小到不会改变任何非并列的名次，
# 只保证同分时的顺序可复现。
TIE_BREAK_PRESENCE = 1e-6
TIE_BREAK_DIGIT = 1e-9


def zu6_digit_scores(numbers, presence_windows):
    """组六选池用的「出现与否」评分，不分位。

    **一注里重复的数字只算一次**：四码组六池只关心某个数字在不在，
    豹子里的三个 7 不比组六里的一个 7 更说明问题。

    **所有形态的开奖都计入**，不只筛组六的那些——每一期都是对「下期各数字
    出现概率」的有效观测，只留组六会丢掉四分之一的样本。
    """
    if not numbers:
        return _zeros()

    windows = [window for window in presence_windows if window > 0]
    score = _zeros()
    for window in windows:
        recent = history._recent(numbers, window)
        presence = Counter(digit for draw in recent for digit in set(draw))
        for digit in _digits():
            score[digit] += presence[digit] / max(1, len(recent))

    shortest = min(windows) if windows else len(numbers)
    short_presence = Counter(
        digit for draw in history._recent(numbers, shortest) for digit in set(draw))
    for digit in _digits():
        score[digit] += short_presence[digit] * TIE_BREAK_PRESENCE - digit * TIE_BREAK_DIGIT
    return score
