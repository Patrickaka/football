"""历史开奖序列上的统计：遗漏、冷热、马尔可夫、和值趋势、对频、形态切换。

**序列是旧在前、新在后**——`numbers[-1]` 是最近一期。这与
`domain/numeric/statistics.py` 的 `draws`（新在前）相反，两边直接对接会让
冷热与遗漏全部倒过来，而且不会报错，只是结论反了。这里所有函数都按
「旧在前」写，`_recent(...)` 取的是**尾部**。

**这里没有一个量能提高中奖概率。** 3D 是公平摇奖，遗漏久了不会更容易开出。
这些统计只用来让推荐的结构（和值、形态、冷热分布）贴近真实开奖的样子。
`rebound`、`miss_cycle` 这类「欠账加分」是刻意保留的弱先验，权重都很小，
调大它们等于开始追冷号。
"""
import math
from collections import Counter, defaultdict

from src.domain.numeric.lottery3d import draw as draw_props
from src.domain.numeric.lottery3d.space import DIGIT_SPACE, POSITIONS, SUM_MAX, SUM_MIN


def digits():
    return DIGIT_SPACE.numbers()


def _recent(series, window):
    """取最近 window 期。**尾部**，因为序列是旧在前。"""
    return series[-window:] if len(series) > window else list(series)


def _expected_per_digit(window):
    """理论出现次数：每期开 POSITIONS 个数字，均分给空间里的每个数字。"""
    return (POSITIONS * window) / DIGIT_SPACE.size


# ─── 遗漏 ───

def miss_value(numbers, digit, position=None):
    """距上次出现隔了几期。0 表示最近一期就有。从未出现记为序列长度。

    `position=None` 看整注（任一位命中即可），给定位置则只看那一位——
    百位的 7 和个位的 7 是两个不同的量。
    """
    for index in range(len(numbers) - 1, -1, -1):
        current = numbers[index]
        hit = digit in current if position is None else current[position] == digit
        if hit:
            return len(numbers) - 1 - index
    return len(numbers)


def form_miss(forms, target):
    """距上次出现该形态隔了几期。与 `miss_value` 同一套语义。"""
    for index in range(len(forms) - 1, -1, -1):
        if forms[index] == target:
            return len(forms) - 1 - index
    return len(forms)


DEFAULT_MISS_CYCLE = 7.0     # 数据不足时的平均遗漏周期。10 个数字、每期开 3 个
MIN_PERIODS_FOR_CYCLE = 10   # 少于这么多期，算出来的周期只是噪声


def average_miss_cycle(numbers, digit, window):
    """该数字平均隔几期出现一次。

    **末尾那段还没结束的遗漏不计入**——它只说明「到目前为止还没开」，
    把它当成一个完整周期会系统性拉高平均值，而这个量正是用来判断
    「当前遗漏算不算超期」的分母。
    """
    if len(numbers) < MIN_PERIODS_FOR_CYCLE:
        return DEFAULT_MISS_CYCLE

    completed, current = [], 0
    for current_draw in _recent(numbers, window):
        if digit in current_draw:
            completed.append(current)
            current = 0
        else:
            current += 1
    return sum(completed) / len(completed) if completed else DEFAULT_MISS_CYCLE


def miss_cycle_bonus(numbers, window, over_ratio_threshold, over_bonus):
    """超期遗漏的弱加分：当前遗漏超过平均周期若干倍才给。

    这是**结构性的**弱先验，不是「该出了」——公平摇奖里遗漏久了不会更容易
    开出。倍率越高给得越多，所以阈值和系数一起决定了追冷的力度。
    """
    bonus = {}
    for digit in digits():
        average = average_miss_cycle(numbers, digit, window)
        if average <= 0:
            bonus[digit] = 0.0
            continue
        ratio = miss_value(numbers, digit) / average
        bonus[digit] = (over_bonus * (ratio - over_ratio_threshold + 1)
                        if ratio > over_ratio_threshold else 0.0)
    return bonus


# ─── 冷热 ───

def digit_counts(numbers, window):
    """最近 window 期里每个数字出现了几次，按位计重。"""
    counts = Counter()
    for current in _recent(numbers, window):
        counts.update(current)
    return counts


def rebound_bonus(numbers, window, threshold, bonus):
    """欠账回补：出现次数明显低于理论值的数字给一点分。

    与 `miss_cycle_bonus` 是同一类弱先验，区别是它看**次数**、那个看**间隔**。
    窗口不足时一律给 0——用半个窗口算出来的比值会把正常波动判成欠账。
    """
    if len(numbers) < window:
        return {digit: 0.0 for digit in digits()}

    counts = digit_counts(numbers, window)
    expected = _expected_per_digit(window)
    return {digit: (bonus if _ratio(counts.get(digit, 0), expected) < threshold else 0.0)
            for digit in digits()}


# 冷热分档的阈值：相对理论出现次数的倍率。
# **注意不要与 `config.HOT_RATIO`/`WARM_RATIO` 混淆**——那两个是「选池里热温冷
# 各留多少成」的构成比例（0.4/0.4/0.2），与这里的分档倍率是两回事，名字却很像。
HOT_THRESHOLD = 1.2
WARM_THRESHOLD = 0.8


def classify_by_hot(numbers, window,
                    hot_ratio=HOT_THRESHOLD, warm_ratio=WARM_THRESHOLD):
    """按出现次数与理论值的比值分成热 / 温 / 冷三档。

    窗口不足时**全部算热**而不是全部算冷——下游拿冷号名单去加分，
    凭空造出十个冷号会让推荐整体偏掉；全热等于这一项没有意见。
    """
    if len(numbers) < window:
        return list(digits()), [], []

    counts = digit_counts(numbers, window)
    expected = _expected_per_digit(window)
    hot, warm, cold = [], [], []
    for digit in digits():
        ratio = _ratio(counts.get(digit, 0), expected)
        target = hot if ratio >= hot_ratio else warm if ratio >= warm_ratio else cold
        target.append(digit)
    return hot, warm, cold


def _ratio(actual, expected):
    return actual / expected if expected > 0 else 0


def exp_weighted_counts(series, decay):
    """指数加权计数：越近的一项权重越大。

    从尾部往前走，每退一期乘一次衰减系数——序列旧在前，所以要 `reversed`。
    """
    counts = Counter()
    weight = 1.0
    for item in reversed(series):
        counts[item] += weight
        weight *= decay
    return counts


# ─── 马尔可夫 ───

def build_markov(numbers, position):
    """一阶转移计数：某位上 a 之后出现 b 的次数。"""
    transitions = defaultdict(Counter)
    for index in range(len(numbers) - 1):
        transitions[numbers[index][position]][numbers[index + 1][position]] += 1
    return transitions


def build_markov2(numbers, position):
    """二阶转移计数：某位上 (a, b) 之后出现 c 的次数。"""
    transitions = defaultdict(Counter)
    for index in range(len(numbers) - 2):
        key = (numbers[index][position], numbers[index + 1][position])
        transitions[key][numbers[index + 2][position]] += 1
    return transitions


def markov_prob_smoothed(row, states, alpha):
    """转移概率，拉普拉斯平滑。

    **没见过的转移不能给 0**：0 会被下游当成「不可能」，而实际只是样本没覆盖到。
    平滑系数决定了「没见过」离均匀分布有多近。
    """
    states = list(states)
    denominator = sum(row.values()) + alpha * len(states)
    return {state: (row.get(state, 0) + alpha) / denominator for state in states}


# ─── 和值 ───

def sum_trend(numbers, window, adjust):
    """和值中心与趋势方向。

    把窗口切两半比平均值：后半明显高于前半算上行。**判定用的是固定阈值
    而不是比例**，因为和值本身有天然量纲（0~27），比例阈值在中心值附近
    会变得过于敏感。
    """
    neutral_center = (SUM_MIN + SUM_MAX) / 2
    if len(numbers) < window:
        return neutral_center, 'oscillate'

    sums = [sum(current) for current in numbers[-window:]]
    half = window // 2
    earlier = sum(sums[:half]) / half if half > 0 else 0
    later = sum(sums[half:]) / (window - half) if window - half > 0 else 0
    overall = sum(sums) / window

    if later > earlier + TREND_MARGIN:
        direction, center = 'up', overall + adjust
    elif later < earlier - TREND_MARGIN:
        direction, center = 'down', overall - adjust
    else:
        direction, center = 'oscillate', overall
    return max(SUM_MIN, min(SUM_MAX, center)), direction


# 判定上行/下行所需的和值差。小于它算震荡。
TREND_MARGIN = 1.5


def sum_interval(numbers, window, width, bonus, extreme_penalty):
    """和值回归区间：中心附近加分，两端极值降权。

    极端和值（三位都很小或都很大）确实罕见，但**罕见不等于不会开**——
    降权是为了让推荐的和值分布像真实开奖，不是断言它开不出来。
    """
    if len(numbers) < window:
        return {'center': (SUM_MIN + SUM_MAX) / 2, 'low': 10, 'high': 17, 'bonus': {}}

    sums = [sum(current) for current in numbers[-window:]]
    center = sum(sums) / len(sums)
    low = max(SUM_MIN, int(center - width))
    high = min(SUM_MAX, int(center + width))
    return {'center': center, 'low': low, 'high': high,
            'bonus': {total: _interval_score(total, low, high, bonus, extreme_penalty)
                      for total in range(SUM_MAX + 1)}}


# 两端各多少算「极端」。和值 0~5 与 25~27 合计不到全部组合的 2%。
EXTREME_LOW = 5
EXTREME_HIGH = 25


def _interval_score(total, low, high, bonus, extreme_penalty):
    if low <= total <= high:
        return bonus
    if total <= EXTREME_LOW or total >= EXTREME_HIGH:
        return -extreme_penalty
    return 0.0


# ─── 对频 ───

def pair_frequency(numbers, window):
    """数字对在最近若干期里出现的频率。

    **一注内先去重**：`117` 只贡献 (1,7) 一对，不贡献 (1,1)。对子说的是
    「两个不同的数字同时开出」，把重复位算进来会让豹子、组三凭空多出对子。
    """
    recent = _recent(numbers, window)
    if not recent:
        return {}

    counts = Counter()
    for current in recent:
        distinct = sorted(set(current))
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                counts[(distinct[i], distinct[j])] += 1
    return {pair: count / len(recent) for pair, count in counts.items()}


def high_freq_pairs(numbers, windows, threshold):
    """多个窗口里频率超过阈值的对子，取并集。

    并集而不是交集：不同窗口捕捉的是不同时间尺度上的热度，要求同时满足
    等于只留下最长窗口的结论。
    """
    high = set()
    for window in windows:
        high.update(pair for pair, frequency in pair_frequency(numbers, window).items()
                    if frequency > threshold)
    return high


def pair_bonus(triple, high_pairs, bonus):
    """这一注里命中了几个高频对子，每个给一份分。"""
    distinct = sorted(set(triple))
    return bonus * sum(1 for i in range(len(distinct))
                       for j in range(i + 1, len(distinct))
                       if (distinct[i], distinct[j]) in high_pairs)


# ─── 形态 ───

MIN_PERIODS_FOR_FORM_SWITCH = 5


def form_switch_bonus(numbers, weight, zu6_threshold, zu3_threshold):
    """同一形态连开多期后，给另一种形态加分。

    **只看结尾那一段连续**，不需要遍历整段历史——迁移前这里每次都把全部
    1999 期都归一遍类，而答案只取决于末尾的连续段。

    这不是「该换了」：形态之间没有记忆。加分是为了让推荐池的形态构成不至于
    跟着最近一段连开而整体倾斜。
    """
    if len(numbers) < MIN_PERIODS_FOR_FORM_SWITCH:
        return {draw_props.ZU3: 0.0, draw_props.ZU6: 0.0}

    last_form = draw_props.classify_form(numbers[-1])
    streak = _tail_streak(numbers, last_form)

    bonus = {draw_props.ZU3: 0.0, draw_props.ZU6: 0.0}
    if last_form == draw_props.ZU6 and streak >= zu6_threshold:
        bonus[draw_props.ZU3] = weight * (streak - zu6_threshold + 1)
    elif last_form == draw_props.ZU3 and streak >= zu3_threshold:
        bonus[draw_props.ZU6] = weight * (streak - zu3_threshold + 1)
    return bonus


def _tail_streak(numbers, form):
    """结尾连续多少期是同一形态。"""
    streak = 0
    for index in range(len(numbers) - 1, -1, -1):
        if draw_props.classify_form(numbers[index]) != form:
            break
        streak += 1
    return streak


def form_recent_p(forms, window, decay):
    """最近若干期里各形态的指数加权占比。"""
    counts = exp_weighted_counts(_recent(forms, window), decay)
    total = sum(counts.values()) or 1.0
    return {form: counts.get(form, 0) / total for form in draw_props.THEORY_FORM_P}


# ─── 工具 ───

def gaussian_score(value, center, sigma):
    """离中心越远分越低的钟形分。sigma 非正时返回 0——那不是一个分布。"""
    if sigma <= 0:
        return 0.0
    z = (value - center) / sigma
    return math.exp(-0.5 * z * z)
