"""开奖号码的统计。

这些量是数字彩票分析的共同起点：某个号码开了多少次、多久没开、最近是转热
还是转冷、哪些号码爱一起出现、号码在区间/012 路/奇偶/大小上的分布。
换一种彩票也是同一批概念，变的只是号码空间——所以按 `NumberSpace` 参数化，
而不是把 1~80 写死，方便同一套统计支持不同的号码范围。

统一约定：输入是**号码列表的列表**，按期倒序（第一条是最近一期）。不接
`Draw` 对象是为了让回测能直接喂切片，不必为每个窗口重新构造对象。
"""
from collections import Counter, defaultdict
from dataclasses import dataclass

# 冷热趋势要把窗口对半劈开比较，样本太少时两半都不足以说明问题。
MIN_TREND_DRAWS = 40

# 跨期转移的 Beta 先验。相当于「先验地见过 5 次命中、20 次触发」，
# 把小样本拉回公平基线 0.25——否则偶然的 1/1、2/2 会被当成强规律。
TRANSITION_PRIOR_HITS = 5.0
TRANSITION_PRIOR_TRIALS = 20.0
TRANSITION_BASELINE = TRANSITION_PRIOR_HITS / TRANSITION_PRIOR_TRIALS


@dataclass(frozen=True)
class NumberSpace:
    """号码空间。快乐8使用 1~80，也允许调用方传入其他范围。"""

    low: int
    high: int

    def __post_init__(self):
        if self.high < self.low:
            raise ValueError(f'号码空间上下界颠倒: low={self.low} high={self.high}')

    @property
    def size(self):
        return self.high - self.low + 1

    def numbers(self):
        return range(self.low, self.high + 1)

    def contains(self, number):
        return self.low <= number <= self.high


def number_frequency(draws, space=None):
    """每个号码开出的次数。

    只统计出现过的号码，**不给未出现的补零**——「没开过」和「开过 0 次」
    在下游是两种处理，混为一谈会让「从未开出」这个信号消失。需要全空间
    覆盖的量（比如遗漏）另有函数。
    """
    freq = Counter()
    for draw in draws:
        freq.update(draw)
    if space is not None:
        return {n: c for n, c in freq.items() if space.contains(n)}
    return dict(freq)


def gaps(draws, space):
    """每个号码距上次开出隔了几期。0 表示最近一期就开了。

    **覆盖整个号码空间**，与 `number_frequency` 相反：漏掉的号码在下游会被
    当成遗漏 0，也就是「刚开过」——恰好与事实相反。从未开出的记为窗口长度。
    """
    result = {}
    for number in space.numbers():
        gap = 0
        for draw in draws:
            if number in draw:
                break
            gap += 1
        result[number] = gap
    return result


def trend(draws, space, min_draws=MIN_TREND_DRAWS):
    """冷热趋势：后半段频率减前半段频率。正数表示近期转热。

    样本不足时全部返回 0——**不给出趋势，而不是给一个用半个窗口算出来的
    假趋势**。后者更危险：它看起来是个正常的数，下游无从分辨。
    """
    total = len(draws)
    if total < min_draws:
        return {number: 0 for number in space.numbers()}

    mid = total // 2
    older = Counter(n for draw in draws[mid:] for n in draw)
    newer = Counter(n for draw in draws[:mid] for n in draw)
    return {number: newer.get(number, 0) - older.get(number, 0)
            for number in space.numbers()}


def pair_cooccurrence(draws):
    """号码两两同时开出的次数。键固定为 (小, 大)。

    不排序的话同一对会被记成两个不同的键，共现次数被劈成两半。
    """
    pairs = defaultdict(int)
    for draw in draws:
        ordered = sorted(draw)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                pairs[(ordered[i], ordered[j])] += 1
    return dict(pairs)


def average_cooccurrence(pairs, space):
    """每个号码与其余号码的平均共现次数。覆盖整个号码空间。"""
    numbers = list(space.numbers())
    result = {}
    for number in numbers:
        others = [n for n in numbers if n != number]
        if not others:
            result[number] = 0
            continue
        total = sum(pairs.get((min(number, n), max(number, n)), 0) for n in others)
        result[number] = total / len(others)
    return result


def transition_probability(draws, space, prior_hits=TRANSITION_PRIOR_HITS,
                           prior_trials=TRANSITION_PRIOR_TRIALS):
    """给定最近一期开出的号码，估计每个号码下一期出现的概率。

    与同期共现含义不同：那是「一起开」，这是「先后开」。历史按
    「较老一期 → 紧接着的较新一期」统计。

    返回 (概率, 支持度)。**没有任何支持度时给公平基线而不是 0**——0 会被
    下游当成「几乎不可能开出」，那是凭空造出来的结论。
    """
    if not draws:
        baseline = prior_hits / prior_trials
        return ({n: baseline for n in space.numbers()},
                {n: 0 for n in space.numbers()})

    transitions = defaultdict(Counter)
    triggers = Counter()
    for older in range(1, len(draws)):
        source = set(draws[older])
        following = set(draws[older - 1])
        for trigger in source:
            triggers[trigger] += 1
            transitions[trigger].update(following)

    last_numbers = set(draws[0])
    probability, support = {}, {}
    for target in space.numbers():
        weighted = 0.0
        total = 0
        for trigger in last_numbers:
            trials = triggers.get(trigger, 0)
            if trials <= 0:
                continue
            hits = transitions[trigger].get(target, 0)
            weighted += ((hits + prior_hits) / (trials + prior_trials)) * trials
            total += trials
        probability[target] = (weighted / total if total
                               else prior_hits / prior_trials)
        support[target] = total
    return probability, support


def adjacent_frequency(frequency, space):
    """号码左右相邻两号的平均开出次数。

    边界号码只有一个邻居。按两个算会把 1 和 80 的邻号频率腰斩，
    而这两个号本就容易被边界效应影响。
    """
    result = {}
    for number in space.numbers():
        neighbours = [n for n in (number - 1, number + 1) if space.contains(n)]
        result[number] = (sum(frequency.get(n, 0) for n in neighbours) / len(neighbours)
                          if neighbours else 0)
    return result


def zone_frequency(draws, size, space):
    """按等宽区间统计。区间编号从 1 开始。

    `size` 不写死：kl8 同时在用 5 码区（与残差粒度一致）与 20 码区（四分区）。
    """
    freq = defaultdict(int)
    for draw in draws:
        for number in draw:
            freq[(number - space.low) // size + 1] += 1
    return dict(freq)


def road_frequency(draws, modulus=3):
    """012 路：号码对 3 取余的分布。"""
    freq = defaultdict(int)
    for draw in draws:
        for number in draw:
            freq[number % modulus] += 1
    return dict(freq)


def parity_frequency(draws):
    freq = defaultdict(int)
    for draw in draws:
        for number in draw:
            freq['odd' if number % 2 == 1 else 'even'] += 1
    return dict(freq)


def high_low_frequency(draws, threshold):
    """大小分布。**分界值是排他的**：等于 threshold 的号码算小。

    差一位会让两侧计数整体偏移，而结果看上去仍然「正常」。
    """
    freq = defaultdict(int)
    for draw in draws:
        for number in draw:
            freq['big' if number > threshold else 'small'] += 1
    return dict(freq)
