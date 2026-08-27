"""3D 机器学习模型的特征工程：把一注号码变成 54 个数。

**特征名与特征值在同一次遍历里产出。** 迁移前它们是两个各一百来行的独立
函数——一个 append 值、一个 append 名字，顺序全靠手工对齐。加一个特征要
改两处，漏改一处就会错位，而错位的后果是页面上的「特征重要性」张冠李戴：
`digit_0_exp_interval` 那一栏显示的其实是别的特征的分数。**全程不报错。**
现在 `describe()` 产出 (名字, 值) 对，另外两个方法各取一半，错位不再可能。

**这一层不读全局配置**：窗口、衰减、冷热分档的阈值一律由调用方传入。
迁移前它们硬编码在类里，其中冷热阈值是 1.3/0.7，而同一个项目的规则模型
用的是 1.2/0.8——两处各写各的，名字还都叫「热号」。

**特征多不等于预测准。** 3D 直选恒定 1/1000，这 54 个数只是让模型有东西
可学；线上实测的平均排名是 710，比随机期望的 500 还差。
"""
from collections import Counter, defaultdict
from typing import NamedTuple

from src.domain.numeric.lottery3d import draw as _draw
from src.domain.numeric.lottery3d import history as _history
from src.domain.numeric.lottery3d.space import DIGIT_SPACE, POSITIONS

# 质数就是质数，不是调参对象，所以留在领域层
PRIMES = frozenset({2, 3, 5, 7})
# 大中小三档的分界：0~2 小、3~5 中、6~9 大。与 `draw.BIG_SMALL_THRESHOLD`
# 的两分法是**两个不同的特征**，不是同一件事的两种写法
SIZE_TIERS = (2, 5)
COLD, WARM, HOT = 0, 1, 2
# 拿特征名时喂进去的那注号码。名字与它无关，随便哪一注都行
NAME_PROBE = (0, 0, 0)


class FeatureSettings(NamedTuple):
    """特征工程的可调项。**全部由调用方传入**——这些是配置，不是领域知识。"""

    history_window: int      # 统计只看最近这么多期
    decay: float             # 指数加权的衰减系数
    markov_alpha: float      # 转移概率的拉普拉斯平滑系数
    fallback_prob: float     # 转移概率查不到时的兜底。1/10 即均匀分布
    hot_ratio: float         # 出现次数达到均值这么多倍算热号
    warm_ratio: float        # 达到这么多倍算温号，更低算冷号
    trend_window: int        # 和值/跨度趋势看最近这么多期
    default_sum_mean: float  # 历史不足时的和值均值兜底
    default_span_mean: float  # 历史不足时的跨度均值兜底
    default_deviation: float  # 历史不足时的标准差兜底。0 会让偏离度除零


def distinct_digit_overlap(left, right):
    """两注**去重后**共有几个数字。

    与 `draw.digit_overlap` 是两个不同的量，不是同一件事写了两遍：
    那个按重数算（`112` 与 `111` 共有两个 1），这个按集合算（共有一个数字）。
    ML 特征要的是「上期开出的数字里，这注沾了几个」——一个数字沾一次还是
    两次不改变这个问题的答案，所以按集合。回测里的重合度则要按重数，
    否则「开出两个 1」和「开出一个 1」会被算成一回事。
    """
    return len(set(left) & set(right))


def cross_position_reuse(triple, last_draw):
    """上期开过、但**不在同一位**的数字个数。

    与「同位重复」互补：同一个数字换了位置出现，和它待在原位不动，
    在号码走势里是两种不同的现象。
    """
    return sum(1 for index in range(len(triple))
               if triple[index] in last_draw and triple[index] != last_draw[index])


def size_category(digit, tiers=SIZE_TIERS):
    """大中小分档，从 1 开始。"""
    for offset, bound in enumerate(tiers):
        if digit <= bound:
            return offset + 1
    return len(tiers) + 1


class FeatureEngineer:
    """把历史序列预计算成统计量，再按需为任意一注构造特征向量。

    预计算与逐注构造分开是有原因的：预测一次要为全部 1000 注建特征，
    而那些统计量对这 1000 注是同一份。混在一起会把 1000 次重复计算
    藏在一个看起来很正常的循环里。
    """

    def __init__(self, numbers, settings):
        self.settings = settings
        self.numbers = numbers
        self.recent = (numbers[-settings.history_window:]
                       if len(numbers) > settings.history_window else list(numbers))
        self.last_draw = numbers[-1] if numbers else None
        self.last_two = numbers[-2] if len(numbers) >= 2 else None
        self._precompute()

    # ─── 预计算 ───

    def _precompute(self):
        settings = self.settings
        self.freq_global = _history.exp_weighted_counts(
            [digit for number in self.recent for digit in number], settings.decay)
        self.total_global = sum(self.freq_global.values()) or 1.0

        self.freq_pos, self.total_pos = [], []
        for position in range(POSITIONS):
            frequency = _history.exp_weighted_counts(
                [number[position] for number in self.recent], settings.decay)
            self.freq_pos.append(frequency)
            self.total_pos.append(sum(frequency.values()) or 1.0)

        # 马尔可夫用**全量**历史而不是 recent：转移矩阵要的是尽可能多的样本，
        # 十个状态的一阶矩阵有一百个格子，二阶有一千个
        self.markov = [_history.build_markov(self.numbers, position)
                       for position in range(POSITIONS)]
        self.markov2 = [self._second_order(position) for position in range(POSITIONS)]

        self.miss_global = {digit: _history.miss_value(self.numbers, digit)
                            for digit in DIGIT_SPACE.numbers()}
        self.miss_pos = [
            {digit: _history.miss_value(self.numbers, digit, position=position)
             for digit in DIGIT_SPACE.numbers()}
            for position in range(POSITIONS)]

        self.interval_stats = {digit: self._intervals(digit)
                               for digit in DIGIT_SPACE.numbers()}

        self.oe_freq = Counter(_draw.odd_even_key(n) for n in self.recent)
        self.bs_freq = Counter(_draw.big_small_key(n) for n in self.recent)
        self.forms = [_draw.classify_form(number) for number in self.recent]
        self.form_streaks = self._streaks()

        self.consec_count = sum(1 for number in self.recent
                                if _draw.has_consecutive_digits(*number))
        self.consec_rate = self.consec_count / len(self.recent) if self.recent else 0.0

        self.sums = [sum(number) for number in self.recent]
        self.spans = [_draw.span(number) for number in self.recent]
        self.sum_freq = Counter(self.sums)
        self.span_freq = Counter(self.spans)
        self._precompute_trends()
        self._precompute_tiers()

        self.road_freq = Counter(_draw.road(digit)
                                 for number in self.recent for digit in number)
        self.road_total = sum(self.road_freq.values()) or 1

    def _second_order(self, position):
        """二阶转移计数。历史不足三期时给一个空表，查询自然落到兜底概率。"""
        if len(self.numbers) < POSITIONS:
            return defaultdict(Counter)
        return _history.build_markov2(self.numbers, position)

    def _intervals(self, digit):
        """该数字在窗口内相邻两次出现的间隔统计。

        **末尾那段还没结束的等待不计入**——它只说明「到现在还没开」，
        算成一个完整间隔会系统性拉高均值，而这个均值正是用来判断
        「当前遗漏算不算超期」的分母。
        """
        gaps, last_seen = [], None
        for index, number in enumerate(self.recent):
            if digit in number:
                if last_seen is not None:
                    gaps.append(index - last_seen)
                last_seen = index
        if not gaps:
            # 一次没出现过（或只出现一次）时用窗口长度兜底：那是间隔的上界
            return {'mean': len(self.recent), 'count': 0}
        return {'mean': sum(gaps) / len(gaps), 'count': len(gaps)}

    def _streaks(self):
        """形态的连续段：[(形态, 连续几期), ...]，按时间顺序。"""
        streaks, current, length = [], None, 0
        for form in self.forms:
            if form == current:
                length += 1
                continue
            if current is not None:
                streaks.append((current, length))
            current, length = form, 1
        if current is not None:
            streaks.append((current, length))
        return streaks

    def _precompute_trends(self):
        settings = self.settings
        window = settings.trend_window
        if len(self.sums) >= window:
            recent_sums = self.sums[-window:]
            self.sum_mean = sum(recent_sums) / window
            self.sum_std = (sum((value - self.sum_mean) ** 2
                                for value in recent_sums) / window) ** 0.5
            self.sum_trend = self._slope(recent_sums, self.sum_mean) if self.sum_std > 0 else 0.0
        else:
            self.sum_mean = (sum(self.sums) / len(self.sums) if self.sums
                             else settings.default_sum_mean)
            self.sum_std = settings.default_deviation
            self.sum_trend = 0.0

        if len(self.spans) >= window:
            recent_spans = self.spans[-window:]
            self.span_mean = sum(recent_spans) / window
            self.span_std = (sum((value - self.span_mean) ** 2
                                 for value in recent_spans) / window) ** 0.5
        else:
            self.span_mean = (sum(self.spans) / len(self.spans) if self.spans
                              else settings.default_span_mean)
            self.span_std = settings.default_deviation

    @staticmethod
    def _slope(values, mean):
        """最小二乘斜率，横轴取以中点为原点的期序。"""
        centre = (len(values) - 1) / 2
        numerator = sum((index - centre) * (values[index] - mean)
                        for index in range(len(values)))
        denominator = sum((index - centre) ** 2 for index in range(len(values)))
        return numerator / denominator if denominator else 0.0

    def _precompute_tiers(self):
        """冷/温/热三档。分界是**相对均值的倍数**，不是绝对次数——
        窗口一变，绝对次数跟着变，倍数才是稳定的那个量。
        """
        average = (sum(self.freq_global.values()) or 1) / DIGIT_SPACE.size
        self.digit_tier = {}
        for digit in DIGIT_SPACE.numbers():
            frequency = self.freq_global.get(digit, 0)
            if frequency >= average * self.settings.hot_ratio:
                self.digit_tier[digit] = HOT
            elif frequency >= average * self.settings.warm_ratio:
                self.digit_tier[digit] = WARM
            else:
                self.digit_tier[digit] = COLD

    # ─── 特征 ───

    def describe(self, triple):
        """一注的全部特征，(名字, 值) 成对产出。

        **名字与值在这一处同时定下来**，不再有两份需要手工对齐的列表。
        """
        features = []
        for group in (self._digit_heat, self._markov, self._misses, self._intervals_of,
                      self._tiers, self._sum_span, self._ratios, self._similarity,
                      self._roads, self._forms_of, self._composition):
            features.extend(group(tuple(triple)))
        return features

    def build_features(self, *triple):
        """特征向量。接受 `(a, b, c)` 或一个三元组，与迁移前的调用方式兼容。"""
        if len(triple) == 1:
            triple = tuple(triple[0])
        return [value for _, value in self.describe(triple)]

    def get_feature_names(self):
        """特征名，顺序与 `build_features` 一一对应——它们来自同一次遍历。"""
        return [name for name, _ in self.describe(NAME_PROBE)]

    def _digit_heat(self, triple):
        rows = [(f'digit_{index}_global_freq',
                 self.freq_global.get(digit, 0) / self.total_global)
                for index, digit in enumerate(triple)]
        rows += [(f'pos_{index}_freq',
                  self.freq_pos[index].get(digit, 0) / self.total_pos[index])
                 for index, digit in enumerate(triple)]
        return rows

    def _markov(self, triple):
        rows = []
        for index, digit in enumerate(triple):
            previous = self.last_draw[index] if self.last_draw else 0
            row = self.markov[index].get(previous, Counter())
            rows.append((f'pos_{index}_markov1', self._transition(row, digit)))
        for index, digit in enumerate(triple):
            if self.last_two and self.last_draw:
                key = (self.last_two[index], self.last_draw[index])
                probability = self._transition(self.markov2[index].get(key, Counter()), digit)
            else:
                probability = self.settings.fallback_prob
            rows.append((f'pos_{index}_markov2', probability))
        return rows

    def _transition(self, row, digit):
        smoothed = _history.markov_prob_smoothed(
            row, DIGIT_SPACE.numbers(), self.settings.markov_alpha)
        return smoothed.get(digit, self.settings.fallback_prob)

    def _misses(self, triple):
        rows = [(f'digit_{index}_miss_global', self.miss_global.get(digit, 0))
                for index, digit in enumerate(triple)]
        rows += [(f'pos_{index}_miss', self.miss_pos[index].get(digit, 0))
                 for index, digit in enumerate(triple)]
        return rows

    def _intervals_of(self, triple):
        rows = []
        for index, digit in enumerate(triple):
            stats = self.interval_stats[digit]
            # 平均间隔减去当前遗漏：还差几期到「该出了」。负数没有意义，夹到 0
            overdue = max(0, stats['mean'] - self.miss_global.get(digit, 0))
            rows.append((f'digit_{index}_exp_interval', overdue))
            rows.append((f'digit_{index}_appear_count', stats['count']))
        return rows

    def _tiers(self, triple):
        return [(f'digit_{index}_tier', self.digit_tier.get(digit, WARM))
                for index, digit in enumerate(triple)]

    def _sum_span(self, triple):
        total = sum(triple)
        span = _draw.span(triple)
        return [
            ('sum', total),
            ('sum_freq', self.sum_freq.get(total, 0)),
            ('sum_deviation', abs(total - self.sum_mean) / max(self.sum_std, 1.0)),
            ('span', span),
            ('span_freq', self.span_freq.get(span, 0)),
            ('span_deviation', abs(span - self.span_mean) / max(self.span_std, 1.0)),
        ]

    def _ratios(self, triple):
        odd_even = _draw.odd_even_key(triple)
        big_small = _draw.big_small_key(triple)
        return [
            ('odd_count', odd_even[0]),
            ('oe_freq', self.oe_freq.get(odd_even, 0)),
            ('big_count', big_small[0]),
            ('bs_freq', self.bs_freq.get(big_small, 0)),
            ('has_consecutive', 1 if _draw.has_consecutive_digits(*triple) else 0),
        ]

    def _similarity(self, triple):
        if not self.last_draw:
            rows = [('pos_repeat', 0), ('digit_overlap', 0), ('repeat_count', 0)]
        else:
            rows = [
                ('pos_repeat', sum(1 for index in range(POSITIONS)
                                   if triple[index] == self.last_draw[index])),
                ('digit_overlap', distinct_digit_overlap(triple, self.last_draw)),
                # 名字叫 repeat_count，量的其实是**跨位复用**。线上的特征重要性
                # 表里就是这个名字，改名会让那张表和历史记录对不上
                ('repeat_count', cross_position_reuse(triple, self.last_draw)),
            ]
        rows.append(('overlap_last2',
                     distinct_digit_overlap(triple, self.last_two) if self.last_two else 0))
        return rows

    def _roads(self, triple):
        roads = [_draw.road(digit) for digit in triple]
        match = sum(self.road_freq.get(road, 0) / self.road_total
                    for road in roads) / POSITIONS
        neighbours = set()
        for digit in (self.last_draw or ()):
            neighbours |= _draw.neighbor(digit)
        return [
            ('road_sum', sum(roads)),
            ('road_max', max(Counter(roads).values())),
            ('road_match', match),
            ('neighbor_overlap', len(set(triple) & neighbours)),
        ]

    def _forms_of(self, triple):
        distinct = len(set(triple))
        rows = [
            ('is_baozi', 1 if distinct == 1 else 0),
            ('is_zu3', 1 if distinct == 2 else 0),
        ]
        if self.form_streaks:
            streak_form, streak_length = self.form_streaks[-1]
            rows.append(('form_streak', streak_length))
            rows.append(('form_switch',
                         1 if _draw.classify_form(triple) != streak_form else 0))
        else:
            rows.extend([('form_streak', 0), ('form_switch', 0)])
        return rows

    def _composition(self, triple):
        rows = [('prime_count', sum(1 for digit in triple if digit in PRIMES))]
        rows += [(f'digit_{index}_size_cat', size_category(digit))
                 for index, digit in enumerate(triple)]
        return rows
