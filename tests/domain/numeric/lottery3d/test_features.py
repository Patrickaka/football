"""福彩 3D 的基础特征层：单注属性、历史统计、斜连、推荐去重。

这一层喂给评分与选号，**算错了不会报错，只会换一组推荐号**。参照物是从
迁移前的实现生成的黄金文件（`tests/fixtures/golden/lottery3d_features.json.gz`，
913 条），语料按七种长度的历史序列（1999 期全量 / 200 / 30 / 5 / 2 / 1 / 空）
× 十二个覆盖各种形态的三元组 × 十个数字铺开。

另有一组手写用例守住语义，重点是三处「反过来也不会报错」的地方：
序列的时间方向、连号与邻号的绕回差异、重合度按重数而非集合算。
"""
import gzip
import json
import pathlib
import unittest

from src.domain.numeric.lottery3d import draw, history, recommendations, slope
from src.domain.numeric.lottery3d.space import (
    DIGIT_SPACE, POSITIONS, POSITION_NAMES, SUM_MAX, SUM_MIN,
)
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'


def _load(name):
    with gzip.open(FIXTURES / name, 'rt', encoding='utf-8') as fh:
        return json.load(fh)


GOLDEN = _load('golden/lottery3d_features.json.gz')
NUMBERS = [tuple(r['digits'])
           for r in _load('numeric/lottery3d_history.json.gz')['results']]

SERIES = {'full': NUMBERS, 'recent200': NUMBERS[-200:], 'recent30': NUMBERS[-30:],
          'recent5': NUMBERS[-5:], 'tiny2': NUMBERS[-2:], 'one': NUMBERS[-1:],
          'empty': []}
TRIPLES = [(0, 0, 0), (9, 9, 9), (1, 2, 3), (3, 2, 1), (0, 5, 9), (4, 4, 7),
           (7, 4, 4), (5, 6, 7), (9, 0, 1), (2, 2, 8), (6, 6, 6), (0, 1, 2)]
DIGITS = list(range(10))

# 领域层不读全局配置，参数一律由调用方传入。这里写死的是**迁移当时
# `src/lottery3d/config.py` 里生效的那组值**——黄金文件就是用它们生成的。
# 不 import 配置：那样配置一改，期望值会跟着挪，黄金文件就白设了（判据 12）。
EXP_DECAY = 0.96
REBOUND_WINDOW, REBOUND_THRESHOLD, REBOUND_BONUS = 30, 0.5, 0.5
HOT_WINDOW = 20
SUM_TREND_WINDOW, SUM_TREND_ADJUST = 20, 0.0
MISS_CYCLE_WINDOW, MISS_OVER_RATIO, MISS_OVER_BONUS = 200, 2.5, 1.0
PAIR_WINDOWS, PAIR_THRESHOLD, PAIR_BONUS = (50, 100, 200), 0.15, 2.5
FORM_SWITCH_WEIGHT, ZU6_STREAK, ZU3_STREAK = 0.0, 8, 4
SUM_INTERVAL_WINDOW, SUM_INTERVAL_WIDTH = 5, 3
SUM_INTERVAL_BONUS, SUM_EXTREME_PENALTY = 0.0, 0.0
SLOPE_MIN_CHAIN, SLOPE_MAX_CHAIN, W_SLOPE_MATCH = 3, 6, 1.2
RECENT_WINDOW, RECENT_PENALTY, RECENT_CONSECUTIVE_PENALTY = 5, 5.0, 16.0


def _key(triple):
    return ''.join(map(str, triple))


def golden_entries():
    """按 (键, 值) 逐条产出全部语料。

    **测试与重生成脚本共用这一个生成器。** 两边各写一套语料，比对的就不是
    生成时的那批输入了——而那种不一致只会表现为「黄金文件里有些键从来没被
    验过」，不会报错。
    """
    yield from _draw_entries()
    yield from _miss_entries()
    yield from _series_entries()
    yield from _slope_entries()
    yield from _gaussian_entries()
    yield from _recommendation_entries()


def _draw_entries():
    for triple in TRIPLES:
        key = _key(triple)
        yield f'calc_span:{key}', draw.span(triple)
        yield f'odd_even_key:{key}', draw.odd_even_key(triple)
        yield f'big_small_key:{key}', draw.big_small_key(triple)
        yield f'has_consecutive:{key}', draw.has_consecutive_digits(*triple)
        yield f'classify_form:{key}', draw.classify_form(triple)
    for digit in DIGITS:
        yield f'neighbor:{digit}', sorted(draw.neighbor(digit))
        yield f'road:{digit}', draw.road(digit)
    for ratio in ((0, 3), (3, 0), (1, 2), (2, 1)):
        for kind in ('oe', 'bs', 'other'):
            yield f'ratio_label:{ratio[0]}{ratio[1]}:{kind}', draw.ratio_label(ratio, kind)
    yield 'form_labels', draw.FORM_LABELS
    yield 'theory_form_p', draw.THEORY_FORM_P


def _miss_entries():
    for name, series in SERIES.items():
        for position in (None, 0, 1, 2):
            for digit in DIGITS:
                yield (f'miss_value:{name}:{position}:{digit}',
                       history.miss_value(series, digit, position))
        for digit in DIGITS:
            for window in (30, 100):
                yield (f'avg_miss_cycle:{name}:{digit}:{window}',
                       history.average_miss_cycle(series, digit, window))
        forms = [draw.classify_form(t) for t in series]
        for target in ('zu6', 'zu3', 'baozi'):
            yield f'form_miss:{name}:{target}', history.form_miss(forms, target)
        for window in (5, 30, 100):
            yield (f'form_recent_p:{name}:{window}',
                   history.form_recent_p(forms, window, EXP_DECAY))


def _series_entries():
    for name, series in SERIES.items():
        for position in range(POSITIONS):
            yield f'markov:{name}:{position}', history.build_markov(series, position)
            yield f'markov2:{name}:{position}', history.build_markov2(series, position)
        yield (f'rebound:{name}',
               history.rebound_bonus(series, REBOUND_WINDOW, REBOUND_THRESHOLD,
                                     REBOUND_BONUS))
        yield f'hot_classify:{name}', history.classify_by_hot(series, HOT_WINDOW)
        yield (f'sum_trend:{name}',
               history.sum_trend(series, SUM_TREND_WINDOW, SUM_TREND_ADJUST))
        yield (f'miss_cycle_bonus:{name}',
               history.miss_cycle_bonus(series, MISS_CYCLE_WINDOW, MISS_OVER_RATIO,
                                        MISS_OVER_BONUS))
        yield (f'high_freq_pairs:{name}',
               sorted(map(list, history.high_freq_pairs(series, PAIR_WINDOWS,
                                                        PAIR_THRESHOLD))))
        yield (f'form_switch:{name}',
               history.form_switch_bonus(series, FORM_SWITCH_WEIGHT, ZU6_STREAK,
                                         ZU3_STREAK))
        yield (f'sum_interval:{name}',
               history.sum_interval(series, SUM_INTERVAL_WINDOW, SUM_INTERVAL_WIDTH,
                                    SUM_INTERVAL_BONUS, SUM_EXTREME_PENALTY))
        for window in (10, 50):
            yield f'pair_freq:{name}:{window}', history.pair_frequency(series, window)
        flat = [d for t in series for d in t]
        for decay in (0.9, 1.0, 0.5):
            yield (f'exp_weighted:{name}:{decay}',
                   history.exp_weighted_counts(flat, decay))
        for window in (0, 1, 5, 30, 100, 10000):
            yield (f'recent_slice:{name}:{window}',
                   [list(x) for x in history._recent(series, window)])


def _slope_entries():
    for name, series in SERIES.items():
        yield f'slope_patterns:{name}', slope.analyze(series, SLOPE_MIN_CHAIN,
                                                      SLOPE_MAX_CHAIN)
        yield f'cross_slope:{name}', slope.cross_period_signals(series)
    for left in (0, 3, 9):
        for right in (0, 4, 9):
            yield f'slope_step:{left}:{right}', slope.step_between(left, right)
    for name in ('full', 'recent200', 'recent30', 'recent5'):
        for position in range(POSITIONS):
            column = [t[position] for t in SERIES[name]]
            yield (f'slope_chain:{name}:{position}',
                   slope.detect_chain(column, SLOPE_MIN_CHAIN, SLOPE_MAX_CHAIN))
    for name in ('full', 'recent200', 'recent30'):
        analysis = slope.analyze(SERIES[name], SLOPE_MIN_CHAIN, SLOPE_MAX_CHAIN)
        pairs = history.high_freq_pairs(SERIES[name], PAIR_WINDOWS, PAIR_THRESHOLD)
        for triple in TRIPLES:
            yield (f'slope_bonus:{name}:{_key(triple)}',
                   round(slope.triplet_bonus(triple, analysis, W_SLOPE_MATCH), 10))
            yield (f'pair_bonus:{name}:{_key(triple)}',
                   round(history.pair_bonus(triple, pairs, PAIR_BONUS), 10))


def _gaussian_entries():
    for value in (0, 5, 13.5, 27, -3, 100):
        for center in (13.5, 0, 27):
            for sigma in (1, 5, 0.1):
                yield (f'gaussian:{value}:{center}:{sigma}',
                       history.gaussian_score(value, center, sigma))


def _recommendation_entries():
    pools = [[(1.0, '123'), (0.9, '456'), (0.8, '789')],
             [(2.0, '000'), (1.5, '111')], []]
    recents = [[], [{'numbers': ['123']}],
               [{'numbers': ['123', '456']}, {'numbers': ['789']}]]
    for i, pool in enumerate(pools):
        for j, recent in enumerate(recents):
            yield (f'recent_penalty:{i}:{j}',
                   recommendations.penalise_repeats(pool, recent, RECENT_WINDOW,
                                                    RECENT_PENALTY,
                                                    RECENT_CONSECUTIVE_PENALTY))
    for actual in ('123', '000', '987'):
        for candidates in (['123', '456'], ['321'], [], ['111', '122']):
            yield (f'max_overlap:{actual}:{"-".join(candidates) or "none"}',
                   recommendations.max_digit_overlap(actual, candidates))


class GoldenTests(unittest.TestCase):
    """迁移前后逐条比对。任何一条对不上，都意味着推荐的号变了。"""

    def test_matches_golden(self):
        seen = set()
        for key, value in golden_entries():
            seen.add(key)
            with self.subTest(case=key):
                self.assertEqual(as_comparable(value), GOLDEN[key])
        # 黄金文件里有语料不再覆盖的键，说明语料被删过而文件没跟着重生成——
        # 那些键从此再也不会被验证，而少了断言不会有任何提示。
        self.assertEqual(sorted(set(GOLDEN) - seen), [])


class DisabledEntropyTests(unittest.TestCase):
    """`entropy_model` 恒为 0，且**刻意**留在 `src/lottery3d/features.py`。

    它不是领域逻辑，是一条结论：「长期未出现」不会提高下一期出现的概率，
    所以熵值奖励被关掉了。`digit_scores` 仍在把它加进去——删掉函数等于把
    这条结论也删掉，下次有人想「加个冷号奖励」时就读不到了。
    """

    def test_entropy_bonus_is_zero_for_every_digit(self):
        from src.lottery3d.features import entropy_model
        for series in (NUMBERS, NUMBERS[-5:], []):
            with self.subTest(length=len(series)):
                self.assertEqual(entropy_model(series),
                                 {d: 0.0 for d in range(10)})


class TimeDirectionTests(unittest.TestCase):
    """序列是**旧在前、新在后**。方向反了不会报错，只会让结论整个颠倒。"""

    def test_miss_counts_from_the_tail(self):
        """最近一期就有的数字，遗漏是 0。"""
        series = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
        self.assertEqual(history.miss_value(series, 3), 0)
        self.assertEqual(history.miss_value(series, 1), 2)

    def test_recent_slice_takes_the_tail(self):
        series = list(range(10))
        self.assertEqual(history._recent(series, 3), [7, 8, 9])

    def test_exp_weight_favours_the_newest(self):
        """最后一项权重 1，往前每退一步乘一次衰减。"""
        counts = history.exp_weighted_counts([1, 2], 0.5)
        self.assertEqual(counts[2], 1.0)
        self.assertEqual(counts[1], 0.5)

    def test_markov_reads_older_to_newer(self):
        """转移是「先出现的 → 后出现的」。反了会把走势整个照镜子。"""
        series = [(1, 0, 0), (2, 0, 0), (3, 0, 0)]
        transitions = history.build_markov(series, 0)
        self.assertEqual(transitions[1][2], 1)
        self.assertEqual(transitions[2][1], 0)

    def test_markov2_reads_two_steps_back(self):
        series = [(1, 0, 0), (2, 0, 0), (3, 0, 0)]
        self.assertEqual(history.build_markov2(series, 0)[(1, 2)][3], 1)


class DrawPropertyTests(unittest.TestCase):

    def test_consecutive_does_not_wrap_but_neighbour_does(self):
        """两个概念故意不一样：连号比数值，邻号是转盘上的相邻。"""
        self.assertFalse(draw.has_consecutive_digits(9, 0, 5))
        self.assertIn(9, draw.neighbor(0))
        self.assertIn(1, draw.neighbor(0))

    def test_consecutive_finds_any_pair_not_only_adjacent_positions(self):
        """`3_4` 隔着中间一位也算连号——比的是数字，不是位置。"""
        self.assertTrue(draw.has_consecutive_digits(3, 8, 4))

    def test_form_classification_covers_all_three(self):
        self.assertEqual(draw.classify_form((1, 2, 3)), draw.ZU6)
        self.assertEqual(draw.classify_form((1, 1, 3)), draw.ZU3)
        self.assertEqual(draw.classify_form((1, 1, 1)), draw.BAOZI)

    def test_theory_form_probabilities_sum_to_one(self):
        """它们是组合数算出来的常数，不是拟合值——加起来必须是 1。"""
        self.assertAlmostEqual(sum(draw.THEORY_FORM_P.values()), 1.0, places=6)

    def test_overlap_counts_multiplicity(self):
        """`112` 与 `122` 共有一个 1、一个 2，不是三个。"""
        self.assertEqual(draw.digit_overlap('112', '122'), 2)

    def test_overlap_of_identical_draws_is_the_whole_draw(self):
        self.assertEqual(draw.digit_overlap('111', '111'), 3)

    def test_both_ends_of_the_digit_space_are_covered(self):
        self.assertEqual(draw.road(DIGIT_SPACE.low), 0)
        self.assertEqual(draw.span((DIGIT_SPACE.low, DIGIT_SPACE.high, 5)),
                         DIGIT_SPACE.high - DIGIT_SPACE.low)


class HotClassificationTests(unittest.TestCase):
    """分档阈值与 `config.HOT_RATIO` 名字像、含义完全不同，容易接错。"""

    def _series(self, digit, times, filler=0):
        """构造一段：某个数字出现 times 次，其余用 filler 补满。"""
        rows = [(digit, digit, digit)] * times
        rows += [(filler, filler, filler)] * (30 - times)
        return rows

    def test_over_represented_digit_is_hot(self):
        hot, _, _ = history.classify_by_hot(self._series(7, 20), 30)
        self.assertIn(7, hot)

    def test_under_represented_digit_is_cold(self):
        _, _, cold = history.classify_by_hot(self._series(7, 1), 30)
        self.assertIn(7, cold)

    def test_thresholds_are_the_ratio_not_the_pool_share(self):
        """1.2 / 0.8 是「相对理论值的倍率」。误接成 0.4/0.4 会让温号档消失。"""
        self.assertEqual((history.HOT_THRESHOLD, history.WARM_THRESHOLD), (1.2, 0.8))

    def test_short_history_calls_everything_hot(self):
        """全冷会凭空造出十个冷号，让下游的冷号加分整体走偏。"""
        hot, warm, cold = history.classify_by_hot(NUMBERS[-3:], 30)
        self.assertEqual((hot, warm, cold), (list(range(10)), [], []))

    def test_every_digit_lands_in_exactly_one_bucket(self):
        hot, warm, cold = history.classify_by_hot(NUMBERS, 30)
        self.assertEqual(sorted(hot + warm + cold), list(range(10)))


class SumTests(unittest.TestCase):

    def test_trend_needs_a_margin_not_just_any_difference(self):
        """差一点点算震荡。没有余量的话趋势会每期翻来覆去。"""
        flat = [(4, 4, 5)] * 10 + [(4, 5, 5)] * 10
        self.assertEqual(history.sum_trend(flat, 20, 1.0)[1], 'oscillate')

    def test_rising_sums_are_reported_as_up(self):
        rising = [(0, 0, 0)] * 10 + [(9, 9, 9)] * 10
        self.assertEqual(history.sum_trend(rising, 20, 1.0)[1], 'up')

    def test_falling_sums_are_reported_as_down(self):
        """上下两个方向都要有样本，只测一边接反了也发现不了。"""
        falling = [(9, 9, 9)] * 10 + [(0, 0, 0)] * 10
        self.assertEqual(history.sum_trend(falling, 20, 1.0)[1], 'down')

    def test_center_is_clamped_into_the_possible_range(self):
        rising = [(0, 0, 0)] * 10 + [(9, 9, 9)] * 10
        center, _ = history.sum_trend(rising, 20, 99.0)
        self.assertLessEqual(center, SUM_MAX)
        self.assertGreaterEqual(center, SUM_MIN)

    def test_interval_rewards_the_centre_and_penalises_the_extremes(self):
        result = history.sum_interval([(4, 5, 5)] * 10, 5, 4, 0.8, 0.5)
        self.assertEqual(result['bonus'][14], 0.8)
        self.assertEqual(result['bonus'][0], -0.5)
        self.assertEqual(result['bonus'][SUM_MAX], -0.5)

    def test_interval_covers_every_possible_sum(self):
        result = history.sum_interval([(4, 5, 5)] * 10, 5, 4, 0.8, 0.5)
        self.assertEqual(sorted(result['bonus']), list(range(SUM_MIN, SUM_MAX + 1)))


class PairTests(unittest.TestCase):

    def test_repeated_digits_do_not_form_a_pair_with_themselves(self):
        """`117` 只贡献 (1,7)。算上 (1,1) 会让豹子和组三凭空多出对子。"""
        frequency = history.pair_frequency([(1, 1, 7)], 10)
        self.assertEqual(sorted(frequency), [(1, 7)])

    def test_frequency_is_per_draw_not_per_occurrence(self):
        self.assertEqual(history.pair_frequency([(1, 2, 5)] * 4, 10)[(1, 2)], 1.0)

    def test_high_freq_pairs_unions_the_windows(self):
        """并集而非交集：要求所有窗口都满足，等于只剩最长那个窗口的结论。"""
        # 窗口刻意选得一大一小：线上那组（50/100/200）都比这段语料长，三个
        # 窗口看到的是同一段，并集与交集就分不出来了。
        # 序列旧在前，所以短窗口看到的是**末尾**那三期。
        series = [(3, 4, 9)] * 30 + [(1, 2, 9)] * 3
        both = history.high_freq_pairs(series, (3, 33), 0.5)
        self.assertIn((1, 2), both)
        self.assertIn((3, 4), both)

    def test_pair_bonus_counts_every_matching_pair(self):
        self.assertEqual(history.pair_bonus((1, 2, 3), {(1, 2), (2, 3)}, 0.5), 1.0)

    def test_pair_bonus_is_zero_without_matches(self):
        self.assertEqual(history.pair_bonus((1, 2, 3), {(4, 5)}, 0.5), 0.0)


class FormSwitchTests(unittest.TestCase):
    """线上 `FORM_SWITCH_WEIGHT` 现在是 0，这一项等于关着。机制类用例用自己的
    非零权重——拿 0 去测，两个方向接反了也看不出来。"""

    WEIGHT, ZU6_MIN, ZU3_MIN = 0.5, 5, 3

    def test_long_zu6_streak_rewards_zu3(self):
        series = [(1, 2, 3), (4, 5, 6), (1, 3, 5), (2, 4, 6), (3, 5, 7), (0, 2, 4)]
        bonus = history.form_switch_bonus(series, self.WEIGHT, self.ZU6_MIN, self.ZU3_MIN)
        self.assertGreater(bonus[draw.ZU3], 0)
        self.assertEqual(bonus[draw.ZU6], 0)

    def test_long_zu3_streak_rewards_zu6(self):
        """两个方向对称，只测一个的话反了也发现不了。"""
        series = [(1, 1, 3)] * 6
        bonus = history.form_switch_bonus(series, self.WEIGHT, self.ZU6_MIN, self.ZU3_MIN)
        self.assertGreater(bonus[draw.ZU6], 0)
        self.assertEqual(bonus[draw.ZU3], 0)

    def test_short_streak_gives_nothing(self):
        series = [(1, 1, 3), (1, 2, 3), (4, 5, 6), (1, 1, 2), (7, 8, 9)]
        self.assertEqual(history.form_switch_bonus(series, self.WEIGHT, self.ZU6_MIN, self.ZU3_MIN),
                         {draw.ZU3: 0.0, draw.ZU6: 0.0})

    def test_too_little_history_gives_nothing(self):
        self.assertEqual(
            history.form_switch_bonus([(1, 2, 3)] * 4, self.WEIGHT,
                                      self.ZU6_MIN, self.ZU3_MIN),
            {draw.ZU3: 0.0, draw.ZU6: 0.0})

    def test_streak_only_reads_the_tail(self):
        """迁移前这里把全部历史都归一遍类，而答案只取决于末尾那一段。"""
        series = [(1, 1, 1)] * 500 + [(1, 2, 3)] * 6
        self.assertEqual(history._tail_streak(series, draw.ZU6), 6)


class SlopeTests(unittest.TestCase):

    def test_step_rejects_the_wrap_around(self):
        """9→0 不是斜连。认了它，「等差」就跟邻号混成一件事。"""
        self.assertIsNone(slope.step_between(9, 0))
        self.assertEqual(slope.step_between(3, 4), 1)
        self.assertEqual(slope.step_between(4, 3), -1)

    def test_longest_chain_wins(self):
        chain = slope.detect_chain([1, 2, 3, 4, 5], 3, 6)
        self.assertEqual(chain['length'], 5)
        self.assertEqual(chain['predict_digit'], 6)

    def test_chain_is_rejected_when_the_prediction_leaves_the_space(self):
        """5→6→7→8→9 的下一个是 10，不存在，所以这条链不成立。"""
        self.assertIsNone(slope.detect_chain([7, 8, 9], 3, 3))

    def test_descending_chain_is_detected_too(self):
        chain = slope.detect_chain([5, 4, 3], 3, 6)
        self.assertEqual((chain['step'], chain['predict_digit']), (-1, 2))

    def test_mixed_steps_are_not_a_chain(self):
        self.assertIsNone(slope.detect_chain([1, 2, 1], 3, 6))

    def test_position_hints_are_keyed_by_the_real_position_names(self):
        """键写成「百位」会让加分恒为 0，而且不报错。"""
        analysis = slope.analyze(NUMBERS, SLOPE_MIN_CHAIN, SLOPE_MAX_CHAIN)
        self.assertEqual(sorted(analysis['position_hints']), sorted(POSITION_NAMES))

    def test_triplet_bonus_weights_by_hint_strength(self):
        analysis = {'position_hints': {POSITION_NAMES[0]: [{'digit': 7, 'strength': 2.0}]}}
        self.assertEqual(slope.triplet_bonus((7, 0, 0), analysis, 1.5), 3.0)

    def test_triplet_bonus_ignores_hints_for_other_positions(self):
        analysis = {'position_hints': {POSITION_NAMES[0]: [{'digit': 7, 'strength': 2.0}]}}
        self.assertEqual(slope.triplet_bonus((0, 7, 0), analysis, 1.5), 0.0)

    def test_missing_analysis_is_not_an_error(self):
        self.assertEqual(slope.triplet_bonus((1, 2, 3), None, 1.5), 0.0)


class RecommendationTests(unittest.TestCase):

    def test_both_history_formats_are_understood(self):
        """线上新旧两种格式都还在，只认一种会让一半历史被当成空。"""
        new_format = [{'period': '1', 'recommendations': ['123']}]
        old_format = [['123']]
        for history_entries in (new_format, old_format):
            with self.subTest(shape=history_entries):
                seen, _ = recommendations.recent_numbers(history_entries, 5)
                self.assertEqual(seen, {'123'})

    def test_repeated_recommendation_is_penalised(self):
        pool = [(1.0, '123'), (1.0, '456')]
        result = recommendations.penalise_repeats(pool, [['123']], 5, 0.5, 0.3)
        self.assertEqual(result, [(0.5, '123'), (1.0, '456')])

    def test_consecutive_recommendation_is_penalised_twice(self):
        pool = [(1.0, '123')]
        result = recommendations.penalise_repeats(pool, [['123'], ['123']], 5, 0.5, 0.3)
        self.assertAlmostEqual(result[0][0], 0.2)

    def test_only_the_window_is_considered(self):
        pool = [(1.0, '123')]
        old = [['123']] + [['999']] * 5
        self.assertEqual(recommendations.penalise_repeats(pool, old, 5, 0.5, 0.3), pool)

    def test_empty_history_returns_the_pool_unchanged(self):
        pool = [(1.0, '123')]
        self.assertIs(recommendations.penalise_repeats(pool, [], 5, 0.5, 0.3), pool)

    def test_max_overlap_of_no_candidates_is_zero(self):
        self.assertEqual(recommendations.max_digit_overlap('123', []), 0)


if __name__ == '__main__':
    unittest.main()
