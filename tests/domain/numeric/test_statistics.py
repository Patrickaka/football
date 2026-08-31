"""数字彩票的号码统计。

`KL8Analyzer` 是一个 2170 行的类，混着数据加载、统计、特征评分、模型、
预测编排、快照结算、排除重算至少六种职责。这一批先把**统计层**摘出来——
它是纯计算，也是唯一真正属于基座的部分：区间分布、012 路、奇偶、大小、
遗漏、冷热趋势、共现，换一种数字彩票也是同一批概念，变的只是号码空间。

所以这里按 `NumberSpace` 参数化，让同一套统计支持不同的号码空间。

**正确性由差分测试保证**：旧实现仍在线（要到端点切换才删），对同一份真实
历史同时跑新旧两份、断言输出逐字相等。
"""
import gzip
import json
import pathlib
import unittest

from tests.domain.numeric.test_kl8_statistics_parity import (
    _legacy_big_small, _legacy_parity, _legacy_road, _legacy_zone,
)
from src.domain.numeric.statistics import (
    NumberSpace, adjacent_frequency, average_cooccurrence, gaps,
    high_low_frequency, number_frequency, pair_cooccurrence, parity_frequency,
    road_frequency, transition_probability, trend, zone_frequency,
)

# 夹具取自线上真实历史的最近 300 期（gzip，10KB），**提交进仓库**。
# 原先直接读 `data/kl8_history.json`，而那个文件在 .gitignore 里——本地跑得
# 好好的，CI 上直接 FileNotFoundError。测试不该依赖未跟踪的本地数据。
DATA = (pathlib.Path(__file__).resolve().parents[2]
        / 'fixtures' / 'numeric' / 'kl8_history.json.gz')
KL8 = NumberSpace(low=1, high=80)


def _history():
    with gzip.open(DATA, 'rt', encoding='utf-8') as fh:
        raw = json.load(fh)
    records = raw['results'] if isinstance(raw, dict) else raw
    return [r['numbers'] for r in records]


HISTORY = _history()
RECENT = HISTORY[:250]


class _LegacyBase(unittest.TestCase):
    """用真实历史与迁移前的实现对拍。

    参照物取自 `test_kl8_statistics_parity`——那里存着迁移前
    `KL8Analyzer` 各个私有方法的逐行抄本。分析器本身的那几个方法在替换时
    已经删掉了，抄本就是那一刻行为的存档。
    """

    @classmethod
    def setUpClass(cls):
        cls.records = [{'numbers': nums} for nums in RECENT]


class ZoneFrequencyParityTests(_LegacyBase):
    def test_matches_legacy(self):
        """kl8 把 80 个号码切成 16 个 5 码区，与 position_residual 粒度一致。"""
        self.assertEqual(zone_frequency(RECENT, size=5, space=KL8),
                         _legacy_zone(self.records))

    def test_zone_index_starts_at_one(self):
        self.assertEqual(min(zone_frequency(RECENT, size=5, space=KL8)), 1)
        self.assertEqual(max(zone_frequency(RECENT, size=5, space=KL8)), 16)

    def test_counts_add_up(self):
        total = sum(zone_frequency(RECENT, size=5, space=KL8).values())
        self.assertEqual(total, sum(len(d) for d in RECENT))

    def test_other_zone_sizes(self):
        """20 码区（四分区）也是常用粒度，不能写死 5。"""
        by_20 = zone_frequency(RECENT, size=20, space=KL8)
        self.assertEqual(sorted(by_20), [1, 2, 3, 4])
        self.assertEqual(sum(by_20.values()), sum(len(d) for d in RECENT))


class RoadFrequencyParityTests(_LegacyBase):
    def test_matches_legacy(self):
        self.assertEqual(road_frequency(RECENT), _legacy_road(self.records))

    def test_keys_are_the_residues(self):
        self.assertEqual(sorted(road_frequency(RECENT)), [0, 1, 2])


class ParityFrequencyParityTests(_LegacyBase):
    def test_matches_legacy(self):
        self.assertEqual(parity_frequency(RECENT),
                         _legacy_parity(self.records))


class HighLowFrequencyParityTests(_LegacyBase):
    def test_matches_legacy(self):
        """kl8 的分界是 40：大于 40 为大。分界值属于号码空间，不该写死。"""
        self.assertEqual(high_low_frequency(RECENT, threshold=40),
                         _legacy_big_small(self.records))

    def test_threshold_is_exclusive(self):
        """40 本身算小。边界差一位会让两侧的计数全部偏移。"""
        counts = high_low_frequency([[40, 41]], threshold=40)
        self.assertEqual(counts, {'small': 1, 'big': 1})


class NumberFrequencyTests(unittest.TestCase):
    def test_counts_every_number_in_the_window(self):
        freq = number_frequency(RECENT, space=KL8)
        self.assertEqual(sum(freq.values()), sum(len(d) for d in RECENT))

    def test_numbers_outside_the_space_are_dropped(self):
        """越界号码不进频率表。

        一条被改坏的开奖记录可能带着 81 号；让它流进频率表，下游按号码
        索引时会拿到一个本不该存在的键，而不会报错。`Draw` 在构造处就会
        拒掉这种记录，这里是第二道闸——统计函数直接吃号码列表，绕开了那道闸。
        """
        self.assertEqual(number_frequency([[1, 81, 0, 80]], space=KL8),
                         {1: 1, 80: 1})

    def test_without_a_space_nothing_is_dropped(self):
        """不给空间时不过滤——回测切片有时用的是自定义号码集。"""
        self.assertEqual(number_frequency([[1, 81]]), {1: 1, 81: 1})

    def test_unseen_numbers_are_absent_not_zero(self):
        """只出现过的号码才有键——补零会让「没开过」与「开过 0 次」混为一谈，
        而后者根本不存在。"""
        freq = number_frequency([[1, 2, 3]], space=KL8)
        self.assertEqual(freq, {1: 1, 2: 1, 3: 1})


class GapTests(unittest.TestCase):
    """遗漏：距上次开出过了几期。0 表示最近一期就开了。"""

    def test_matches_legacy_shape(self):
        records = [{'numbers': nums} for nums in RECENT]
        expected = {}
        for num in range(1, 81):
            expected[num] = 0
            for record in records:
                if num in record['numbers']:
                    break
                expected[num] += 1
        self.assertEqual(gaps(RECENT, space=KL8), expected)

    def test_zero_when_in_the_latest_draw(self):
        self.assertEqual(gaps([[1, 2], [3, 4]], space=NumberSpace(1, 4))[1], 0)

    def test_counts_draws_since_last_seen(self):
        self.assertEqual(gaps([[1], [1], [2]], space=NumberSpace(1, 2))[2], 2)

    def test_never_seen_equals_the_window_length(self):
        """从未开出时遗漏等于窗口长度——不是无穷大，也不是 0。"""
        self.assertEqual(gaps([[1], [1]], space=NumberSpace(1, 2))[2], 2)

    def test_every_number_in_the_space_has_an_entry(self):
        """遗漏必须覆盖整个号码空间：漏掉的号码在下游会被当成遗漏 0。"""
        self.assertEqual(sorted(gaps([[1]], space=NumberSpace(1, 5))), [1, 2, 3, 4, 5])


class TrendTests(unittest.TestCase):
    """冷热趋势：后半段频率减前半段频率。正数表示近期转热。"""

    def test_matches_legacy(self):
        recent = RECENT
        mid = len(recent) // 2
        from collections import Counter

        first = Counter(n for d in recent[mid:] for n in d)
        second = Counter(n for d in recent[:mid] for n in d)
        expected = {num: second.get(num, 0) - first.get(num, 0) for num in range(1, 81)}
        self.assertEqual(trend(recent, space=KL8), expected)

    def test_all_zero_when_the_window_is_too_short(self):
        """样本不足时不给出趋势，而不是给一个用半个窗口算出来的假趋势。"""
        self.assertEqual(set(trend(RECENT[:39], space=KL8).values()), {0})

    def test_threshold_is_forty_draws(self):
        self.assertNotEqual(set(trend(RECENT[:40], space=KL8).values()), {0})

    def test_positive_means_recently_hotter(self):
        space = NumberSpace(1, 2)
        draws = [[1]] * 20 + [[2]] * 20
        self.assertGreater(trend(draws, space=space, min_draws=2)[1], 0)
        self.assertLess(trend(draws, space=space, min_draws=2)[2], 0)


class PairCooccurrenceTests(unittest.TestCase):
    def test_counts_unordered_pairs(self):
        self.assertEqual(pair_cooccurrence([[3, 1, 2]]),
                         {(1, 2): 1, (1, 3): 1, (2, 3): 1})

    def test_accumulates_across_draws(self):
        self.assertEqual(pair_cooccurrence([[1, 2], [2, 1]])[(1, 2)], 2)

    def test_key_is_always_ordered(self):
        """键固定为 (小, 大)。不排序的话同一对会被记成两个不同的键。"""
        pairs = pair_cooccurrence([[5, 1]])
        self.assertIn((1, 5), pairs)
        self.assertNotIn((5, 1), pairs)

    def test_matches_legacy_on_real_history(self):
        expected = {}
        for nums in RECENT:
            ordered = sorted(nums)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    key = (ordered[i], ordered[j])
                    expected[key] = expected.get(key, 0) + 1
        self.assertEqual(pair_cooccurrence(RECENT), expected)


class AverageCooccurrenceTests(unittest.TestCase):
    """每个号码与其余号码的平均共现次数。"""

    def test_matches_legacy_on_real_history(self):
        pairs = pair_cooccurrence(RECENT)
        expected = {}
        for num in range(1, 81):
            total = 0
            count = 0
            for other in range(1, 81):
                if num != other:
                    total += pairs.get((min(num, other), max(num, other)), 0)
                    count += 1
            expected[num] = total / count if count > 0 else 0
        self.assertEqual(average_cooccurrence(pairs, space=KL8), expected)

    def test_covers_the_whole_space(self):
        avg = average_cooccurrence({}, space=NumberSpace(1, 5))
        self.assertEqual(sorted(avg), [1, 2, 3, 4, 5])
        self.assertEqual(set(avg.values()), {0})

    def test_single_number_space_has_no_partner(self):
        """只有一个号码时没有「其余号码」，平均值定义为 0 而不是除零。"""
        self.assertEqual(average_cooccurrence({}, space=NumberSpace(1, 1)), {1: 0})


class TransitionProbabilityTests(unittest.TestCase):
    """跨期条件关联：给定最近一期开出的号码，估计每个号码下一期出现的概率。

    与同期共现含义不同——那是「一起开」，这是「先后开」。
    Beta(5,15) 收缩把小样本拉回公平基线 0.25，免得偶然的 1/1、2/2 被当成强规律。
    """

    def test_matches_legacy_on_real_history(self):
        from collections import Counter, defaultdict

        recent = RECENT
        last_numbers = set(recent[0])
        transition_counts = defaultdict(Counter)
        trigger_counts = Counter()
        for older in range(1, len(recent)):
            source = set(recent[older])
            following = set(recent[older - 1])
            for trigger in source:
                trigger_counts[trigger] += 1
                for target in following:
                    transition_counts[trigger][target] += 1

        expected_prob, expected_support = {}, {}
        for target in range(1, 81):
            weighted = 0.0
            total = 0
            for trigger in last_numbers:
                support = trigger_counts.get(trigger, 0)
                if support <= 0:
                    continue
                hits = transition_counts[trigger].get(target, 0)
                weighted += ((hits + 5.0) / (support + 20.0)) * support
                total += support
            expected_prob[target] = weighted / total if total else 0.25
            expected_support[target] = total

        probability, support = transition_probability(recent, space=KL8)
        self.assertEqual(probability, expected_prob)
        self.assertEqual(support, expected_support)

    def test_falls_back_to_the_fair_baseline_without_support(self):
        """没有任何支持度时给公平基线 0.25，而不是 0——0 会被下游当成
        「几乎不可能开出」，那是凭空造出来的结论。"""
        probability, support = transition_probability([[1]], space=NumberSpace(1, 3))
        self.assertEqual(set(probability.values()), {0.25})
        self.assertEqual(set(support.values()), {0})

    def test_shrinkage_pulls_small_samples_toward_the_baseline(self):
        """一次一中（1/1）不该被判成 100%。"""
        draws = [[2], [1]]
        probability, _ = transition_probability(draws, space=NumberSpace(1, 2))
        self.assertLess(probability[2], 0.5)

    def test_covers_the_whole_space(self):
        probability, support = transition_probability(RECENT, space=NumberSpace(1, 80))
        self.assertEqual(len(probability), 80)
        self.assertEqual(len(support), 80)


class AdjacentFrequencyTests(unittest.TestCase):
    """邻号频率：号码左右相邻两个号的平均开出次数。"""

    def test_matches_legacy_on_real_history(self):
        freq = number_frequency(RECENT, space=KL8)
        expected = {}
        for num in range(1, 81):
            neighbours = [n for n in (num - 1, num + 1) if 1 <= n <= 80]
            expected[num] = (sum(freq.get(n, 0) for n in neighbours) / len(neighbours)
                             if neighbours else 0)
        self.assertEqual(adjacent_frequency(freq, space=KL8), expected)

    def test_edges_have_only_one_neighbour(self):
        """1 和 80 各只有一个邻居。按两个算会把它们的邻号频率腰斩。"""
        freq = {1: 0, 2: 10, 79: 10, 80: 0}
        adjacent = adjacent_frequency(freq, space=KL8)
        self.assertEqual(adjacent[1], 10)
        self.assertEqual(adjacent[80], 10)

    def test_interior_averages_both_sides(self):
        self.assertEqual(adjacent_frequency({1: 4, 3: 8}, space=NumberSpace(1, 3))[2], 6)


class NumberSpaceTests(unittest.TestCase):
    """号码空间不应在统计函数内部写死为 1~80。"""

    def test_numbers_covers_the_range(self):
        self.assertEqual(list(NumberSpace(1, 3).numbers()), [1, 2, 3])

    def test_size(self):
        self.assertEqual(NumberSpace(1, 80).size, 80)

    def test_rejects_an_inverted_range(self):
        with self.assertRaises(ValueError):
            NumberSpace(10, 1)

    def test_contains(self):
        space = NumberSpace(1, 80)
        self.assertTrue(space.contains(80))
        self.assertFalse(space.contains(81))
        self.assertFalse(space.contains(0))


if __name__ == '__main__':
    unittest.main()
