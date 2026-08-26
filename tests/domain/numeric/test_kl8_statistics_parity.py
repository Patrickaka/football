"""`update_statistics` 迁到 domain/numeric 后的整体差分。

单个统计函数各有自己的差分测试，但**整份统计字典**才是下游真正消费的东西：
16 个键里任何一个形状变了、少了、多了，特征评分与模型都会跟着变，而且不会
报错。所以这里把迁移前的算法原样抄进来对拍。

参照物是迁移前 `KL8Analyzer.update_statistics` 的实现（见提交
「统计层迁入 domain/numeric」之前的版本）。抄写而非 import，是因为原实现
已经被替换掉了——留在测试里的这份副本就是那一刻行为的存档。
"""
import gzip
import json
import pathlib
import unittest
from collections import Counter, defaultdict

from src.kl8.analyzer import KL8Analyzer
from src.kl8.config import (
    KL8_DEFAULT_HISTORY, KL8_DRAW_COUNT, KL8_EXPECTED_GAP, KL8_NUM_RANGE,
)

# 夹具取自线上真实历史的最近 300 期（gzip，10KB），**提交进仓库**。
# 原先直接读 `data/kl8_history.json`，而那个文件在 .gitignore 里——本地跑得
# 好好的，CI 上直接 FileNotFoundError。测试不该依赖未跟踪的本地数据。
DATA = (pathlib.Path(__file__).resolve().parents[2]
        / 'fixtures' / 'numeric' / 'kl8_history.json.gz')


def _history():
    with gzip.open(DATA, 'rt', encoding='utf-8') as fh:
        raw = json.load(fh)
    return raw['results'] if isinstance(raw, dict) else raw


def legacy_statistics(history_data):
    """迁移前的实现，逐行抄写。"""
    if not history_data:
        return {}

    n = len(history_data)
    recent = min(n, KL8_DEFAULT_HISTORY)
    recent_data = history_data[:recent]

    freq = Counter()
    for record in recent_data:
        for num in record['numbers']:
            freq[num] += 1

    gap = {}
    for num in range(1, 81):
        gap[num] = 0
        for record in recent_data:
            if num in record['numbers']:
                break
            gap[num] += 1

    last_numbers = set(recent_data[0]['numbers']) if recent_data else set()

    trend_freq = {}
    if recent >= 40:
        mid = recent // 2
        first_freq = Counter()
        for record in recent_data[mid:]:
            for num in record['numbers']:
                first_freq[num] += 1
        second_freq = Counter()
        for record in recent_data[:mid]:
            for num in record['numbers']:
                second_freq[num] += 1
        for num in range(1, 81):
            trend_freq[num] = second_freq.get(num, 0) - first_freq.get(num, 0)
    else:
        for num in range(1, 81):
            trend_freq[num] = 0

    pair_cooccurrence = {}
    for record in recent_data:
        nums = sorted(record['numbers'])
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                key = (nums[i], nums[j])
                pair_cooccurrence[key] = pair_cooccurrence.get(key, 0) + 1

    avg_cooccurrence = {}
    for num in range(1, 81):
        cooc_sum = 0
        cooc_count = 0
        for other in range(1, 81):
            if num != other:
                key = (min(num, other), max(num, other))
                cooc_sum += pair_cooccurrence.get(key, 0)
                cooc_count += 1
        avg_cooccurrence[num] = cooc_sum / cooc_count if cooc_count > 0 else 0

    transition_counts = defaultdict(Counter)
    trigger_counts = Counter()
    for older_idx in range(1, len(recent_data)):
        source = set(recent_data[older_idx]['numbers'])
        following = set(recent_data[older_idx - 1]['numbers'])
        for trigger in source:
            trigger_counts[trigger] += 1
            for target in following:
                transition_counts[trigger][target] += 1

    next_transition_probability = {}
    next_transition_support = {}
    for target in range(1, 81):
        weighted_probability = 0.0
        total_support = 0
        for trigger in last_numbers:
            support = trigger_counts.get(trigger, 0)
            if support <= 0:
                continue
            hits = transition_counts[trigger].get(target, 0)
            posterior = (hits + 5.0) / (support + 20.0)
            weighted_probability += posterior * support
            total_support += support
        next_transition_probability[target] = (
            weighted_probability / total_support if total_support else 0.25
        )
        next_transition_support[target] = total_support

    adjacent_freq = {}
    for num in range(1, 81):
        adj_nums = [x for x in [num - 1, num + 1] if 1 <= x <= 80]
        adjacent_freq[num] = (sum(freq.get(x, 0) for x in adj_nums) / len(adj_nums)
                              if adj_nums else 0)

    return {
        'frequency': dict(freq),
        'gap': gap,
        'trend': trend_freq,
        'pair_cooccurrence': pair_cooccurrence,
        'avg_cooccurrence': avg_cooccurrence,
        'next_transition_probability': next_transition_probability,
        'next_transition_support': next_transition_support,
        'adjacent_freq': adjacent_freq,
        'total_periods': recent,
        'expected_freq': recent * KL8_DRAW_COUNT / KL8_NUM_RANGE,
        'expected_gap': KL8_EXPECTED_GAP,
        'last_numbers': last_numbers,
        'freq_by_zone': _legacy_zone(recent_data),
        'freq_by_road': _legacy_road(recent_data),
        'freq_by_odd_even': _legacy_parity(recent_data),
        'freq_by_big_small': _legacy_big_small(recent_data),
    }


def _legacy_zone(data):
    freq = defaultdict(int)
    for record in data:
        for num in record['numbers']:
            freq[(num - 1) // 5 + 1] += 1
    return dict(freq)


def _legacy_road(data):
    freq = defaultdict(int)
    for record in data:
        for num in record['numbers']:
            freq[num % 3] += 1
    return dict(freq)


def _legacy_parity(data):
    freq = defaultdict(int)
    for record in data:
        for num in record['numbers']:
            freq['odd' if num % 2 == 1 else 'even'] += 1
    return dict(freq)


def _legacy_big_small(data):
    freq = defaultdict(int)
    for record in data:
        for num in record['numbers']:
            freq['big' if num > 40 else 'small'] += 1
    return dict(freq)


def _statistics(history_data):
    analyzer = KL8Analyzer.__new__(KL8Analyzer)
    analyzer.history_data = history_data
    analyzer.statistics = {}
    analyzer.update_statistics()
    return analyzer.statistics


class WholeStatisticsParityTests(unittest.TestCase):
    HISTORY = _history()

    def test_matches_legacy_on_real_history(self):
        self.assertEqual(_statistics(self.HISTORY), legacy_statistics(self.HISTORY))

    def test_matches_legacy_on_a_short_history(self):
        """窗口不足 40 期时冷热趋势整体归零，是一条独立的分支。"""
        short = self.HISTORY[:30]
        self.assertEqual(_statistics(short), legacy_statistics(short))

    def test_matches_legacy_at_the_trend_threshold(self):
        for size in (39, 40, 41):
            with self.subTest(size=size):
                window = self.HISTORY[:size]
                self.assertEqual(_statistics(window), legacy_statistics(window))

    def test_matches_legacy_on_a_single_draw(self):
        one = self.HISTORY[:1]
        self.assertEqual(_statistics(one), legacy_statistics(one))

    def test_empty_history_yields_empty_statistics(self):
        self.assertEqual(_statistics([]), {})

    def test_key_set_is_unchanged(self):
        """16 个键少一个、多一个，下游都不会报错，只会安静地算出别的结果。"""
        self.assertEqual(sorted(_statistics(self.HISTORY)),
                         sorted(legacy_statistics(self.HISTORY)))

    def test_window_is_capped_at_the_configured_size(self):
        stats = _statistics(self.HISTORY)
        self.assertEqual(stats['total_periods'],
                         min(len(self.HISTORY), KL8_DEFAULT_HISTORY))


if __name__ == '__main__':
    unittest.main()
