"""福彩 3D 的展示层与可用性判断。

展示层不做计算，只做**取舍与四舍五入**——但它决定页面上出现的每个数字，
改错了不会报错，只会让人看到一个不一样的数。可用性判断（数据够不够干净、
ML 缓存还算不算数）则决定要不要让 ML 参与出号。

参照物是从迁移前的实现生成的黄金文件
（`tests/fixtures/golden/lottery3d_prediction.json.gz`，35 条）。
"""
import gzip
import json
import pathlib
import time
import unittest
from collections import Counter

from src.domain.numeric.lottery3d import presentation, quality
from src.lottery3d import prediction as adapter
from src.lottery3d import scoring as scoring_adapter
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'


def _load(name):
    with gzip.open(FIXTURES / name, 'rt', encoding='utf-8') as fh:
        return json.load(fh)


GOLDEN = _load('golden/lottery3d_prediction.json.gz')
HISTORY = _load('numeric/lottery3d_history.json.gz')['results']
DATA = [(r['issue'], r['date'], tuple(r['digits'])) for r in HISTORY]
NUMBERS = [row[2] for row in DATA]

# 迁移当时 config.py 里生效的值，写死而不是 import（判据 12）
MIN_PERIODS_FOR_ML = 200
POSITION_BASELINE = 0.1
POSITION_NAMES = ('百', '十', '个')

QUALITY_CASES = {
    'full': DATA,
    'recent200': DATA[-200:],
    'recent50': DATA[-50:],
    'one': DATA[-1:],
    'empty': [],
    'dup': DATA[-10:] + DATA[-1:],
    'gap': DATA[-10:-5] + DATA[-3:],
    'cross_year': [('2025365', '2025-12-31', (1, 2, 3)),
                   ('2026001', '2026-01-01', (4, 5, 6))],
    'bad_year': [('2025365', '2025-12-31', (1, 2, 3)),
                 ('2026002', '2026-01-02', (4, 5, 6))],
    'nonnumeric': [('abcdefg', '2026-01-01', (1, 2, 3)),
                   ('2026001', '2026-01-02', (4, 5, 6))],
}


def _ml_cases():
    from src.lottery3d.config import ML_CACHE_MAX_AGE_SECONDS, ML_MODEL_VERSION
    now = time.time()

    def stamp(offset):
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now - offset))

    return {
        'valid': ({'base_period': '2026228', 'model_version': ML_MODEL_VERSION,
                   'created_at': stamp(60)}, '2026228'),
        'no_cache': (None, '2026228'),
        'empty_cache': ({}, '2026228'),
        'wrong_period': ({'base_period': '2026227', 'model_version': ML_MODEL_VERSION,
                          'created_at': stamp(60)}, '2026228'),
        'wrong_version': ({'base_period': '2026228', 'model_version': 'v0',
                           'created_at': stamp(60)}, '2026228'),
        'no_timestamp': ({'base_period': '2026228',
                          'model_version': ML_MODEL_VERSION}, '2026228'),
        'expired': ({'base_period': '2026228', 'model_version': ML_MODEL_VERSION,
                     'created_at': stamp(ML_CACHE_MAX_AGE_SECONDS + 60)}, '2026228'),
        'just_fresh': ({'base_period': '2026228', 'model_version': ML_MODEL_VERSION,
                        'created_at': stamp(ML_CACHE_MAX_AGE_SECONDS - 60)}, '2026228'),
        'bad_timestamp': ({'base_period': '2026228', 'model_version': ML_MODEL_VERSION,
                           'created_at': '不是时间'}, '2026228'),
    }


def golden_entries():
    """按 (键, 值) 逐条产出全部语料，测试与重生成脚本共用。"""
    for name, data in QUALITY_CASES.items():
        yield f'quality:{name}', adapter.assess_data_quality(data)
    for name, (cache, period) in _ml_cases().items():
        yield f'ml_cache:{name}', adapter.is_ml_prediction_cache_valid(cache, period)
    for name, series in (('full', NUMBERS), ('recent200', NUMBERS[-200:]),
                         ('recent30', NUMBERS[-30:]), ('tiny2', NUMBERS[-2:])):
        for wname, weights in (('default', scoring_adapter.default_window_weights()),
                               ('flat', {30: 1.0})):
            lag1 = scoring_adapter.ensemble_lag1_dynamics(series, weights)
            pattern = scoring_adapter.ensemble_patterns(series, weights)
            dynamic = scoring_adapter.derive_dynamic_weights(lag1, pattern['consec_rate'])
            yield (f'transition:{name}:{wname}',
                   adapter._transition_for_api(lag1, dynamic))
            yield (f'transition_names:{name}:{wname}',
                   adapter._transition_for_api(lag1, dynamic, ('A', 'B', 'C')))


class GoldenTests(unittest.TestCase):
    """迁移前后逐条比对。"""

    def test_matches_golden(self):
        seen = set()
        for key, value in golden_entries():
            seen.add(key)
            with self.subTest(case=key):
                self.assertEqual(as_comparable(value), GOLDEN[key])
        self.assertEqual(sorted(set(GOLDEN) - seen), [])


class DeadCacheTests(unittest.TestCase):
    """只写不读的模块级预测缓存已删除，连同它的两个从没被传过的开关。"""

    def test_run_prediction_no_longer_takes_a_cache_flag(self):
        import inspect
        params = inspect.signature(adapter.run_prediction).parameters
        self.assertNotIn('use_prediction_cache', params)

    def test_the_module_no_longer_exports_the_cache_globals(self):
        """迁移前 `__init__.py` 还把它们导出着——那是导入时的值快照。"""
        import src.lottery3d as package
        for name in ('_prediction_cache', '_cache_time', 'clear_cache'):
            with self.subTest(name=name):
                self.assertFalse(hasattr(package, name))

    def test_the_prediction_module_keeps_no_module_level_cache(self):
        for name in ('_prediction_cache', '_cache_time'):
            with self.subTest(name=name):
                self.assertFalse(hasattr(adapter, name))


class TransitionViewTests(unittest.TestCase):

    def _lag1(self, **over):
        base = {'pairs': 100, 'pos_repeat_rate': [0.12, 0.10, 0.08],
                'repeat_dist': {0: 0.7, 1: 0.25, 2: 0.05},
                'digit_reuse_rate': 0.271, 'full_repeat_rate': 0.001,
                'same_set_rate': 0.01, 'ge2_overlap_rate': 0.05}
        base.update(over)
        return base

    def _view(self, **over):
        return presentation.transition_view(
            self._lag1(**over), {'w_pos_repeat': 2.4567, 'pos_mult': [1.2345, 1.0, 0.8]},
            POSITION_NAMES, POSITION_BASELINE)

    def test_vs_random_divides_by_the_baseline(self):
        """光看「同位复刻率 12%」说明不了什么，除以理论基线才知道它偏高。"""
        view = self._view()
        self.assertEqual(view['pos_repeat_rate'][0]['vs_random'], 1.2)
        self.assertEqual(view['pos_repeat_rate'][1]['vs_random'], 1.0)

    def test_positions_keep_their_names_and_order(self):
        view = self._view()
        self.assertEqual([item['name'] for item in view['pos_repeat_rate']],
                         list(POSITION_NAMES))

    def test_repeat_distribution_is_shown_as_a_percentage(self):
        self.assertEqual(self._view()['repeat_dist']['1位同'], 25.0)

    def test_dynamic_weights_are_rounded_to_three_places(self):
        view = self._view()
        self.assertEqual(view['dynamic']['w_pos_repeat'], 2.457)
        self.assertEqual(view['dynamic']['pos_mult'], [1.234, 1.0, 0.8])

    def test_probabilities_keep_four_places(self):
        """千分之一量级的问题上，第四位仍然有意义。位数写字面量。"""
        view = self._view(digit_reuse_rate=0.271234)
        self.assertEqual(view['digit_reuse_rate'], 0.2712)


class ViewHelperTests(unittest.TestCase):

    def test_position_top_ranks_each_position_separately(self):
        scores = [[0, 0, 9, 0, 0, 0, 0, 0, 0, 0],
                  [0, 8, 0, 0, 0, 0, 0, 0, 0, 0],
                  [7, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        result = presentation.position_top(scores, POSITION_NAMES, 1)
        self.assertEqual([item['digits'][0]['digit'] for item in result], [2, 1, 0])

    def test_long_miss_only_lists_digits_past_the_threshold(self):
        """全列出来那张表就没有信息量了——遗漏三五期是常态。"""
        misses = {0: 3, 1: 8, 2: 20, 3: 7}
        result = presentation.long_miss_digits(misses, 8)
        self.assertEqual([item['digit'] for item in result], [2, 1])

    def test_long_miss_is_sorted_by_severity(self):
        result = presentation.long_miss_digits({0: 10, 1: 30, 2: 20}, 8)
        self.assertEqual([item['miss'] for item in result], [30, 20, 10])

    def test_position_miss_top_is_per_position(self):
        misses = [{d: d for d in range(10)}, {d: 9 - d for d in range(10)},
                  {d: 0 for d in range(10)}]
        result = presentation.position_miss_top(misses, POSITION_NAMES, 1)
        self.assertEqual([item['digits'][0]['digit'] for item in result[:2]], [9, 0])

    def test_weighted_digits_keeps_the_top_n(self):
        counter = Counter({1: 9.55, 2: 8.44, 3: 1.0})
        result = presentation.weighted_digits(counter, 2)
        self.assertEqual(result, [{'digit': 1, 'weight': 9.6},
                                  {'digit': 2, 'weight': 8.4}])

    def test_sum_span_view_rounds_the_centres(self):
        view = presentation.sum_span_view({'sum_center': 13.55, 'hot_sums': [13],
                                           'span_center': 5.44, 'hot_spans': [5]})
        self.assertEqual((view['sum_center'], view['span_center']), (13.6, 5.4))

    def test_stability_view_rounds_to_two_places(self):
        view = presentation.stability_view(0.06666, 'low', 0.08333)
        self.assertEqual((view['score'], view['adjusted_exploration_rate']),
                         (0.07, 0.08))


class HistoryQualityTests(unittest.TestCase):

    def _assess(self, periods, dates=None, minimum=200):
        return quality.assess_history(periods, dates or periods, minimum)

    def test_clean_history_allows_ml_fusion(self):
        periods = [f'2026{i:03d}' for i in range(1, 301)]
        self.assertTrue(self._assess(periods)['ml_fusion_allowed'])

    def test_duplicates_block_ml_fusion(self):
        periods = [f'2026{i:03d}' for i in range(1, 301)] + ['2026001']
        result = self._assess(periods)
        self.assertEqual(result['duplicate_periods'], 1)
        self.assertFalse(result['ml_fusion_allowed'])

    def test_gaps_block_ml_fusion(self):
        """ML 把序列当连续的来学，断期对它特别敏感。"""
        periods = [f'2026{i:03d}' for i in range(1, 301) if i != 100]
        result = self._assess(periods)
        self.assertEqual(result['period_gaps'], 1)
        self.assertFalse(result['ml_fusion_allowed'])

    def test_short_history_blocks_ml_fusion(self):
        periods = [f'2026{i:03d}' for i in range(1, 100)]
        result = self._assess(periods)
        self.assertIn('history_too_short_for_ml_fusion', result['warnings'])
        self.assertFalse(result['ml_fusion_allowed'])

    def test_a_clean_year_rollover_is_not_a_gap(self):
        """跨年时序号从 1 重来，那是正常的，不是断期。"""
        self.assertEqual(self._assess(['2025365', '2026001'])['period_gaps'], 0)

    def test_a_year_rollover_not_starting_at_one_is_a_gap(self):
        self.assertEqual(self._assess(['2025365', '2026002'])['period_gaps'], 1)

    def test_unparseable_periods_are_skipped_not_counted_as_gaps(self):
        """格式异常是另一类问题，混进断期数会让两种毛病说不清。"""
        self.assertEqual(self._assess(['abcdefg', '2026001'])['period_gaps'], 0)

    def test_empty_history_reports_no_boundaries(self):
        result = self._assess([], [])
        self.assertIsNone(result['first_period'])
        self.assertIsNone(result['last_period'])
        self.assertEqual(result['periods'], 0)

    def test_problems_are_counted_not_fixed(self):
        """把问题摆出来是这一层的职责；补数据是抓取层的事。"""
        periods = ['2026001', '2026001', '2026005']
        result = self._assess(periods, minimum=1)
        self.assertEqual(result['periods'], 3)
        self.assertEqual(result['duplicate_periods'], 1)


class MlCacheValidityTests(unittest.TestCase):
    """默认不可用。用一份来路不明的缓存出号，比多训练一次贵得多。"""

    NOW = 1_000_000.0
    MAX_AGE = 3600

    def _valid(self, cache, period='p1', version='v1'):
        return quality.is_cache_valid(cache, period, version, self.MAX_AGE,
                                      self.NOW, float)

    def _cache(self, **over):
        base = {'base_period': 'p1', 'model_version': 'v1',
                'created_at': str(self.NOW - 60)}
        base.update(over)
        return base

    def test_a_fresh_matching_cache_is_valid(self):
        self.assertTrue(self._valid(self._cache()))

    def test_no_cache_is_invalid(self):
        self.assertFalse(self._valid(None))
        self.assertFalse(self._valid({}))

    def test_a_different_period_is_invalid(self):
        self.assertFalse(self._valid(self._cache(base_period='p0')))

    def test_a_different_model_version_is_invalid(self):
        self.assertFalse(self._valid(self._cache(model_version='v0')))

    def test_an_expired_cache_is_invalid(self):
        self.assertFalse(self._valid(self._cache(created_at=str(self.NOW - 3601))))

    def test_a_cache_just_inside_the_age_limit_is_valid(self):
        """两侧都要断言：只测过期的话，把上限改小也发现不了。"""
        self.assertTrue(self._valid(self._cache(created_at=str(self.NOW - 3599))))

    def test_a_missing_timestamp_keeps_the_cache(self):
        """期号与版本都对上了，缺个时间戳不足以推翻它——迁移前的行为。"""
        cache = self._cache()
        del cache['created_at']
        self.assertTrue(self._valid(cache))

    def test_an_unparseable_timestamp_invalidates_the_cache(self):
        """解析不了说明这份缓存本身有问题，不再信任它。"""
        self.assertFalse(self._valid(self._cache(created_at='不是时间')))


if __name__ == '__main__':
    unittest.main()
