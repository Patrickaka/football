"""福彩 3D 的窗口权重与预测记录。

这一层的主题是**把副作用挡在领域层之外**：权重怎么算、记录怎么结算是纯计算，
缓存、kv 读写、时间戳全留在 `src/lottery3d/` 的适配层。分开的理由很实在——
只要结算逻辑自己去 `kv_store.load()`，测一行判断都得先造一套存储。

参照物是从迁移前的实现生成的黄金文件
（`tests/fixtures/golden/lottery3d_records.json.gz`，85 条）。

**黄金值里有一处是有意与迁移前不同的**：`recommendation_stability` 在当前
推荐为空时**除零崩溃**（`len(current_set)` 为 0）。空推荐意味着「没有可比的
东西」，与「历史为空」是同一种情况，现在走同一条兜底路径返回 0.0。
"""
import gzip
import json
import pathlib
import unittest

from src.domain.numeric.lottery3d import records, window_weights
from src.domain.numeric.lottery3d.recommendations import max_digit_overlap
from src.lottery3d import records as records_adapter
from src.lottery3d import scoring as adapter
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'


def _load(name):
    with gzip.open(FIXTURES / name, 'rt', encoding='utf-8') as fh:
        return json.load(fh)


GOLDEN = _load('golden/lottery3d_records.json.gz')
HISTORY = _load('numeric/lottery3d_history.json.gz')['results']
NUMBERS = [tuple(r['digits']) for r in HISTORY]
PERIODS = [r['issue'] for r in HISTORY]

# 迁移当时 config.py 里生效的值，写死而不是 import（判据 12）
RECENT_WINDOWS = (30, 45, 60, 90)
WINDOW_PRIOR = 2.0
ZU6_RECENT_WINDOW = 5
EXPLORATION_RATE = 0.0


def _record(top3, top30, version='v1'):
    return {'version': version, 'period': '2026228', 'last_draw': '399',
            'zhixuan_top3': list(top3), 'zhixuan': list(top30),
            'danma': [5, 8], 'kill': [6, 4], 'actual': None, 'settled': False,
            'hit_top3': False, 'hit_top30': False, 'ge2_digit': False,
            'created_at': '2026-08-27 09:23:22'}


def _settled(top3_hit, top30_hit, ge2, version='v1'):
    return {'version': version, 'settled': True, 'hit_top3': top3_hit,
            'hit_top30': top30_hit, 'ge2_digit': ge2}


SETTLE_CASES = [
    (('582', '852', '528'), ['582', '852', '528', '385'], (5, 8, 2)),
    (('582', '852', '528'), ['582', '852', '528', '385'], (3, 8, 5)),
    (('582', '852', '528'), ['582', '852', '528', '385'], (1, 2, 3)),
    (('582', '852', '528'), ['582', '852', '528', '385'], (1, 1, 1)),
    ((), [], (5, 8, 2)),
    (('000',), ['000'], (0, 0, 0)),
]
STAT_CASES = {
    'empty': [],
    'all_unsettled': [{'settled': False}] * 3,
    'mixed': [_settled(True, True, True), _settled(False, True, True),
              _settled(False, False, False), {'settled': False}],
    'two_versions': [_settled(True, True, True, 'v1'), _settled(False, True, True, 'v2'),
                     _settled(False, False, True, 'v2')],
    'all_hit': [_settled(True, True, True)] * 5,
}
HISTORY_SHAPES = {
    'empty': [],
    'new_format': [{'period': '1', 'recommendations': ['123', '456']},
                   {'period': '2', 'recommendations': ['123', '789']}],
    'old_format': [['123', '456'], ['123', '789']],
    'with_blank': [{'period': '1', 'recommendations': []},
                   {'period': '2', 'recommendations': ['123']}],
    'long': [{'period': str(i), 'recommendations': ['123', str(100 + i)]}
             for i in range(12)],
}
ZU6_HISTORY = {
    'empty': [],
    'one': [{'period': '1', 'digits': [0, 1, 2, 3]}],
    'three': [{'period': str(i), 'digits': [i, i + 1, i + 2, i + 3]} for i in range(3)],
    'raw_list': [[0, 1, 2, 3], [4, 5, 6, 7]],
    'out_of_range': [{'period': '1', 'digits': [0, 9, 15, -1]}],
    'long': [{'period': str(i), 'digits': [0, 1, 2, 3]} for i in range(10)],
}
SCORE = [10.0 - i for i in range(10)]


def golden_entries():
    """按 (键, 值) 逐条产出全部语料，测试与重生成脚本共用。"""
    yield 'default_weights', adapter.default_window_weights()
    for name, series in (('full', NUMBERS), ('recent200', NUMBERS[-200:]),
                         ('recent100', NUMBERS[-100:]), ('short', NUMBERS[-50:])):
        for trials in (10, 30):
            yield (f'compute_weights:{name}:{trials}',
                   adapter.compute_window_weights(series, trials=trials,
                                                  enable_cache=False))
    for index, (top3, top30, actual) in enumerate(SETTLE_CASES):
        yield (f'settle:{index}',
               records_adapter.settle_prediction(_record(top3, top30), actual))
    original = records_adapter.load_online_predictions
    try:
        for name, rows in STAT_CASES.items():
            records_adapter.load_online_predictions = lambda r=rows: list(r)
            yield f'stats:{name}', records_adapter.calculate_online_stats()
    finally:
        records_adapter.load_online_predictions = original
    for name, history in HISTORY_SHAPES.items():
        for current in (['123', '456'], ['999'], []):
            yield (f'stability:{name}:{"".join(current) or "none"}',
                   records_adapter.recommendation_stability(current, history))
    for value in (0.0, 0.29, 0.3, 0.5, 0.8, 0.81, 1.0):
        yield f'stability_level:{value}', records_adapter.get_stability_level(value)
        yield f'exploration_rate:{value}', records_adapter.adjust_exploration_rate(value)
    for name, history in ZU6_HISTORY.items():
        for base in (1.0, 3.0, 5.0):
            for decay in (0.5, 0.9):
                yield (f'zu6_penalty:{name}:{base}:{decay}',
                       records_adapter.recent_zu6_digit_penalty(SCORE, history,
                                                                base, decay))


class GoldenTests(unittest.TestCase):
    """迁移前后逐条比对。"""

    def test_matches_golden(self):
        seen = set()
        for key, value in golden_entries():
            seen.add(key)
            with self.subTest(case=key):
                self.assertEqual(as_comparable(value), GOLDEN[key])
        self.assertEqual(sorted(set(GOLDEN) - seen), [])


# 领域层不许依赖的东西。**按 import 判定，不按文本**——按文本的话，
# docstring 里写一句「把 time.time() 混进来会怎样」都会让守卫报警，
# 而这种误报最后只会被加白名单绕过去（判据 8）。
FORBIDDEN_IMPORTS = {'time', 'os', 'pathlib', 'src.common.kv_store',
                     'src.common.repositories', 'src.foundation.store'}


class NoSideEffectTests(unittest.TestCase):
    """领域层不碰存储、不读时钟——这是这一批的全部目的。"""

    def _imports(self, path):
        import ast
        tree = ast.parse(pathlib.Path(path).read_text())
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module)
                found.update(f'{node.module}.{a.name}' for a in node.names)
        return found

    def test_domain_records_imports_nothing_stateful(self):
        found = self._imports('src/domain/numeric/lottery3d/records.py')
        self.assertEqual(found & FORBIDDEN_IMPORTS, set())

    def test_domain_window_weights_imports_nothing_stateful(self):
        found = self._imports('src/domain/numeric/lottery3d/window_weights.py')
        self.assertEqual(found & FORBIDDEN_IMPORTS, set())

    def test_the_guard_would_catch_a_real_violation(self):
        """守卫本身要能被证伪：拿适配层来试，它**应该**命中。"""
        found = self._imports('src/lottery3d/records.py')
        self.assertNotEqual(found & FORBIDDEN_IMPORTS, set())

    def test_settle_needs_no_storage(self):
        """喂一条记录就能测，不必先造一套 kv。"""
        record = _record(('123',), ['123', '456'])
        records.settle(record, (1, 2, 3), max_digit_overlap)
        self.assertTrue(record['hit_top3'])


class StabilityTests(unittest.TestCase):

    def test_empty_current_returns_zero_instead_of_crashing(self):
        """迁移前这里直接除以 `len(current)` 而崩掉。"""
        self.assertEqual(records.stability([], [['123']]), 0.0)

    def test_empty_history_returns_zero(self):
        self.assertEqual(records.stability(['123'], []), 0.0)

    def test_identical_recommendation_is_fully_stable(self):
        self.assertEqual(records.stability(['1', '2'], [['1', '2']]), 1.0)

    def test_disjoint_recommendation_is_fully_unstable(self):
        self.assertEqual(records.stability(['1', '2'], [['3', '4']]), 0.0)

    def test_overlap_is_measured_against_the_current_pick(self):
        """分母是当前推荐的注数——问的是「这次有多少是老面孔」。"""
        self.assertEqual(records.stability(['1', '2'], [['1', '9', '8', '7']]), 0.5)

    def test_blank_history_entries_are_skipped_not_counted_as_zero(self):
        """把空历史算成 0 分重叠会把稳定度平白拉低。"""
        with_blank = records.stability(['1'], [[], ['1']])
        self.assertEqual(with_blank, 1.0)

    def test_both_history_formats_are_understood(self):
        for shape in ([{'period': '1', 'recommendations': ['1']}], [['1']]):
            with self.subTest(shape=shape):
                self.assertEqual(records.stability(['1'], shape), 1.0)

    def test_only_the_last_seven_entries_count(self):
        history = [['9']] * 20 + [['1']]
        self.assertEqual(records.stability(['1'], history), 1 / 7)

    def test_levels_split_at_both_ends(self):
        """两头都不好：太稳是同一批号，太随机是看不出主张。界限写字面量。"""
        self.assertEqual(records.stability_level(0.81), 'high')
        self.assertEqual(records.stability_level(0.8), 'normal')
        self.assertEqual(records.stability_level(0.3), 'normal')
        self.assertEqual(records.stability_level(0.29), 'low')

    def test_exploration_moves_opposite_to_stability(self):
        """太稳就多探索打散，太随机就少探索收敛——方向反了会自我强化。"""
        self.assertEqual(records.exploration_rate(0.9, 0.15), 0.25)
        self.assertEqual(records.exploration_rate(0.1, 0.15), 0.08)
        self.assertEqual(records.exploration_rate(0.5, 0.15), 0.15)


class SettleTests(unittest.TestCase):

    def _settle(self, top3, top30, actual):
        record = _record(top3, top30)
        return records.settle(record, actual, max_digit_overlap)

    def test_top3_hit_implies_top30_hit(self):
        result = self._settle(('123',), ['123', '456'], (1, 2, 3))
        self.assertTrue(result['hit_top3'])
        self.assertTrue(result['hit_top30'])

    def test_top30_hit_without_top3(self):
        result = self._settle(('999',), ['123', '999'], (1, 2, 3))
        self.assertFalse(result['hit_top3'])
        self.assertTrue(result['hit_top30'])

    def test_ge2_needs_two_shared_digits(self):
        two = self._settle((), ['189'], (1, 8, 5))
        one = self._settle((), ['189'], (1, 5, 5))
        self.assertTrue(two['ge2_digit'])
        self.assertFalse(one['ge2_digit'])

    def test_actual_is_stored_as_a_string(self):
        self.assertEqual(self._settle((), [], (0, 5, 9))['actual'], '059')

    def test_settle_marks_the_record(self):
        self.assertTrue(self._settle((), [], (1, 2, 3))['settled'])


class SettleAllTests(unittest.TestCase):
    """一条记录预测的是**它自己期号的下一期**。差一位会把全部记录对错开奖号。"""

    def _rows(self):
        return [{'period': PERIODS[0], 'zhixuan_top3': [], 'zhixuan': [],
                 'settled': False},
                {'period': PERIODS[1], 'zhixuan_top3': [], 'zhixuan': [],
                 'settled': False}]

    def test_record_is_settled_against_the_following_draw(self):
        rows = self._rows()
        records.settle_all(rows, PERIODS, NUMBERS, max_digit_overlap)
        self.assertEqual(rows[0]['actual'], ''.join(map(str, NUMBERS[1])))
        self.assertEqual(rows[0]['draw_period'], PERIODS[1])

    def test_already_settled_records_are_left_alone(self):
        rows = self._rows()
        rows[0]['settled'] = True
        rows[0]['actual'] = 'kept'
        count, _ = records.settle_all(rows, PERIODS, NUMBERS, max_digit_overlap)
        self.assertEqual(rows[0]['actual'], 'kept')
        self.assertEqual(count, 1)

    def test_the_latest_period_has_no_draw_yet(self):
        rows = [{'period': PERIODS[-1], 'zhixuan_top3': [], 'zhixuan': [],
                 'settled': False}]
        count, changed = records.settle_all(rows, PERIODS, NUMBERS, max_digit_overlap)
        self.assertEqual((count, changed), (0, False))
        self.assertFalse(rows[0]['settled'])

    def test_an_unknown_period_is_skipped(self):
        rows = [{'period': '9999999', 'zhixuan_top3': [], 'zhixuan': [],
                 'settled': False}]
        self.assertEqual(records.settle_all(rows, PERIODS, NUMBERS, max_digit_overlap),
                         (0, False))


class StatsTests(unittest.TestCase):

    def test_unsettled_records_stay_out_of_the_denominator(self):
        """算进去等于用「还没开奖」冲淡命中率。"""
        rows = [_settled(True, True, True), {'settled': False}]
        stats = records.online_stats(rows)
        self.assertEqual(stats['settled_count'], 1)
        self.assertEqual(stats['hit_top3_rate'], 1.0)

    def test_unsettled_count_is_still_reported(self):
        """否则看的人分不清是没中还是没结算。"""
        stats = records.online_stats([{'settled': False}] * 3)
        self.assertEqual((stats['total_records'], stats['unsettled_count']), (3, 3))

    def test_no_settled_records_yields_zero_rates(self):
        stats = records.online_stats([])
        self.assertEqual(stats['hit_top3_rate'], 0.0)
        self.assertEqual(stats['by_version'], {})

    def test_versions_are_counted_separately(self):
        """换了版本的记录不该混在一起比——那是两个模型。"""
        rows = [_settled(True, True, True, 'v1'),
                _settled(False, False, False, 'v2')]
        by_version = records.online_stats(rows)['by_version']
        self.assertEqual(by_version['v1']['hit_top3_rate'], 1.0)
        self.assertEqual(by_version['v2']['hit_top3_rate'], 0.0)


class Zu6RotationTests(unittest.TestCase):

    def test_recent_digits_are_penalised_most(self):
        history = [{'digits': [0]}, {'digits': [1]}]
        adjusted = records.recent_zu6_penalty(SCORE, history, 5, 1.0, 0.5)
        # 最后一期（1）罚满，前一期（0）罚一半
        self.assertAlmostEqual(SCORE[1] - adjusted[1], 1.0, places=9)
        self.assertAlmostEqual(SCORE[0] - adjusted[0], 0.5, places=9)

    def test_repeated_digits_accumulate(self):
        history = [{'digits': [0]}, {'digits': [0]}]
        adjusted = records.recent_zu6_penalty(SCORE, history, 5, 1.0, 0.5)
        self.assertAlmostEqual(SCORE[0] - adjusted[0], 1.5, places=9)

    def test_penalty_is_capped(self):
        """再高就不是轮换而是禁用了。上限写字面量。"""
        adjusted = records.recent_zu6_penalty(SCORE, [{'digits': [0]}], 5, 99.0, 1.0)
        self.assertAlmostEqual(SCORE[0] - adjusted[0], 3.0, places=9)

    def test_out_of_range_digits_are_ignored(self):
        adjusted = records.recent_zu6_penalty(SCORE, [{'digits': [15, -1]}], 5, 1.0, 1.0)
        self.assertEqual(adjusted, SCORE)

    def test_empty_history_leaves_the_score_alone(self):
        self.assertEqual(records.recent_zu6_penalty(SCORE, [], 5, 1.0, 1.0), SCORE)

    def test_only_the_window_is_considered(self):
        history = [{'digits': [5]}] + [{'digits': [0]}] * 5
        adjusted = records.recent_zu6_penalty(SCORE, history, 5, 1.0, 1.0)
        self.assertEqual(adjusted[5], SCORE[5])

    def test_both_history_formats_are_understood(self):
        for shape in ([{'digits': [0]}], [[0]]):
            with self.subTest(shape=shape):
                adjusted = records.recent_zu6_penalty(SCORE, shape, 5, 1.0, 1.0)
                self.assertAlmostEqual(SCORE[0] - adjusted[0], 1.0, places=9)


class UpsertTests(unittest.TestCase):
    """推荐历史以「期」为单位，不是以「页面被访问了几次」为单位。"""

    def test_same_period_is_updated_not_appended(self):
        history = [{'period': '1', 'recommendations': ['a']}]
        result = records.upsert_by_period(history, '1', {'recommendations': ['b']}, 5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['recommendations'], ['b'])

    def test_a_new_period_is_appended(self):
        history = [{'period': '1', 'recommendations': ['a']}]
        result = records.upsert_by_period(history, '2', {'recommendations': ['b']}, 5)
        self.assertEqual([e['period'] for e in result], ['1', '2'])

    def test_only_the_window_is_kept(self):
        history = [{'period': str(i), 'recommendations': []} for i in range(10)]
        result = records.upsert_by_period(history, '10', {'recommendations': []}, 5)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[-1]['period'], '10')

    def test_the_input_history_is_not_mutated(self):
        history = [{'period': '1', 'recommendations': ['a']}]
        records.upsert_by_period(history, '2', {'recommendations': ['b']}, 5)
        self.assertEqual(len(history), 1)

    def test_stamps_are_applied_when_given(self):
        result = records.upsert_by_period([], '1', {'recommendations': []}, 5,
                                          {'created_at': 'T'})
        self.assertEqual(result[0]['created_at'], 'T')


class WindowWeightTests(unittest.TestCase):

    def test_default_weights_are_uniform(self):
        """没有证据时就该均分，而不是偏向某一个窗口。"""
        weights = window_weights.default_weights(RECENT_WINDOWS)
        self.assertEqual(set(weights.values()), {0.25})
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=9)

    def test_prior_keeps_a_zero_scoring_window_alive(self):
        """一次零命中的回测不该把某个窗口压到 0——那可能只是运气差。"""
        raw = {30: 0.0, 45: 10.0}
        weights = window_weights.normalise(raw, (30, 45), WINDOW_PRIOR)
        self.assertGreater(weights[30], 0)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=9)

    def test_a_better_window_gets_more_weight(self):
        weights = window_weights.normalise({30: 1.0, 45: 9.0}, (30, 45), WINDOW_PRIOR)
        self.assertGreater(weights[45], weights[30])

    def test_equal_scores_give_equal_weights(self):
        weights = window_weights.normalise({30: 5.0, 45: 5.0}, (30, 45), WINDOW_PRIOR)
        self.assertEqual(weights[30], weights[45])

    def test_full_hit_outscores_a_partial_one(self):
        """部分分不是安慰奖：只按全中打分的话，几十期里各窗口都是 0。"""
        self.assertEqual(window_weights._period_score('123', ['123']), 1.0)
        self.assertEqual(window_weights._period_score('123', ['124']), 0.25)
        self.assertEqual(window_weights._period_score('123', ['456']), 0.0)

    def test_scoring_only_ever_sees_earlier_draws(self):
        seen = []

        def predict(train, window):
            seen.append(len(train))
            return []

        window_weights.score_windows(NUMBERS[-60:], (30,), 10, predict, str)
        self.assertEqual(seen, sorted(seen))
        self.assertLess(max(seen), 60)

    def test_a_window_longer_than_the_training_set_is_skipped(self):
        called = []
        window_weights.score_windows(NUMBERS[-40:], (30, 90), 5,
                                     lambda t, w: called.append(w) or [], str)
        self.assertEqual(set(called), {30})

    def test_trial_count_never_drops_below_the_minimum(self):
        self.assertEqual(window_weights.trial_count(NUMBERS, RECENT_WINDOWS, 1), 10)

    def test_short_history_has_not_enough_data(self):
        self.assertFalse(window_weights.has_enough_history(NUMBERS[:50], RECENT_WINDOWS))
        self.assertTrue(window_weights.has_enough_history(NUMBERS, RECENT_WINDOWS))


class WeightsCacheTests(unittest.TestCase):
    """缓存留在适配层。迁移前它是三个模块级全局量，而 `__init__.py` 还把它们
    `from ... import` 了出去——那是导入时的值快照，永远是 None。"""

    def setUp(self):
        self.cache = adapter._WeightsCache()

    def test_a_miss_returns_none(self):
        self.assertIsNone(self.cache.get('fp'))

    def test_a_hit_returns_the_stored_value(self):
        self.cache.put('fp', ('weights', 'scores'))
        self.assertEqual(self.cache.get('fp'), ('weights', 'scores'))

    def test_a_different_fingerprint_misses(self):
        """数据变了就该重算，哪怕还没到期。"""
        self.cache.put('fp', 'value')
        self.assertIsNone(self.cache.get('other'))

    def test_an_expired_entry_misses(self):
        self.cache.put('fp', 'value')
        self.cache._stamped_at -= self.cache.TTL_SECONDS
        self.assertIsNone(self.cache.get('fp'))

    def test_clear_drops_everything(self):
        self.cache.put('fp', 'value')
        self.cache.clear()
        self.assertIsNone(self.cache.get('fp'))

    def test_the_module_no_longer_exports_the_old_globals(self):
        import src.lottery3d as package
        for name in ('_window_weights_cache', '_window_weights_cache_time',
                     '_window_weights_cache_numbers_hash'):
            with self.subTest(name=name):
                self.assertFalse(hasattr(package, name))


if __name__ == '__main__':
    unittest.main()
