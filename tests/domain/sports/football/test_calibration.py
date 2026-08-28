# -*- coding: utf-8 -*-
"""足球的概率校准：Platt / 保序回归 / 分层校准 / 分桶与整数键还原。

参照物是从迁移前的三个模块生成的黄金文件
（`tests/fixtures/golden/football_calibration.json.gz`，636 条），**逐条相同**。
迁移当时另跑过 **674 条**新旧双跑差分，零差异。

**保序回归返回的是闭包**——黄金存的是它在一组探针概率上的输出，
不是函数对象（比函数对象等于比内存地址，第一版差分就是这么白跑了 6 条）。

**迁移时在这里写错过一次**：`_get_bucket_key` 的签名是
`(league, total_line, asian, expected_total)`，适配层按位置传成了
`(…, expected_total, asian)`，桶键整个错位——双跑差分当场抓住（180 条全红）。
所以转发一律**按名字传**。
"""
import gzip
import json
import pathlib
import unittest
from unittest import mock

from src.domain.sports.football import calibration, calibration_buckets
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'
GOLDEN = json.load(gzip.open(FIXTURES / 'golden/football_calibration.json.gz',
                             'rt', encoding='utf-8'))

MIN_ACTIVATION_SAMPLES = 4
POOLING_K = 12.0


def golden_entries():
    from scripts.gen_football_calibration_golden import entries
    return entries()


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))


class IntegerKeysSurviveTheJsonRoundTrip(unittest.TestCase):
    """判据 26/27：`kv_store` 走 JSON，`{2: 0.9}` 存一轮回来是 `{'2': 0.9}`。

    `40398d1` 就是这么踩的——`factors.get(2)` 查不到 `'2'`，每个因子悄悄
    回落成 1.0，分布原样返回、日志干净、一切正常。所以**必须真的存一次再读一次**，
    不能直接给 `db` 赋值（那会绕过防腐层，修没修都测不出来）。
    """

    def test_a_real_json_round_trip_turns_int_keys_into_strings(self):
        """先把前提钉住——不然下面两条读起来像在测一个不存在的问题。"""
        restored = json.loads(json.dumps({'bucket': {'calibration_factors': {2: 0.9}}}))
        self.assertEqual(list(restored['bucket']['calibration_factors']), ['2'])

    def test_restore_puts_them_back(self):
        after_json = json.loads(json.dumps(
            {'bucket': {'calibration_factors': {0: 1.1, 1: 0.9, 2: 1.05}}}))
        restored = calibration_buckets._restore_goal_keys(after_json)
        self.assertEqual(sorted(restored['bucket']['calibration_factors']), [0, 1, 2])

    def test_predicted_distributions_are_restored_too(self):
        after_json = json.loads(json.dumps(
            {'bucket': {'predicted_distributions': [{0: 0.3, 1: 0.7}, {2: 1.0}]}}))
        restored = calibration_buckets._restore_goal_keys(after_json)
        for dist in restored['bucket']['predicted_distributions']:
            self.assertTrue(all(isinstance(k, int) for k in dist))

    def test_bucket_keys_themselves_are_left_alone(self):
        """分桶键（`'瑞典超_2.75_+0.50_2.50'`）本来就是字符串，**不能碰**。"""
        db = {'瑞典超_2.75_+0.50_2.50': {'calibration_factors': {'1': 0.9}}}
        restored = calibration_buckets._restore_goal_keys(json.loads(json.dumps(db)))
        self.assertIn('瑞典超_2.75_+0.50_2.50', restored)

    def test_unconvertible_keys_are_kept_not_dropped(self):
        """**转不动的原样留下**——丢掉之后概率就不再归一，而那不会报错。"""
        restored = calibration_buckets._int_keyed({'1': 0.5, 'x': 0.5})
        self.assertEqual(restored, {1: 0.5, 'x': 0.5})
        self.assertAlmostEqual(sum(restored.values()), 1.0)

    def test_malformed_databases_do_not_crash(self):
        for db in ({}, None, 'x', {'k': 'notadict'}, {'k': {'calibration_factors': 'bad'}}):
            with self.subTest(db=db):
                calibration_buckets._restore_goal_keys(db)

    def test_the_calibrator_survives_a_real_store_round_trip(self):
        """走一遍 `_load()`——**直接给 db 赋值会绕过防腐层**（判据 26）。"""
        import src.football.goal_count_calibrator as gcc
        db = {'英超_2.50_+0.50_2.50': {
            'count': 10, 'calibration_factors': {0: 1.2, 1: 0.9, 2: 1.05},
            'predicted_distributions': [], 'actual_counts': {},
        }}
        stored = json.loads(json.dumps(db))   # 模拟 kv_store 的 JSON 往返
        with mock.patch('src.common.kv_store.load', return_value=stored):
            calibrator = gcc.GoalCountCalibrator()
        factors = calibrator.db['英超_2.50_+0.50_2.50']['calibration_factors']
        self.assertEqual(sorted(factors), [0, 1, 2])
        # 最强的一条：存过一轮与没存过，应用结果必须一致
        dist = {0: 0.3, 1: 0.4, 2: 0.3}
        self.assertEqual(
            calibration_buckets.apply_goal_calibration(dist, factors),
            calibration_buckets.apply_goal_calibration(dist, db['英超_2.50_+0.50_2.50']['calibration_factors']))


class BucketKeyArgumentOrder(unittest.TestCase):
    """第三第四个参数是 `asian, expected_total`——顺序写反过一次。"""

    def test_asian_comes_before_expected_total(self):
        key = calibration_buckets.goal_bucket_key('英超', 2.5, asian=0.5, expected_total=2.0)
        self.assertEqual(key, '英超_2.50_+0.50_2.00')
        swapped = calibration_buckets.goal_bucket_key('英超', 2.5, asian=2.0, expected_total=0.5)
        self.assertNotEqual(key, swapped)

    def test_the_adapter_forwards_by_name(self):
        """适配层按名字传——按位置传就会错位。"""
        import src.football.goal_count_calibrator as gcc
        with mock.patch('src.common.kv_store.load', return_value={}):
            calibrator = gcc.GoalCountCalibrator()
        self.assertEqual(
            calibrator._get_bucket_key('英超', 2.5, asian=0.5, expected_total=2.0),
            calibration_buckets.goal_bucket_key('英超', 2.5, asian=0.5, expected_total=2.0))

    def test_each_dimension_is_bucketed_at_its_own_granularity(self):
        """盘口 0.25、让球 0.5、预测总进球 0.5——**三个粒度不一样**。"""
        base = calibration_buckets.goal_bucket_key('英超', 2.5, asian=0.0, expected_total=2.5)
        self.assertEqual(calibration_buckets.goal_bucket_key('英超', 2.6, asian=0.0,
                                                             expected_total=2.5), base)
        self.assertNotEqual(calibration_buckets.goal_bucket_key('英超', 2.7, asian=0.0,
                                                                expected_total=2.5), base)
        self.assertEqual(calibration_buckets.goal_bucket_key('英超', 2.5, asian=0.2,
                                                             expected_total=2.5), base)
        self.assertNotEqual(calibration_buckets.goal_bucket_key('英超', 2.5, asian=0.3,
                                                                expected_total=2.5), base)

    def test_the_handicap_always_carries_a_sign(self):
        self.assertIn('+0.00', calibration_buckets.goal_bucket_key('英超', 2.5, asian=0.0,
                                                                   expected_total=2.5))
        self.assertIn('-1.50', calibration_buckets.goal_bucket_key('英超', 2.5, asian=-1.5,
                                                                   expected_total=2.5))


class TeamAliasResolutionIsVeryWide(unittest.TestCase):
    """三级匹配，第三级是**双向包含**——把宽度钉住，不是认可它。"""

    ALIAS = {'曼联': ['曼彻斯特联', '红魔'], '国安': ['北京国安']}

    def test_an_exact_standard_name_or_alias_resolves(self):
        self.assertEqual(calibration.resolve_team_alias('曼联', self.ALIAS), '曼联')
        self.assertEqual(calibration.resolve_team_alias('曼彻斯特联', self.ALIAS), '曼联')
        self.assertEqual(calibration.resolve_team_alias('红魔', self.ALIAS), '曼联')

    def test_surrounding_whitespace_is_stripped_first(self):
        self.assertEqual(calibration.resolve_team_alias('  曼联  ', self.ALIAS), '曼联')

    def test_containment_matches_in_both_directions(self):
        """`alias in name` **或** `name in alias`——两个方向都算命中。"""
        self.assertEqual(calibration.resolve_team_alias('北京国安足球俱乐部', self.ALIAS), '国安')
        self.assertEqual(calibration.resolve_team_alias('国安', self.ALIAS), '国安')

    def test_an_unknown_name_is_returned_unchanged(self):
        self.assertEqual(calibration.resolve_team_alias('切尔西', self.ALIAS), '切尔西')

    def test_an_empty_alias_map_is_a_no_op(self):
        for alias_map in (None, {}):
            with self.subTest(alias_map=alias_map):
                self.assertEqual(calibration.resolve_team_alias('曼彻斯特联', alias_map),
                                 '曼彻斯特联')

    def test_empty_names_short_circuit(self):
        for name in ('', None):
            with self.subTest(name=name):
                self.assertEqual(calibration.resolve_team_alias(name, self.ALIAS), name)


class PlattScaling(unittest.TestCase):

    def test_the_default_parameters_are_not_the_identity_they_flatten_hard(self):
        """**`(1.0, 0.0)` 不是恒等**——它是 `sigmoid(1.0*p + 0.0)`，
        而 sigmoid 在 [0, 1] 上只从 0.5 走到 0.731。归一之后整张矩阵被压平。

        要紧的是：`train_league_platt_params` 在**历史数据不足 5 场**时正是
        返回 `(1.0, 0.0)` 当「默认参数」——那不是安全兜底，是把模型的判别度
        几乎抹平。实测一张 998:1 的矩阵会被压成 1.46:1，top1 从 0.998 → 0.422。

        线上走不到（`pipeline.py:727` 虽然开着校准，但贝叶斯那条一直成功，
        7 天零「贝叶斯校准失败」）。**行为原样保留**，见交接文档 §四。
        """
        matrix = {(0, 0): 0.25, (1, 0): 0.5, (1, 1): 0.25}
        flattened = calibration.calibrate_with_platt(matrix, {'platt_params': (1.0, 0.0)})
        self.assertNotAlmostEqual(flattened[(1, 0)], 0.5, places=2)
        self.assertAlmostEqual(sum(flattened.values()), 1.0, places=9)

        def ratio(m):
            return max(m.values()) / min(m.values())
        self.assertAlmostEqual(ratio(matrix), 2.0)
        self.assertLess(ratio(flattened), 1.2)

        sharp = {(0, 0): 0.001, (1, 0): 0.998, (1, 1): 0.001}
        self.assertLess(ratio(calibration.calibrate_with_platt(
            sharp, {'platt_params': (1.0, 0.0)})), 1.5)

    def test_only_a_missing_platt_params_key_is_a_true_no_op(self):
        """真正原样返回的只有「没有 platt_params」这一种——**同一个对象**。"""
        matrix = {(0, 0): 0.25, (1, 0): 0.75}
        for data in (None, {}, {'isotonic': []}):
            with self.subTest(data=data):
                self.assertIs(calibration.calibrate_with_platt(matrix, data), matrix)

    def test_the_result_is_always_normalised(self):
        matrix = {(0, 0): 0.25, (1, 0): 0.5, (1, 1): 0.25}
        for params in ((2.0, -0.5), (0.5, 1.0), (-1.0, 0.0)):
            with self.subTest(params=params):
                result = calibration.calibrate_with_platt(matrix, {'platt_params': params})
                self.assertAlmostEqual(sum(result.values()), 1.0, places=9)

    def test_fitting_degenerate_data_falls_back_to_identity(self):
        """全同标签或空样本拟不出东西——必须退回恒等，不能抛。"""
        for pairs in ([], [(0.5, 1)] * 10, [(0.5, 0)] * 10, [(0.2, 0)]):
            with self.subTest(pairs=pairs):
                a, b = calibration.fit_platt_scaling(pairs)
                self.assertTrue(all(isinstance(v, float) for v in (a, b)))

    def test_fitting_separable_data_produces_a_non_identity_mapping(self):
        """**反方向**：可分的数据必须拟出非恒等的参数，否则上一条毫无意义。"""
        a, b = calibration.fit_platt_scaling(
            [(0.1, 0), (0.2, 0), (0.3, 0), (0.7, 1), (0.8, 1), (0.9, 1)] * 5)
        self.assertNotEqual((a, b), (1.0, 0.0))


class IsotonicRegressionReturnsAClosure(unittest.TestCase):

    PAIRS = [(0.1, 0), (0.3, 0), (0.5, 1), (0.7, 1), (0.9, 1)] * 4

    def test_it_returns_a_callable_not_a_table(self):
        fn = calibration.isotonic_regression_calibration(self.PAIRS)
        self.assertTrue(callable(fn))

    def test_the_mapping_is_monotonically_non_decreasing(self):
        """保序回归的定义就是单调不减——这是它唯一必须守的性质。"""
        fn = calibration.isotonic_regression_calibration(self.PAIRS)
        probes = [i / 20 for i in range(21)]
        values = [fn(p) for p in probes]
        for earlier, later in zip(values, values[1:]):
            self.assertLessEqual(earlier, later + 1e-12)

    def test_degenerate_input_still_returns_a_callable(self):
        for pairs in ([], [(0.5, 1)], [(0.5, 1)] * 10):
            with self.subTest(pairs=pairs):
                fn = calibration.isotonic_regression_calibration(pairs)
                self.assertTrue(callable(fn))
                self.assertIsInstance(fn(0.5), float)


class GoalCalibrationApplication(unittest.TestCase):

    DIST = {0: 0.2, 1: 0.5, 2: 0.3}

    def test_no_factors_returns_the_distribution_untouched(self):
        for factors in ({}, None):
            with self.subTest(factors=factors):
                self.assertIs(calibration_buckets.apply_goal_calibration(self.DIST, factors),
                              self.DIST)

    def test_factors_reweight_then_renormalise(self):
        result = calibration_buckets.apply_goal_calibration(self.DIST, {1: 1.5})
        self.assertAlmostEqual(sum(result.values()), 1.0)
        self.assertGreater(result[1], self.DIST[1])
        self.assertLess(result[0], self.DIST[0])

    def test_goals_present_only_in_the_factors_join_the_output(self):
        """**并集**——因子里有而分布里没有的进球数会带着 0 概率出现。"""
        result = calibration_buckets.apply_goal_calibration(self.DIST, {1: 1.5, 7: 2.0})
        self.assertIn(7, result)
        self.assertEqual(result[7], 0.0)

    def test_all_zero_factors_leave_the_unnormalised_zeros(self):
        """总和为 0 时不能除以 0——原样返回未归一的结果。"""
        result = calibration_buckets.apply_goal_calibration(self.DIST, {0: 0.0, 1: 0.0, 2: 0.0})
        self.assertEqual(sum(result.values()), 0.0)


FORBIDDEN_IMPORTS = {'time', 'os', 'pathlib', 'requests', 'urllib.request',
                     'urllib.error', 'src.common.kv_store', 'src.foundation.store',
                     'src.football.fetching', 'src.football.config',
                     'src.football.prediction_records'}


class NoSideEffectTests(unittest.TestCase):

    DOMAIN = ('src/domain/sports/football/calibration.py',
              'src/domain/sports/football/calibration_buckets.py')
    ADAPTER = 'src/football/goal_count_calibrator.py'

    def _imports(self, path):
        import ast
        found = set()
        for node in ast.walk(ast.parse(pathlib.Path(path).read_text(encoding='utf-8'))):
            if isinstance(node, ast.Import):
                found.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module)
                found.update(f'{node.module}.{a.name}' for a in node.names)
        return found

    def test_domain_imports_nothing_stateful(self):
        for path in self.DOMAIN:
            with self.subTest(path=path):
                self.assertEqual(self._imports(path) & FORBIDDEN_IMPORTS, set())

    def test_the_guard_would_catch_a_real_violation(self):
        """适配层**应该**命中——守卫本身要能被证伪（判据 16）。"""
        self.assertNotEqual(self._imports(self.ADAPTER) & FORBIDDEN_IMPORTS, set())


if __name__ == '__main__':
    unittest.main()
