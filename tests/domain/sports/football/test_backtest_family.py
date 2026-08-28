# -*- coding: utf-8 -*-
"""回测族：样本质量、策略口径、历史校准、动态权重、回测。

参照物是黄金文件（`tests/fixtures/golden/football_backtest.json.gz`，1314 条）。
迁移当时另跑过 **1314 条**新旧双跑差分，零差异。

**记录语料是冻结的**（`tests/fixtures/football_backtest_records.json.gz`）：
从生产历史抽的 68 条按「被读到的字段」裁剪后固定下来，再补 24 条造出来的边角。
不读实时历史——读了黄金就会随每次赛果回填漂动，也就不再是参照物。
裁剪本身也验证过：用裁剪后的语料重跑双跑差分仍是零差异，
说明裁掉的 `market_timeline`（单条 377 KB）这些字段确实没人读。

## 这一批修掉的两处真实回归

1. `_expand_param_grid` 漏了 `import itertools`（差分抓到，旧的出网格新的 NameError）。
2. 时间层权重原本是「延迟 import 失败就退回 `1.0/0.5` 两档」的兜底。
   那条兜底**从来走不到**，而真值是 0.35/0.55/0.75/0.9/1.0——四个层全不一样。
   同一件事实现两遍就会漂（判据 11）；现在直接引领域实现。
"""
import ast
import gzip
import json
import pathlib
import unittest
from datetime import datetime
from unittest import mock

from src.domain.sports.football import backtest, calibration_history, policy
from src.domain.sports.football import quality, settlement, weights
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'
GOLDEN = json.load(gzip.open(FIXTURES / 'golden/football_backtest.json.gz',
                             'rt', encoding='utf-8'))


class _FrozenDatetime(datetime):
    """一个"现在"停在两年后的 datetime——用来证明没有漏网的时钟读取。"""

    @classmethod
    def now(cls, tz=None):
        return datetime(2028, 3, 1, 4, 30, 0)


def golden_entries():
    from scripts.gen_football_backtest_golden import entries
    return entries()


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))

    def test_the_golden_does_not_move_when_the_wall_clock_does(self):
        """**换一个"现在"重跑，黄金必须一字不差。**

        这一族自己不读时钟，但它调得到 `settlement`——顺着调用链漏进来的
        时钟依赖症状是「本地绿、CI 红」（CI 跑 UTC，本地东八区）。
        """
        with mock.patch.object(settlement, 'datetime', _FrozenDatetime):
            for key, value in golden_entries():
                with self.subTest(key=key):
                    self.assertEqual(GOLDEN[key], as_comparable(value))


class SampleQuality(unittest.TestCase):

    SETTLED = {'settled': True, 'sync_status': 'synced', 'actual_score': '2-1',
               'predicted_scores': {'1-0': 0.5, '2-1': 0.5},
               'predicted_1x2': {'H': 0.5, 'D': 0.3, 'A': 0.2},
               'league': '英超', 'result_quality': {'grade': 'high'},
               'asian_line': -0.5, 'total_line': 2.5,
               'odds_snapshot': {'asian': {'handicap': 0.9}}}

    def test_a_complete_settled_record_scores_high(self):
        self.assertEqual(quality.assess_record_quality(self.SETTLED)['grade'], 'high')

    def test_each_missing_piece_costs_score(self):
        """**逐个**抽掉字段——只测一个的话，别的扣分删了也全绿。"""
        full = quality.assess_record_quality(self.SETTLED)['score']
        for field in ('actual_score', 'predicted_scores', 'predicted_1x2'):
            with self.subTest(field=field):
                degraded = dict(self.SETTLED)
                degraded[field] = None
                self.assertLess(quality.assess_record_quality(degraded)['score'], full)

    def test_a_failed_sync_is_downgraded_but_not_rejected(self):
        """`failed`/`ignored` 掉到 medium（权重仍有 0.525），**不是** reject。

        判掉它的是别处：`calibration_history._quality_weight` 对这类记录
        直接给 0——同一个"能不能用"在两层有两套口径。
        """
        for status in ('failed', 'ignored'):
            with self.subTest(status=status):
                verdict = quality.assess_record_quality(
                    dict(self.SETTLED, sync_status=status))
                self.assertEqual(verdict['grade'], 'medium')
                self.assertGreater(verdict['calibration_weight'], 0.0)
        self.assertEqual(
            quality.assess_record_quality(self.SETTLED)['grade'], 'high')

    def test_friendlies_are_recognised_in_either_field(self):
        for field in ('league', 'match_type'):
            with self.subTest(field=field):
                self.assertTrue(quality._is_friendly({field: '国际友谊'}))
        self.assertFalse(quality._is_friendly({'league': '英超'}))

    def test_the_grade_filter_is_a_floor_not_an_exact_match(self):
        records = [dict(self.SETTLED, match_id='keep'),
                   dict(self.SETTLED, match_id='drop', actual_score=None,
                        predicted_scores={}, settled=False)]
        kept, _ = quality.filter_quality_records(records, 'medium')
        self.assertEqual([r['match_id'] for r in kept], ['keep'])

    def test_a_stricter_floor_keeps_strictly_less(self):
        """**反方向**：门槛提高，留下的只能更少。"""
        records = [dict(self.SETTLED, result_quality={'grade': 'medium'})]
        loose, _ = quality.filter_quality_records(records, 'low')
        strict, _ = quality.filter_quality_records(records, 'high')
        self.assertGreaterEqual(len(loose), len(strict))

    def test_the_grade_ranking_is_ordered(self):
        ranks = [quality.GRADE_RANK[g] for g in ('reject', 'low', 'medium', 'high')]
        self.assertEqual(ranks, sorted(ranks))


class PolicyBuckets(unittest.TestCase):

    def test_total_buckets_partition_the_line(self):
        buckets = [policy.get_total_bucket(x) for x in (1.5, 2.25, 2.5, 3.0, 4.5)]
        self.assertEqual(len(set(buckets)), len(set(buckets)))
        self.assertTrue(all(isinstance(b, str) and b for b in buckets))

    def test_a_missing_line_still_gets_a_bucket(self):
        self.assertTrue(policy.get_total_bucket(None))
        self.assertTrue(policy.get_handicap_bucket(None))

    def test_the_bucket_key_combines_both_dimensions(self):
        keys = {policy.policy_bucket_key(h, t)
                for h in (-1.0, 0.0, 1.0) for t in (2.0, 3.0)}
        self.assertEqual(len(keys), 6)

    def test_canonical_params_resolve_aliases(self):
        self.assertEqual(policy._canonical_params({'market_db_weight': 0.5}),
                         policy._canonical_params({'static_market_cap': 0.5}))

    def test_canonical_params_clamp_into_the_declared_range(self):
        """`PARAM_RANGES` 是领域契约（判据 29）——越界要夹回来，两端都要。"""
        for key, (low, high) in policy.PARAM_RANGES.items():
            with self.subTest(key=key):
                self.assertEqual(policy._canonical_params({key: high + 99})[key], high)
                self.assertEqual(policy._canonical_params({key: low - 99})[key], low)

    def test_unknown_and_unparseable_params_are_dropped(self):
        self.assertEqual(policy._canonical_params({'不认识的键': 1}), {})
        self.assertEqual(policy._canonical_params({'draw_bias': '不是数'}), {})

    def test_blending_at_the_extremes_returns_each_side(self):
        left = {'1-0': 1.0}
        right = {'2-1': 1.0}
        self.assertAlmostEqual(policy.blend_score_matrices(left, right, 0.0)['1-0'], 1.0)
        self.assertAlmostEqual(policy.blend_score_matrices(left, right, 1.0)['2-1'], 1.0)

    def test_normalising_makes_the_matrix_sum_to_one(self):
        self.assertAlmostEqual(
            sum(policy.normalize_score_matrix({'1-0': 2.0, '2-1': 2.0}).values()), 1.0)

    CANDIDATES = [((1, 0), 0.2), ((2, 1), 0.3), ((0, 0), 0.1),
                  ((1, 1), 0.25), ((3, 0), 0.15)]

    def test_the_primary_scenario_follows_the_aggregate_direction(self):
        """主选比分跟的是 1X2 的**合计**方向，不是全局众数。

        `1-1` 是这组里的众数，但主胜方向合计更大，所以主选必须是个主胜比分。
        """
        candidates = [((1, 1), 0.30), ((2, 1), 0.25), ((3, 1), 0.25), ((0, 1), 0.20)]
        primary = policy.select_diverse_score_scenarios(candidates, 3)[0][0]
        self.assertGreater(primary[0], primary[1])

    def test_the_primary_is_the_first_match_in_input_order_not_the_likeliest(self):
        """**方向内取的是"输入顺序里第一个"**，不是该方向里概率最大的。

        候选没按概率排过序的话，主选就不是最可能的那个比分——
        这里 `(3, 1)` 概率更高，出来的却是 `(2, 1)`。
        """
        candidates = [((1, 1), 0.10), ((2, 1), 0.20), ((3, 1), 0.50), ((0, 1), 0.20)]
        self.assertEqual(
            policy.select_diverse_score_scenarios(candidates, 3)[0][0], (2, 1))

    def test_the_limit_is_an_upper_bound_with_a_floor_of_two(self):
        """`limit` 挡不住前两个：主选和全局众数是**无条件**放进去的。

        传 1 也会拿回 2 条——展示层若按 limit 排版就会多出一格。
        """
        for limit in (0, 1, 2):
            with self.subTest(limit=limit):
                self.assertEqual(len(policy.select_diverse_score_scenarios(
                    self.CANDIDATES, limit)), 2)
        for limit in (3, 5):
            with self.subTest(limit=limit):
                self.assertEqual(len(policy.select_diverse_score_scenarios(
                    self.CANDIDATES, limit)), limit)

    def test_it_never_returns_more_than_it_was_given(self):
        self.assertEqual(len(policy.select_diverse_score_scenarios(
            self.CANDIDATES, 10)), len(self.CANDIDATES))

    def test_the_extra_slots_prefer_distinct_total_goals(self):
        """名额优先给**不同总进球数**的剧本，避免只呈现一种比赛走势。"""
        picked = policy.select_diverse_score_scenarios(self.CANDIDATES, 4)
        totals = [sum(score) for score, _ in picked]
        self.assertEqual(len(set(totals)), len(totals))

    def test_no_candidates_yields_nothing(self):
        self.assertEqual(policy.select_diverse_score_scenarios([], 3), [])


class DynamicWeights(unittest.TestCase):

    def test_the_three_sources_sum_to_one_without_ml(self):
        for confidence in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0):
            with self.subTest(confidence=confidence):
                self.assertAlmostEqual(sum(weights.confidence_weights(confidence)), 1.0)

    def test_market_weight_rises_with_confidence(self):
        market = [weights.confidence_weights(c)[0] for c in (0.0, 0.3, 0.5, 0.7, 1.0)]
        for earlier, later in zip(market, market[1:]):
            self.assertLessEqual(earlier, later)

    def test_team_weight_falls_with_confidence(self):
        """**反方向**：市场涨，球队就得让。"""
        team = [weights.confidence_weights(c)[1] for c in (0.0, 0.3, 0.5, 0.7, 1.0)]
        for earlier, later in zip(team, team[1:]):
            self.assertGreaterEqual(earlier, later)

    def test_the_plateaus_are_flat_beyond_the_thresholds(self):
        self.assertEqual(weights.confidence_weights(0.0),
                         weights.confidence_weights(weights.LOW_CONFIDENCE))
        self.assertEqual(weights.confidence_weights(1.0),
                         weights.confidence_weights(weights.HIGH_CONFIDENCE))

    def test_the_interpolation_is_continuous_at_the_knots(self):
        for knot in (weights.LOW_CONFIDENCE, 0.5, weights.HIGH_CONFIDENCE):
            with self.subTest(knot=knot):
                below = weights.confidence_weights(knot - 1e-9)
                above = weights.confidence_weights(knot + 1e-9)
                for a, b in zip(below, above):
                    self.assertAlmostEqual(a, b, places=6)

    def test_ml_takes_its_share_out_of_the_other_three(self):
        market, team, elo, ml = weights.get_dynamic_weights(0.5, 0.2)
        self.assertAlmostEqual(market + team + elo + ml, 1.0)
        self.assertAlmostEqual(ml, 0.2)

    def test_a_full_ml_weight_is_left_alone(self):
        """`ml_weight >= 1.0` 原样返回——权重和会大于 1，由下游归一化兜住。"""
        self.assertEqual(weights.get_dynamic_weights(0.5, 1.0)[:3],
                         weights.confidence_weights(0.5))

    def test_fusion_normalises_and_covers_every_score(self):
        fused = weights.fuse_predictions({'1-0': 1.0}, {'2-1': 1.0}, {'0-0': 1.0})
        self.assertAlmostEqual(sum(fused.values()), 1.0)
        self.assertEqual(set(fused), {'1-0', '2-1', '0-0'})

    def test_an_all_zero_fusion_does_not_divide_by_zero(self):
        self.assertEqual(weights.fuse_predictions({'1-0': 0.0}, {}, {}), {'1-0': 0.0})


class HistoryCalibration(unittest.TestCase):

    RECORDS = json.load(gzip.open(FIXTURES / 'football_backtest_records.json.gz',
                                  'rt', encoding='utf-8'))

    def test_too_little_history_does_not_calibrate(self):
        profile = calibration_history.estimate_history_calibration(self.RECORDS[:5])
        self.assertFalse(profile.get('applied'))

    def test_the_sample_floor_is_what_gates_it(self):
        """**反方向**：把语料喂够，它就该动手。"""
        enough = self.RECORDS * 4
        self.assertGreaterEqual(len(enough), calibration_history.MIN_HISTORY_SAMPLES)
        self.assertTrue(
            calibration_history.estimate_history_calibration(enough).get('applied'))

    CANDIDATES = [((1, 0), 0.4), ((2, 1), 0.4), ((3, 3), 0.2)]

    def test_an_inactive_profile_leaves_the_candidates_alone(self):
        """收的是**候选列表** `[(比分元组, 概率)]`，返回 `(候选, 说明)` 两元组。"""
        adjusted, meta = calibration_history.apply_history_calibration(
            list(self.CANDIDATES), {})
        self.assertEqual(adjusted, self.CANDIDATES)
        self.assertFalse(meta['applied'])

    def test_calibration_keeps_the_probabilities_normalised(self):
        profile = calibration_history.estimate_history_calibration(self.RECORDS * 4)
        adjusted, meta = calibration_history.apply_history_calibration(
            list(self.CANDIDATES), profile)
        self.assertTrue(meta['applied'])
        self.assertAlmostEqual(sum(p for _, p in adjusted), 1.0)

    def test_calibration_preserves_the_1x2_marginals(self):
        """校准只动比分**形状**，1X2 的边际必须原样保住。"""
        profile = calibration_history.estimate_history_calibration(self.RECORDS * 4)
        adjusted, _ = calibration_history.apply_history_calibration(
            list(self.CANDIDATES), profile)

        def marginals(items):
            masses = {'H': 0.0, 'D': 0.0, 'A': 0.0}
            for score, probability in items:
                masses[calibration_history._outcome(score)] += probability
            return masses

        before, after = marginals(self.CANDIDATES), marginals(adjusted)
        for outcome in before:
            with self.subTest(outcome=outcome):
                self.assertAlmostEqual(before[outcome], after[outcome], places=6)

    def test_the_goal_beta_stays_within_its_cap(self):
        profile = calibration_history.estimate_history_calibration(self.RECORDS * 4)
        self.assertLessEqual(abs(profile.get('goal_beta', 0.0)),
                             calibration_history.MAX_GOAL_BETA)

    def test_excluded_records_weigh_nothing(self):
        self.assertEqual(calibration_history._quality_weight(
            {'exclude_from_calibration': True, 'settled': True,
             'sync_status': 'synced', 'actual_score': '1-0'}), 0.0)

    def test_legacy_records_keep_a_small_but_nonzero_weight(self):
        """老快照缺赔率元数据，但比分形状够用——全判 0 会让运行时校准永久停摆。"""
        legacy = {'settled': True, 'sync_status': 'synced', 'actual_score': '2-1',
                  'predicted_scores': {'2-1': 1.0}}
        self.assertEqual(calibration_history._quality_weight(legacy), 0.35)

    def test_a_failed_sync_is_never_admitted_even_as_legacy(self):
        legacy = {'settled': True, 'sync_status': 'failed', 'actual_score': '2-1',
                  'predicted_scores': {'2-1': 1.0}}
        self.assertEqual(calibration_history._quality_weight(legacy), 0.0)

    def test_the_outcome_of_a_score_follows_the_goal_difference(self):
        self.assertEqual(calibration_history._outcome((2, 1)), 'H')
        self.assertEqual(calibration_history._outcome((1, 1)), 'D')
        self.assertEqual(calibration_history._outcome((0, 2)), 'A')


class Backtest(unittest.TestCase):

    RECORDS = json.load(gzip.open(FIXTURES / 'football_backtest_records.json.gz',
                                  'rt', encoding='utf-8'))

    def test_the_report_counts_only_what_survives_the_quality_filter(self):
        """报告默认**开着**质量过滤，所以计数不等于"有比分的记录数"。"""
        report = backtest.run_backtest_report(self.RECORDS)
        with_score = [r for r in self.RECORDS if r.get('actual_score')]
        self.assertLessEqual(report['summary']['total_matches'], len(with_score))
        loose = backtest.run_backtest_report(self.RECORDS, quality_filter=False)
        self.assertGreaterEqual(loose['summary']['total_matches'],
                                report['summary']['total_matches'])

    def test_the_hit_rates_are_nested(self):
        summary = backtest.run_backtest_report(self.RECORDS)['summary']
        self.assertLessEqual(summary['top1_hit_rate'], summary['top3_hit_rate'])
        self.assertLessEqual(summary['top3_hit_rate'], summary['top5_hit_rate'])

    def test_the_time_layer_weights_come_from_the_domain(self):
        """兜底那两档（1.0/0.5）从来走不到，真值是五个不同的数（判据 11）。"""
        actual = [backtest.time_layer_weight(l)
                  for l in ('T-24h', 'T-6h', 'T-1h', 'T-15min', 'final')]
        self.assertEqual(actual, [0.35, 0.55, 0.75, 0.9, 1.0])
        self.assertIs(backtest.time_layer_weight, settlement.time_layer_weight)

    def test_goals_are_summed_from_the_score_text(self):
        self.assertEqual(backtest._actual_goals('3-2'), 5)

    def test_an_unparseable_score_counts_as_zero_goals(self):
        for bad in ('', None, 'abc', '3'):
            with self.subTest(bad=bad):
                self.assertEqual(backtest._actual_goals(bad), 0)

    def test_a_draw_is_recognised_only_when_the_sides_are_equal(self):
        self.assertTrue(backtest._is_draw_score('1-1'))
        self.assertFalse(backtest._is_draw_score('2-1'))
        self.assertFalse(backtest._is_draw_score(''))

    def test_the_half_full_sample_needs_real_half_time_data(self):
        usable = {'half_time_data_quality': 'real',
                  'result_quality': {'grade': 'high', 'usable_for_calibration': True}}
        self.assertTrue(backtest._has_real_half_full_sample(usable))
        self.assertFalse(backtest._has_real_half_full_sample(
            dict(usable, half_time_data_quality='estimated')))

    def test_the_param_grid_is_the_cartesian_product(self):
        """`itertools` 曾经漏了 import——差分抓到的。"""
        grid = backtest._expand_param_grid({'a': [1, 2], 'b': [3, 4]})
        self.assertEqual(len(grid), 4)
        self.assertIn({'a': 2, 'b': 3}, grid)

    def test_the_quality_filter_can_be_switched_off(self):
        kept_on, _ = backtest._quality_filter(list(self.RECORDS), True, 'high')
        kept_off, _ = backtest._quality_filter(list(self.RECORDS), False, 'high')
        self.assertEqual(len(kept_off), len(self.RECORDS))
        self.assertLessEqual(len(kept_on), len(kept_off))

    def test_rolling_windows_report_one_entry_each(self):
        rolling = backtest.rolling_backtest_report(self.RECORDS, (10, 30))
        self.assertEqual(len(rolling['windows']), 2)

    def test_tuning_suggestions_stay_inside_the_declared_ranges(self):
        report = backtest.run_backtest_report(self.RECORDS)
        plan = backtest.build_diagnostic_tuning_plan(report)
        for key, value in (plan.get('param_deltas') or {}).items():
            with self.subTest(key=key):
                low, high = policy.PARAM_RANGES[key]
                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)


FORBIDDEN_IMPORTS = {'os', 'pathlib', 'requests', 'urllib.request',
                     'src.common.kv_store', 'src.common.repositories',
                     'src.football.config', 'src.football.result_sync'}

DOMAIN_FILES = {
    'quality': 'src/domain/sports/football/quality.py',
    'policy': 'src/domain/sports/football/policy.py',
    'calibration_history': 'src/domain/sports/football/calibration_history.py',
    'weights': 'src/domain/sports/football/weights.py',
    'backtest': 'src/domain/sports/football/backtest.py',
}


class NoSideEffectTests(unittest.TestCase):

    ADAPTER = 'src/football/result_sync.py'

    def _imports(self, path):
        found = set()
        for node in ast.walk(ast.parse(pathlib.Path(path).read_text(encoding='utf-8'))):
            if isinstance(node, ast.Import):
                found.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module)
                found.update(f'{node.module}.{a.name}' for a in node.names)
        return found

    def test_no_domain_module_imports_anything_stateful(self):
        for name, path in DOMAIN_FILES.items():
            with self.subTest(module=name):
                self.assertEqual(self._imports(path) & FORBIDDEN_IMPORTS, set())

    def test_no_domain_module_has_deferred_relative_imports(self):
        """延迟的相对 import 搬进领域包后解析不到，会**静默落到兜底**。"""
        siblings = set(DOMAIN_FILES) | {'settlement', 'markets', 'parsing',
                                        'scoring_model'}
        for name, path in DOMAIN_FILES.items():
            tree = ast.parse(pathlib.Path(path).read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level:
                    with self.subTest(module=name, imported=node.module):
                        self.assertIn(node.module, siblings)

    def test_no_domain_module_reads_the_wall_clock(self):
        for name, path in DOMAIN_FILES.items():
            with self.subTest(module=name):
                self.assertNotIn('datetime.now()',
                                 pathlib.Path(path).read_text(encoding='utf-8'))

    def test_the_guard_would_catch_a_real_violation(self):
        self.assertNotEqual(self._imports(self.ADAPTER) & FORBIDDEN_IMPORTS, set())


if __name__ == '__main__':
    unittest.main()
