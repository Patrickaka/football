# -*- coding: utf-8 -*-
"""赛果判定、命中统计、结算质量与 ML 融合资格。

参照物是黄金文件（`tests/fixtures/golden/football_settlement.json.gz`，300 条），
**逐条相同**。迁移当时另跑过 **340 条**新旧双跑差分，零差异。

**时间解析的「当前年」已注入**（判据 16）：`_parse_match_datetime` 对不带年的
时间串要补当前年，跨年时还要把 12 月/1 月的边界修正回来。不注入的话黄金
**跨年就红**。

**`calculate_logloss` 对非 `H`/`D`/`A` 的赛果故意返回 NaN**——而 NaN 不等于
自身，所以黄金里存成字符串 `'nan'`。这不是边角情况，是正常路径上的返回值：
第一版差分因此报了 75 条假差异（两边都是 nan）。

## 计划要求的核查：football 的回填链是接上的

北单那条「预测 → 赛果回填 → 校准」从投产起就没接上（500 条 `settled` 全是
False，交接文档 §四）。**football 不是同一个病**——线上 239 条历史里
234 条 `settled=True` 且有实际比分。所以这里的判定逻辑是活的。
"""
import ast
import gzip
import json
import math
import pathlib
import unittest
from datetime import datetime, timedelta
from unittest import mock

from src.domain.sports.football import settlement
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'
GOLDEN = json.load(gzip.open(FIXTURES / 'golden/football_settlement.json.gz',
                             'rt', encoding='utf-8'))
NOW = datetime(2026, 8, 28, 12, 0, 0)
PROBS = {'H': 0.5, 'D': 0.3, 'A': 0.2}


class _FrozenDatetime(datetime):
    """一个"现在"停在两年后的 datetime——用来证明没有漏网的时钟读取。"""

    @classmethod
    def now(cls, tz=None):
        return datetime(2028, 3, 1, 4, 30, 0)


def golden_entries():
    from scripts.gen_football_settlement_golden import entries
    return entries()


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))

    def test_the_golden_does_not_move_when_the_wall_clock_does(self):
        """**换一个"现在"重跑，黄金必须一字不差。**

        漏注入一处时钟的症状是「本地绿、CI 红」：CI 跑在 UTC、本地在东八区，
        `_assess_result_quality` 判「比赛到期没」两边就能得出不同答案。
        黄金写死一个 `NOW` 挡不住这个——挡得住的是这条：
        把模块看到的真实时钟换掉，结果不变才说明没有漏网的 `datetime.now()`。
        """
        with mock.patch.object(settlement, 'datetime', _FrozenDatetime):
            for key, value in golden_entries():
                with self.subTest(key=key):
                    self.assertEqual(GOLDEN[key], as_comparable(value))


class ScoringMetrics(unittest.TestCase):

    def test_logloss_rewards_a_confident_correct_call(self):
        confident = settlement.calculate_logloss({'H': 0.9, 'D': 0.05, 'A': 0.05}, 'H')
        hedged = settlement.calculate_logloss({'H': 0.34, 'D': 0.33, 'A': 0.33}, 'H')
        self.assertLess(confident, hedged)

    def test_logloss_punishes_a_confident_wrong_call(self):
        """**反方向**：自信押错要比模棱两可更糟。"""
        wrong = settlement.calculate_logloss({'H': 0.9, 'D': 0.05, 'A': 0.05}, 'A')
        hedged = settlement.calculate_logloss({'H': 0.34, 'D': 0.33, 'A': 0.33}, 'A')
        self.assertGreater(wrong, hedged)

    def test_only_hda_are_valid_results(self):
        """`home`/`draw`/`away` **不是**合法赛果——契约是 `H`/`D`/`A`。

        第一版差分语料用了 `home`，于是 42 条全落到 NaN 分支
        （判据 23：看着"通过"其实什么也没测）。
        """
        for bad in ('home', 'draw', 'away', '主胜', '', None):
            with self.subTest(bad=bad):
                self.assertTrue(math.isnan(settlement.calculate_logloss(PROBS, bad)))
        for good in ('H', 'D', 'A'):
            with self.subTest(good=good):
                self.assertFalse(math.isnan(settlement.calculate_logloss(PROBS, good)))

    def test_zero_probability_is_clamped_instead_of_diverging(self):
        """概率为 0 时 `log(0)` 会发散——必须夹紧。"""
        loss = settlement.calculate_logloss({'H': 0.0, 'D': 0.0, 'A': 1.0}, 'H')
        self.assertTrue(math.isfinite(loss))

    def test_brier_is_zero_for_a_perfect_call(self):
        self.assertAlmostEqual(
            settlement.calculate_brier_score({'H': 1.0, 'D': 0.0, 'A': 0.0}, 'H'),
            0.0, places=9)

    def test_hit_follows_the_argmax(self):
        self.assertTrue(settlement.calculate_hit(PROBS, 'H'))
        self.assertFalse(settlement.calculate_hit(PROBS, 'A'))


class ProbabilityNormalisation(unittest.TestCase):

    def test_it_makes_them_sum_to_one(self):
        normalised = settlement.normalize_1x2_probs({'H': 2.0, 'D': 1.0, 'A': 1.0})
        self.assertAlmostEqual(sum(normalised.values()), 1.0)
        self.assertAlmostEqual(normalised['H'], 0.5)

    def test_an_all_zero_input_does_not_divide_by_zero(self):
        settlement.normalize_1x2_probs({'H': 0, 'D': 0, 'A': 0})

    def test_it_is_the_single_shared_implementation(self):
        """**三份逐字相同的实现已合成一份**（F-1 用 AST 确认）。"""
        import src.football.prediction_records as records
        import src.football.result_sync as sync
        self.assertIs(records.normalize_1x2_probs, settlement.normalize_1x2_probs)
        self.assertIs(sync.normalize_1x2_probs, settlement.normalize_1x2_probs)


class MatchTimeNeedsAnInjectedClock(unittest.TestCase):

    def test_a_yearless_time_takes_the_injected_year(self):
        parsed = settlement._parse_match_datetime('08-28 20:00', now=datetime(2020, 5, 1))
        self.assertEqual(parsed.year, 2020)

    def test_a_december_january_boundary_rolls_forward(self):
        """12 月看到 1 月的比赛 → 那是**明年**的（跨年修正）。"""
        parsed = settlement._parse_match_datetime('01-05 20:00', now=datetime(2026, 12, 20))
        self.assertEqual(parsed.year, 2027)

    def test_a_january_december_boundary_rolls_back(self):
        """**反方向**：1 月看到 12 月的比赛 → 那是**去年**的。"""
        parsed = settlement._parse_match_datetime('12-28 20:00', now=datetime(2026, 1, 5))
        self.assertEqual(parsed.year, 2025)

    def test_a_full_timestamp_ignores_the_clock(self):
        parsed = settlement._parse_match_datetime('2019-03-15 20:00', now=NOW)
        self.assertEqual(parsed.year, 2019)

    def test_unparseable_input_returns_none(self):
        for bad in ('', 'bad', None):
            with self.subTest(bad=bad):
                self.assertIsNone(settlement._parse_match_datetime(bad, now=NOW))


class SettleDueGate(unittest.TestCase):

    KICKOFF = '2026-08-28 08:00:00'

    def test_a_match_is_due_only_after_the_grace_period(self):
        self.assertTrue(settlement._is_match_settle_due(self.KICKOFF, 180, NOW))
        self.assertFalse(settlement._is_match_settle_due(
            self.KICKOFF, 180, datetime(2026, 8, 28, 10, 0)))

    def test_the_grace_period_is_configurable(self):
        """同一时刻，窗口越长越不到期——**两个方向都测**。"""
        self.assertFalse(settlement._is_match_settle_due(self.KICKOFF, 300, NOW))
        self.assertTrue(settlement._is_match_settle_due(self.KICKOFF, 60, NOW))

    def test_an_unparseable_kickoff_is_never_due(self):
        self.assertFalse(settlement._is_match_settle_due('', 180, NOW))


class MlFusionEligibility(unittest.TestCase):

    PASSING = {'overall': {'sample_count': 150,
                           'base_1x2_logloss': 1.00, 'base_1x2_brier': 0.200,
                           'ml_1x2_logloss': 0.95, 'ml_1x2_brier': 0.190,
                           'fused_5pct_logloss': 0.98, 'fused_5pct_brier': 0.195}}

    def test_too_few_samples_is_ineligible(self):
        self.assertFalse(
            settlement.check_ml_fusion_eligibility({'overall': {}}, 0)['eligible'])

    def test_a_fully_passing_model_is_eligible(self):
        """**反方向**：只测不合格那边，把闸门整个删掉也全绿。"""
        self.assertTrue(
            settlement.check_ml_fusion_eligibility(self.PASSING, 200)['eligible'])

    def test_each_required_condition_alone_can_disqualify(self):
        """逐个否掉必需条件——少判一条就漏一条。"""
        vetoes = {
            'test_set_samples': dict(overall=self.PASSING['overall']),
            'fused_5pct_logloss_better': {'overall': dict(
                self.PASSING['overall'], fused_5pct_logloss=1.10)},
            'fused_5pct_brier_not_worse': {'overall': dict(
                self.PASSING['overall'], fused_5pct_brier=0.300)},
        }
        for name, stats in vetoes.items():
            with self.subTest(condition=name):
                samples = 0 if name == 'test_set_samples' else 200
                self.assertFalse(
                    settlement.check_ml_fusion_eligibility(stats, samples)['eligible'])

    def test_raw_ml_metrics_stay_diagnostic(self):
        """裸 ML 指标变差**不该**否掉资格——把关的是 5% 融合后的指标。"""
        stats = {'overall': dict(self.PASSING['overall'],
                                 ml_1x2_logloss=1.50, ml_1x2_brier=0.400)}
        verdict = settlement.check_ml_fusion_eligibility(stats, 200)
        self.assertFalse(verdict['conditions']['ml_logloss_better']['passed'])
        self.assertTrue(verdict['eligible'])

    def test_shadow_samples_below_one_hundred_disqualify(self):
        stats = {'overall': dict(self.PASSING['overall'], sample_count=99)}
        self.assertFalse(
            settlement.check_ml_fusion_eligibility(stats, 200)['eligible'])
        stats['overall']['sample_count'] = 100
        self.assertTrue(
            settlement.check_ml_fusion_eligibility(stats, 200)['eligible'])

    def test_the_weight_is_zero_when_ineligible(self):
        self.assertEqual(settlement.get_ml_fusion_weight(False, 1000, 0.0), 0.0)

    def test_the_weight_grows_with_the_shadow_sample_count(self):
        weights = [settlement.get_ml_fusion_weight(True, n, 0.0)
                   for n in (0, 50, 200, 1000)]
        for earlier, later in zip(weights, weights[1:]):
            self.assertLessEqual(earlier, later)


class CalibrationSampleWeightNeedsAnInjectedAssessor(unittest.TestCase):
    """**迁移时踩过**：`from .sample_quality import` 是延迟 import，
    搬进领域包后解析不到会**静默落到兜底**（旧 0.0 / 新 0.7），
    双跑差分抓到了。现在改成注入。
    """

    def test_an_excluded_record_weighs_nothing(self):
        self.assertEqual(
            settlement._calibration_sample_weight({'exclude_from_calibration': True}), 0.0)

    def test_without_an_assessor_it_falls_back_by_source(self):
        for source, expected in (('live_fid', 1.0), ('live_team', 0.85),
                                 ('shuju', 0.60), ('别的来源', 0.70)):
            with self.subTest(source=source):
                self.assertEqual(
                    settlement._calibration_sample_weight(
                        {'result_quality': {'source': source}}), expected)

    def test_a_rejected_grade_weighs_nothing_whatever_the_source(self):
        for grade in ('reject', 'low'):
            with self.subTest(grade=grade):
                self.assertEqual(
                    settlement._calibration_sample_weight(
                        {'result_quality': {'grade': grade, 'source': 'live_fid'}}), 0.0)

    def test_an_injected_assessor_wins(self):
        """**反方向**：注入的评估器必须真的被用上。"""
        weight = settlement._calibration_sample_weight(
            {'result_quality': {'source': 'shuju'}},
            assess_record_quality=lambda record: {'calibration_weight': 0.42})
        self.assertAlmostEqual(weight, 0.42)

    def test_the_injected_weight_is_clamped_into_zero_one(self):
        for raw, expected in ((-5.0, 0.0), (5.0, 1.0)):
            with self.subTest(raw=raw):
                self.assertEqual(
                    settlement._calibration_sample_weight(
                        {}, assess_record_quality=lambda r: {'calibration_weight': raw}),
                    expected)


FORBIDDEN_IMPORTS = {'os', 'pathlib', 'requests', 'urllib.request',
                     'src.common.kv_store', 'src.common.repositories',
                     'src.football.config', 'src.football.sample_quality'}


class NoSideEffectTests(unittest.TestCase):

    DOMAIN = 'src/domain/sports/football/settlement.py'
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

    def test_domain_imports_nothing_stateful(self):
        self.assertEqual(self._imports(self.DOMAIN) & FORBIDDEN_IMPORTS, set())

    def test_the_domain_has_no_deferred_relative_imports(self):
        """**这一批就是被这个坑到的**——延迟的 `from .sample_quality import`
        搬进领域包后解析不到，静默落到兜底。
        """
        source = pathlib.Path(self.DOMAIN).read_text(encoding='utf-8')
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.level:
                with self.subTest(module=node.module):
                    self.assertIn(node.module, {'markets', 'parsing', 'scoring_model'},
                                  f'领域层不该有指向适配层的相对 import: {node.module}')

    def test_the_domain_never_reads_the_wall_clock_unguarded(self):
        source = pathlib.Path(self.DOMAIN).read_text(encoding='utf-8')
        for line in source.splitlines():
            if 'datetime.now()' in line:
                self.assertIn('or datetime.now()', line,
                              f'不可注入的时钟: {line.strip()}')

    def test_the_guard_would_catch_a_real_violation(self):
        self.assertNotEqual(self._imports(self.ADAPTER) & FORBIDDEN_IMPORTS, set())


if __name__ == '__main__':
    unittest.main()
