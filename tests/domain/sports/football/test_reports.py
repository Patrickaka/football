# -*- coding: utf-8 -*-
"""报告层：监控、联赛闸门、就绪度、策略验证、临场情报、渲染。

参照物是黄金文件（`tests/fixtures/golden/football_reports.json.gz`，399 条）。
迁移当时另跑过 **399 条**新旧双跑差分，零差异。

## 这一批断掉的环

迁移前 `production_league_gate` 顶层 `from .professional_monitoring import
wilson_interval`，而 `professional_monitoring` 又在函数体里延迟 import
`production_league_gate.build_production_league_spf_policies`——两个模块互相咬，
只能靠延迟 import 绕开。`wilson_interval` 提到共用的 `stats` 之后依赖变成单向，
那两条延迟 import 都改成了顶层 import。

## 这一批抓到的三处静默降级

延迟 import 搬进领域包后解析不到，`except Exception` 把 ImportError 吞掉，
结果**看着正常、内容是空的**：
- `build_professional_monitoring` 的 `league_spf_validation` 变成
  `{'error': "No module named ...production_league_gate"}`，三个联赛的判定全没了。
- 同一个函数的 `rqspf_independent_validation` 同样退化。
- 另有三处漏 import（`math` / `json` / `os`）在差分里现形。

差分之外没有任何东西会红——这就是为什么每批都要双跑。
"""
import ast
import gzip
import json
import pathlib
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from src.domain.sports.football import accuracy_gate, context, league_gate
from src.domain.sports.football import monitoring, readiness, reporting
from src.domain.sports.football import stats, validation
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'
GOLDEN = json.load(gzip.open(FIXTURES / 'golden/football_reports.json.gz',
                             'rt', encoding='utf-8'))
NOW = datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc)


def golden_entries():
    from scripts.gen_football_reports_golden import entries
    return entries()


class _FrozenDatetime(datetime):
    """一个"现在"停在两年后的 datetime——用来证明没有漏网的时钟读取。"""

    @classmethod
    def now(cls, tz=None):
        return datetime(2028, 3, 1, 4, 30, 0, tzinfo=tz or timezone.utc)


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))

    def test_the_golden_does_not_move_when_the_wall_clock_does(self):
        """**换一个"现在"重跑，黄金必须一字不差。**

        `assess_live_context` 判「这条情报够不够新」要跟当前时间比，
        漏注入的症状是「本地绿、CI 红」——CI 跑 UTC，本地在东八区。
        """
        with mock.patch.object(context, 'datetime', _FrozenDatetime):
            for key, value in golden_entries():
                with self.subTest(key=key):
                    self.assertEqual(GOLDEN[key], as_comparable(value))


class WilsonInterval(unittest.TestCase):

    def test_it_is_not_the_naive_point_estimate(self):
        """8/10 的点估计是 0.8，区间必须比它宽——这就是用 Wilson 的理由。"""
        low, high = stats.wilson_interval(8, 10)
        self.assertLess(low, 0.8)
        self.assertGreater(high, 0.8)

    def test_no_samples_yields_no_interval(self):
        self.assertEqual(stats.wilson_interval(0, 0), (0.0, 0.0))

    def test_the_interval_narrows_as_samples_grow(self):
        narrow = stats.wilson_interval(800, 1000)
        wide = stats.wilson_interval(8, 10)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_it_stays_inside_zero_one(self):
        for hits, n in ((0, 5), (5, 5), (1, 1), (0, 1)):
            with self.subTest(hits=hits, n=n):
                low, high = stats.wilson_interval(hits, n)
                self.assertGreaterEqual(low, 0.0)
                self.assertLessEqual(high, 1.0)

    def test_a_smaller_z_gives_a_tighter_interval(self):
        tight = stats.wilson_interval(8, 10, 1.0)
        loose = stats.wilson_interval(8, 10, 1.96)
        self.assertLess(tight[1] - tight[0], loose[1] - loose[0])

    def test_the_cycle_is_broken(self):
        """`wilson_interval` 住在 `stats`，两个模块都从那里引——环断了。"""
        source = pathlib.Path('src/domain/sports/football/league_gate.py').read_text(
            encoding='utf-8')
        self.assertIn('from .stats import wilson_interval', source)
        self.assertNotIn('from .monitoring import', source)


class LeagueGate(unittest.TestCase):

    def test_a_league_without_enough_history_is_not_supported(self):
        verdict = league_gate.validate_league_spf_policy([], '英超')
        self.assertFalse(verdict['supported'])

    def test_the_row_needs_a_settled_result_and_a_market_probability(self):
        base = {'settled': True, 'actual_result': 'H',
                'professional_snapshot': {'accuracy_gate': {
                    'spf': {'candidate': '胜', 'market_probability': 0.7}}}}
        self.assertIsNotNone(league_gate._gate_row(base))
        for missing in ('settled', 'actual_result'):
            with self.subTest(missing=missing):
                self.assertIsNone(league_gate._gate_row(
                    {k: v for k, v in base.items() if k != missing}))

    def test_a_market_probability_outside_zero_one_is_rejected(self):
        """`0 < p <= 1` 是**两侧**都要挡的。"""
        for bad in (0.0, -0.1, 1.5, 'x', None):
            with self.subTest(bad=bad):
                self.assertIsNone(league_gate._gate_row(
                    {'settled': True, 'actual_result': 'H',
                     'professional_snapshot': {'accuracy_gate': {
                         'spf': {'candidate': '胜', 'market_probability': bad}}}}))

    def test_chinese_candidate_labels_map_to_hda(self):
        self.assertEqual(league_gate._candidate_label('胜'), 'H')
        self.assertEqual(league_gate._candidate_label('平'), 'D')
        self.assertEqual(league_gate._candidate_label('负'), 'A')

    def test_an_already_english_label_passes_through(self):
        self.assertEqual(league_gate._candidate_label('H'), 'H')

    def test_an_empty_label_is_none(self):
        for empty in ('', None, 0):
            with self.subTest(empty=empty):
                self.assertIsNone(league_gate._candidate_label(empty))

    def test_league_names_are_normalised_before_grouping(self):
        self.assertEqual(league_gate._normalise_league('  英超  '),
                         league_gate._normalise_league('英超'))

    def test_a_higher_threshold_selects_no_more_than_a_lower_one(self):
        rows = [{'hit': i % 2 == 0, 'market_probability': 0.3 + i * 0.015,
                 'base_eligible': True, 'time': '', 'match_id': str(i)}
                for i in range(40)]
        loose = league_gate._metrics(rows, 0.3)
        strict = league_gate._metrics(rows, 0.9)
        self.assertGreater(loose['sample_count'], strict['sample_count'])
        self.assertGreater(loose['coverage'], strict['coverage'])

    def test_a_row_blocked_by_its_contemporaneous_guards_is_not_selectable(self):
        """当时被别的闸门挡下的场次，回头统计也不能算进来。"""
        rows = [{'hit': True, 'market_probability': 0.9, 'base_eligible': False,
                 'time': '', 'match_id': 'x'}]
        self.assertEqual(league_gate._metrics(rows, 0.5)['sample_count'], 0)


class Validation(unittest.TestCase):

    def test_probabilities_normalise_to_one(self):
        self.assertAlmostEqual(
            sum(validation.normalize_probabilities({'H': 2, 'D': 1, 'A': 1}).values()), 1.0)

    def test_an_all_zero_input_does_not_divide_by_zero(self):
        validation.normalize_probabilities({'H': 0, 'D': 0, 'A': 0})

    def test_odds_become_probabilities_that_sum_to_one(self):
        probs = validation.probabilities_from_odds({'H': 2.0, 'D': 4.0, 'A': 4.0})
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertGreater(probs['H'], probs['D'])

    def test_zero_odds_do_not_blow_up(self):
        validation.probabilities_from_odds({'H': 0, 'D': 0, 'A': 0})

    def test_the_drawdown_of_a_rising_curve_is_zero(self):
        self.assertEqual(validation._max_drawdown([1, 2, 3, 4]), 0.0)

    def test_the_drawdown_measures_the_deepest_trough(self):
        self.assertAlmostEqual(validation._max_drawdown([1, 3, 1.5, 4]), 1.5)

    def test_an_empty_curve_has_no_drawdown(self):
        self.assertEqual(validation._max_drawdown([]), 0.0)

    def test_blending_at_the_extremes_returns_each_side(self):
        record = {'probabilities': {'H': 1.0, 'D': 0.0, 'A': 0.0},
                  'odds': {'H': 2.0, 'D': 4.0, 'A': 4.0}}
        pure_model = validation.blend_record_with_market(dict(record), 1.0)
        pure_market = validation.blend_record_with_market(dict(record), 0.0)
        self.assertAlmostEqual(pure_model['probabilities']['H'], 1.0)
        self.assertLess(pure_market['probabilities']['H'], 1.0)

    def test_a_stricter_threshold_never_takes_more_bets(self):
        records = [{'probabilities': {'H': 0.3 + i * 0.02, 'D': 0.35, 'A': 0.35},
                    'odds': {'H': 2.0, 'D': 4.0, 'A': 4.0},
                    'actual': 'HDA'[i % 3]} for i in range(30)]
        loose = validation.evaluate_strategy(records, 0.0, 0.0)
        strict = validation.evaluate_strategy(records, 0.9, 0.5)
        self.assertGreaterEqual(loose['bets'], strict['bets'])


class Readiness(unittest.TestCase):

    AUDITED = {'source': '官方', 'ts': NOW.isoformat()}

    def test_live_information_counts_only_with_source_and_timestamp(self):
        """来源和时间戳缺一不可——不能审计的情报不算数。"""
        self.assertTrue(readiness._live_item_verified(dict(self.AUDITED)))
        for missing in ('source', 'ts'):
            with self.subTest(missing=missing):
                self.assertFalse(readiness._live_item_verified(
                    {k: v for k, v in self.AUDITED.items() if k != missing}))

    def test_a_list_counts_only_when_every_item_is_auditable(self):
        """**全部**都要可审计——有一条来路不明就不算。"""
        self.assertTrue(readiness._live_item_verified([dict(self.AUDITED)]))
        self.assertFalse(readiness._live_item_verified(
            [dict(self.AUDITED), {'source': '官方'}]))
        self.assertFalse(readiness._live_item_verified([]))

    def test_presence_treats_empty_containers_as_absent(self):
        for empty in (None, '', [], {}):
            with self.subTest(empty=empty):
                self.assertFalse(readiness._present(empty))
        for filled in ('x', [1], {'a': 1}, 1.0):
            with self.subTest(filled=filled):
                self.assertTrue(readiness._present(filled))

    def test_zero_counts_as_present(self):
        """`0` 是个**有效取值**，不是缺失。"""
        self.assertTrue(readiness._present(0))

    def test_identical_distributions_do_not_diverge(self):
        probs = {'H': 0.5, 'D': 0.3, 'A': 0.2}
        self.assertAlmostEqual(
            readiness._probability_divergence(dict(probs), dict(probs)), 0.0)

    def test_divergence_grows_as_the_two_sides_separate(self):
        model = {'H': 0.9, 'D': 0.05, 'A': 0.05}
        near = readiness._probability_divergence(model, {'H': 0.8, 'D': 0.1, 'A': 0.1})
        far = readiness._probability_divergence(model, {'H': 0.1, 'D': 0.1, 'A': 0.8})
        self.assertLess(near, far)

    def test_a_missing_side_has_no_divergence(self):
        self.assertIsNone(readiness._probability_divergence(None, None))


class AccuracyGate(unittest.TestCase):

    def test_only_the_two_audited_leagues_have_a_static_policy(self):
        """静态策略**只有两个联赛**（判据 10：读真实配置，别按名字猜）。"""
        self.assertEqual(set(accuracy_gate.SPF_LEAGUE_POLICIES), {'SP1', 'D1'})

    def test_every_alias_of_an_audited_league_resolves(self):
        for alias in ('SP1', '西甲', 'La Liga', 'D1', '德甲', 'Bundesliga'):
            with self.subTest(alias=alias):
                self.assertTrue(accuracy_gate.has_static_spf_policy(alias))

    def test_an_unaudited_league_has_none(self):
        for league in ('英超', '德乙', '', None):
            with self.subTest(league=league):
                self.assertFalse(accuracy_gate.has_static_spf_policy(league))

    def test_alias_matching_ignores_case_and_padding(self):
        self.assertTrue(accuracy_gate.has_static_spf_policy('  la liga  '))

    def test_the_top_pick_is_the_argmax_with_its_margin(self):
        pick, probability, margin = accuracy_gate._top_pick(
            {'H': 0.55, 'D': 0.25, 'A': 0.20})
        self.assertEqual(pick, 'H')
        self.assertAlmostEqual(probability, 0.55)
        self.assertAlmostEqual(margin, 0.30)

    def test_an_empty_distribution_picks_nothing(self):
        for empty in ({}, None):
            with self.subTest(empty=empty):
                self.assertIsNone(accuracy_gate._top_pick(empty)[0])

    def test_the_market_agrees_only_when_it_favours_the_same_side(self):
        """收的是**概率**，不是赔率——它对入参取 argmax。"""
        market = {'H': 0.55, 'D': 0.25, 'A': 0.20}
        self.assertTrue(accuracy_gate._market_agrees(market, 'H'))
        self.assertFalse(accuracy_gate._market_agrees(market, 'A'))

    def test_feeding_it_raw_odds_inverts_the_verdict(self):
        """**名字叫 market，喂赔率进去结论恰好相反**：赔率最大的是最不可能的一边。

        `{'H': 1.9, 'D': 3.4, 'A': 4.2}` 明明是主队热门，argmax 却落在 A。
        调用方必须先把赔率转成概率。
        """
        odds = {'H': 1.9, 'D': 3.4, 'A': 4.2}
        self.assertFalse(accuracy_gate._market_agrees(odds, 'H'))
        self.assertTrue(accuracy_gate._market_agrees(odds, 'A'))

    def test_no_market_means_no_agreement(self):
        for market in (None, {}):
            with self.subTest(market=market):
                self.assertFalse(accuracy_gate._market_agrees(market, 'H'))

    def test_reliability_rises_with_both_inputs(self):
        base = accuracy_gate.prediction_reliability(0.5, 0.5)
        self.assertGreater(accuracy_gate.prediction_reliability(0.9, 0.5), base)
        self.assertGreater(accuracy_gate.prediction_reliability(0.5, 1.0), base)


class LiveContext(unittest.TestCase):
    """**时钟由调用方注入**（判据 16）——不注入的话黄金隔天就红。"""

    def _lineup(self, hours_ago):
        return {'lineup': {'confirmed': True, 'verified': True, 'source': '官方',
                           'ts': (NOW - timedelta(hours=hours_ago)).isoformat(),
                           'updated_at': (NOW - timedelta(hours=hours_ago)).isoformat()}}

    def test_fresh_information_scores_higher_than_stale(self):
        fresh = context.assess_live_context(self._lineup(1), NOW)
        stale = context.assess_live_context(self._lineup(48), NOW)
        self.assertGreater(fresh['quality_score'], stale['quality_score'])

    def test_the_age_limit_is_configurable(self):
        ctx = self._lineup(18)
        self.assertGreater(
            context.assess_live_context(ctx, NOW, False, 24.0)['quality_score'],
            context.assess_live_context(ctx, NOW, False, 12.0)['quality_score'])

    def test_requiring_a_confirmed_lineup_can_only_lower_the_verdict(self):
        ctx = {'lineup': {'confirmed': False, 'source': '传闻',
                          'updated_at': NOW.isoformat()}}
        self.assertLessEqual(
            context.assess_live_context(ctx, NOW, True)['quality_score'],
            context.assess_live_context(ctx, NOW, False)['quality_score'])

    def test_an_empty_context_still_allows_official_betting(self):
        """**没有任何情报也放行**：`quality_score` 0.6、`official_bet_allowed` True。

        三项检查全是 `missing`、`freshness` 是 `unknown`，闸门却不拦。
        行为照搬未改，记在这里是因为它不符合直觉。
        """
        verdict = context.assess_live_context({}, NOW)
        self.assertEqual(verdict['quality_score'], 0.6)
        self.assertTrue(verdict['official_bet_allowed'])
        self.assertEqual(verdict['blockers'], [])
        self.assertEqual(verdict['checks']['lineup'], 'missing')

    def test_timestamps_parse_in_both_iso_forms(self):
        self.assertIsNotNone(context._parse_timestamp('2026-08-29T12:00:00Z'))
        self.assertIsNotNone(context._parse_timestamp('2026-08-29T12:00:00+00:00'))

    def test_unparseable_timestamps_are_none(self):
        for bad in ('bad', '', None, 123):
            with self.subTest(bad=bad):
                self.assertIsNone(context._parse_timestamp(bad))


class ContextualFusion(unittest.TestCase):

    CANDIDATES = [((1, 0), 0.25), ((2, 1), 0.30), ((1, 1), 0.20),
                  ((0, 1), 0.15), ((3, 0), 0.10)]

    def test_an_empty_context_leaves_the_candidates_alone(self):
        adjusted, meta = context.apply_contextual_fusion(list(self.CANDIDATES), {})
        self.assertEqual(adjusted, self.CANDIDATES)
        self.assertFalse(meta.get('applied'))

    def test_fusion_keeps_the_probabilities_normalised(self):
        adjusted, _ = context.apply_contextual_fusion(
            list(self.CANDIDATES),
            {'motivation': {'home': 'must_win', 'away': 'nothing'},
             'h2h': {'home_wins': 4, 'draws': 1, 'away_wins': 1}})
        self.assertAlmostEqual(sum(p for _, p in adjusted), 1.0)

    def test_numbers_parse_from_text_and_fall_back_when_they_cannot(self):
        self.assertAlmostEqual(context._number('1.5'), 1.5)
        self.assertAlmostEqual(context._number('x', 9.0), 9.0)
        self.assertAlmostEqual(context._number(None, 9.0), 9.0)


FULL_REPORT = {
    'match': {'league': '英超', 'home': '主队', 'away': '客队',
              'time': '20:00', 'num': '周三001', 'label': '英超', 'available': True},
    'wdl': {'home': 0.55, 'draw': 0.25, 'away': 0.20},
    'p0': {'home': 0.50, 'draw': 0.28, 'away': 0.22},
    'league': reporting.league_specifics({'avg_goals': 2.7, 'home_advantage': 0.3}, '英超'),
    'ts': '2026-08-29T12:00:00', 'module_version': 'v1',
    'confidence_label': '中', 'confidence_score': 0.6,
    'risk_level': 'medium', 'risks': ['主力伤停'],
    'scripts': [], 'tactical': reporting.tactical_context({}),
    'tool_log': [],
    'update': reporting.likelihood_update(
        {'home': 0.55, 'draw': 0.25, 'away': 0.20}, {}, '英超'),
    'trap_warn': reporting.possession_trap_warning({}),
    'injury_conflict': {'available': False, 'items': []},
}


class Reporting(unittest.TestCase):

    ODDS = {'home': 1.9, 'draw': 3.4, 'away': 4.2}

    def test_odds_and_probabilities_are_both_accepted(self):
        """>1 当赔率取倒数，<1 当概率直接归一——两条路都要走到。"""
        from_odds = reporting._to_implied(dict(self.ODDS))
        from_probs = reporting._to_implied({'home': 0.5, 'draw': 0.3, 'away': 0.2})
        self.assertAlmostEqual(sum(from_odds.values()), 1.0)
        self.assertAlmostEqual(from_probs['home'], 0.5)

    def test_chinese_keys_work_too(self):
        self.assertEqual(reporting._to_implied({'胜': 1.9, '平': 3.4, '负': 4.2}),
                         reporting._to_implied(dict(self.ODDS)))

    def test_unusable_odds_yield_nothing(self):
        for bad in ({}, None, {'home': 'x'}, {'home': 0, 'draw': 0, 'away': 0}):
            with self.subTest(bad=bad):
                self.assertIsNone(reporting._to_implied(bad))

    def test_the_prior_sums_to_one(self):
        self.assertAlmostEqual(
            sum(reporting.derive_prior_p0(dict(self.ODDS)).values()), 1.0)

    def test_percentages_render_as_text(self):
        self.assertEqual(reporting.pct(0.5), '50.0%')

    def test_the_match_id_comes_out_of_the_report_filename(self):
        self.assertEqual(reporting._extract_mid_from_report_path(
            '/a/b/reports/football_bayes_12345.html'), '12345')
        self.assertEqual(reporting._extract_mid_from_report_path(
            'beidan_bayes_abc.html'), 'abc')

    def test_a_filename_in_another_shape_yields_nothing(self):
        for path in ('/x/football_12345.html', 'x.html', ''):
            with self.subTest(path=path):
                self.assertIsNone(reporting._extract_mid_from_report_path(path))

    def test_the_likelihood_update_stays_normalised(self):
        """返回 `{'p1': 后验, 'evidence': 证据行}`——不是后验本身。"""
        updated = reporting.likelihood_update(
            {'home': 0.55, 'draw': 0.25, 'away': 0.20},
            {'injuries': [{'team': '主', 'player': '9号', 'role': '前锋',
                           'status': '伤停', 'impact': '大', 'source': '官方'}]},
            '英超')
        self.assertAlmostEqual(sum(updated['p1'].values()), 1.0, places=6)

    def test_an_injury_leaves_an_auditable_evidence_line(self):
        updated = reporting.likelihood_update(
            {'home': 0.55, 'draw': 0.25, 'away': 0.20},
            {'injuries': [{'team': '主', 'player': '9号', 'role': '前锋',
                           'status': '伤停', 'impact': '大', 'source': '官方'}]},
            '英超')
        self.assertTrue(any('9号' in line for line in updated['evidence']))

    def test_no_live_information_leaves_the_prior_alone(self):
        prior = {'home': 0.55, 'draw': 0.25, 'away': 0.20}
        self.assertEqual(reporting.likelihood_update(dict(prior), {}, '英超')['p1'],
                         prior)

    def test_the_rendered_report_is_html(self):
        html = reporting.render_html(FULL_REPORT)
        self.assertIn('<', html)
        self.assertIn('主队', html)


DOMAIN_FILES = {name: f'src/domain/sports/football/{name}.py'
                for name in ('stats', 'monitoring', 'readiness', 'validation',
                             'league_gate', 'reporting', 'accuracy_gate', 'context')}
FORBIDDEN_IMPORTS = {'os.path', 'pathlib', 'requests', 'urllib.request', 'pickle',
                     'src.common.kv_store', 'src.common.repositories',
                     'src.football.config', 'src.football.result_sync'}


class NoSideEffectTests(unittest.TestCase):

    ADAPTER = 'src/football/bayes_report.py'

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

    def test_every_relative_import_resolves_inside_the_domain_package(self):
        """**这一批被这个坑了两次**：延迟 import 搬进领域包后解析不到，
        `except Exception` 把 ImportError 吞掉，报告字段静默变空。
        """
        package = pathlib.Path('src/domain/sports/football')
        siblings = {p.stem for p in package.glob('*.py')}
        for name, path in DOMAIN_FILES.items():
            for node in ast.walk(ast.parse(
                    pathlib.Path(path).read_text(encoding='utf-8'))):
                if isinstance(node, ast.ImportFrom) and node.level:
                    with self.subTest(module=name, imported=node.module):
                        self.assertIn(node.module, siblings)

    def test_no_domain_module_opens_a_file(self):
        for name, path in DOMAIN_FILES.items():
            source = pathlib.Path(path).read_text(encoding='utf-8')
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    with self.subTest(module=name, call=node.func.id):
                        self.assertNotIn(node.func.id, {'open', 'input'})

    def test_the_guard_would_catch_a_real_violation(self):
        self.assertNotEqual(self._imports(self.ADAPTER) & FORBIDDEN_IMPORTS, set())


if __name__ == '__main__':
    unittest.main()
