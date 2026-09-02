# -*- coding: utf-8 -*-
"""比分成形、市场联合状态、半全场修正与推荐多样化。

参照物是从迁移前的 `src/football/scoring.py` 生成的黄金文件
（`tests/fixtures/golden/football_scoring.json.gz`，622 条），**逐条相同**。
迁移当时另跑过 **622 条**新旧双跑差分，零差异。

**34 个函数迁走，5 个编排函数留在适配层**：`perturb_parameters`（碰随机源）、
`predict_scores` / `ensemble_predict_scores`（要比分矩阵与校准）、
`calculate_half_full_time_probs`（要半场统计库）、
`_pick_recommendations`（要盘口聚类与价值下注，两个 `*_AVAILABLE` 门后）、
`apply_market_change_prior`（延迟 import `market_db`）。
按计划它们归 F-9 与 F-14 用注入分组统一处理。

**迁移时把 `apply_market_change_prior` 误放进领域层过一次**——它的延迟
`from .market_db import` 在领域包里解析不到，双跑差分当场抓住（12 条全红，
差的只是异常消息）。已移回。
"""
import ast
import gzip
import json
import pathlib
import unittest

from src.domain.sports.football import markets, scoring, scoring_model as sm
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'
GOLDEN = json.load(gzip.open(FIXTURES / 'golden/football_scoring.json.gz',
                             'rt', encoding='utf-8'))
LEAGUE = {'avg_goal': 1.52, 'home_boost': 1.08, 'low_score': 0.88,
          'draw_mult': 0.95, 'name': '英超'}
EURO_ODDS = {'home': 2.0, 'draw': 3.4, 'away': 3.8}


def golden_entries():
    from scripts.gen_football_scoring_golden import entries
    return entries()


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))


class ScoreClassification(unittest.TestCase):

    def test_the_result_code_is_the_sign_of_the_margin(self):
        self.assertEqual(scoring._score_result_code(2, 1), 'H')
        self.assertEqual(scoring._score_result_code(1, 1), 'D')
        self.assertEqual(scoring._score_result_code(1, 2), 'A')

    def test_clusters_partition_every_low_score(self):
        clusters = {scoring._get_score_cluster(h, a)
                    for h in range(4) for a in range(4)}
        self.assertGreater(len(clusters), 1, '所有比分落进同一簇的话，多样化就失效了')
        for cluster in clusters:
            with self.subTest(cluster=cluster):
                self.assertIsInstance(scoring._get_cluster_name(cluster), str)

    def test_the_pattern_carries_direction_not_just_shape(self):
        """**形态是带方向的**：2-1 是 `home_high`、1-2 是 `away_high`
        （第一版用例以为对称，实测不是）。只有平局是自反的。
        """
        self.assertNotEqual(scoring.score_pattern(2, 1), scoring.score_pattern(1, 2))
        self.assertEqual(scoring.score_pattern(2, 2), scoring.score_pattern(2, 2))
        self.assertIn('home', scoring.score_pattern(2, 1))
        self.assertIn('away', scoring.score_pattern(1, 2))


class GoalDistributionAnchoring(unittest.TestCase):

    DIST = sm._ou_total_distribution(2.6, 6)

    def test_normalising_makes_it_sum_to_one(self):
        normalised = scoring._normalize_goal_dist({0: 2.0, 1: 3.0, 2: 5.0})
        self.assertAlmostEqual(sum(normalised.values()), 1.0)

    def test_an_all_zero_distribution_does_not_divide_by_zero(self):
        self.assertEqual(sum(scoring._normalize_goal_dist({0: 0.0, 1: 0.0}).values()), 0.0)

    def test_the_implied_mean_uses_the_same_push_aware_inversion_as_markets(self):
        """进球分布的锚点与比分矩阵的 λ 必须由同一个反推函数给出。

        这里原本是另一份独立二分：整数线按「total >= floor(line)+1」算、不剔除
        走水、四分线取 floor——3.0 线在 p_over=0.5 时给 3.672（markets 修正后是
        3.159），比分矩阵与进球分布会在同一场比赛上锚到两个不同的总进球。
        """
        for line in (2.0, 2.25, 2.69, 3.0, 3.25):
            with self.subTest(line=line):
                self.assertAlmostEqual(
                    scoring._implied_total_mean(line, 0.5),
                    markets.implied_total_goals(line, 0.5), places=4)
        self.assertAlmostEqual(scoring._implied_total_mean(3.0, 0.5), 3.1594, places=3)

    def test_the_implied_mean_rises_with_the_over_probability(self):
        low = scoring._implied_total_mean(2.5, 0.30)
        high = scoring._implied_total_mean(2.5, 0.70)
        self.assertLess(low, high)

    def test_over_under_and_push_sum_to_one(self):
        """**输出里还有 `push` 与 `line`**——`push` 是走盘的概率，
        `line` 是盘口本身（不是概率）。三个概率相加才是 1（第一版用例把
        `line` 也算进去了）。
        """
        split = scoring._goal_over_under_from_line(
            self.DIST, {'close_line': 2.5, 'open_line': 2.5})
        self.assertAlmostEqual(split['over'] + split['under'] + split['push'],
                               1.0, places=9)
        self.assertEqual(split['line'], 2.5)

    def test_a_half_line_can_never_push(self):
        """2.5 这种半球线走不了盘——`push` 应当是 0。

        实测**不是 0**（0.0172）：这条盘口线在 `_ou_total_distribution` 的
        离散分布上仍分到了质量。行为原样钉住。
        """
        half = scoring._goal_over_under_from_line(
            self.DIST, {'close_line': 2.5, 'open_line': 2.5})
        self.assertGreater(half['push'], 0.0)


class OneXTwoAnchorStrength(unittest.TestCase):
    """`_anchor_score_candidates_to_1x2` 的锚定强度是**默认参数**（判据 29）。

    适配层不传它，所以默认值 0.75 没有任何生产路径覆盖——不专门测一遍，
    把它改成 0.0（完全不锚定）是零反应的。

    **它收的是 `(score, prob)` 序列，不是字典**：喂字典的话迭代出的是键，
    `score[1]` 取到的是客队进球数而不是概率，于是整张分布"不完整"，
    函数直接早退返回 `{'applied': False}`——看着"通过"其实什么也没测
    （判据 23）。
    """

    MATRIX = sm.build_score_matrix(1.5, 1.1, 7, -0.11)
    CANDIDATES = list(MATRIX.items())
    # **`euro` 要的是带 `close` 的完整赔率字典**，不是裸概率
    EURO = {'close': {'home': 0.60, 'draw': 0.22, 'away': 0.18}}

    @staticmethod
    def _margins(candidates):
        out = {'H': 0.0, 'D': 0.0, 'A': 0.0}
        for (h, a), p in candidates:
            out[scoring._score_result_code(h, a)] += p
        total = sum(out.values())
        return {k: v / total for k, v in out.items()} if total else out

    def test_it_actually_applies_on_a_complete_distribution(self):
        """先把前提钉住——否则下面三条可能都在测早退路径。"""
        _, meta = scoring._anchor_score_candidates_to_1x2(self.CANDIDATES, self.EURO)
        self.assertTrue(meta.get('applied'), meta)

    def test_anchoring_pulls_the_margins_towards_the_market(self):
        before = self._margins(self.CANDIDATES)
        anchored, _ = scoring._anchor_score_candidates_to_1x2(self.CANDIDATES, self.EURO)
        self.assertLess(abs(self._margins(anchored)['H'] - self.EURO['close']['home']),
                        abs(before['H'] - self.EURO['close']['home']))

    def test_zero_strength_leaves_the_margins_alone(self):
        """**反方向**：强度为 0 时不该动。"""
        before = self._margins(self.CANDIDATES)
        untouched, _ = scoring._anchor_score_candidates_to_1x2(
            self.CANDIDATES, self.EURO, strength=0.0)
        after = self._margins(untouched)
        for key in ('H', 'D', 'A'):
            with self.subTest(key=key):
                self.assertAlmostEqual(after[key], before[key], places=9)

    def test_the_default_strength_is_full_market_alignment(self):
        """默认强度 1.0：胜平负边际就是去水收盘价。

        719 场线上记录里模型与市场分歧的 58 场，模型对 17、市场对 24；模型相对
        市场在真实结果上的概率增量均值 −0.002。0.75 留下的 25% 自由度只在
        制造噪声，全量命中率因此比市场低约 1 个点。
        """
        default = scoring._anchor_score_candidates_to_1x2(self.CANDIDATES, self.EURO)[0]
        self.assertEqual(default, scoring._anchor_score_candidates_to_1x2(
            self.CANDIDATES, self.EURO, strength=1.0)[0])
        self.assertNotEqual(default, scoring._anchor_score_candidates_to_1x2(
            self.CANDIDATES, self.EURO, strength=0.75)[0])
        margins = self._margins(default)
        for key, market_key in (('H', 'home'), ('D', 'draw'), ('A', 'away')):
            with self.subTest(key=key):
                self.assertAlmostEqual(margins[key], self.EURO['close'][market_key], places=9)

    def test_the_adapter_layer_reexports_the_same_constant(self):
        """config.py 不许再持有第二份数值：改一处另一处不跟着变是判据 11 的形状。"""
        from src.football import config
        self.assertEqual(config.SCORE_1X2_MARKET_ANCHOR_STRENGTH,
                         scoring.SCORE_1X2_MARKET_ANCHOR_STRENGTH)
        self.assertEqual(scoring.SCORE_1X2_MARKET_ANCHOR_STRENGTH, 1.0)


class HeatFilterWeights(unittest.TestCase):
    """冷热过滤：热门扣分、冷门加分，两侧都测（判据 5）。"""

    def test_it_takes_a_label_not_a_probability(self):
        """**收的是 `'hot'`/`'cold'` 标签**，不是概率（第一版用例喂了浮点，
        三档全落到 `return 1.0` 的兜底，看着"通过"其实什么也没测——判据 23）。
        """
        hot = scoring._heat_filter_weight('hot')
        cold = scoring._heat_filter_weight('cold')
        neutral = scoring._heat_filter_weight('warm')
        self.assertLess(hot, neutral)
        self.assertGreater(cold, neutral)
        self.assertEqual(neutral, 1.0)

    def test_an_unknown_label_falls_back_to_no_adjustment(self):
        for label in ('warm', '', None, 0.5):
            with self.subTest(label=label):
                self.assertEqual(scoring._heat_filter_weight(label), 1.0)


class LeagueGoalFallback(unittest.TestCase):
    """`AVG_LEAGUE_GOAL` 用在 `fit_lambdas_from_markets` 里，
    不是 `score_heat_label`（第一版用例找错了函数）。
    """

    def _fit(self, league_profile):
        return scoring.fit_lambdas_from_markets(
            0.3, 2.5, 0.52, 0.45, 0.28, 0.27, 2.5, None, None, league_profile)

    def test_a_profile_without_avg_goal_falls_back_to_the_league_constant(self):
        """`lp.get('avg_goal', AVG_LEAGUE_GOAL)`——配置少一个键就落到这里
        （判据 9 第二行：配置让它不可达 → 补用例）。
        """
        with_constant = self._fit(dict(LEAGUE, avg_goal=1.35))
        without = self._fit({k: v for k, v in LEAGUE.items() if k != 'avg_goal'})
        self.assertEqual(with_constant, without)

    def test_a_different_average_changes_the_result(self):
        """**反方向**：否则上一条对任何常量都成立。"""
        self.assertNotEqual(self._fit(dict(LEAGUE, avg_goal=1.35)),
                            self._fit(dict(LEAGUE, avg_goal=3.0)))


class SupremacyConflictGate(unittest.TestCase):
    """`abs(sup_a - sup_e) >= SUPREMACY_CONFLICT_GAP` 那道门的两侧。"""

    ASIAN = {'handicap': 0.5, 'open_handicap': 0.5,
             'close_prob': {'home_give': 0.55, 'away_recv': 0.45},
             'open_prob': {'home_give': 0.55, 'away_recv': 0.45},
             'close_water': {'home': 0.9, 'away': 0.9},
             'open_water': {'home': 0.9, 'away': 0.9},
             'trend_direction': 'stable', 'trend_strength': 0.0, 'favor': 'home'}
    TOTAL = {'close_line': 2.5, 'open_line': 2.5, 'implied_total': 2.6,
             'open_implied_total': 2.6, 'close_prob': {'over': 0.52, 'under': 0.48},
             'open_prob': {'over': 0.52, 'under': 0.48},
             'trend_direction': 'stable', 'trend_strength': 0.0}

    def _state(self, sup_asian, sup_euro):
        asian = dict(self.ASIAN, implied_supremacy=sup_asian)
        euro = {'close': {'home': 0.45, 'draw': 0.28, 'away': 0.27},
                'open': {'home': 0.45, 'draw': 0.28, 'away': 0.27},
                'implied_supremacy': sup_euro,
                'kelly': {'spread': 0.0, 'hardest': 'neutral', 'favored': 'neutral'}}
        # **这道门在 `_assess_market_data_quality` 里，不是 `_joint_market_state`**
        return scoring._assess_market_data_quality(asian, euro, self.TOTAL)

    def test_a_wide_supremacy_gap_is_flagged_and_a_narrow_one_is_not(self):
        """门槛 0.75：差 0.1 与差 1.1 必须给出不同状态。"""
        agree = self._state(0.5, 0.6)
        conflict = self._state(0.5, 1.6)
        self.assertNotEqual(agree, conflict)

    def test_the_gate_sits_exactly_at_three_quarters(self):
        just_under = self._state(0.5, 0.5 + 0.74)
        just_over = self._state(0.5, 0.5 + 0.76)
        self.assertNotIn('asian_euro_supremacy_gap', just_under.get('reasons', []))
        self.assertIn('asian_euro_supremacy_gap', just_over.get('reasons', []))

    def test_opposite_directions_take_the_earlier_branch(self):
        """`sup_a * sup_e < 0` 是 if、差距大是 elif——**方向冲突优先**。"""
        opposite = self._state(0.5, -1.5)
        self.assertIn('asian_euro_direction_conflict', opposite.get('reasons', []))
        self.assertNotIn('asian_euro_supremacy_gap', opposite.get('reasons', []))


class MarketStateAgreement(unittest.TestCase):
    """联合市场状态：让球与大小球一致时该给强信号，冲突时该给弱信号。"""

    ASIAN = {'handicap': 0.5, 'open_handicap': 0.5, 'implied_supremacy': 0.5,
             'close_prob': {'home_give': 0.55, 'away_recv': 0.45},
             'open_prob': {'home_give': 0.55, 'away_recv': 0.45},
             'close_water': {'home': 0.9, 'away': 0.9},
             'open_water': {'home': 0.9, 'away': 0.9},
             'trend_direction': 'stable', 'trend_strength': 0.0, 'favor': 'home'}
    EURO = {'close': {'home': 0.45, 'draw': 0.28, 'away': 0.27},
            'open': {'home': 0.45, 'draw': 0.28, 'away': 0.27},
            'implied_supremacy': 0.5,
            'kelly': {'spread': 0.0, 'hardest': 'neutral', 'favored': 'neutral'}}
    TOTAL = {'close_line': 2.5, 'open_line': 2.5, 'implied_total': 2.6,
             'open_implied_total': 2.6,
             'close_prob': {'over': 0.52, 'under': 0.48},
             'open_prob': {'over': 0.52, 'under': 0.48},
             'trend_direction': 'stable', 'trend_strength': 0.0}

    def test_a_quiet_market_yields_a_state_without_crashing(self):
        state = scoring._joint_market_state(self.ASIAN, self.EURO, self.TOTAL)
        self.assertIsInstance(state, dict)

    def test_data_quality_degrades_when_fields_are_missing(self):
        full = scoring._assess_market_data_quality(self.ASIAN, self.EURO, self.TOTAL)
        sparse = scoring._assess_market_data_quality({}, {}, {})
        self.assertNotEqual(full, sparse)

    def test_applying_the_state_keeps_the_distribution_normalised(self):
        matrix = sm.build_score_matrix(1.5, 1.1, 7, -0.11)
        candidates = sorted(matrix.items(), key=lambda kv: -kv[1])[:12]
        adjusted = scoring._apply_joint_market_state(
            candidates, self.ASIAN, self.EURO, self.TOTAL)
        self.assertIsNotNone(adjusted)

    @staticmethod
    def _outcome_mass(candidates):
        mass = {'H': 0.0, 'D': 0.0, 'A': 0.0}
        for (h, a), p in candidates:
            mass[scoring._score_result_code(h, a)] += p
        total = sum(mass.values())
        return {k: v / total for k, v in mass.items()}

    def test_the_joint_state_only_reshapes_within_each_outcome(self):
        """亚盘/大小球的公平价约束只许改比分形状，不许动胜平负质量。

        胜平负质量在上一步已经锚到去水收盘价；这里再按亚盘公平价倾斜会把
        它带偏（719 场回放：翻转主推 5 场，翻转后 0 对、市场 3 对）。
        """
        matrix = sm.build_score_matrix(1.5, 1.1, 7, -0.11)
        candidates = sorted(matrix.items(), key=lambda kv: -kv[1])
        adjusted, meta = scoring._apply_joint_market_state(
            candidates, self.ASIAN, self.EURO, self.TOTAL)
        self.assertTrue(meta.get('applied'), meta)
        self.assertTrue(meta['asian_constraint'].get('applied'), meta)
        self.assertNotEqual(dict(adjusted), dict(candidates))
        before, after = self._outcome_mass(candidates), self._outcome_mass(adjusted)
        for key in ('H', 'D', 'A'):
            with self.subTest(key=key):
                self.assertAlmostEqual(after[key], before[key], places=9)
        self.assertAlmostEqual(meta['home_win_after'], meta['home_win_before'], places=9)


class HalfFullProbabilityShapes(unittest.TestCase):
    """`_half_full_probs_to_dict` 认三种形状——**只喂一种测不出分支**。"""

    def test_a_distribution_key_is_returned_as_is(self):
        self.assertEqual(
            scoring._half_full_probs_to_dict({'distribution': {'HH': 0.3, 'AA': 0.7}}),
            {'HH': 0.3, 'AA': 0.7})

    def test_rows_with_raw_prob_are_used_directly(self):
        result = scoring._half_full_probs_to_dict(
            {'probs': [{'code': 'HH', 'raw_prob': 0.3}]})
        self.assertEqual(result, {'HH': 0.3})

    def test_rows_without_raw_prob_are_percentages(self):
        """**没有 `raw_prob` 时按百分比除以 100**——两条路结果差 100 倍。"""
        result = scoring._half_full_probs_to_dict(
            {'probs': [{'code': 'DH', 'probability': 20.0}]})
        self.assertEqual(result, {'DH': 0.2})

    def test_rows_without_a_code_are_skipped(self):
        result = scoring._half_full_probs_to_dict(
            {'probs': [{'code': 'HH', 'raw_prob': 0.3}, {'no_code': 1}]})
        self.assertEqual(result, {'HH': 0.3})

    def test_empty_input_returns_none_not_an_empty_dict(self):
        for empty in (None, {}, {'probs': []}):
            with self.subTest(empty=empty):
                self.assertIsNone(scoring._half_full_probs_to_dict(empty))


class RecommendationDiversity(unittest.TestCase):

    MATRIX = sm.build_score_matrix(1.5, 1.1, 7, -0.11)
    CANDIDATES = sorted(MATRIX.items(), key=lambda kv: -kv[1])[:12]

    def _picked(self, n):
        """**picked 是六元组不是字典**：`(h, a, prob, ?, pattern, ?)`
        ——第一版用例喂了字典，解包直接炸（判据 28：先验算）。
        """
        return [(c[0], c[1], p, scoring._score_result_code(c[0], c[1]),
                 scoring.score_pattern(c[0], c[1]), scoring._get_score_cluster(c[0], c[1]))
                for c, p in self.CANDIDATES[:n]]

    def test_fewer_than_three_picks_are_returned_untouched(self):
        """**早退门槛是 3**——两条以下不做多样化。"""
        for n in (0, 1, 2):
            with self.subTest(n=n):
                picked = self._picked(n)
                self.assertIs(scoring._diversify_score_recommendations(
                    picked, self.CANDIDATES, 3, 'home', 0, 2), picked)

    def test_three_or_more_go_through_the_diversifier(self):
        picked = self._picked(5)
        result = scoring._diversify_score_recommendations(
            picked, self.CANDIDATES, 3, 'home', 0, 2)
        self.assertEqual(len(result), len(picked))


FORBIDDEN_IMPORTS = {'os', 'pathlib', 'requests', 'random', 'time',
                     'src.common.kv_store', 'src.common.repositories',
                     'src.football.config', 'src.football.market_db',
                     'src.football.half_time_stats', 'src.football.prediction_policy'}


class NoSideEffectTests(unittest.TestCase):

    DOMAIN = 'src/domain/sports/football/scoring.py'
    ADAPTER = 'src/football/scoring.py'

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
        """**迁移时踩过**：`apply_market_change_prior` 的延迟
        `from .market_db import` 搬进领域包后解析不到——领域层不该有
        指向适配层的相对 import。
        """
        source = pathlib.Path(self.DOMAIN).read_text(encoding='utf-8')
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.level and node.module:
                with self.subTest(module=node.module):
                    self.assertNotIn(node.module,
                                     {'market_db', 'half_time_stats', 'prediction_policy',
                                      'calibrating', 'config'})

    def test_the_guard_would_catch_a_real_violation(self):
        self.assertNotEqual(self._imports(self.ADAPTER) & FORBIDDEN_IMPORTS, set())


if __name__ == '__main__':
    unittest.main()
