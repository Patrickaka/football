# -*- coding: utf-8 -*-
"""足球的 Elo 评分纯计算与爆冷评估。

参照物是从迁移前的 `elo.py` / `upset.py` 生成的黄金文件
（`tests/fixtures/golden/football_elo_upset.json.gz`，3207 条），**逐条相同**。
迁移当时另跑过 **15693 条**新旧双跑差分，零差异。

**与北单的 upset 不是同一个形状**（判据 28 要求重新验算，不许照抄结论）：
北单那三个门槛是 `AND` 且互相耦合；这里是**先把十来个独立信号累加成
risk_score，再用 `OR` 分档**（`fav_p < 0.45 或 risk_score >= 0.55` → high）。
任一条单独成立即可，不存在那种耦合不可达。

**`elo.py` 与 `dynamic_elo.py` 各有一个 `get_elo_system`，同名不同物**——
前者返回 `ELORatingSystem`，后者返回 `DynamicELO`，是两个不同类的单例。
计划里点名提醒过「别当重复删掉」，这里用例把它钉住。
"""
import gzip
import json
import pathlib
import unittest

from src.domain.sports.football import elo, upset
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'
GOLDEN = json.load(gzip.open(FIXTURES / 'golden/football_elo_upset.json.gz',
                             'rt', encoding='utf-8'))

# 迁移当时的真实取值。**写死不 import**（判据 4）
INITIAL_ELO = 1500
HOME_ADVANTAGE = 50
K_LEAGUE, K_FRIENDLY, K_WORLD_CUP = 25, 20, 40


def golden_entries():
    from scripts.gen_football_elo_upset_golden import entries
    return entries()


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))


class TwoDifferentEloSingletons(unittest.TestCase):
    """同名不同物——合并它们会把两套评分体系搅在一起。"""

    def test_they_return_different_classes(self):
        import src.football.dynamic_elo as dyn
        import src.football.elo as static
        self.assertIsNot(static.get_elo_system, dyn.get_elo_system)
        self.assertIs(static.get_elo_system.__annotations__['return'],
                      static.ELORatingSystem)
        self.assertIs(dyn.get_elo_system.__annotations__['return'], dyn.DynamicELO)

    def test_the_static_one_has_a_lock_and_the_dynamic_one_does_not(self):
        """静态那份要给并发预测用，所以带锁；这也是它们不是一回事的证据。"""
        import src.football.dynamic_elo as dyn
        import src.football.elo as static
        self.assertTrue(hasattr(static, '_elo_system_lock'))
        self.assertFalse(hasattr(dyn, '_elo_system_lock'))


class ExpectedScore(unittest.TestCase):

    def test_equal_ratings_give_an_even_match(self):
        self.assertAlmostEqual(elo.expected_score(1500, 1500), 0.5)

    def test_it_is_symmetric_around_a_half(self):
        for a, b in ((1600, 1500), (1800, 1200), (2100, 2000)):
            with self.subTest(a=a, b=b):
                self.assertAlmostEqual(elo.expected_score(a, b) + elo.expected_score(b, a),
                                       1.0, places=12)

    def test_a_four_hundred_point_gap_is_ten_to_one(self):
        """Elo 的定义：差 400 分，胜率约 10:1。"""
        self.assertAlmostEqual(elo.expected_score(1900, 1500), 10 / 11, places=6)

    def test_it_is_monotonic_in_the_rating_gap(self):
        values = [elo.expected_score(r, 1500) for r in (1200, 1400, 1500, 1600, 1800)]
        for earlier, later in zip(values, values[1:]):
            self.assertLess(earlier, later)


class KFactorAndLeagueWeight(unittest.TestCase):

    def test_each_match_type_has_its_own_k(self):
        self.assertEqual(elo.k_factor('友谊赛'), 20)
        self.assertEqual(elo.k_factor('联赛'), 25)
        self.assertEqual(elo.k_factor('世界杯'), 40)

    def test_bigger_competitions_move_ratings_faster(self):
        """K 值随赛事级别单调递增——这是它存在的理由。"""
        ks = [elo.k_factor(t) for t in ('友谊赛', '联赛', '杯赛', '洲际杯', '世界杯')]
        for earlier, later in zip(ks, ks[1:]):
            self.assertLess(earlier, later)

    def test_an_unknown_type_falls_back_to_the_league_value(self):
        for unknown in ('未知赛事', '', None):
            with self.subTest(unknown=unknown):
                self.assertEqual(elo.k_factor(unknown), 25)

    def test_top_leagues_weigh_more_than_unknown_ones(self):
        self.assertGreater(elo.league_weight('英超'), elo.league_weight('某某丙级联赛'))
        self.assertEqual(elo.league_weight('英超'), 1.1)


class TeamNameSanitising(unittest.TestCase):

    def test_surrounding_whitespace_is_stripped(self):
        self.assertEqual(elo.sanitize_team_name('  曼联 FC  '), '曼联 FC')

    def test_empty_and_none_are_rejected(self):
        for name in ('', '   ', None):
            with self.subTest(name=name):
                self.assertIsNone(elo.sanitize_team_name(name))

    def test_an_absurdly_long_name_is_only_warned_about_not_rejected(self):
        """**它只记一条 warning，照样把名字返回**——不是拒绝（判据 9c：
        第一版用例断言返回 None，实测不是）。名字过长会一路带进评分表。
        """
        long_name = '队' * 60
        self.assertEqual(elo.sanitize_team_name(long_name), long_name)

    def test_a_normal_name_passes_through(self):
        for name in ('曼联', 'Manchester United', 'AC米兰', '1860慕尼黑'):
            with self.subTest(name=name):
                self.assertEqual(elo.sanitize_team_name(name), name)


class UpsetThresholdsAreOrNotAnd(unittest.TestCase):
    """**分档是 OR**——任一条单独成立即可。北单那种耦合不可达在这里不存在。

    分档在 `assess_football_upset` 里做，`_evaluate_upset_profile` 只给
    `risk_score` / `signals` / `favorite*`——**第一版用例找错了函数**
    （判据 28：先验算再断言）。
    """

    CANDIDATES = [((1, 0), 0.14), ((2, 0), 0.11), ((1, 1), 0.13), ((0, 0), 0.10),
                  ((0, 1), 0.09), ((2, 1), 0.10), ((1, 2), 0.07)]
    ASIAN = {'handicap': 0.5, 'open_handicap': 0.5,
             'close_water': {'home': 0.9, 'away': 0.9},
             'open_water': {'home': 0.9, 'away': 0.9}, 'favor': 'home'}
    TOTAL = {'close_line': 2.5, 'open_line': 2.5}

    def _assess(self, home, draw, away, **kw):
        euro = {'close': {'home': home, 'draw': draw, 'away': away},
                'open': {'home': home, 'draw': draw, 'away': away},
                'kelly': {'hardest': 'neutral', 'favored': 'neutral'}}
        return upset.assess_football_upset(self.ASIAN, euro, None, self.CANDIDATES,
                                           self.TOTAL, **kw)

    def test_a_weak_favourite_alone_reaches_high(self):
        """`fav_p < 0.45` 一条就够——风险分只有 0.43，**够不到 0.55**。"""
        result = self._assess(0.40, 0.32, 0.28)
        self.assertEqual(result['level'], 'high')
        self.assertLess(result['risk_score'], 0.55)

    def test_a_strong_favourite_with_no_signals_stays_low(self):
        result = self._assess(0.70, 0.18, 0.12)
        self.assertEqual(result['level'], 'low')
        self.assertFalse(result['alert'])
        self.assertEqual(result['risk_score'], 0.0)

    def test_the_two_cut_points_are_forty_five_and_fifty_two_hundredths(self):
        self.assertEqual(self._assess(0.44, 0.30, 0.26)['level'], 'high')
        self.assertEqual(self._assess(0.46, 0.29, 0.25)['level'], 'medium')
        self.assertEqual(self._assess(0.51, 0.27, 0.22)['level'], 'medium')
        self.assertEqual(self._assess(0.53, 0.26, 0.21)['level'], 'low')

    def test_risk_signals_can_lift_the_level_on_their_own(self):
        """**反方向**：热门 0.62 远超 0.52，靠信号累加照样进 high。

        不测这一条的话，把 `or risk_score >= 0.55` 整个删掉也全绿——
        OR 的另一半就没守住（判据 5）。
        """
        asian = {'handicap': 0.5, 'open_handicap': 1.25,        # 让球被削
                 'close_water': {'home': 1.02, 'away': 0.78},   # 热门水位大涨
                 'open_water': {'home': 0.88, 'away': 0.92},
                 'favor': 'home'}
        euro = {'close': {'home': 0.62, 'draw': 0.22, 'away': 0.16},
                'open': {'home': 0.70, 'draw': 0.18, 'away': 0.12},
                'kelly': {'hardest': 'home', 'favored': 'away'}}
        strong = upset.assess_football_upset(
            asian, euro, None, self.CANDIDATES, {'close_line': 2.25, 'open_line': 2.75},
            {'deviation': 0.60}, {'summary': {'dominant_signal': 'trap'}})
        self.assertGreaterEqual(strong['favorite_prob'], 0.55)
        self.assertGreaterEqual(strong['risk_score'], 0.55)
        self.assertEqual(strong['level'], 'high')

    def test_the_confident_tier_is_an_and_of_four_conditions(self):
        """`confident` 才是 AND：未预警 **且** 热门 ≥0.58 **且** 领先 ≥0.20
        **且** 风险分 <0.3——缺一不可（判据 7）。
        """
        self.assertTrue(self._assess(0.62, 0.22, 0.16)['confident'])
        # 热门 0.55 < 0.58
        self.assertFalse(self._assess(0.55, 0.25, 0.20)['confident'])
        # 热门够强但领先 0.02 < 0.20（且已被预警）
        self.assertFalse(self._assess(0.42, 0.40, 0.18)['confident'])


class UpsetAlternativeScores(unittest.TestCase):
    """预警时给的备选比分要真的与热门相反。"""

    CANDIDATES = UpsetThresholdsAreOrNotAnd.CANDIDATES
    ASIAN = UpsetThresholdsAreOrNotAnd.ASIAN
    TOTAL = UpsetThresholdsAreOrNotAnd.TOTAL

    def _assess(self, home, draw, away):
        euro = {'close': {'home': home, 'draw': draw, 'away': away},
                'open': {'home': home, 'draw': draw, 'away': away},
                'kelly': {'hardest': 'neutral', 'favored': 'neutral'}}
        return upset.assess_football_upset(self.ASIAN, euro, None, self.CANDIDATES,
                                           self.TOTAL)

    def test_a_home_favourite_gets_only_away_wins_as_outright_upsets(self):
        result = self._assess(0.40, 0.32, 0.28)
        self.assertEqual(result['level'], 'high')
        self.assertTrue(result['outright_candidates'])
        for pick in result['outright_candidates']:
            home_goals, away_goals = (int(x) for x in pick['score'].split('-'))
            with self.subTest(score=pick['score']):
                self.assertLess(home_goals, away_goals)
                self.assertEqual(pick['scenario'], 'outright_upset')

    def test_draw_candidates_are_all_level_scores(self):
        result = self._assess(0.40, 0.32, 0.28)
        self.assertTrue(result['draw_candidates'])
        for pick in result['draw_candidates']:
            home_goals, away_goals = (int(x) for x in pick['score'].split('-'))
            with self.subTest(score=pick['score']):
                self.assertEqual(home_goals, away_goals)
                self.assertEqual(pick['scenario'], 'draw_cover')

    def test_candidates_are_computed_even_without_an_alert(self):
        """**备选比分与是否预警无关**——`alert` 只影响别的字段。

        第一版用例断言「没预警就没备选」，实测不是：low 档照样给出
        2 条客胜、2 条平局。这个落差没有别的东西盯着（判据 12）。
        """
        result = self._assess(0.70, 0.18, 0.12)
        self.assertFalse(result['alert'])
        self.assertEqual(result['level'], 'low')
        self.assertTrue(result['outright_candidates'])
        self.assertTrue(result['draw_candidates'])

    def test_an_away_favourite_flips_which_scores_count_as_upsets(self):
        """**对称的两条分支只测一条不够**（判据 7）。"""
        euro = {'close': {'home': 0.28, 'draw': 0.32, 'away': 0.40},
                'open': {'home': 0.28, 'draw': 0.32, 'away': 0.40},
                'kelly': {'hardest': 'neutral', 'favored': 'neutral'}}
        result = upset.assess_football_upset(
            {'handicap': -0.5, 'open_handicap': -0.5,
             'close_water': {'home': 0.9, 'away': 0.9},
             'open_water': {'home': 0.9, 'away': 0.9}, 'favor': 'away'},
            euro, None, self.CANDIDATES, self.TOTAL)
        self.assertEqual(result['level'], 'high')
        for pick in result['outright_candidates']:
            home_goals, away_goals = (int(x) for x in pick['score'].split('-'))
            with self.subTest(score=pick['score']):
                self.assertGreater(home_goals, away_goals)


FORBIDDEN_IMPORTS = {'time', 'os', 'pathlib', 'requests', 'urllib.request',
                     'src.common.kv_store', 'src.common.repositories',
                     'src.football.config', 'src.football.fetching', 'threading'}


class NoSideEffectTests(unittest.TestCase):

    DOMAIN = ('src/domain/sports/football/elo.py',
              'src/domain/sports/football/upset.py')
    ADAPTER = 'src/football/elo.py'

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
        self.assertNotEqual(self._imports(self.ADAPTER) & FORBIDDEN_IMPORTS, set())

    def test_the_test_helper_is_no_longer_in_production_code(self):
        """`elo.py:780` 原本挂着 `test_error_handling()`——测试代码留在生产模块里。"""
        import src.football.elo as adapter
        self.assertFalse(hasattr(adapter, 'test_error_handling'))


if __name__ == '__main__':
    unittest.main()
