"""三个分析函数（胜负 / 让分胜负 / 大小分）迁入领域层。

主体仍是差分测试（裁决 D）。与走势那批不同的是，分析函数依赖 Elo 与校准器，
而旧实现在函数体内 `from .elo import get_elo_system` 取全局单例——那个单例会
真的去读 kv_store。所以差分的前提是**两侧注入同一组假依赖**：旧实现打桩它的
单例工厂，新实现直接构造注入。这样两份代码面对的是同一份 Elo 输出，差异只可能
来自算法本身。
"""
import unittest
from unittest import mock

import src.basketball as legacy
from src.basketball import calibration as legacy_calibration
from src.basketball import elo as legacy_elo
from src.domain.sports.basketball.analysis import BasketballAnalyzer


class FakeElo:
    """按球队名给出确定性的 Elo 输出，便于差分与断言。"""

    def __init__(self, home_prob=0.62, margin=4.5, total=214.0,
                 home_games=30, away_games=30):
        self.home_prob = home_prob
        self.margin = margin
        self.total = total
        self.home_games = home_games
        self.away_games = away_games
        self.calls = []

    def predict_win_prob(self, home, away, league):
        self.calls.append(('win', home, away, league))
        return {'home_prob': self.home_prob, 'away_prob': 1 - self.home_prob,
                'home_games': self.home_games, 'away_games': self.away_games}

    def predict_margin(self, home, away, league):
        return {'expected_margin': self.margin}

    def predict_total_score(self, home, away, league):
        return {'expected_total': self.total}


class ExplodingElo:
    def predict_win_prob(self, *a, **k):
        raise RuntimeError('elo 库挂了')

    predict_margin = predict_total_score = predict_win_prob


class FakeCalibrator:
    """按固定倍率缩放，足以区分「校准生效」与「原样透传」。"""

    def __init__(self, factor=1.05):
        self.factor = factor
        self.calls = []

    def calibrate(self, bet_type, predicted_prob, league, confidence):
        self.calls.append((bet_type, round(predicted_prob, 6), league, confidence))
        return predicted_prob * self.factor


class ExplodingCalibrator:
    def calibrate(self, *a, **k):
        raise RuntimeError('校准库挂了')


MATCHES = [
    {'id': 'm0', 'home': '湖人', 'away': '凯尔特人', 'league': 'NBA'},
    {'id': 'm1', 'home': '湖人', 'away': '凯尔特人', 'league': 'NBA',
     'spf_home': 1.75, 'spf_away': 2.10, 'rqspf_home': 1.90, 'rqspf_away': 1.90,
     'handicap': -3.5, 'dx_over': 1.85, 'dx_under': 1.95, 'total_line': 221.5},
    {'id': 'm2', 'home': '广东', 'away': '辽宁', 'league': 'CBA',
     'spf_home': 2.40, 'spf_away': 1.55, 'rqspf_home': 2.05, 'rqspf_away': 1.78,
     'handicap': 4.5, 'dx_over': 2.00, 'dx_under': 1.80, 'total_line': 185.5},
    {'id': 'm3', 'home': 'A 队', 'away': 'B 队', 'league': '没见过的联赛',
     'spf_home': 1.90, 'spf_away': 1.90, 'rqspf_home': 1.90, 'rqspf_away': 1.90,
     'handicap': '不是数字', 'dx_over': 1.90, 'dx_under': 1.90, 'total_line': None},
    {'id': 'm4', 'home': '天津', 'away': '上海', 'league': 'WNBA',
     'spf_home': 3.20, 'spf_away': 1.33, 'rqspf_home': 1.95, 'rqspf_away': 1.85,
     'handicap': None, 'dx_over': 1.72, 'dx_under': 2.08, 'total_line': 150.5},
    {'id': 'm5', 'home': 'C 队', 'away': 'D 队', 'league': 'NCAAB',
     'spf_home': 1.90, 'rqspf_away': 1.90, 'dx_under': 1.90, 'total_line': 140.5},
    # 盘口高出联赛均值 15 分：大小分那条「偏离超过 5 分就向回归方向让一点」
    # 的先验有升/降两个方向，只测降的一侧，把升的阈值改成 50 都测不出来。
    {'id': 'm6', 'home': 'E 队', 'away': 'F 队', 'league': 'NBA',
     'spf_home': 1.80, 'spf_away': 2.00, 'rqspf_home': 1.88, 'rqspf_away': 1.92,
     'handicap': -2.5, 'dx_over': 1.90, 'dx_under': 1.90, 'total_line': 235.0},
    # 恰好等于均值：两条分支都不该走
    {'id': 'm7', 'home': 'G 队', 'away': 'H 队', 'league': 'CBA',
     'spf_home': 2.00, 'spf_away': 1.80, 'rqspf_home': 1.90, 'rqspf_away': 1.90,
     'handicap': 1.5, 'dx_over': 1.95, 'dx_under': 1.85, 'total_line': 190.0},
]

MOVEMENTS = [
    None,
    {},
    {'available': False},
    {'available': True, 'side': 'flat', 'strength': 0.0, 'samples': 5,
     'stale': False, 'steam': False, 'signal_conflict': False,
     'water_side': 'flat', 'line_side': 'flat', 'signal_agreement': False,
     'line_move': 0.0, 'home_move': 0.0, 'away_move': 0.0, 'kind': 'ml'},
    {'available': True, 'side': 'home', 'strength': 0.55, 'samples': 6,
     'stale': False, 'steam': False, 'signal_conflict': False,
     'water_side': 'home', 'line_side': 'home', 'signal_agreement': True,
     'line_move': -1.0, 'home_move': -0.12, 'away_move': 0.06, 'kind': 'ah',
     'opening_line': -3.5, 'current_line': -4.5},
    {'available': True, 'side': 'away', 'strength': 0.80, 'samples': 9,
     'stale': False, 'steam': True, 'signal_conflict': False,
     'water_side': 'away', 'line_side': 'away', 'signal_agreement': True,
     'line_move': 1.5, 'home_move': 0.08, 'away_move': -0.15, 'kind': 'ah'},
    {'available': True, 'side': 'home', 'strength': 0.9, 'samples': 1,
     'stale': False, 'steam': True, 'signal_conflict': False,
     'water_side': 'home', 'line_side': 'flat', 'signal_agreement': False,
     'line_move': 0.0, 'home_move': -0.2, 'away_move': 0.1, 'kind': 'ml'},
    {'available': True, 'side': 'home', 'strength': 0.7, 'samples': 8,
     'stale': True, 'steam': False, 'signal_conflict': False,
     'water_side': 'home', 'line_side': 'flat', 'signal_agreement': False,
     'line_move': 0.0, 'home_move': -0.15, 'away_move': 0.05, 'kind': 'ml'},
    {'available': True, 'side': 'over', 'strength': 0.6, 'samples': 7,
     'stale': False, 'steam': False, 'signal_conflict': True,
     'water_side': 'over', 'line_side': 'under', 'signal_agreement': False,
     'line_move': -1.0, 'home_move': -0.1, 'away_move': 0.04, 'kind': 'ou'},
    {'available': True, 'side': 'under', 'strength': 0.2, 'samples': 5,
     'stale': False, 'steam': False, 'signal_conflict': False,
     'water_side': 'under', 'line_side': 'flat', 'signal_agreement': False,
     'line_move': 0.0, 'home_move': 0.03, 'away_move': -0.06, 'kind': 'ou'},
    {'available': True, 'side': 'over', 'strength': 0.5, 'samples': 6,
     'stale': False, 'steam': False, 'signal_conflict': False,
     'water_side': 'over', 'line_side': 'over', 'signal_agreement': True,
     'line_move': 2.0, 'home_move': -0.1, 'away_move': 0.05, 'kind': 'ou',
     'opening_line': 210.5, 'current_line': 212.5},
]

ELO_SETUPS = [
    FakeElo(),
    FakeElo(home_prob=0.35, margin=-6.0, total=190.0),
    # 冷启动：样本不足，trust 应把 Elo 的影响压到接近零
    FakeElo(home_prob=0.9, margin=15.0, total=250.0, home_games=2, away_games=40),
    FakeElo(home_prob=0.5, margin=0.0, total=200.0, home_games=0, away_games=0),
]

CALIBRATORS = [FakeCalibrator(1.0), FakeCalibrator(1.08), FakeCalibrator(0.93)]


class _ParityBase(unittest.TestCase):
    def _compare(self, method_name, legacy_fn):
        for elo in ELO_SETUPS:
            for calibrator in CALIBRATORS:
                analyzer = BasketballAnalyzer(elo=elo, calibrator=calibrator)
                for match in MATCHES:
                    for movement in MOVEMENTS:
                        with self.subTest(mid=match['id'],
                                          side=(movement or {}).get('side'),
                                          games=elo.home_games,
                                          factor=calibrator.factor):
                            with mock.patch.object(
                                    legacy_elo, 'get_elo_system', lambda: elo), \
                                 mock.patch.object(
                                     legacy_calibration, 'get_calibrator',
                                     lambda: calibrator):
                                expected = legacy_fn(match, movement)
                            actual = getattr(analyzer, method_name)(match, movement)
                            self.assertEqual(actual, expected)


class AnalyzeSpfParityTests(_ParityBase):
    def test_matches_legacy(self):
        self._compare('analyze_spf', legacy.analyze_spf)


class AnalyzeRqspfParityTests(_ParityBase):
    def test_matches_legacy(self):
        self._compare('analyze_rqspf', legacy.analyze_rqspf)


class AnalyzeDaxiaoParityTests(_ParityBase):
    def test_matches_legacy(self):
        self._compare('analyze_daxiao', legacy.analyze_daxiao)


class DependencyFailureTests(unittest.TestCase):
    """依赖不可用时必须降级，而不是让整场比赛的分析失败。"""

    MATCH = MATCHES[1]

    def test_missing_elo_falls_back_to_market_only(self):
        analyzer = BasketballAnalyzer(elo=None, calibrator=FakeCalibrator(1.0))
        result = analyzer.analyze_spf(self.MATCH)
        self.assertTrue(result['available'])
        self.assertEqual(result['elo_trust'], 0.0)
        self.assertIsNone(result['elo_home_prob'])

    def test_exploding_elo_falls_back_to_market_only(self):
        analyzer = BasketballAnalyzer(elo=ExplodingElo(),
                                      calibrator=FakeCalibrator(1.0))
        result = analyzer.analyze_spf(self.MATCH)
        self.assertTrue(result['available'])
        self.assertEqual(result['elo_trust'], 0.0)

    def test_exploding_calibrator_keeps_raw_probability(self):
        analyzer = BasketballAnalyzer(elo=None, calibrator=ExplodingCalibrator())
        result = analyzer.analyze_spf(self.MATCH)
        self.assertTrue(result['available'])
        self.assertEqual(result['pick_prob'], result['raw_pick_prob'])

    def test_missing_calibrator_keeps_raw_probability(self):
        analyzer = BasketballAnalyzer(elo=None, calibrator=None)
        result = analyzer.analyze_daxiao(self.MATCH)
        self.assertEqual(result['pick_prob'], result['raw_pick_prob'])


class ColdStartTrustTests(unittest.TestCase):
    """Elo 冷启动不得稀释信息量更大的市场价格。"""

    MATCH = MATCHES[1]

    def test_trust_scales_with_the_thinner_side_history(self):
        analyzer = BasketballAnalyzer(
            elo=FakeElo(home_games=2, away_games=40), calibrator=None)
        self.assertEqual(analyzer.analyze_spf(self.MATCH)['elo_trust'], 0.1)

    def test_trust_caps_at_one(self):
        analyzer = BasketballAnalyzer(
            elo=FakeElo(home_games=200, away_games=200), calibrator=None)
        self.assertEqual(analyzer.analyze_spf(self.MATCH)['elo_trust'], 1.0)

    def test_cold_start_elo_moves_probability_less_than_warm_elo(self):
        market_only = BasketballAnalyzer(elo=None, calibrator=None)
        baseline = market_only.analyze_spf(self.MATCH)['home_prob']
        cold = BasketballAnalyzer(
            elo=FakeElo(home_prob=0.95, home_games=1, away_games=1),
            calibrator=None).analyze_spf(self.MATCH)['home_prob']
        warm = BasketballAnalyzer(
            elo=FakeElo(home_prob=0.95, home_games=50, away_games=50),
            calibrator=None).analyze_spf(self.MATCH)['home_prob']
        self.assertLess(abs(cold - baseline), abs(warm - baseline))


class NoLegacyImportTests(unittest.TestCase):
    def test_analysis_does_not_import_legacy_package(self):
        import ast
        import inspect

        from src.domain.sports.basketball import analysis

        tree = ast.parse(inspect.getsource(analysis))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertFalse(alias.name.startswith('src.basketball'))
            elif isinstance(node, ast.ImportFrom):
                module = ('.' * (node.level or 0)) + (node.module or '')
                self.assertFalse(module.startswith('src.basketball'), module)
                self.assertFalse(module.startswith('.'), f'不该有相对导入: {module}')


if __name__ == '__main__':
    unittest.main()
