"""整条预测管线的回归判据。

原本是 `tests/test_basketball_prediction_pipeline.py`，直接测 `src/basketball`。
旧模块删除后原样移植到领域层——这些用例编码的是**业务规则**而不是实现细节，
迁移不该让它们消失：Elo 冷启动不许动市场价格、让分与大小分必须有新鲜且确认的
走势才出票、强走势可以翻转弱模型、开赛后一律撤下推荐。

打桩方式随之改变：旧实现只能 patch 模块级函数，领域层直接注入假依赖——
这正是当初把全局单例改成注入的理由。
"""
import unittest
from unittest import mock

from src.domain.sports.basketball.analysis import BasketballAnalyzer
from src.domain.sports.basketball.prediction import (
    PredictionService, find_value_bets,
)

MATCH = {
    'league': 'NBA', 'home': 'Home', 'away': 'Away',
    'spf_home': 1.55, 'spf_away': 2.35,
    'handicap': '-4.5', 'rqspf_home': 1.70, 'rqspf_away': 1.95,
    'total_line': 215.5, 'dx_over': 1.70, 'dx_under': 1.95,
}


class _Elo:
    """按给定值作答的假 Elo。trust 由场次数换算，与真实实现一致。"""

    def __init__(self, home_prob=0.5, margin=0.0, total=200.0, games=0):
        self.home_prob = home_prob
        self.margin = margin
        self.total = total
        self.games = games

    def predict_win_prob(self, home, away, league):
        return {'home_prob': self.home_prob, 'away_prob': 1 - self.home_prob,
                'home_games': self.games, 'away_games': self.games}

    def predict_margin(self, home, away, league):
        return {'expected_margin': self.margin}

    def predict_total_score(self, home, away, league):
        return {'expected_total': self.total}


class _IdentityCalibrator:
    """原样返回。校准本身另有测试，这里要隔离掉它的影响。"""

    def calibrate(self, bet_type, predicted_prob, league, confidence):
        return predicted_prob


def _analyzer(elo=None):
    return BasketballAnalyzer(elo=elo, calibrator=_IdentityCalibrator())


COLD_ELO = _Elo(home_prob=0.1, margin=-20, total=180, games=0)
MATURE_ELO = _Elo(home_prob=0.8, margin=10, total=235, games=40)


class EloTrustTests(unittest.TestCase):
    def test_cold_start_elo_does_not_move_market(self):
        """双方都没打过球时，Elo 的评分毫无信息量，不许动市场价格。"""
        result = _analyzer(COLD_ELO).analyze_spf(MATCH)
        self.assertAlmostEqual(result['home_prob'], result['market_home_prob'],
                               places=4)
        self.assertEqual(result['elo_trust'], 0.0)

    def test_mature_elo_is_blended_into_all_markets(self):
        analyzer = _analyzer(MATURE_ELO)
        spf = analyzer.analyze_spf(MATCH)
        rqspf = analyzer.analyze_rqspf(MATCH)
        dx = analyzer.analyze_daxiao(MATCH)
        self.assertGreater(spf['home_prob'], spf['market_home_prob'])
        self.assertGreater(rqspf['home_prob'], rqspf['market_home_prob'])
        self.assertGreater(dx['over_prob'], dx['market_over_prob'])


class MovementGateTests(unittest.TestCase):
    """让分与大小分的官方推荐要求走势佐证——准确率优先于出票量。"""

    STRONG_HOME = {**MATCH, 'rqspf_home': 1.30, 'rqspf_away': 3.20}

    def _analyze_rqspf(self, movement=None, match=None):
        return _analyzer().analyze_rqspf(match or self.STRONG_HOME, movement)

    def test_missing_movement_blocks_the_pick(self):
        result = self._analyze_rqspf()
        self.assertFalse(result['official'])
        self.assertEqual(result['skip_reason'], 'movement_unavailable')

    def test_contrary_movement_blocks_the_pick(self):
        result = self._analyze_rqspf({
            'available': True, 'side': 'away', 'strength': .5,
            'samples': 4, 'stale': False, 'steam': False})
        self.assertFalse(result['official'])
        self.assertEqual(result['skip_reason'], 'movement_conflicts_with_model')

    def test_confirming_movement_lets_the_pick_through(self):
        result = self._analyze_rqspf({
            'available': True, 'side': 'home', 'strength': .5,
            'samples': 4, 'stale': False, 'steam': False})
        self.assertTrue(result['official'])

    def test_stale_movement_blocks_totals(self):
        result = _analyzer().analyze_daxiao(MATCH, {
            'available': True, 'side': 'under', 'strength': .6,
            'samples': 5, 'stale': True, 'steam': False})
        self.assertFalse(result['official'])
        self.assertEqual(result['skip_reason'], 'movement_stale')


class WaterInferenceTests(unittest.TestCase):
    """强走势可以翻转弱模型——但只在样本充分、未过期、水位与盘口不冲突时。"""

    SPREAD_FLOW = {
        'available': True, 'side': 'away', 'strength': .8, 'samples': 5,
        'stale': False, 'steam': False, 'water_side': 'away',
        'line_side': 'away', 'signal_agreement': True, 'signal_conflict': False,
    }
    TOTAL_FLOW = {
        'available': True, 'side': 'under', 'strength': .8, 'samples': 5,
        'stale': False, 'steam': False, 'water_side': 'under',
        'line_side': 'under', 'signal_agreement': True, 'signal_conflict': False,
    }

    def test_reverses_weak_spread_model(self):
        result = _analyzer().analyze_rqspf(MATCH, self.SPREAD_FLOW)
        self.assertEqual(result['recommendation'], '让负')
        self.assertTrue(result['movement_led'])
        self.assertTrue(result['official'])

    def test_reverses_weak_total_model(self):
        result = _analyzer().analyze_daxiao(MATCH, self.TOTAL_FLOW)
        self.assertEqual(result['recommendation'], '小分')
        self.assertTrue(result['movement_led'])
        self.assertTrue(result['official'])


class ValueBetTests(unittest.TestCase):
    def test_value_list_excludes_non_official_opinions(self):
        """模型的看法与「可以出票」是两件事，价值榜只收后者。"""
        rows = [{'match': MATCH, 'spf': {
            'available': True, 'playable': False, 'home_prob': 0.8,
            'away_prob': 0.2, 'recommendation': '主胜',
        }, 'rqspf': None, 'dx': None}]
        self.assertEqual(find_value_bets(rows), [])


class StartedMatchTests(unittest.TestCase):
    def test_started_match_is_visible_but_never_official(self):
        """已开赛的场次仍然展示，但一律不出票——赔率已停更。"""
        started = {**MATCH, 'id': 'm1', 'status': 'in_progress', 'time': '10:00'}
        service = PredictionService(
            analyzer=_analyzer(),
            schedule_sources={'500': lambda date: [dict(started)]},
            today_fn=lambda: '2026-08-27')
        payload = service.generate(source='500')
        self.assertEqual(payload['count'], 1)
        self.assertFalse(payload['results'][0]['rqspf']['official'])
        self.assertEqual(payload['results'][0]['rqspf']['skip_reason'],
                         'match_already_started')


if __name__ == '__main__':
    unittest.main()
