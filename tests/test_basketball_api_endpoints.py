"""五个篮球端点切到领域层。

这是整个 basketball 迁移里唯一改动线上行为的一步：此前六批只新增代码，
端点仍走 `src/basketball`。所以这里测的是**响应形状没变**——前端字段一个
不少、错误路径仍然返回 `{'error': ...}` 而不是抛出去。
"""
import logging
import unittest
from unittest import mock

from src.webapp import basketball_service
from src.webapp.basketball_api import BasketballApiMixin


class _Handler(BasketballApiMixin):
    def __init__(self):
        self._log = logging.getLogger('test.basketball_api')


ANALYSIS = {
    'available': True, 'recommendation': '主胜', 'confidence': 'high',
    'home_prob': 0.62, 'away_prob': 0.38, 'home_odds': 1.75, 'away_odds': 2.10,
    'over_prob': 0.55, 'under_prob': 0.45, 'over_odds': 1.85, 'under_odds': 1.95,
    'handicap': '-3.5', 'total_line': 221.5,
    'line_movement': {'side': 'home'}, 'water_inference': {'actionable': True},
    'movement_led': False, 'sharp_confirmed': True,
    'official': True, 'skip_reason': None, 'playable': True,
}

PAYLOAD = {
    'date': '2026-08-27',
    'count': 1,
    'version': 'v-test',
    'source': 'okooo',
    'movement_stats': {'with_movement': 3},
    'results': [{
        # handicap / total_line 挂在 match 上而非玩法上——解析器给的就是
        # 这个形状，价值投注拼展示标签时直接取它们
        'match': {'home': '湖人', 'away': '凯尔特人', 'league': 'NBA',
                  'time': '09:00', 'status': 'not_started',
                  'handicap': '-3.5', 'total_line': 221.5},
        'spf': dict(ANALYSIS), 'rqspf': dict(ANALYSIS), 'dx': dict(ANALYSIS),
        'market_analysis': {'available': True, 'verdict': '一致'},
    }],
}

SCHEDULE = [{'id': 'm1', 'home': '湖人', 'away': '凯尔特人', 'time': '09:00'}]
SNAPSHOTS = {
    'm1': [
        {'ts': '2026-08-26T09:00:00', 'spf_home': 2.00, 'spf_away': 1.80},
        {'ts': '2026-08-26T11:00:00', 'spf_home': 1.70, 'spf_away': 2.10},
    ],
    'm2': [{'ts': '2026-08-26T09:00:00', 'spf_home': None, 'spf_away': None}],
}


def _context(prediction=None, tracker=None, history=None):
    ctx = mock.Mock()
    ctx.prediction = prediction or mock.Mock()
    ctx.tracker = tracker
    ctx.history = history
    return ctx


class _Base(unittest.TestCase):
    def setUp(self):
        self.handler = _Handler()
        self.addCleanup(basketball_service.reset)

    def _with(self, ctx):
        patcher = mock.patch('src.webapp.basketball_api.get_context',
                             lambda: ctx)
        patcher.start()
        self.addCleanup(patcher.stop)
        return ctx


class RecommendationEndpointTests(_Base):
    def _payload(self, params=None):
        prediction = mock.Mock()
        prediction.generate.return_value = PAYLOAD
        self.prediction = prediction
        self._with(_context(prediction=prediction))
        return self.handler._basketball_payload(params or {})

    def test_response_shape_is_unchanged(self):
        result = self._payload()['result']
        self.assertEqual(result['date'], '2026-08-27')
        self.assertEqual(result['total_matches'], 1)
        self.assertEqual(result['version'], 'v-test')
        self.assertEqual(result['source'], 'okooo')
        self.assertEqual(result['movement_stats'], {'with_movement': 3})

    def test_每个玩法的字段都在(self):
        match = self._payload()['result']['matches'][0]
        self.assertEqual(match['home'], '湖人')
        self.assertTrue(match['official_open'])
        self.assertEqual(match['spf']['prediction'], '主胜')
        self.assertEqual(match['spf']['probabilities'], {'主胜': 0.62, '客胜': 0.38})
        self.assertEqual(match['rqspf']['handicap'], '-3.5')
        self.assertEqual(match['daxiao']['total'], 221.5)
        self.assertEqual(match['daxiao']['probabilities'], {'大分': 0.55, '小分': 0.45})

    def test_defaults_match_the_previous_behaviour(self):
        self._payload()
        self.prediction.generate.assert_called_once_with(
            date=None, bet_types=['spf', 'rqspf', 'dx'], source='okooo',
            use_movement=True)

    def test_unknown_source_falls_back_to_okooo(self):
        self._payload({'source': ['随便写的']})
        self.assertEqual(self.prediction.generate.call_args.kwargs['source'],
                         'okooo')

    def test_bet_types_are_taken_from_the_query(self):
        self._payload({'types': ['spf,dx']})
        self.assertEqual(self.prediction.generate.call_args.kwargs['bet_types'],
                         ['spf', 'dx'])

    def test_unavailable_market_reports_its_reason(self):
        prediction = mock.Mock()
        payload = dict(PAYLOAD)
        payload['results'] = [{
            'match': {'home': 'A', 'away': 'B', 'league': 'CBA',
                      'time': '19:35', 'status': 'in_progress'},
            'spf': {'available': False, 'reason': 'missing_odds'},
            'rqspf': None, 'dx': None, 'market_analysis': None,
        }]
        prediction.generate.return_value = payload
        self._with(_context(prediction=prediction))
        match = self.handler._basketball_payload({})['result']['matches'][0]
        self.assertEqual(match['spf'], {'error': 'missing_odds'})
        self.assertEqual(match['rqspf'], {'error': 'no_data'})
        self.assertFalse(match['official_open'])

    def test_failure_is_reported_not_raised(self):
        prediction = mock.Mock()
        prediction.generate.side_effect = RuntimeError('炸了')
        self._with(_context(prediction=prediction))
        self.assertIn('error', self.handler._basketball_payload({}))


class ScheduleEndpointTests(_Base):
    def test_returns_the_schedule(self):
        prediction = mock.Mock()
        prediction.fetch_schedule.return_value = SCHEDULE
        self._with(_context(prediction=prediction))
        self.assertEqual(self.handler._basketball_matches_payload({}),
                         {'matches': SCHEDULE})
        prediction.fetch_schedule.assert_called_once_with(date=None)

    def test_failure_is_reported_not_raised(self):
        prediction = mock.Mock()
        prediction.fetch_schedule.side_effect = IOError('源站挂了')
        self._with(_context(prediction=prediction))
        self.assertIn('error', self.handler._basketball_matches_payload({}))


class ValueBetEndpointTests(_Base):
    def test_uses_the_threshold_from_the_query(self):
        prediction = mock.Mock()
        prediction.generate.return_value = PAYLOAD
        self._with(_context(prediction=prediction))
        loose = self.handler._basketball_value_payload({'threshold': ['0.0']})
        strict = self.handler._basketball_value_payload({'threshold': ['0.9']})
        self.assertTrue(loose['result'])
        self.assertEqual(strict['result'], [])

    def test_failure_is_reported_not_raised(self):
        prediction = mock.Mock()
        prediction.generate.side_effect = RuntimeError('炸了')
        self._with(_context(prediction=prediction))
        self.assertIn('error', self.handler._basketball_value_payload({}))


class TrackEndpointTests(_Base):
    def test_triggers_one_capture(self):
        tracker = mock.Mock()
        tracker.track.return_value = 7
        self._with(_context(tracker=tracker))
        result = self.handler._basketball_track_payload({'date': ['2026-08-27']})
        self.assertEqual(result['result'], {'tracked': 7, 'date': '2026-08-27'})
        tracker.track.assert_called_once_with('2026-08-27')

    def test_reports_when_the_database_is_missing(self):
        """采集的全部意义就是落盘，没有库时明说，而不是假装采集成功。"""
        self._with(_context(tracker=None))
        self.assertIn('error', self.handler._basketball_track_payload({}))

    def test_failure_is_reported_not_raised(self):
        tracker = mock.Mock()
        tracker.track.side_effect = IOError('库挂了')
        self._with(_context(tracker=tracker))
        self.assertIn('error', self.handler._basketball_track_payload({}))


class MovementEndpointTests(_Base):
    def _store(self):
        store = mock.Mock()
        store.load.return_value = {k: list(v) for k, v in SNAPSHOTS.items()}
        store.history_for.side_effect = lambda mid: list(SNAPSHOTS.get(mid, []))
        return store

    def test_single_match_returns_its_snapshots(self):
        self._with(_context(history=self._store()))
        result = self.handler._basketball_movement_payload({'match_id': ['m1']})
        self.assertEqual(result['result']['match_id'], 'm1')
        self.assertEqual(len(result['result']['snapshots']), 2)

    def test_summary_reports_the_move_per_match(self):
        self._with(_context(history=self._store()))
        detail = self.handler._basketball_movement_payload({})['result']['detail']
        by_id = {d['match_id']: d for d in detail}
        self.assertAlmostEqual(by_id['m1']['spf_home_move'], -0.3)
        self.assertAlmostEqual(by_id['m1']['spf_away_move'], 0.3)
        self.assertNotIn('spf_home_move', by_id['m2'],
                         '没有有效赔率的场次不该编出位移')

    def test_reports_when_the_database_is_missing(self):
        self._with(_context(history=None))
        self.assertIn('error', self.handler._basketball_movement_payload({}))


class ContextTests(unittest.TestCase):
    def setUp(self):
        basketball_service.reset()
        self.addCleanup(basketball_service.reset)

    def test_context_is_a_process_singleton(self):
        """连接池、熔断器状态都是进程级的。每请求重建会让熔断器永远回到
        初始状态、形同虚设。"""
        self.assertIs(basketball_service.get_context(),
                      basketball_service.get_context())

    def test_database_failure_degrades_instead_of_raising(self):
        with mock.patch('src.foundation.store.make_engine',
                        side_effect=RuntimeError('连不上')):
            basketball_service.reset()
            ctx = basketball_service.get_context()
        self.assertIsNone(ctx.db)
        self.assertIsNone(ctx.tracker)
        self.assertIsNotNone(ctx.prediction, '数据库不可用不该让推荐服务缺席')

    def test_cache_failure_degrades_instead_of_raising(self):
        with mock.patch('src.api.deps.build_cache', side_effect=RuntimeError('炸了')):
            basketball_service.reset()
            ctx = basketball_service.get_context()
        self.assertIsNone(ctx.cache)
        self.assertIsNotNone(ctx.prediction)


if __name__ == '__main__':
    unittest.main()
