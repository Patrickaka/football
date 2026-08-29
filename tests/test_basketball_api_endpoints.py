"""五个篮球端点切到领域层。

这是整个 basketball 迁移里唯一改动线上行为的一步：此前六批只新增代码，
端点仍走 `src/basketball`。所以这里测的是**响应形状没变**——前端字段一个
不少、错误路径仍然返回 `{'error': ...}` 而不是抛出去。
"""
import logging
import threading
import unittest
from unittest import mock

from src.webapp import basketball_service
from src.api.services import basketball as service


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
        self.addCleanup(basketball_service.reset)

    def _with(self, ctx):
        # 业务逻辑已迁至 `src.api.services.basketball`，新旧入口共用同一份
        # （判据 11）。mock 要打在它现在住的地方——打在旧模块上不会报错，
        # 只是**什么也没替换掉**。
        patcher = mock.patch('src.api.services.basketball.get_context',
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
        return service.basketball_payload(params or {})

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
        match = service.basketball_payload({})['result']['matches'][0]
        self.assertEqual(match['spf'], {'error': 'missing_odds'})
        self.assertEqual(match['rqspf'], {'error': 'no_data'})
        self.assertFalse(match['official_open'])

    def test_failure_is_reported_not_raised(self):
        prediction = mock.Mock()
        prediction.generate.side_effect = RuntimeError('炸了')
        self._with(_context(prediction=prediction))
        self.assertIn('error', service.basketball_payload({}))


class ScheduleEndpointTests(_Base):
    def test_returns_the_schedule(self):
        prediction = mock.Mock()
        prediction.fetch_schedule.return_value = SCHEDULE
        self._with(_context(prediction=prediction))
        self.assertEqual(service.basketball_matches_payload({}),
                         {'matches': SCHEDULE})
        prediction.fetch_schedule.assert_called_once_with(date=None)

    def test_failure_is_reported_not_raised(self):
        prediction = mock.Mock()
        prediction.fetch_schedule.side_effect = IOError('源站挂了')
        self._with(_context(prediction=prediction))
        self.assertIn('error', service.basketball_matches_payload({}))


class ValueBetEndpointTests(_Base):
    def test_uses_the_threshold_from_the_query(self):
        prediction = mock.Mock()
        prediction.generate.return_value = PAYLOAD
        self._with(_context(prediction=prediction))
        loose = service.basketball_value_payload({'threshold': ['0.0']})
        strict = service.basketball_value_payload({'threshold': ['0.9']})
        self.assertTrue(loose['result'])
        self.assertEqual(strict['result'], [])

    def test_failure_is_reported_not_raised(self):
        prediction = mock.Mock()
        prediction.generate.side_effect = RuntimeError('炸了')
        self._with(_context(prediction=prediction))
        self.assertIn('error', service.basketball_value_payload({}))


class TrackEndpointTests(_Base):
    def test_triggers_one_capture(self):
        tracker = mock.Mock()
        tracker.track.return_value = 7
        self._with(_context(tracker=tracker))
        result = service.basketball_track_payload({'date': ['2026-08-27']})
        self.assertEqual(result['result'], {'tracked': 7, 'date': '2026-08-27'})
        tracker.track.assert_called_once_with('2026-08-27')

    def test_reports_when_the_database_is_missing(self):
        """采集的全部意义就是落盘，没有库时明说，而不是假装采集成功。"""
        self._with(_context(tracker=None))
        self.assertIn('error', service.basketball_track_payload({}))

    def test_failure_is_reported_not_raised(self):
        tracker = mock.Mock()
        tracker.track.side_effect = IOError('库挂了')
        self._with(_context(tracker=tracker))
        self.assertIn('error', service.basketball_track_payload({}))


class MovementEndpointTests(_Base):
    def _store(self):
        store = mock.Mock()
        store.load.return_value = {k: list(v) for k, v in SNAPSHOTS.items()}
        store.history_for.side_effect = lambda mid: list(SNAPSHOTS.get(mid, []))
        return store

    def test_single_match_returns_only_that_match_snapshots(self):
        """断言内容而不是条数：全量字典恰好也有两个键，只比长度分辨不出
        「这一场的快照」和「所有场次的字典」。"""
        store = self._store()
        self._with(_context(history=store))
        result = service.basketball_movement_payload({'match_id': ['m1']})
        self.assertEqual(result['result']['match_id'], 'm1')
        self.assertEqual(result['result']['snapshots'], SNAPSHOTS['m1'])
        store.load.assert_not_called()

    def test_summary_reports_the_move_per_match(self):
        self._with(_context(history=self._store()))
        detail = service.basketball_movement_payload({})['result']['detail']
        by_id = {d['match_id']: d for d in detail}
        self.assertAlmostEqual(by_id['m1']['spf_home_move'], -0.3)
        self.assertAlmostEqual(by_id['m1']['spf_away_move'], 0.3)
        self.assertNotIn('spf_home_move', by_id['m2'],
                         '没有有效赔率的场次不该编出位移')

    def test_reports_when_the_database_is_missing(self):
        self._with(_context(history=None))
        self.assertIn('error', service.basketball_movement_payload({}))


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


class OddsTrackingSchedulerTests(unittest.TestCase):
    """赔率快照的周期采样。

    走势要靠一天里反复采样攒出来。夜里没人看的时候盘口照样在动，而那段
    变化正是开盘到临场的主要部分——只在有人请求时才采是攒不出来的。

    迁移过程中这条一度差点丢掉：旧的采样器由 `server.py` 启动，而我按
    `src/` 目录下的 grep 判断它「没有调用方」，差点当作死代码删掉。
    """

    def setUp(self):
        from src.webapp import background

        basketball_service.reset()
        background.reset()
        self.addCleanup(basketball_service.reset)
        self.addCleanup(background.reset)

    def _with_tracker(self, tracker):
        """打桩 `_build_context` 而不是 `get_context`。

        差别很关键：打桩 `get_context` 会把真正的取锁路径整个绕过去，
        而那里正好藏过一个持锁死锁——`start_odds_tracking` 在锁内调用
        `get_context()`，后者要拿同一把不可重入的锁。那个 bug 上线过一次，
        五个端点全部 120 秒超时，而当时的测试全绿。
        """
        ctx = mock.Mock()
        ctx.tracker = tracker
        patcher = mock.patch.object(basketball_service, '_build_context',
                                    lambda: ctx)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _call(self, fn, *args, **kwargs):
        """带超时地调用。

        这一整组用例都可能撞上持锁死锁，而**挂住的测试是坏测试**——它把
        「有 bug」伪装成「跑得慢」，在 CI 上表现为整个任务超时，没人能一眼
        看出是哪里坏了。所以统一走后台线程 + 超时断言。
        """
        box = {}
        thread = threading.Thread(
            target=lambda: box.setdefault('value', fn(*args, **kwargs)),
            daemon=True)
        thread.start()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive(), f'{fn.__name__} 卡住了（疑似死锁）')
        return box.get('value')

    def test_register_does_not_deadlock_on_a_cold_context(self):
        """首次登记时上下文还没建好，这正是死锁发生的时机。"""
        self._with_tracker(mock.Mock())
        self.assertTrue(self._call(basketball_service.register_odds_tracking))

    def test_context_stays_reachable_after_register(self):
        """死锁的后果不止是登记卡住——锁没释放，之后每个请求都会一起卡住。"""
        self._with_tracker(mock.Mock())
        self._call(basketball_service.register_odds_tracking)
        self.assertIsNotNone(self._call(basketball_service.get_context))

    def test_registers_one_periodic_task(self):
        from src.webapp import background

        self._with_tracker(mock.Mock())
        self.assertTrue(self._call(basketball_service.register_odds_tracking))
        self.assertEqual(background.task_count(), 1)

    def test_registered_task_runs_after_start(self):
        """登记了却不跑，是这类接线最典型的无声失败。"""
        from src.webapp import background

        tracker = mock.Mock()
        ran = threading.Event()
        tracker.track.side_effect = lambda date: ran.set()
        self._with_tracker(tracker)
        self._call(basketball_service.register_odds_tracking)
        background.start()
        self.assertTrue(ran.wait(timeout=5), '登记的采样任务没有被执行')
        self.assertTrue(basketball_service.is_odds_tracking_running())

    def test_repeated_registration_is_rejected(self):
        self._with_tracker(mock.Mock())
        self.assertTrue(self._call(basketball_service.register_odds_tracking))
        self.assertFalse(self._call(basketball_service.register_odds_tracking))

    def test_not_registered_without_a_database(self):
        """采集的全部意义就是落盘。没有库就别假装在采。"""
        from src.webapp import background

        self._with_tracker(None)
        self.assertFalse(self._call(basketball_service.register_odds_tracking))
        self.assertEqual(background.task_count(), 0)

    def test_interval_is_floored_at_one_minute(self):
        """采样间隔有下限：okooo 与 500 都有限速，采太密只是挤占抓取配额。"""
        self._with_tracker(mock.Mock())
        self.assertTrue(self._call(basketball_service.register_odds_tracking,
                                   interval_minutes=0))


if __name__ == '__main__':
    unittest.main()
