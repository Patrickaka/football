"""装配。

装配代码本身没有算法，但它是**唯一**可能把 A 的实现接到 B 的端口上的地方，
接错了不会报错、只会安静地给出错误结果。所以这里测的全是「接到了谁」。
"""
import unittest
from datetime import datetime
from unittest import mock

from src.domain.sports.basketball import factory
from src.domain.sports.basketball.analysis import BasketballAnalyzer
from src.domain.sports.basketball.movement_map import MovementMapBuilder
from src.domain.sports.basketball.odds_history import OddsTracker
from src.domain.sports.basketball.prediction import PredictionService
from src.domain.sports.basketball.repository import create_all
from src.foundation.cache import Cache, MemoryBackend
from src.foundation.store import Database, make_engine

NOW = datetime(2026, 8, 26, 12, 0, 0)


class _Base(unittest.TestCase):
    def setUp(self):
        self.db = Database(make_engine('sqlite+pysqlite:///:memory:'))
        create_all(self.db)
        self.requested = []

    def transport(self, url, timeout=20):
        self.requested.append(url)
        return ''


class AnalyzerWiringTests(_Base):
    def test_analyzer_reads_elo_and_calibration_from_the_database(self):
        analyzer = factory.build_analyzer(self.db)
        self.assertIsInstance(analyzer, BasketballAnalyzer)
        self.assertIsNotNone(analyzer._elo)
        self.assertIsNotNone(analyzer._calibrator)

    def test_analyzer_survives_an_empty_database(self):
        """全新部署时三张表都是空的，此时应退化为纯市场价格而不是报错。"""
        result = factory.build_analyzer(self.db).analyze_spf(
            {'home': 'A', 'away': 'B', 'league': 'NBA',
             'spf_home': 1.8, 'spf_away': 2.0})
        self.assertTrue(result['available'])
        self.assertEqual(result['elo_trust'], 0.0)


class ScheduleSourceWiringTests(_Base):
    def test_source_names_match_the_request_parameter(self):
        sources = factory.build_schedule_sources(self.transport)
        self.assertEqual(set(sources), {'500', 'okooo'})

    def test_each_source_hits_its_own_site(self):
        """两个源接反了不会报错，只会安静地拿另一家的数据。"""
        sources = factory.build_schedule_sources(self.transport)
        sources['500']('2026-08-27')
        self.assertTrue(any('500.com' in u for u in self.requested), self.requested)

        self.requested.clear()
        sources['okooo']('2026-08-27')
        self.assertTrue(any('okooo.com' in u for u in self.requested), self.requested)


class PredictionServiceWiringTests(_Base):
    def test_builds_a_service_with_every_port_filled(self):
        service = factory.build_prediction_service(
            self.db, cache=Cache(l1=MemoryBackend(), l2=MemoryBackend()),
            transport=self.transport, recorder=mock.Mock())
        self.assertIsInstance(service, PredictionService)
        self.assertIsInstance(service._movement_provider, MovementMapBuilder)
        self.assertEqual(set(service._schedules), {'500', 'okooo'})
        self.assertIsNotNone(service._cache)

    def test_ttl_defaults_to_the_service_default(self):
        from src.domain.sports.basketball.prediction import DEFAULT_TTL

        service = factory.build_prediction_service(self.db, transport=self.transport)
        self.assertEqual(service._ttl, DEFAULT_TTL)

    def test_ttl_can_be_overridden(self):
        service = factory.build_prediction_service(
            self.db, transport=self.transport, ttl=30)
        self.assertEqual(service._ttl, 30)

    def test_generates_a_payload_end_to_end_without_network(self):
        """空页面 + 空库：整条链路要走通并给出空结果，而不是抛异常。"""
        service = factory.build_prediction_service(
            self.db, transport=self.transport, recorder=None)
        payload = service.generate(date='2026-08-27')
        self.assertEqual(payload['count'], 0)
        self.assertEqual(payload['date'], '2026-08-27')

    def test_cache_makes_the_second_call_free(self):
        service = factory.build_prediction_service(
            self.db, cache=Cache(l1=MemoryBackend(), l2=MemoryBackend()),
            transport=self.transport)
        service.generate(date='2026-08-27')
        before = len(self.requested)
        service.generate(date='2026-08-27')
        self.assertEqual(len(self.requested), before, '第二次请求又去抓了一遍')


class TrackerWiringTests(_Base):
    def test_tracker_uses_the_500_source(self):
        """采集只针对 500 源：澳客自带盘路历史，不需要我们攒。"""
        tracker = factory.build_odds_tracker(self.db, transport=self.transport)
        self.assertIsInstance(tracker, OddsTracker)
        tracker.track('2026-08-27')
        self.assertTrue(all('500.com' in u for u in self.requested), self.requested)

    def test_history_store_reads_the_same_table_the_tracker_writes(self):
        tracker = factory.build_odds_tracker(self.db, transport=self.transport)
        tracker._store.save({'m': [{'ts': '2026-08-26T09:00:00', 'spf_home': 1.8}]})
        self.assertIn('m', factory.build_odds_history_store(self.db).load())


class RecorderWiringTests(_Base):
    """结算的意义就是回喂 Elo 与校准器——接不上它们，记录就只是个日志。"""

    def test_recorder_is_wired_with_elo_and_calibrator(self):
        recorder = factory.build_recorder(self.db)
        self.assertIsNotNone(recorder._elo)
        self.assertIsNotNone(recorder._calibrator)

    def test_recorder_is_absent_without_a_database(self):
        self.assertIsNone(factory.build_recorder(None))

    def test_prediction_service_gets_a_recorder_by_default(self):
        service = factory.build_prediction_service(self.db, transport=self.transport)
        self.assertIsNotNone(service._recorder)

    def test_explicit_recorder_wins(self):
        sentinel = mock.Mock()
        service = factory.build_prediction_service(
            self.db, transport=self.transport, recorder=sentinel)
        self.assertIs(service._recorder, sentinel)

    def test_settlement_reaches_the_calibrator_end_to_end(self):
        """一条完整链路：记录 → 结算 → 校准器落库。中间断在哪一环都测不出来，
        除非真的走一遍。"""
        from src.domain.sports.basketball.calibration_store import CalibrationStore

        recorder = factory.build_recorder(self.db)
        recorder.save('2026-08-27', [{
            'match': {'id': 'm1', 'home': '甲', 'away': '乙', 'league': 'NBA'},
            'spf': {'available': True, 'recommendation': '主胜', 'playable': True,
                    'home_prob': 0.62, 'away_prob': 0.38, 'confidence': 'high'},
            'rqspf': None, 'dx': None,
        }], 'v1')
        outcome = recorder.settle('m1', 110, 90, 'NBA')
        self.assertTrue(outcome['ok'])
        self.assertEqual(outcome['calibration_samples'], 1)
        self.assertTrue(CalibrationStore(self.db).load(), '校准样本没落库')


class MovementProviderWiringTests(_Base):
    """走势构建器与采集器必须共用同一份快照仓储。各拿一份不会报错，
    只会让「刚采到的快照」在同一次请求里读不出来。"""

    MATCH = {'id': '2026-08-27_甲_乙', 'home': '甲', 'away': '乙',
             'league': 'NBA', 'source': '500'}

    def test_provider_reads_the_snapshots_it_just_captured(self):
        factory.build_odds_history_store(self.db).save({
            self.MATCH['id']: [
                {'ts': '2026-08-26T09:00:00', 'spf_home': 2.00, 'spf_away': 1.80},
                {'ts': '2026-08-26T11:30:00', 'spf_home': 1.70, 'spf_away': 2.10},
            ]})
        provider = factory.build_movement_provider(
            self.db, self.transport, now_fn=lambda: NOW)
        result = provider([dict(self.MATCH)], '500', '2026-08-27')
        movement = result[self.MATCH['id']]['spf']
        self.assertIsNotNone(movement, '走势构建器没读到快照仓储里的数据')
        self.assertEqual(movement['side'], 'home')


class DatabaseUnavailableTests(_Base):
    """MySQL 连不上时必须降级，不能让端点整个失败。

    迁移前 kv_store 在这种情况下会退回 JSON 文件——「少一点信息，
    但仍然出结果」。这条降级路径不能在迁移中悄悄丢掉。
    """

    def test_analyzer_degrades_to_market_only(self):
        result = factory.build_analyzer(None).analyze_spf(
            {'home': 'A', 'away': 'B', 'league': 'NBA',
             'spf_home': 1.8, 'spf_away': 2.0})
        self.assertTrue(result['available'])
        self.assertEqual(result['elo_trust'], 0.0)

    def test_movement_provider_degrades_to_schedule_trends_only(self):
        provider = factory.build_movement_provider(None, self.transport,
                                                    now_fn=lambda: NOW)
        result = provider([{'id': 'm', 'home': 'A', 'away': 'B', 'source': '500'}],
                          '500', '2026-08-27')
        self.assertEqual(result['m'], {'spf': None, 'rqspf': None, 'dx': None})

    def test_prediction_service_still_produces_a_payload(self):
        service = factory.build_prediction_service(None, transport=self.transport)
        payload = service.generate(date='2026-08-27')
        self.assertEqual(payload['count'], 0)

    def test_odds_history_store_is_absent_rather_than_broken(self):
        self.assertIsNone(factory.build_odds_history_store(None))

    def test_tracker_is_absent_rather_than_broken(self):
        """采集的全部意义就是落盘。给它一个空仓储只会在真正写入时才炸，
        那时错误信息离原因已经很远了。"""
        self.assertIsNone(factory.build_odds_tracker(None))


class TransportWiringTests(unittest.TestCase):
    def test_dispatches_by_hostname(self):
        okooo_calls, default_calls = [], []
        transport = __import__(
            'src.domain.sports.basketball.fetching', fromlist=['x']
        ).dispatch_transport(
            okooo=lambda url, timeout: okooo_calls.append(url) or 'ok',
            default=lambda url, timeout: default_calls.append(url) or 'default')
        transport('https://www.okooo.com/a', 5)
        transport('https://trade.500.com/b', 5)
        self.assertEqual(len(okooo_calls), 1)
        self.assertEqual(len(default_calls), 1)

    def test_build_transport_returns_a_callable_get(self):
        self.assertTrue(callable(factory.build_transport()))

    def test_build_transport_dispatches_okooo_to_its_own_implementation(self):
        """okooo 要 Session 预热与 gb2312 解码，用普通 urllib 打过去只会
        拿到 WAF 页或乱码——而且不会报错。"""
        from src.domain.sports.basketball import fetching

        okooo_urls, default_urls = [], []

        class _FakeOkoooTransport:
            def __call__(self, url, timeout):
                okooo_urls.append(url)
                return 'okooo'

        with mock.patch.object(fetching, 'OkoooTransport', _FakeOkoooTransport), \
             mock.patch.object(fetching, 'urllib_get',
                               lambda url, timeout: default_urls.append(url) or 'd'):
            get = factory.build_transport()
            get('https://www.okooo.com/jingcailanqiu/hunhe/')
            get('https://trade.500.com/jclq/')

        self.assertEqual(len(okooo_urls), 1, '澳客的请求没走专用实现')
        self.assertEqual(len(default_urls), 1)


if __name__ == '__main__':
    unittest.main()
