"""北单缓存：键构造、刷新档位，以及「有缓存绝不阻塞请求」这条硬约束

线上曾因整页重算 160 秒超过网关超时而 504：缓存按最早开赛场次判过期，
傍晚 TTL 缩到 2 分钟，等于每次打开都同步重算。这里锁定修复后的行为。
"""

import threading
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import server
import src.webapp.beidan_api as beidan_api
import src.webapp.beidan_cache as beidan_cache
from src.webapp.beidan_cache import (
    beidan_cache_key, beidan_earliest_kickoff, beidan_refresh_after,
)


def _rec(offset_minutes):
    when = datetime.now() + timedelta(minutes=offset_minutes)
    return {'date': when.strftime('%Y-%m-%d'), 'time': when.strftime('%H:%M')}


class BeidanCacheKeyTests(unittest.TestCase):
    def test_key_is_stable_regardless_of_bet_type_order(self):
        self.assertEqual(beidan_cache_key('2026-08-21', 'okooo', ['zjq', 'spf', 'rqspf']),
                         beidan_cache_key('2026-08-21', 'okooo', ['spf', 'rqspf', 'zjq']))

    def test_blank_date_and_none_share_the_today_key(self):
        """预热用 None、接口层用 ''，必须落在同一个键上，否则预热热不到点上"""
        self.assertEqual(beidan_cache_key(None, 'okooo', ['spf']),
                         beidan_cache_key('', 'okooo', ['spf']))

    def test_source_and_date_separate_keys(self):
        self.assertNotEqual(beidan_cache_key('2026-08-21', 'okooo', ['spf']),
                            beidan_cache_key('2026-08-21', 'jczq', ['spf']))
        self.assertNotEqual(beidan_cache_key('2026-08-21', 'okooo', ['spf']),
                            beidan_cache_key('2026-08-22', 'okooo', ['spf']))


class BeidanEarliestKickoffTests(unittest.TestCase):
    def test_picks_the_soonest_future_match(self):
        expected = (datetime.now() + timedelta(minutes=90)).strftime('%Y-%m-%d %H:%M')
        self.assertEqual(beidan_earliest_kickoff(
            {'recommendations': [_rec(600), _rec(90), _rec(300)]}), expected)

    def test_ignores_already_started_matches(self):
        expected = (datetime.now() + timedelta(minutes=240)).strftime('%Y-%m-%d %H:%M')
        self.assertEqual(beidan_earliest_kickoff(
            {'recommendations': [_rec(-120), _rec(-10), _rec(240)]}), expected)

    def test_returns_none_when_nothing_upcoming(self):
        self.assertIsNone(beidan_earliest_kickoff({'recommendations': [_rec(-30)]}))
        self.assertIsNone(beidan_earliest_kickoff({}))


class BeidanRefreshAfterTests(unittest.TestCase):
    def test_closer_kickoff_refreshes_more_often(self):
        imminent = beidan_refresh_after({'recommendations': [_rec(5)]})
        soon = beidan_refresh_after({'recommendations': [_rec(45)]})
        later = beidan_refresh_after({'recommendations': [_rec(120)]})
        distant = beidan_refresh_after({'recommendations': [_rec(600)]})
        self.assertLess(imminent, soon)
        self.assertLess(soon, later)
        self.assertLess(later, distant)

    def test_all_tiers_stay_well_above_zero(self):
        """刷新档位只决定后台节奏，任何一档都不该退化成「每次请求都刷」"""
        for offset in (1, 30, 120, 600):
            self.assertGreaterEqual(beidan_refresh_after({'recommendations': [_rec(offset)]}), 60)


class BeidanSingleFlightTests(unittest.TestCase):
    def setUp(self):
        beidan_cache._refreshing.clear()

    def tearDown(self):
        beidan_cache._refreshing.clear()

    def test_second_refresh_is_rejected_while_first_runs(self):
        release = threading.Event()
        started = threading.Event()

        def slow():
            started.set()
            release.wait(5)
            return {'recommendations': []}

        with patch.object(beidan_cache, 'write_beidan_cache'):
            self.assertTrue(beidan_cache.refresh_beidan_async('k', slow))
            started.wait(5)
            self.assertFalse(beidan_cache.refresh_beidan_async('k', slow),
                             '同键刷新进行中时不应再起一轮')
            release.set()
            for _ in range(50):
                if 'k' not in beidan_cache._refreshing:
                    break
                time.sleep(0.1)
        self.assertNotIn('k', beidan_cache._refreshing, '刷新结束后必须释放闸门')

    def test_gate_is_released_even_when_refresh_raises(self):
        with patch.object(beidan_cache, 'write_beidan_cache'):
            beidan_cache.refresh_beidan_async('boom', lambda: (_ for _ in ()).throw(RuntimeError('x')))
            for _ in range(50):
                if 'boom' not in beidan_cache._refreshing:
                    break
                time.sleep(0.1)
        self.assertNotIn('boom', beidan_cache._refreshing)


class BeidanPayloadNeverBlocksTests(unittest.TestCase):
    def setUp(self):
        self.handler = server.Handler.__new__(server.Handler)
        self.handler._log = server.log
        self.params = {'source': ['okooo'], 'types': ['spf,rqspf,zjq']}

    def _run(self, cached, fresh, force_refresh=False):
        params = dict(self.params)
        if force_refresh:
            params['force_refresh'] = ['true']
        sync_calls = []

        def fake_generate(date=None, bet_types=None, source=None):
            sync_calls.append(source)
            return {'recommendations': [], 'source': source}

        with patch.object(beidan_api, 'read_beidan_cache', return_value=(cached, fresh)), \
             patch.object(beidan_api, 'write_beidan_cache'), \
             patch.object(beidan_api, 'refresh_beidan_async', return_value=True) as bg, \
             patch.object(beidan_api, '_load_beidan_helpers',
                          return_value=(fake_generate, None, None)), \
             patch.object(beidan_api, '_BAYES_REPORT_AVAILABLE', False), \
             patch.object(beidan_api, '_attach_bayes_report_url'), \
             patch.object(beidan_api, '_trigger_beidan_report_sync'):
            payload = self.handler._beidan_payload(params)
        return payload, sync_calls, bg

    def test_stale_cache_is_served_without_recomputing(self):
        """核心回归：缓存过期也必须立刻返回，重算只能进后台，否则网关 504"""
        payload, sync_calls, bg = self._run(cached={'recommendations': [], 'tag': 'old'},
                                            fresh=False)
        self.assertEqual(sync_calls, [], '过期缓存不得触发同步重算')
        self.assertEqual(payload['result']['tag'], 'old')
        bg.assert_called_once()
        self.assertTrue(payload['result']['refreshing'])

    def test_fresh_cache_does_not_trigger_refresh(self):
        payload, sync_calls, bg = self._run(cached={'recommendations': [], 'tag': 'new'},
                                            fresh=True)
        self.assertEqual(sync_calls, [])
        bg.assert_not_called()
        self.assertNotIn('refreshing', payload['result'])

    def test_force_refresh_returns_at_once_and_refreshes_in_background(self):
        """刷新按钮同样不能阻塞：整页重算远超网关超时"""
        payload, sync_calls, bg = self._run(cached={'recommendations': [], 'tag': 'old'},
                                            fresh=True, force_refresh=True)
        self.assertEqual(sync_calls, [], 'force_refresh 也不得同步重算')
        bg.assert_called_once()
        self.assertTrue(payload['result']['refreshing'])

    def test_reports_refreshing_even_when_joining_an_inflight_refresh(self):
        """已有同键刷新在跑时单飞会返回 False，但数据确实在刷新，
        仍须报 refreshing，否则前端不会回来取更新后的数据。"""
        params = dict(self.params)
        params['force_refresh'] = ['true']
        with patch.object(beidan_api, 'read_beidan_cache',
                          return_value=({'recommendations': [], 'tag': 'old'}, True)), \
             patch.object(beidan_api, 'write_beidan_cache'), \
             patch.object(beidan_api, 'refresh_beidan_async', return_value=False), \
             patch.object(beidan_api, '_load_beidan_helpers',
                          return_value=(lambda **kw: {}, None, None)), \
             patch.object(beidan_api, '_BAYES_REPORT_AVAILABLE', False), \
             patch.object(beidan_api, '_attach_bayes_report_url'), \
             patch.object(beidan_api, '_trigger_beidan_report_sync'):
            payload = self.handler._beidan_payload(params)
        self.assertTrue(payload['result']['refreshing'])

    def test_cold_start_computes_synchronously(self):
        """从来没算过时别无选择，只能同步算这一次"""
        payload, sync_calls, bg = self._run(cached=None, fresh=False)
        self.assertEqual(sync_calls, ['okooo'])
        bg.assert_not_called()

    def test_error_on_cold_start_is_not_cached(self):
        def failing(date=None, bet_types=None, source=None):
            return {'error': '未获取到比赛数据'}

        with patch.object(beidan_api, 'read_beidan_cache', return_value=(None, False)), \
             patch.object(beidan_api, 'write_beidan_cache') as writer, \
             patch.object(beidan_api, '_load_beidan_helpers',
                          return_value=(failing, None, None)):
            payload = self.handler._beidan_payload(dict(self.params))

        self.assertIn('error', payload)
        writer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
