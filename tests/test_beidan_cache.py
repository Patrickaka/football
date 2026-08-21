"""北单结果级缓存：键构造、最早开赛时间推导、接口层命中与强制刷新"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import server
import src.webapp.beidan_api as beidan_api
from src.webapp.beidan_cache import beidan_cache_key, beidan_earliest_kickoff


def _rec(offset_minutes, **extra):
    when = datetime.now() + timedelta(minutes=offset_minutes)
    rec = {'date': when.strftime('%Y-%m-%d'), 'time': when.strftime('%H:%M')}
    rec.update(extra)
    return rec


class BeidanCacheKeyTests(unittest.TestCase):
    def test_key_is_stable_regardless_of_bet_type_order(self):
        self.assertEqual(beidan_cache_key('2026-08-21', 'okooo', ['zjq', 'spf', 'rqspf']),
                         beidan_cache_key('2026-08-21', 'okooo', ['spf', 'rqspf', 'zjq']))

    def test_blank_date_and_none_share_the_today_key(self):
        """预热用 None、接口层用 ''，两者必须落在同一个键上，否则预热热不到点上"""
        self.assertEqual(beidan_cache_key(None, 'okooo', ['spf']),
                         beidan_cache_key('', 'okooo', ['spf']))

    def test_source_and_date_separate_keys(self):
        self.assertNotEqual(beidan_cache_key('2026-08-21', 'okooo', ['spf']),
                            beidan_cache_key('2026-08-21', 'jczq', ['spf']))
        self.assertNotEqual(beidan_cache_key('2026-08-21', 'okooo', ['spf']),
                            beidan_cache_key('2026-08-22', 'okooo', ['spf']))


class BeidanEarliestKickoffTests(unittest.TestCase):
    def test_picks_the_soonest_future_match(self):
        result = {'recommendations': [_rec(600), _rec(90), _rec(300)]}
        expected = (datetime.now() + timedelta(minutes=90)).strftime('%Y-%m-%d %H:%M')
        self.assertEqual(beidan_earliest_kickoff(result), expected)

    def test_ignores_already_started_matches(self):
        """已开赛场次不该把 TTL 拖成最短档，否则整份缓存永远 2 分钟过期"""
        result = {'recommendations': [_rec(-120), _rec(-10), _rec(240)]}
        expected = (datetime.now() + timedelta(minutes=240)).strftime('%Y-%m-%d %H:%M')
        self.assertEqual(beidan_earliest_kickoff(result), expected)

    def test_returns_none_when_nothing_upcoming(self):
        self.assertIsNone(beidan_earliest_kickoff({'recommendations': [_rec(-30)]}))
        self.assertIsNone(beidan_earliest_kickoff({'recommendations': []}))
        self.assertIsNone(beidan_earliest_kickoff({}))

    def test_skips_malformed_timestamps(self):
        result = {'recommendations': [
            {'date': '', 'time': ''},
            {'date': '不是日期', 'time': '18:00'},
            _rec(180),
        ]}
        expected = (datetime.now() + timedelta(minutes=180)).strftime('%Y-%m-%d %H:%M')
        self.assertEqual(beidan_earliest_kickoff(result), expected)


class BeidanPayloadCacheTests(unittest.TestCase):
    def setUp(self):
        self.handler = server.Handler.__new__(server.Handler)
        self.handler._log = server.log
        self.params = {'source': ['okooo'], 'types': ['spf,rqspf,zjq']}

    def _run(self, cached, force_refresh=False):
        params = dict(self.params)
        if force_refresh:
            params['force_refresh'] = ['true']
        calls = []

        def fake_generate(date=None, bet_types=None, source=None):
            calls.append(source)
            return {'recommendations': [], 'source': source}

        with patch.object(beidan_api, 'read_beidan_cache', return_value=cached), \
             patch.object(beidan_api, 'write_beidan_cache') as writer, \
             patch.object(beidan_api, '_load_beidan_helpers',
                          return_value=(fake_generate, None, None)), \
             patch.object(beidan_api, '_BAYES_REPORT_AVAILABLE', False), \
             patch.object(beidan_api, '_attach_bayes_report_url'), \
             patch.object(beidan_api, '_trigger_beidan_report_sync'):
            payload = self.handler._beidan_payload(params)
        return payload, calls, writer

    def test_cache_hit_skips_recompute(self):
        payload, calls, writer = self._run(cached={'recommendations': [], 'cached': True})
        self.assertEqual(calls, [], '命中缓存时不应再调用重算')
        self.assertTrue(payload['result']['cached'])
        writer.assert_not_called()

    def test_cache_miss_computes_and_stores(self):
        payload, calls, writer = self._run(cached=None)
        self.assertEqual(calls, ['okooo'])
        writer.assert_called_once()
        self.assertEqual(payload['result']['source'], 'okooo')

    def test_force_refresh_bypasses_cache(self):
        payload, calls, writer = self._run(cached={'recommendations': [], 'cached': True},
                                           force_refresh=True)
        self.assertEqual(calls, ['okooo'], 'force_refresh 必须绕过缓存重算')
        self.assertNotIn('cached', payload['result'])
        writer.assert_called_once()

    def test_error_result_is_not_cached(self):
        def failing(date=None, bet_types=None, source=None):
            return {'error': '未获取到比赛数据'}

        with patch.object(beidan_api, 'read_beidan_cache', return_value=None), \
             patch.object(beidan_api, 'write_beidan_cache') as writer, \
             patch.object(beidan_api, '_load_beidan_helpers',
                          return_value=(failing, None, None)):
            payload = self.handler._beidan_payload(dict(self.params))

        self.assertIn('error', payload)
        writer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
