from src.api.services import beidan as service
import logging
"""北单缓存：键构造、刷新档位，以及「有缓存绝不阻塞请求」这条硬约束

线上曾因整页重算 160 秒超过网关超时而 504：缓存按最早开赛场次判过期，
傍晚 TTL 缩到 2 分钟，等于每次打开都同步重算。这里锁定修复后的行为。
"""

import threading
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

# 业务逻辑已迁至 `src.api.services.beidan`，新旧入口共用一份（判据 11）。
# patch 要打在它现在住的地方——打在旧模块上不会报错，只是什么也没替换掉。
import src.api.services.beidan as beidan_api
import src.api.runtime.beidan_cache as beidan_cache
from src.foundation.cache import Cache, MemoryBackend
from src.api.runtime.beidan_cache import (
    beidan_cache_key, beidan_earliest_kickoff, beidan_refresh_after, prune_beidan_payload,
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


class BeidanPrunePayloadTests(unittest.TestCase):
    """history_summary.latest 是 30 条完整历史记录，占整份响应四成以上，前端从不读"""

    def test_drops_unused_history_latest(self):
        result = {'history_summary': {'latest': [{'x': 1}] * 30, 'total_records': 200,
                                      'quality_levels': {'strong': 3}}}
        prune_beidan_payload(result)
        self.assertNotIn('latest', result['history_summary'])

    def test_keeps_the_counters_the_frontend_renders(self):
        result = {'history_summary': {'latest': [1], 'total_records': 200,
                                      'settled_records': 10, 'pending_records': 5,
                                      'quality_levels': {'strong': 3}}}
        prune_beidan_payload(result)
        hs = result['history_summary']
        self.assertEqual(hs['total_records'], 200)
        self.assertEqual(hs['settled_records'], 10)
        self.assertEqual(hs['pending_records'], 5)
        self.assertEqual(hs['quality_levels'], {'strong': 3})

    def test_leaves_recommendations_untouched(self):
        recs = [{'match_id': '1', 'spf': {'analysis': 'x'}}]
        result = {'recommendations': recs, 'history_summary': {'latest': [1]}}
        prune_beidan_payload(result)
        self.assertEqual(result['recommendations'], recs)

    def test_tolerates_missing_or_odd_history_summary(self):
        for payload in ({}, {'history_summary': None}, {'history_summary': 'x'},
                        {'history_summary': {}}):
            with self.subTest(payload=payload):
                prune_beidan_payload(dict(payload))

    def test_write_cache_prunes_before_storing(self):
        """预热线程与接口计算都经由 write_beidan_cache，剔除必须发生在落盘前"""
        result = {'history_summary': {'latest': [1] * 30}, 'recommendations': []}
        cache = Cache(l1=MemoryBackend(), l2=MemoryBackend(), default_ttl=60)
        with patch.object(beidan_cache, 'get_shared_cache', return_value=cache):
            beidan_cache.write_beidan_cache('k', result)
            stored, _ = beidan_cache.read_beidan_cache('k')
        self.assertNotIn('latest', stored['history_summary'])


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

    def test_error_result_is_not_written_to_cache(self):
        """冷启动现在完全依赖后台计算，它把错误结果写进缓存的话，
        坏数据会一直顶在那儿直到下一轮刷新。"""
        with patch.object(beidan_cache, 'write_beidan_cache') as writer:
            beidan_cache.refresh_beidan_async('err', lambda: {'error': '未获取到比赛数据'})
            for _ in range(50):
                if 'err' not in beidan_cache._refreshing:
                    break
                time.sleep(0.1)
        writer.assert_not_called()

    def test_successful_refresh_writes_cache(self):
        with patch.object(beidan_cache, 'write_beidan_cache') as writer:
            beidan_cache.refresh_beidan_async('ok', lambda: {'recommendations': [{'match_id': '1'}]})
            for _ in range(50):
                if 'ok' not in beidan_cache._refreshing:
                    break
                time.sleep(0.1)
        writer.assert_called_once()

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
             patch.object(beidan_api, 'refresh_beidan_async', return_value=True) as bg, \
             patch.object(beidan_api, 'finalize_beidan_recs') as finalize, \
             patch.object(beidan_api, '_load_beidan_helpers',
                          return_value=(fake_generate, None, None)):
            payload = service.beidan_payload(params)
        self.finalize = finalize
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
             patch.object(beidan_api, 'refresh_beidan_async', return_value=False), \
             patch.object(beidan_api, 'finalize_beidan_recs'), \
             patch.object(beidan_api, '_load_beidan_helpers',
                          return_value=(lambda **kw: {}, None, None)):
            payload = service.beidan_payload(params)
        self.assertTrue(payload['result']['refreshing'])

    def test_cache_hit_does_not_run_persist_or_report_side_effects(self):
        """事故回归：落盘 340 个 JSON + 整批报告生成此前挂在每次请求上，
        北单变快后被反复触发，把服务器 CPU 与磁盘打满到 SSH 都连不上。
        读缓存必须完全不碰这些副作用。"""
        self._run(cached={'recommendations': [{'match_id': '1'}], 'tag': 'x'}, fresh=True)
        self.finalize.assert_not_called()

    def test_stale_cache_hit_also_skips_side_effects(self):
        self._run(cached={'recommendations': [{'match_id': '1'}]}, fresh=False)
        self.finalize.assert_not_called()

    def test_cold_start_defers_side_effects_to_the_background(self):
        """冷启动的落盘/报告生成也应发生在后台计算里，不在请求线程"""
        self._run(cached=None, fresh=False)
        self.finalize.assert_not_called()

    def test_cold_start_does_not_block_either(self):
        """没有任何缓存时也不能同步硬算：整页重算一两分钟，一样会撞网关超时。
        改为转后台算并立刻回 computing，由前端轮询。"""
        payload, sync_calls, bg = self._run(cached=None, fresh=False)
        self.assertEqual(sync_calls, [], '冷启动不得在请求线程里同步重算')
        bg.assert_called_once()
        self.assertTrue(payload['result']['computing'])
        self.assertTrue(payload['result']['refreshing'])
        self.assertEqual(payload['result']['recommendations'], [])

    def test_cold_start_response_carries_query_context(self):
        """计算中的占位结果也要带上 date/source，前端才能正确显示上下文"""
        payload, _, _ = self._run(cached=None, fresh=False)
        self.assertEqual(payload['result']['source'], 'okooo')
        self.assertIn('date', payload['result'])

    def test_no_request_path_ever_computes_synchronously(self):
        """把三条路径一起锁死：无论有无缓存、是否强制刷新，
        请求线程都不许跑重算——这正是先前 504 的成因。"""
        for cached, fresh, force in (
            (None, False, False),
            ({'recommendations': []}, False, False),
            ({'recommendations': []}, True, True),
            ({'recommendations': []}, True, False),
        ):
            with self.subTest(cached=bool(cached), fresh=fresh, force=force):
                _, sync_calls, _ = self._run(cached=cached, fresh=fresh, force_refresh=force)
                self.assertEqual(sync_calls, [])


if __name__ == '__main__':
    unittest.main()
