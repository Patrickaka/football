"""kl8 预测缓存改用 foundation/cache。

**问题**：`_PERSIST_KEYS` 只含 3d 系列，kl8 的缓存纯进程内存——每次部署重启
都清零，用户在发版之后的第一个请求要等 5.6 秒（线上实测冷启动 3.5~5.6s，
命中 0.01s）。

**做法**：把失效条件编进缓存 key（最新期号 + 预测器版本），而不是读出来
再判断。新开奖或版本变更自然产生新 key，旧值随 TTL 自行淘汰——比「读回来
逐字段校验版本，不符就丢掉」少一整条容易出错的路径。
"""
import unittest
from unittest import mock

from src.foundation.cache import Cache, MemoryBackend
from src.webapp import kl8_cache


class CacheKeyTests(unittest.TestCase):
    """key 必须同时区分期号与版本——少任何一个都会让旧结果活过它该活的时候。"""

    def test_includes_issue_and_version(self):
        key = kl8_cache.cache_key('2026227', 'v9.3')
        self.assertIn('2026227', key)
        self.assertIn('v9.3', key)

    def test_new_issue_gives_a_new_key(self):
        self.assertNotEqual(kl8_cache.cache_key('2026227', 'v9.3'),
                            kl8_cache.cache_key('2026228', 'v9.3'))

    def test_new_version_gives_a_new_key(self):
        self.assertNotEqual(kl8_cache.cache_key('2026227', 'v9.3'),
                            kl8_cache.cache_key('2026227', 'v9.4'))

    def test_is_namespaced(self):
        """与其它业务共用一个 Redis，前缀不能省。"""
        self.assertTrue(kl8_cache.cache_key('1', 'v').startswith('kl8:'))


class PredictTests(unittest.TestCase):
    def setUp(self):
        self.cache = Cache(l1=MemoryBackend(), l2=MemoryBackend(), default_ttl=60)
        self.calls = []

    def _predict(self, issue='2026227', version='v9.3', compute=None, **kwargs):
        def default_compute():
            self.calls.append(1)
            return {'based_on_issue': issue, 'statistics': {'version': version}}

        return kl8_cache.predict(
            compute_fn=compute or default_compute,
            latest_issue=issue, version=version, cache=self.cache, **kwargs)

    def test_computes_once_then_hits(self):
        first = self._predict()
        second = self._predict()
        self.assertEqual(first, second)
        self.assertEqual(len(self.calls), 1)

    def test_new_issue_recomputes(self):
        """新开奖之后必须重算，否则会一直返回上一期的预测。"""
        self._predict(issue='2026227')
        self._predict(issue='2026228')
        self.assertEqual(len(self.calls), 2)

    def test_new_version_recomputes(self):
        self._predict(version='v9.3')
        self._predict(version='v9.4')
        self.assertEqual(len(self.calls), 2)

    def test_concurrent_cold_start_computes_once(self):
        """冷启动 5.6 秒，并发请求各算一遍是这个端点历史上最贵的一次事故。"""
        import threading
        import time

        barrier = threading.Barrier(5)

        def slow():
            self.calls.append(1)
            time.sleep(0.3)
            return {'ok': True}

        results = []
        lock = threading.Lock()

        def worker():
            barrier.wait()
            value = self._predict(compute=slow)
            with lock:
                results.append(value)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start

        self.assertEqual(len(self.calls), 1, '并发冷启动重复计算了')
        self.assertEqual(len(results), 5)
        self.assertLess(elapsed, 0.9, f'总墙钟 {elapsed:.2f}s 接近串行')

    def test_without_a_cache_every_call_computes(self):
        kl8_cache.predict(compute_fn=lambda: self.calls.append(1) or {'a': 1},
                          latest_issue='1', version='v', cache=None)
        kl8_cache.predict(compute_fn=lambda: self.calls.append(1) or {'a': 1},
                          latest_issue='1', version='v', cache=None)
        self.assertEqual(len(self.calls), 2)

    def test_missing_issue_skips_the_cache(self):
        """期号取不到时不能缓存——否则会把「历史还没加载出来」时算出的结果
        当成某一期的预测存下来，而它根本不对应任何一期。"""
        kl8_cache.predict(compute_fn=lambda: self.calls.append(1) or {'a': 1},
                          latest_issue='', version='v', cache=self.cache)
        kl8_cache.predict(compute_fn=lambda: self.calls.append(1) or {'a': 1},
                          latest_issue='', version='v', cache=self.cache)
        self.assertEqual(len(self.calls), 2)

    def test_payload_is_json_serialisable(self):
        """生产态 L2 是 Redis，不可序列化的值会被静默丢弃，
        表现是「缓存接上了但永远不命中」。"""
        import json

        payload = self._predict()
        self.assertEqual(json.loads(json.dumps(payload)), payload)


class SharedCacheTests(unittest.TestCase):
    def setUp(self):
        from src.webapp import shared_cache

        shared_cache.reset()
        self.addCleanup(shared_cache.reset)

    def test_is_a_process_singleton(self):
        from src.webapp import shared_cache

        self.assertIs(shared_cache.get_cache(), shared_cache.get_cache())

    def test_failure_degrades_to_none(self):
        from src.webapp import shared_cache

        with mock.patch('src.api.deps.build_cache', side_effect=RuntimeError('炸了')):
            shared_cache.reset()
            self.assertIsNone(shared_cache.get_cache())


if __name__ == '__main__':
    unittest.main()


class EndpointWiringTests(unittest.TestCase):
    """端点接线：算错 key 不会报错，只会安静地返回上一期的预测。"""

    def setUp(self):
        import logging

        from src.webapp.kl8_api import KL8ApiMixin

        class _Handler(KL8ApiMixin):
            def __init__(self):
                self._log = logging.getLogger('test.kl8')

        self.handler = _Handler()

    def _with_analyzer(self, history):
        analyzer = mock.Mock()
        analyzer.history_data = history
        return mock.patch('src.webapp.kl8_api.get_kl8_analyzer', lambda: analyzer)

    def test_latest_issue_comes_from_the_analyzer(self):
        with self._with_analyzer([{'issue': '2026227'}, {'issue': '2026226'}]):
            self.assertEqual(self.handler._kl8_latest_issue(), '2026227')

    def test_empty_history_yields_no_issue(self):
        with self._with_analyzer([]):
            self.assertEqual(self.handler._kl8_latest_issue(), '')

    def test_analyzer_failure_yields_no_issue(self):
        """取不到期号就绕过缓存，而不是让端点失败。"""
        with mock.patch('src.webapp.kl8_api.get_kl8_analyzer',
                        side_effect=RuntimeError('历史没加载')):
            self.assertEqual(self.handler._kl8_latest_issue(), '')

    def test_payload_caches_by_issue(self):
        calls = []
        cache = Cache(l1=MemoryBackend(), l2=MemoryBackend(), default_ttl=60)
        with self._with_analyzer([{'issue': '2026227'}]), \
             mock.patch('src.webapp.kl8_api.get_shared_cache', lambda: cache), \
             mock.patch('src.webapp.kl8_api._current_kl8_predictor_version',
                        lambda: 'v9.3'), \
             mock.patch('src.webapp.kl8_api.kl8_run_prediction',
                        lambda force_refresh=False: calls.append(1) or {'ok': True}):
            first = self.handler._kl8_payload()
            second = self.handler._kl8_payload()
        self.assertEqual(first, {'result': {'ok': True}})
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)

    def test_compute_failure_is_reported_not_raised(self):
        cache = Cache(l1=MemoryBackend(), l2=MemoryBackend(), default_ttl=60)
        with self._with_analyzer([{'issue': '2026227'}]), \
             mock.patch('src.webapp.kl8_api.get_shared_cache', lambda: cache), \
             mock.patch('src.webapp.kl8_api._current_kl8_predictor_version',
                        lambda: 'v9.3'), \
             mock.patch('src.webapp.kl8_api.kl8_run_prediction',
                        lambda force_refresh=False: {'error': '数据不足'}):
            self.assertIn('error', self.handler._kl8_payload())
