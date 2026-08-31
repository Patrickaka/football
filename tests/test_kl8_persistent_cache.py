"""kl8 预测缓存改用 foundation/cache。

**问题**：`_PERSIST_KEYS` 只含 3d 系列，kl8 的缓存纯进程内存——每次部署重启
都清零，用户在发版之后的第一个请求要等 5.6 秒（线上实测冷启动 3.5~5.6s，
命中 0.01s）。

**做法**：把失效条件编进缓存 key（最新期号 + 预测器版本），而不是读出来
再判断。新开奖或版本变更自然产生新 key，旧值随 TTL 自行淘汰——比「读回来
逐字段校验版本，不符就丢掉」少一整条容易出错的路径。
"""
from src.api.services import kl8 as service
from src.kl8 import snapshots as prediction_snapshots
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.foundation.cache import Cache, MemoryBackend
from src.api.runtime import kl8_cache


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
        from src.api.runtime import shared_cache

        shared_cache.reset()
        self.addCleanup(shared_cache.reset)

    def test_is_a_process_singleton(self):
        from src.api.runtime import shared_cache

        self.assertIs(shared_cache.get_cache(), shared_cache.get_cache())

    def test_app_can_install_its_existing_cache(self):
        from src.api.runtime import shared_cache

        cache = Cache(l1=MemoryBackend(), l2=MemoryBackend(), default_ttl=60)
        shared_cache.set_cache(cache)
        self.assertIs(shared_cache.get_cache(), cache)

    def test_failure_degrades_to_none(self):
        from src.api.runtime import shared_cache

        with mock.patch('src.api.deps.build_cache', side_effect=RuntimeError('炸了')):
            shared_cache.reset()
            self.assertIsNone(shared_cache.get_cache())


class PredictionDiskCacheTests(unittest.TestCase):
    """Redis 不可用时，预测结果也要跨服务重启复用。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_cache_file = prediction_snapshots._PREDICTION_CACHE_FILE
        self.original_memory_cache = prediction_snapshots._prediction_cache
        prediction_snapshots._PREDICTION_CACHE_FILE = (
            Path(self.temp_dir.name) / 'kl8_prediction_cache.json'
        )
        prediction_snapshots._prediction_cache = {
            'data': None, 'timestamp': 0, 'cache_key': None,
        }

    def tearDown(self):
        prediction_snapshots._PREDICTION_CACHE_FILE = self.original_cache_file
        prediction_snapshots._prediction_cache = self.original_memory_cache
        self.temp_dir.cleanup()

    @staticmethod
    def _analyzer(issue, result):
        analyzer = mock.Mock()
        analyzer.history_data = [{
            'issue': issue,
            'numbers': list(range(1, 21)),
        }]
        analyzer.reload_if_needed.return_value = False
        analyzer.predict_all.return_value = result
        return analyzer

    def test_process_restart_reads_disk_without_initializing_analyzer(self):
        result = {
            'based_on_issue': '2026227',
            'statistics': {'version': prediction_snapshots.KL8_PREDICTOR_VERSION},
        }
        analyzer = self._analyzer('2026227', result)
        with mock.patch.object(prediction_snapshots, '_history_signature',
                               return_value=[123, 456]), \
             mock.patch.object(prediction_snapshots, 'get_kl8_analyzer',
                               return_value=analyzer):
            first = prediction_snapshots.run_prediction(force_refresh=True)

        # 模拟服务重启：进程内对象全部丢失，只保留 data 下的磁盘缓存。
        prediction_snapshots._prediction_cache = {
            'data': None, 'timestamp': 0, 'cache_key': None,
        }
        with mock.patch.object(prediction_snapshots, '_history_signature',
                               return_value=[123, 456]), \
             mock.patch.object(prediction_snapshots, 'get_kl8_analyzer',
                               side_effect=AssertionError('不应初始化分析器')):
            second = prediction_snapshots.run_prediction()

        self.assertEqual(second, first)
        analyzer.predict_all.assert_called_once_with()

    def test_history_change_invalidates_disk_cache(self):
        first_result = {'based_on_issue': '2026227'}
        first_analyzer = self._analyzer('2026227', first_result)
        with mock.patch.object(prediction_snapshots, '_history_signature',
                               return_value=[123, 456]), \
             mock.patch.object(prediction_snapshots, 'get_kl8_analyzer',
                               return_value=first_analyzer):
            prediction_snapshots.run_prediction(force_refresh=True)

        prediction_snapshots._prediction_cache = {
            'data': None, 'timestamp': 0, 'cache_key': None,
        }
        second_result = {'based_on_issue': '2026228'}
        second_analyzer = self._analyzer('2026228', second_result)
        with mock.patch.object(prediction_snapshots, '_history_signature',
                               return_value=[124, 500]), \
             mock.patch.object(prediction_snapshots, 'get_kl8_analyzer',
                               return_value=second_analyzer):
            actual = prediction_snapshots.run_prediction()

        self.assertEqual(actual, second_result)
        second_analyzer.predict_all.assert_called_once_with()

    def test_clear_cache_removes_disk_copy(self):
        prediction_snapshots._PREDICTION_CACHE_FILE.write_text(
            json.dumps({'result': {'ok': True}}), encoding='utf-8')
        prediction_snapshots.clear_cache()
        self.assertFalse(prediction_snapshots._PREDICTION_CACHE_FILE.exists())


if __name__ == '__main__':
    unittest.main()


class EndpointWiringTests(unittest.TestCase):
    """端点接线：算错 key 不会报错，只会安静地返回上一期的预测。"""


    def _with_analyzer(self, history):
        analyzer = mock.Mock()
        analyzer.history_data = history
        return mock.patch('src.api.services.kl8.get_kl8_analyzer', lambda: analyzer)

    def test_latest_issue_comes_from_the_analyzer(self):
        with mock.patch.object(service, '_latest_issue_from_history_file',
                               return_value=''), \
             self._with_analyzer([{'issue': '2026227'}, {'issue': '2026226'}]):
            self.assertEqual(service.kl8_latest_issue(), '2026227')

    def test_empty_history_yields_no_issue(self):
        with mock.patch.object(service, '_latest_issue_from_history_file',
                               return_value=''), self._with_analyzer([]):
            self.assertEqual(service.kl8_latest_issue(), '')

    def test_analyzer_failure_yields_no_issue(self):
        """取不到期号就绕过缓存，而不是让端点失败。"""
        with mock.patch.object(service, '_latest_issue_from_history_file',
                               return_value=''), \
             mock.patch('src.api.services.kl8.get_kl8_analyzer',
                        side_effect=RuntimeError('历史没加载')):
            self.assertEqual(service.kl8_latest_issue(), '')

    def test_history_file_fast_path_skips_analyzer(self):
        with mock.patch.object(service, '_latest_issue_from_history_file',
                               return_value='2026228'), \
             mock.patch('src.api.services.kl8.get_kl8_analyzer',
                        side_effect=AssertionError('不应初始化分析器')):
            self.assertEqual(service.kl8_latest_issue(), '2026228')

    def test_history_file_reader_does_not_depend_on_record_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = Path(temp_dir) / 'kl8_history.json'
            history_file.write_text(json.dumps({'results': [
                {'issue': '2026226'},
                {'issue': '2026228'},
                {'issue': '2026227'},
            ]}), encoding='utf-8')
            with mock.patch.object(service, 'data_path', return_value=str(history_file)):
                self.assertEqual(service._latest_issue_from_history_file(), '2026228')

    def test_payload_caches_by_issue(self):
        calls = []
        cache = Cache(l1=MemoryBackend(), l2=MemoryBackend(), default_ttl=60)
        with self._with_analyzer([{'issue': '2026227'}]), \
             mock.patch('src.api.services.kl8.get_shared_cache', lambda: cache), \
             mock.patch('src.api.services.kl8._current_kl8_predictor_version',
                        lambda: 'v9.3'), \
             mock.patch('src.api.services.kl8.kl8_run_prediction',
                        lambda force_refresh=False: calls.append(1) or {'ok': True}):
            first = service.kl8_payload()
            second = service.kl8_payload()
        self.assertEqual(first, {'result': {'ok': True}})
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)

    def test_compute_failure_is_reported_not_raised(self):
        cache = Cache(l1=MemoryBackend(), l2=MemoryBackend(), default_ttl=60)
        with self._with_analyzer([{'issue': '2026227'}]), \
             mock.patch('src.api.services.kl8.get_shared_cache', lambda: cache), \
             mock.patch('src.api.services.kl8._current_kl8_predictor_version',
                        lambda: 'v9.3'), \
             mock.patch('src.api.services.kl8.kl8_run_prediction',
                        lambda force_refresh=False: {'error': '数据不足'}):
            self.assertIn('error', service.kl8_payload())
