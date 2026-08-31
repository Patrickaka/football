"""快乐8缓存的单飞、SWR 与失效护栏。

原本测的是 `webapp/caching.py` 里那套进程内字典（阶段 0 为消除 6.05 秒惊群
而接入的 `_serve_cached`）。端点改用 foundation/cache 之后，那套机制不再在
这条路径上，**但它守的性质一条不少**，所以整体移植而不是删除：

- 并发冷启动只算一次（6.05 秒惊群的直接来源）
- 陈旧缓存先返回旧值、后台刷新，请求线程不承担重算
- 新开奖后必须重算，不得返回旧期号的预测
- 预测器版本变化后不得复用旧结果
- 计算失败不得污染缓存

新旧实现在**怎么做到**上不同：旧的把结果读回来逐字段校验期号与版本，
不符就丢掉重算；新的把这两者编进 key，新开奖或版本变更自然产生新 key。
所以后两条护栏在新实现里换了断言方式——查的是「有没有重算」，
而不是「校验函数有没有被调用」。
"""
import logging
import threading
import time
import unittest
from unittest.mock import patch

from src.foundation.cache import Cache, MemoryBackend
from src.api.services import kl8 as service




class _StubAnalyzer:
    def __init__(self, issue):
        self.history_data = [{'issue': issue}] if issue else []


def _result(issue, version='v-test'):
    return {
        'based_on_issue': issue,
        'statistics': {'version': version},
        'recommendations': [],
    }


class _Base(unittest.TestCase):
    def setUp(self):
        # 每个用例一份独立缓存：共享单例会让上一个用例算出的值漏到下一个，
        # 表现为「明明打了桩却拿到别的结果」。
        self.cache = Cache(l1=MemoryBackend(), l2=MemoryBackend(),
                           default_ttl=86400)

    def _patches(self, predict, issue, version='v-test'):
        return (
            patch('src.api.services.kl8.kl8_run_prediction', predict),
            patch('src.api.services.kl8.kl8_latest_issue', return_value=issue),
            patch('src.api.services.kl8._current_kl8_predictor_version',
                  return_value=version),
            patch('src.api.services.kl8.get_shared_cache', lambda: self.cache),
        )

    def _payload(self, predict, issue, version='v-test'):
        p1, p2, p3, p4 = self._patches(predict, issue, version)
        with p1, p2, p3, p4:
            return service.kl8_payload()


class SingleFlightTests(_Base):
    def test_concurrent_cold_requests_compute_only_once(self):
        """并发冷启动只允许算一次——这是 6.05 秒惊群的直接来源。"""
        calls = []
        barrier = threading.Barrier(5)

        def slow_predict(force_refresh=False):
            calls.append(1)
            time.sleep(0.2)
            return _result('2026001')

        results = []
        p1, p2, p3, p4 = self._patches(slow_predict, '2026001')

        def worker():
            barrier.wait()
            results.append(service.kl8_payload())

        with p1, p2, p3, p4:
            threads = [threading.Thread(target=worker) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.assertEqual(len(calls), 1,
                         f'并发冷启动应只计算一次，实际 {len(calls)} 次')
        self.assertEqual(len(results), 5)
        for payload in results:
            self.assertEqual(payload['result']['based_on_issue'], '2026001')

    def test_second_request_hits_the_cache(self):
        calls = []

        def predict(force_refresh=False):
            calls.append(1)
            return _result('2026001')

        self._payload(predict, '2026001')
        self._payload(predict, '2026001')
        self.assertEqual(len(calls), 1)


class StaleWhileRevalidateTests(_Base):
    def test_stale_cache_returns_old_value_immediately(self):
        """缓存陈旧时先返回旧值、后台刷新，请求线程不承担重算。"""
        cache = Cache(l1=MemoryBackend(), l2=MemoryBackend(), default_ttl=0)
        self.cache = cache
        self._payload(lambda force_refresh=False: _result('2026001'), '2026001')

        def slow_predict(force_refresh=False):
            time.sleep(0.5)
            return _result('2026001')

        started = time.time()
        payload = self._payload(slow_predict, '2026001')
        elapsed = time.time() - started

        self.assertLess(elapsed, 0.2,
                        f'陈旧缓存应立即返回旧值，实际耗时 {elapsed:.2f}s')
        self.assertEqual(payload['result']['based_on_issue'], '2026001')


class InvalidationTests(_Base):
    """失效护栏。新实现把期号与版本编进 key，所以这里查的是「有没有重算」。"""

    def test_new_issue_invalidates_cache(self):
        """正确性护栏：新开奖后必须重算，不得返回旧期号的预测。"""
        calls = []

        def predict(force_refresh=False):
            calls.append(1)
            return _result(calls and '2026002' or '2026001')

        self._payload(lambda force_refresh=False: _result('2026001'), '2026001')
        payload = self._payload(lambda force_refresh=False: _result('2026002'),
                                '2026002')
        self.assertEqual(payload['result']['based_on_issue'], '2026002')

    def test_predictor_version_change_invalidates_cache(self):
        """正确性护栏：预测器版本变化后不得复用旧结果。"""
        self._payload(lambda force_refresh=False: _result('2026001', 'v-old'),
                      '2026001', version='v-old')
        payload = self._payload(
            lambda force_refresh=False: _result('2026001', 'v-test'),
            '2026001', version='v-test')
        self.assertEqual(payload['result']['statistics']['version'], 'v-test')

    def test_unknown_issue_bypasses_the_cache(self):
        """历史还没加载出来时算出的结果不对应任何一期，存下来只会污染
        下一个真正的请求。"""
        calls = []

        def predict(force_refresh=False):
            calls.append(1)
            return _result('')

        self._payload(predict, '')
        self._payload(predict, '')
        self.assertEqual(len(calls), 2)


class ErrorHandlingTests(_Base):
    def test_prediction_error_is_not_cached(self):
        """计算失败不得污染缓存，下次请求应重新尝试。"""
        calls = []

        def failing_predict(force_refresh=False):
            calls.append(1)
            return {'error': '数据源不可用'}

        first = self._payload(failing_predict, '2026001')
        self.assertIn('error', first)

        second = self._payload(lambda force_refresh=False: _result('2026001'),
                               '2026001')
        self.assertEqual(second['result']['based_on_issue'], '2026001',
                         '失败被写进了缓存')
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
