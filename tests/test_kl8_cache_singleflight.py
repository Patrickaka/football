"""快乐8缓存的单飞与 SWR 保护。

线上实测 /api/kl8 平均 6.05 秒：缓存过期时并发请求各自重算一遍。
同文件的 _serve_cached 早已实现单飞 + stale-while-revalidate（3d/3d_ml 都在用），
kl8 却手搓 _CACHE 字典绕过了它。

期号保护那条是正确性护栏：接入 _serve_cached 时若把 based_on_issue 检查弄丢，
新开奖后会继续返回旧预测。
"""
import logging
import threading
import time
import unittest
from unittest.mock import patch

from src.webapp import caching as webapp_caching
from src.webapp.kl8_api import KL8ApiMixin


class _StubHandler(KL8ApiMixin):
    def __init__(self):
        self._log = logging.getLogger('kl8-cache-test')


class _StubAnalyzer:
    def __init__(self, issue):
        self.history_data = [{'issue': issue}] if issue else []


def _result(issue, version='v-test'):
    return {
        'based_on_issue': issue,
        'statistics': {'version': version},
        'recommendations': [],
    }


class KL8CacheSingleFlightTests(unittest.TestCase):
    def setUp(self):
        self.entry = webapp_caching._CACHE['kl8']
        saved = dict(self.entry)
        self.addCleanup(lambda: self.entry.update(saved))
        self.entry['data'] = None
        self.entry['timestamp'] = 0
        self.handler = _StubHandler()

    def _patches(self, predict, issue):
        return (
            patch('src.webapp.kl8_api.kl8_run_prediction', predict),
            patch.object(webapp_caching, 'get_kl8_analyzer',
                         return_value=_StubAnalyzer(issue)),
            patch.object(webapp_caching, '_current_kl8_predictor_version',
                         return_value='v-test'),
        )

    def test_concurrent_cold_requests_compute_only_once(self):
        """并发冷启动只允许算一次——这是 6.05 秒惊群的直接来源。"""
        calls = []
        barrier = threading.Barrier(5)

        def slow_predict(force_refresh=False):
            calls.append(1)
            time.sleep(0.2)
            return _result('2026001')

        results = []

        def worker():
            barrier.wait()
            results.append(self.handler._kl8_payload())

        p1, p2, p3 = self._patches(slow_predict, '2026001')
        with p1, p2, p3:
            threads = [threading.Thread(target=worker) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.assertEqual(len(calls), 1,
                         f'并发冷启动应只计算一次，实际 {len(calls)} 次')
        self.assertEqual(len(results), 5)
        for payload in results:
            self.assertIn('result', payload)
            self.assertEqual(payload['result']['based_on_issue'], '2026001')

    def test_stale_cache_returns_old_value_immediately(self):
        """缓存陈旧时先返回旧值、后台刷新，请求线程不承担重算。"""
        self.entry['data'] = _result('2026001')
        self.entry['timestamp'] = time.time() - 90000  # 超 TTL 且跨天

        def slow_predict(force_refresh=False):
            time.sleep(0.5)
            return _result('2026001')

        p1, p2, p3 = self._patches(slow_predict, '2026001')
        with p1, p2, p3:
            started = time.time()
            payload = self.handler._kl8_payload()
            elapsed = time.time() - started

        self.assertLess(elapsed, 0.2,
                        f'陈旧缓存应立即返回旧值，实际耗时 {elapsed:.2f}s')
        self.assertEqual(payload['result']['based_on_issue'], '2026001')

    def test_new_issue_invalidates_cache(self):
        """正确性护栏：新开奖后必须重算，不得返回旧期号的预测。"""
        self.entry['data'] = _result('2026001')
        self.entry['timestamp'] = time.time()

        def predict(force_refresh=False):
            return _result('2026002')

        p1, p2, p3 = self._patches(predict, '2026002')
        with p1, p2, p3:
            payload = self.handler._kl8_payload()

        self.assertEqual(payload['result']['based_on_issue'], '2026002')

    def test_predictor_version_change_invalidates_cache(self):
        """正确性护栏：预测器版本变化后不得复用旧结果。"""
        self.entry['data'] = _result('2026001', version='v-old')
        self.entry['timestamp'] = time.time()

        def predict(force_refresh=False):
            return _result('2026001', version='v-test')

        p1, p2, p3 = self._patches(predict, '2026001')
        with p1, p2, p3:
            payload = self.handler._kl8_payload()

        self.assertEqual(payload['result']['statistics']['version'], 'v-test')

    def test_prediction_error_is_not_cached(self):
        """计算失败不得污染缓存，下次请求应重新尝试。"""
        def failing_predict(force_refresh=False):
            return {'error': '数据源不可用'}

        p1, p2, p3 = self._patches(failing_predict, '2026001')
        with p1, p2, p3:
            payload = self.handler._kl8_payload()

        self.assertIn('error', payload)
        self.assertIsNone(self.entry['data'])


if __name__ == '__main__':
    unittest.main()
