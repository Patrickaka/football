"""L2 故障与停机等待的降级行为。

两条都是最终整分支审查发现的：
- L2 持续不可用时，lock() 恒返回 False，没有线程能成为 winner，所有请求
  都轮询到 deadline 才本地计算，且回退路径不写缓存——于是每次请求都要
  重挨一遍完整等待。修 Redis 容错反而把"快速失败"变成了"慢速失败"。
- wait_for_refreshes 逐个 join 且每个套用完整 timeout，N 个卡住的刷新会把
  停机耗时线性放大到 N × timeout，可能超过编排的 SIGTERM 宽限期被 SIGKILL，
  那样恰好复现它本要避免的残留锁。
"""
import threading
import time
import unittest

from src.foundation.cache.backend import MemoryBackend
from src.foundation.cache.cache import Cache


class _DeadL2(MemoryBackend):
    """模拟 Redis 持续不可用：读写静默失效、锁永远拿不到。

    与 RedisBackend 的故障降级行为一致（get→None、set→no-op、lock→False）。
    """

    def get(self, key):
        return None

    def set(self, key, value, ttl, now=None):
        return None

    def delete(self, key):
        return None

    def lock(self, key, timeout):
        return False

    def unlock(self, key):
        return None


class L2OutageTests(unittest.TestCase):
    def setUp(self):
        self.l1 = MemoryBackend()
        self.calls = []

    def _cache(self, **kwargs):
        return Cache(l1=self.l1, l2=_DeadL2(), default_ttl=60, **kwargs)

    def _compute(self, value='computed'):
        def _fn():
            self.calls.append(value)
            return value
        return _fn

    def test_outage_wait_is_not_bound_to_lock_timeout(self):
        """等待超时不得套用 lock_timeout——那是"锁最长持有多久"，不是"等多久"。"""
        cache = self._cache(lock_timeout=30, wait_timeout=0.3)
        started = time.time()
        value = cache.get('k', self._compute())
        elapsed = time.time() - started

        self.assertEqual(value, 'computed')
        self.assertLess(elapsed, 2.0,
                        f'L2 故障时不应等满 lock_timeout，实际 {elapsed:.2f}s')

    def test_outage_result_is_written_to_l1(self):
        """回退计算的结果必须落进 L1，否则每次请求都要重挨一遍等待。"""
        cache = self._cache(lock_timeout=30, wait_timeout=0.2)
        cache.get('k', self._compute())

        entry = self.l1.get('k')
        self.assertIsNotNone(entry, '回退路径必须写入 L1')
        self.assertEqual(entry.value, 'computed')

    def test_second_request_hits_l1_without_waiting(self):
        """第二次请求应直接命中 L1，不再等待、不再重算。"""
        cache = self._cache(lock_timeout=30, wait_timeout=0.2)
        cache.get('k', self._compute())

        started = time.time()
        value = cache.get('k', self._compute('recomputed'))
        elapsed = time.time() - started

        self.assertEqual(value, 'computed')
        self.assertEqual(len(self.calls), 1,
                         f'第二次请求不应重算，compute 共调用 {len(self.calls)} 次')
        self.assertLess(elapsed, 0.1,
                        f'第二次请求应立即命中 L1，实际 {elapsed:.2f}s')

    def test_invalidate_during_outage_still_prevents_revival(self):
        """L2 故障下的回退路径同样要守纪元校验，不能让失效的缓存复活。"""
        cache = self._cache(lock_timeout=30, wait_timeout=0.2)
        started = threading.Event()
        release = threading.Event()

        def slow():
            started.set()
            release.wait(timeout=5)
            return 'stale-value'

        holder = {}

        def worker():
            holder['value'] = cache.get('k', slow)

        t = threading.Thread(target=worker)
        t.start()
        self.assertTrue(started.wait(timeout=5))
        cache.invalidate('k')
        release.set()
        t.join(timeout=5)

        self.assertEqual(holder['value'], 'stale-value', '调用方仍应拿到算出的值')
        self.assertIsNone(self.l1.get('k'), '失效期间算出的值不得写回 L1')


class WaitForRefreshesTests(unittest.TestCase):
    def test_multiple_stuck_refreshes_share_one_deadline(self):
        """N 个卡住的刷新线程必须共享一个 deadline，不能线性放大停机耗时。"""
        cache = Cache(l1=MemoryBackend(), l2=MemoryBackend(), default_ttl=60)
        release = threading.Event()
        self.addCleanup(release.set)

        def stuck():
            release.wait(timeout=30)

        threads = []
        for i in range(4):
            t = threading.Thread(target=stuck, name=f'stuck-{i}', daemon=True)
            t.start()
            threads.append(t)
        with cache._refresh_guard:
            cache._refresh_threads.extend(threads)

        started = time.time()
        cache.wait_for_refreshes(timeout=0.5)
        elapsed = time.time() - started

        self.assertLess(elapsed, 1.5,
                        f'4 个卡住线程应共享 0.5s deadline，实际耗时 {elapsed:.2f}s')

    def test_wait_returns_promptly_when_nothing_running(self):
        cache = Cache(l1=MemoryBackend(), l2=MemoryBackend(), default_ttl=60)
        started = time.time()
        cache.wait_for_refreshes(timeout=5)
        self.assertLess(time.time() - started, 0.1)


if __name__ == '__main__':
    unittest.main()
