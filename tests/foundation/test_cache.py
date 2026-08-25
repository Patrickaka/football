import threading
import time
import unittest

from src.foundation.cache.backend import MemoryBackend
from src.foundation.cache.cache import Cache


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.l1 = MemoryBackend()
        self.l2 = MemoryBackend()
        self.cache = Cache(l1=self.l1, l2=self.l2, default_ttl=60)
        self.calls = []

    def compute(self, value='computed'):
        def _fn():
            self.calls.append(value)
            return value
        return _fn

    def test_miss_computes_and_returns(self):
        self.assertEqual(self.cache.get('k', self.compute()), 'computed')
        self.assertEqual(self.calls, ['computed'])

    def test_hit_does_not_recompute(self):
        self.cache.get('k', self.compute())
        self.cache.get('k', self.compute())
        self.assertEqual(len(self.calls), 1)

    def test_l1_miss_falls_back_to_l2_without_recompute(self):
        self.cache.get('k', self.compute())
        self.l1.clear()
        self.assertEqual(self.cache.get('k', self.compute()), 'computed')
        self.assertEqual(len(self.calls), 1)

    def test_l2_hit_repopulates_l1(self):
        self.cache.get('k', self.compute())
        self.l1.clear()
        self.cache.get('k', self.compute())
        self.assertIsNotNone(self.l1.get('k'))

    def test_invalidate_clears_both_layers(self):
        """旧实现分层清理导致连清三次才生效，此处必须一次贯穿。"""
        self.cache.get('k', self.compute())
        self.cache.invalidate('k')
        self.assertIsNone(self.l1.get('k'))
        self.assertIsNone(self.l2.get('k'))

    def test_invalidate_causes_recompute(self):
        self.cache.get('k', self.compute())
        self.cache.invalidate('k')
        self.cache.get('k', self.compute())
        self.assertEqual(len(self.calls), 2)

    def test_stale_value_returned_immediately(self):
        """SWR：过期后先返回陈旧值，不让请求线程等待重算。"""
        self.cache.set('k', 'old', ttl=1)
        time.sleep(1.1)
        slow_calls = []

        def slow():
            slow_calls.append(1)
            time.sleep(0.3)
            return 'new'

        started = time.time()
        value = self.cache.get('k', slow)
        elapsed = time.time() - started
        self.assertEqual(value, 'old')
        self.assertLess(elapsed, 0.2)

    def test_stale_refresh_eventually_updates(self):
        self.cache.set('k', 'old', ttl=1)
        time.sleep(1.1)
        self.cache.get('k', self.compute('new'))
        self.cache.wait_for_refreshes(timeout=5)
        self.assertEqual(self.cache.get('k', self.compute('unused')), 'new')

    def test_single_flight_on_cold_miss(self):
        """并发冷启动只允许一次计算，其余等待复用结果。"""
        barrier = threading.Barrier(5)
        results = []

        def slow():
            self.calls.append('x')
            time.sleep(0.2)
            return 'value'

        def worker():
            barrier.wait()
            results.append(self.cache.get('k', slow))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(results, ['value'] * 5)
        self.assertEqual(len(self.calls), 1)

    def test_compute_error_propagates_and_releases_lock(self):
        def boom():
            raise RuntimeError('compute failed')

        with self.assertRaises(RuntimeError):
            self.cache.get('k', boom)
        self.assertEqual(self.cache.get('k', self.compute()), 'computed')

    def test_explicit_ttl_overrides_default(self):
        self.cache.get('k', self.compute(), ttl=1)
        entry = self.l2.get('k')
        self.assertEqual(entry.ttl, 1)

    def test_single_flight_recovers_after_winner_failure(self):
        """winner 计算失败释放锁后，等待线程必须立刻接手，而不是等满 lock_timeout。"""
        cache = Cache(l1=MemoryBackend(), l2=MemoryBackend(), default_ttl=60, lock_timeout=1)
        call_count = []
        call_lock = threading.Lock()

        def flaky():
            with call_lock:
                call_count.append(1)
                is_first_call = len(call_count) == 1
            if is_first_call:
                raise RuntimeError('winner compute failed')
            return 'value'

        barrier = threading.Barrier(6)
        elapsed_list = []
        elapsed_lock = threading.Lock()

        def worker():
            barrier.wait()
            started = time.time()
            try:
                cache.get('k', flaky)
            except RuntimeError:
                pass
            elapsed = time.time() - started
            with elapsed_lock:
                elapsed_list.append(elapsed)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertLess(len(call_count), 6, call_count)
        self.assertTrue(all(e < 0.5 for e in elapsed_list), elapsed_list)
        self.assertEqual(cache.get('k', lambda: 'should-not-be-called'), 'value')

    def test_l2_hit_repopulates_l1_preserves_staleness(self):
        """L2 命中回填 L1 时必须沿用原 stored_at/ttl，不能让陈旧数据被误判新鲜。

        对应简报点名的行为：`self.l1.set(key, entry.value, entry.ttl, now=entry.stored_at)`。
        若误写成 now=now，本用例应失败（已做变异验证，见 task-9-report.md）。
        """
        stale_stored_at = time.time() - 100
        self.l2.set('k', 'stale-value', ttl=1, now=stale_stored_at)
        self.l1.clear()

        def slow_refresh():
            time.sleep(0.1)
            return 'refreshed-value'

        value = self.cache.get('k', slow_refresh)

        self.assertEqual(value, 'stale-value')
        repopulated = self.l1.get('k')
        self.assertIsNotNone(repopulated)
        self.assertEqual(repopulated.stored_at, stale_stored_at)
        self.assertFalse(repopulated.is_fresh())

        self.cache.wait_for_refreshes(timeout=2)

    def test_invalidate_during_background_refresh_does_not_resurrect(self):
        """SWR 后台刷新进行中调用 invalidate，刷新完成后不应让缓存自己复活。"""
        self.cache.set('k', 'old', ttl=1)
        time.sleep(1.1)

        refresh_started = threading.Event()
        proceed = threading.Event()

        def slow_refresh():
            refresh_started.set()
            proceed.wait(timeout=2)
            return 'new'

        self.cache.get('k', slow_refresh)
        self.assertTrue(refresh_started.wait(timeout=2))

        self.cache.invalidate('k')
        proceed.set()

        self.cache.wait_for_refreshes(timeout=5)

        self.assertIsNone(self.l1.get('k'))
        self.assertIsNone(self.l2.get('k'))


if __name__ == '__main__':
    unittest.main()
