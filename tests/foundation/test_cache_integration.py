"""Cache + RedisBackend 组合路径的集成测试。

`test_cache.py` 全用 MemoryBackend 当 L1+L2，`test_cache_redis_backend.py`
全用 mock 且从不经过 Cache 门面——两者组合起来的路径（唯一的生产拓扑）
此前一个测试都没有。本文件用一个约 10 行的 dict-backed 假 redis client
驱动 `Cache(l1=MemoryBackend(), l2=RedisBackend(fake))` 跑一遍
get / 单飞 / SWR / invalidate，并覆盖 Redis 运行期故障与序列化边界。
"""
import threading
import time
import unittest

from src.foundation.cache.backend import MemoryBackend
from src.foundation.cache.cache import Cache
from src.foundation.cache.redis_backend import RedisBackend


class FakeRedisClient:
    """dict-backed 假 redis client：仅实现 Cache 组合路径需要的最小接口。"""

    def __init__(self):
        self._store = {}

    def get(self, key):
        self._expire(key)
        item = self._store.get(key)
        return item[0] if item else None

    def set(self, key, value, nx=False, ex=None):
        self._expire(key)
        if nx and key in self._store:
            return None
        expire_at = time.time() + ex if ex else None
        self._store[key] = (value, expire_at)
        return True

    def delete(self, key):
        self._store.pop(key, None)

    def _expire(self, key):
        item = self._store.get(key)
        if item and item[1] is not None and time.time() >= item[1]:
            del self._store[key]


def make_cache(client=None, **cache_kwargs):
    client = client if client is not None else FakeRedisClient()
    cache = Cache(l1=MemoryBackend(), l2=RedisBackend(client), default_ttl=60, **cache_kwargs)
    return cache, client


class CacheRedisBackendIntegrationTests(unittest.TestCase):
    """生产拓扑（L1=MemoryBackend, L2=RedisBackend）的端到端行为。"""

    def test_miss_computes_and_persists_across_both_layers(self):
        cache, client = make_cache()
        calls = []

        def compute():
            calls.append(1)
            return 'computed'

        self.assertEqual(cache.get('k', compute), 'computed')
        self.assertEqual(cache.get('k', compute), 'computed')
        self.assertEqual(len(calls), 1)
        # 真正落到了 L2（假 redis）里，不是只停在 L1。
        self.assertIsNotNone(client.get('fb:v:k'))

    def test_l1_eviction_falls_back_to_l2_without_recompute(self):
        cache, client = make_cache()
        calls = []

        def compute():
            calls.append(1)
            return 'computed'

        cache.get('k', compute)
        cache.l1.clear()
        self.assertEqual(cache.get('k', compute), 'computed')
        self.assertEqual(len(calls), 1)

    def test_single_flight_on_cold_miss_across_redis_backend(self):
        cache, client = make_cache()
        calls = []

        def slow():
            calls.append(1)
            time.sleep(0.2)
            return 'value'

        barrier = threading.Barrier(5)
        results = []

        def worker():
            barrier.wait()
            results.append(cache.get('k', slow))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(results, ['value'] * 5)
        self.assertEqual(len(calls), 1)

    def test_swr_returns_stale_then_refreshes_eventually(self):
        cache, client = make_cache()
        cache.set('k', 'old', ttl=1)
        time.sleep(1.1)

        value = cache.get('k', lambda: 'new')
        self.assertEqual(value, 'old')

        cache.wait_for_refreshes(timeout=5)
        self.assertEqual(cache.get('k', lambda: 'unused'), 'new')

    def test_invalidate_clears_both_layers(self):
        cache, client = make_cache()
        cache.get('k', lambda: 'computed')
        cache.invalidate('k')
        self.assertIsNone(cache.l1.get('k'))
        self.assertIsNone(cache.l2.get('k'))
        self.assertIsNone(client.get('fb:v:k'))


class CacheRedisBackendFaultToleranceTests(unittest.TestCase):
    """Important-2 的两个场景，必须经由 Cache 门面 + 真实 RedisBackend 覆盖，
    而不是孤立地测 RedisBackend 或孤立地测 Cache。
    """

    def test_redis_transient_failure_degrades_to_l1_without_raising(self):
        client = FakeRedisClient()
        cache, _ = make_cache(client=client)

        real_set = client.set
        call_count = {'n': 0}

        def flaky_set(key, value, nx=False, ex=None):
            call_count['n'] += 1
            if call_count['n'] == 1:
                raise ConnectionError('redis 抖动')
            return real_set(key, value, nx=nx, ex=ex)

        client.set = flaky_set

        # 第一次 set 恰好撞上 redis 抖动：不应抛出，且 L1 必须仍被写入。
        cache.set('k', 'v', ttl=60)
        entry = cache.l1.get('k')
        self.assertIsNotNone(entry)
        self.assertEqual(entry.value, 'v')

        # 后续请求不再重算（L1 兜住了），服务不因为 Redis 一次抖动而 500。
        calls = []
        self.assertEqual(cache.get('k', lambda: calls.append(1) or 'recomputed'), 'v')
        self.assertEqual(calls, [])

    def test_non_json_serialisable_value_degrades_gracefully(self):
        client = FakeRedisClient()
        cache, _ = make_cache(client=client)

        class Unserialisable:
            pass

        value = Unserialisable()
        cache.set('k', value, ttl=60)  # 不应抛出 TypeError

        # L1 仍然拿到了这个值（Python 对象本身不需要序列化）。
        entry = cache.l1.get('k')
        self.assertIsNotNone(entry)
        self.assertIs(entry.value, value)

        # L2 没有写入任何东西——序列化失败被吞掉，而不是让 key 半写。
        self.assertIsNone(client.get('fb:v:k'))


if __name__ == '__main__':
    unittest.main()
