import threading
import time
import unittest

from src.foundation.cache.backend import Entry, MemoryBackend


class EntryTests(unittest.TestCase):
    def test_fresh_within_ttl(self):
        entry = Entry(value='v', stored_at=100.0, ttl=60)
        self.assertTrue(entry.is_fresh(now=150.0))

    def test_stale_after_ttl(self):
        entry = Entry(value='v', stored_at=100.0, ttl=60)
        self.assertFalse(entry.is_fresh(now=161.0))

    def test_zero_ttl_never_fresh(self):
        entry = Entry(value='v', stored_at=100.0, ttl=0)
        self.assertFalse(entry.is_fresh(now=100.0))


class MemoryBackendTests(unittest.TestCase):
    def setUp(self):
        self.backend = MemoryBackend()

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.backend.get('absent'))

    def test_set_then_get_roundtrips_entry(self):
        self.backend.set('k', {'a': 1}, ttl=60, now=100.0)
        entry = self.backend.get('k')
        self.assertEqual(entry.value, {'a': 1})
        self.assertEqual(entry.ttl, 60)
        self.assertEqual(entry.stored_at, 100.0)

    def test_get_returns_stale_entry_rather_than_none(self):
        """SWR 需要拿到过期数据，因此过期不等于不可见。"""
        self.backend.set('k', 'v', ttl=1, now=100.0)
        entry = self.backend.get('k')
        self.assertIsNotNone(entry)
        self.assertFalse(entry.is_fresh(now=200.0))

    def test_delete_removes_key(self):
        self.backend.set('k', 'v', ttl=60, now=100.0)
        self.backend.delete('k')
        self.assertIsNone(self.backend.get('k'))

    def test_delete_absent_key_is_noop(self):
        self.backend.delete('absent')

    def test_lock_is_exclusive(self):
        acquired = []

        def worker():
            got = self.backend.lock('k', timeout=0.05)
            acquired.append(got)
            if got:
                self.backend.unlock('k')

        with self.backend_locked('k'):
            t = threading.Thread(target=worker)
            t.start()
            t.join()
        self.assertEqual(acquired, [False])

    def backend_locked(self, key):
        backend = self.backend

        class _Ctx:
            def __enter__(self):
                assert backend.lock(key, timeout=1)

            def __exit__(self, *exc):
                backend.unlock(key)

        return _Ctx()

    def test_lock_does_not_block_when_held(self):
        """timeout 是锁的持有上限，不是等待时间。拿不到必须立即返回，
        否则 Cache 的单飞等待逻辑会被绕过。"""
        self.assertTrue(self.backend.lock('k', timeout=30))
        started = time.time()
        self.assertFalse(self.backend.lock('k', timeout=30))
        self.assertLess(time.time() - started, 0.1)
        self.backend.unlock('k')

    def test_lock_released_can_be_reacquired(self):
        self.assertTrue(self.backend.lock('k', timeout=1))
        self.backend.unlock('k')
        self.assertTrue(self.backend.lock('k', timeout=1))
        self.backend.unlock('k')


if __name__ == '__main__':
    unittest.main()
