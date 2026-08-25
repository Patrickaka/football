import json
import unittest
from unittest import mock

from src.foundation.cache.redis_backend import RedisBackend


class RedisBackendTests(unittest.TestCase):
    def setUp(self):
        self.client = mock.MagicMock()
        self.backend = RedisBackend(self.client, prefix='t:')

    def test_get_missing_returns_none(self):
        self.client.get.return_value = None
        self.assertIsNone(self.backend.get('k'))
        self.client.get.assert_called_once_with('t:v:k')

    def test_set_serialises_entry_as_json(self):
        self.backend.set('k', {'a': 1}, ttl=60, now=100.0)
        args, kwargs = self.client.set.call_args
        self.assertEqual(args[0], 't:v:k')
        payload = json.loads(args[1])
        self.assertEqual(payload, {'value': {'a': 1}, 'stored_at': 100.0, 'ttl': 60})

    def test_set_ttl_exceeds_entry_ttl_so_stale_survives(self):
        """Redis 的物理 TTL 必须长于逻辑 TTL，否则 SWR 拿不到陈旧数据。"""
        self.backend.set('k', 'v', ttl=60, now=100.0)
        _, kwargs = self.client.set.call_args
        self.assertGreater(kwargs['ex'], 60)

    def test_get_deserialises_entry(self):
        self.client.get.return_value = json.dumps(
            {'value': [1, 2], 'stored_at': 100.0, 'ttl': 60}
        )
        entry = self.backend.get('k')
        self.assertEqual(entry.value, [1, 2])
        self.assertEqual(entry.stored_at, 100.0)
        self.assertTrue(entry.is_fresh(now=120.0))

    def test_get_returns_none_on_corrupt_payload(self):
        self.client.get.return_value = 'not-json'
        self.assertIsNone(self.backend.get('k'))

    def test_delete_uses_prefix(self):
        self.backend.delete('k')
        self.client.delete.assert_called_once_with('t:v:k')

    def test_lock_uses_setnx_with_expiry(self):
        self.client.set.return_value = True
        self.assertTrue(self.backend.lock('k', timeout=30))
        args, kwargs = self.client.set.call_args
        self.assertEqual(args[0], 't:lock:k')
        self.assertTrue(kwargs['nx'])
        self.assertEqual(kwargs['ex'], 30)

    def test_lock_returns_false_when_held(self):
        self.client.set.return_value = None
        self.assertFalse(self.backend.lock('k', timeout=30))

    def test_unlock_deletes_lock_key(self):
        self.backend.unlock('k')
        self.client.delete.assert_called_once_with('t:lock:k')

    def test_data_and_lock_namespaces_never_collide(self):
        for key in ['foo', 'lock:foo', 'v:foo', '', 'a:b:c']:
            self.assertNotEqual(self.backend._k(key), self.backend._lock_k(key))
        # 交叉碰撞：任意 key 的数据键不得等于任意其他 key 的锁键
        self.assertNotEqual(self.backend._k('lock:foo'), self.backend._lock_k('foo'))


class RedisBackendRuntimeFaultTests(unittest.TestCase):
    """回归测试：运行期 Redis 抖动/重启不应打穿到调用方——各方法必须
    降级而不是让底层异常冒泡。启动时不可用已有防护（build_cache），
    这里覆盖的是运行中途才出现故障的场景。
    """

    def setUp(self):
        self.client = mock.MagicMock()
        self.backend = RedisBackend(self.client, prefix='t:')

    def test_get_swallows_client_exception_and_treated_as_miss(self):
        self.client.get.side_effect = ConnectionError('redis down')
        self.assertIsNone(self.backend.get('k'))

    def test_set_swallows_client_exception_silently(self):
        self.client.set.side_effect = ConnectionError('redis down')
        self.backend.set('k', 'v', ttl=60, now=100.0)  # 不应抛出

    def test_set_swallows_non_json_serialisable_value(self):
        """np.int64/datetime 等非 JSON 可序列化的值会让 json.dumps 抛
        TypeError；这属于 set 内部的失败，同样必须被吞掉。
        """

        class Unserialisable:
            pass

        self.backend.set('k', Unserialisable(), ttl=60, now=100.0)  # 不应抛出
        self.client.set.assert_not_called()

    def test_delete_swallows_client_exception_silently(self):
        self.client.delete.side_effect = ConnectionError('redis down')
        self.backend.delete('k')  # 不应抛出

    def test_lock_swallows_client_exception_and_returns_false(self):
        self.client.set.side_effect = ConnectionError('redis down')
        self.assertFalse(self.backend.lock('k', timeout=30))

    def test_unlock_swallows_client_exception_silently(self):
        self.client.delete.side_effect = ConnectionError('redis down')
        self.backend.unlock('k')  # 不应抛出


if __name__ == '__main__':
    unittest.main()
