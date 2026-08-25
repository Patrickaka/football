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
        self.client.get.assert_called_once_with('t:k')

    def test_set_serialises_entry_as_json(self):
        self.backend.set('k', {'a': 1}, ttl=60, now=100.0)
        args, kwargs = self.client.set.call_args
        self.assertEqual(args[0], 't:k')
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
        self.client.delete.assert_called_once_with('t:k')

    def test_lock_uses_setnx_with_expiry(self):
        self.client.set.return_value = True
        self.assertTrue(self.backend.lock('k', timeout=30))
        _, kwargs = self.client.set.call_args
        self.assertTrue(kwargs['nx'])
        self.assertEqual(kwargs['ex'], 30)

    def test_lock_returns_false_when_held(self):
        self.client.set.return_value = None
        self.assertFalse(self.backend.lock('k', timeout=30))

    def test_unlock_deletes_lock_key(self):
        self.backend.unlock('k')
        self.client.delete.assert_called_once_with('t:lock:k')


if __name__ == '__main__':
    unittest.main()
