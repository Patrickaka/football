import json
import logging
import time

from .backend import Entry

log = logging.getLogger('foundation.cache')

# Redis 物理 TTL 相对逻辑 TTL 的延长倍数。SWR 需要在逻辑过期后
# 仍能读到陈旧值用于兜底，故物理保留更久。
_STALE_GRACE_FACTOR = 10
_MIN_PHYSICAL_TTL = 300


class RedisBackend:
    """Redis 缓存后端。仅作缓存不作真相源，丢失不影响正确性。"""

    def __init__(self, client, prefix='fb:'):
        self.client = client
        self.prefix = prefix

    def get(self, key):
        raw = self.client.get(self._k(key))
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
            return Entry(
                value=payload['value'],
                stored_at=float(payload['stored_at']),
                ttl=float(payload['ttl']),
            )
        except (ValueError, KeyError, TypeError):
            log.warning('缓存条目损坏，按未命中处理: key=%s', key)
            return None

    def set(self, key, value, ttl, now=None):
        now = time.time() if now is None else now
        payload = json.dumps(
            {'value': value, 'stored_at': now, 'ttl': ttl}, ensure_ascii=False
        )
        physical = max(int(ttl) * _STALE_GRACE_FACTOR, _MIN_PHYSICAL_TTL)
        self.client.set(self._k(key), payload, ex=physical)

    def delete(self, key):
        self.client.delete(self._k(key))

    def lock(self, key, timeout):
        return bool(self.client.set(self._lock_k(key), '1', nx=True, ex=int(timeout)))

    def unlock(self, key):
        """释放锁。不做所有权校验，调用方必须保证只对自己成功 lock() 的
        key 调用 unlock()。这是 Cache 门面 SWR 跨线程释放锁所必需——
        主线程 lock、后台线程完成刷新后 unlock，若加 token 或线程绑定的
        所有权校验会直接破坏这个流程。与 MemoryBackend 语义一致。
        """
        self.client.delete(self._lock_k(key))

    def _k(self, key):
        return f'{self.prefix}{key}'

    def _lock_k(self, key):
        return f'{self.prefix}lock:{key}'
