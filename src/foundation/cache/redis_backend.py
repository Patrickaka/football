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
    """Redis 缓存后端。仅作缓存不作真相源，丢失不影响正确性。

    契约：value 必须 JSON 可序列化（`json.dumps` 能处理的类型）。
    pandas/numpy/datetime 等类型需调用方自行转换，本类不做隐式转换。

    运行时故障防护：get/set/delete/lock/unlock 均对底层 client 调用做
    try/except——Redis 可能在服务运行期间抖动或重启（这台机器上配了
    maxmemory + save ""，重启是常态），任一次调用抛异常都只记
    log.warning 并按下列规则降级，绝不向上抛出：
    - get: 视为未命中，返回 None
    - set/delete/unlock: 静默 no-op
    - lock: 返回 False（拿不到锁，调用方走等待/本地计算路径，安全侧）
    """

    def __init__(self, client, prefix='fb:'):
        self.client = client
        self.prefix = prefix

    def get(self, key):
        try:
            raw = self.client.get(self._k(key))
        except Exception:
            log.warning('Redis get 失败，按未命中处理: key=%s', key, exc_info=True)
            return None
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
        try:
            payload = json.dumps(
                {'value': value, 'stored_at': now, 'ttl': ttl}, ensure_ascii=False
            )
            physical = max(int(ttl) * _STALE_GRACE_FACTOR, _MIN_PHYSICAL_TTL)
            self.client.set(self._k(key), payload, ex=physical)
        except Exception:
            log.warning('Redis set 失败，本次写入丢弃: key=%s', key, exc_info=True)

    def delete(self, key):
        try:
            self.client.delete(self._k(key))
        except Exception:
            log.warning('Redis delete 失败，忽略: key=%s', key, exc_info=True)

    def lock(self, key, timeout):
        try:
            return bool(self.client.set(self._lock_k(key), '1', nx=True, ex=int(timeout)))
        except Exception:
            log.warning('Redis lock 失败，视为未拿到锁: key=%s', key, exc_info=True)
            return False

    def unlock(self, key):
        """释放锁。不做所有权校验，调用方必须保证只对自己成功 lock() 的
        key 调用 unlock()。这是 Cache 门面 SWR 跨线程释放锁所必需——
        主线程 lock、后台线程完成刷新后 unlock，若加 token 或线程绑定的
        所有权校验会直接破坏这个流程。与 MemoryBackend 语义一致。
        """
        try:
            self.client.delete(self._lock_k(key))
        except Exception:
            log.warning('Redis unlock 失败，忽略: key=%s', key, exc_info=True)

    def _k(self, key):
        return f'{self.prefix}v:{key}'

    def _lock_k(self, key):
        return f'{self.prefix}lock:{key}'
