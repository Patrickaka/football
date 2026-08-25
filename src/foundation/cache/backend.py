import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Protocol


@dataclass
class Entry:
    """缓存条目。过期条目仍可读取，SWR 依赖此行为。"""

    value: Any
    stored_at: float
    ttl: float

    def is_fresh(self, now=None):
        now = time.time() if now is None else now
        return (now - self.stored_at) < self.ttl


class CacheBackend(Protocol):
    def get(self, key: str) -> Optional[Entry]: ...

    def set(self, key: str, value: Any, ttl: float, now: Optional[float] = None) -> None: ...

    def delete(self, key: str) -> None: ...

    def lock(self, key: str, timeout: float) -> bool: ...

    def unlock(self, key: str) -> None:
        """释放锁。不做所有权校验，调用方必须保证只对自己成功 lock() 的
        key 调用 unlock()。这与 Redis 后端语义一致（Redis 端同样不用
        token 做 CAS 校验），且是 Cache 门面 SWR 跨线程释放锁所必需——
        主线程 lock、后台线程完成刷新后 unlock，若加线程绑定的所有权
        校验会直接破坏这个流程。
        """
        ...


class MemoryBackend:
    """进程内后端。用作 L1（LRU 有界），也是测试时的 L2 替身。

    _data 与 _locks 均以 OrderedDict 实现 LRU 淘汰，避免在长期运行的
    生产进程里无界增长——本进程曾在 3.6G 内存的机器上因内存问题僵死过。
    """

    def __init__(self, maxsize=512):
        self._data = OrderedDict()
        self._locks = OrderedDict()
        self._maxsize = maxsize
        self._guard = threading.Lock()

    def get(self, key):
        with self._guard:
            entry = self._data.get(key)
            if entry is not None:
                self._data.move_to_end(key)  # 命中即刷新为最近使用
            return entry

    def set(self, key, value, ttl, now=None):
        now = time.time() if now is None else now
        with self._guard:
            self._data[key] = Entry(value=value, stored_at=now, ttl=ttl)
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)  # 淘汰最久未使用

    def delete(self, key):
        with self._guard:
            self._data.pop(key, None)

    def lock(self, key, timeout):
        """非阻塞获取。timeout 表示锁的最长持有时间（与 Redis 后端语义一致），
        不是等待时间 —— 拿不到立即返回 False，由调用方决定等待策略。
        """
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                self._evict_idle_locks()
                lock = threading.Lock()
                self._locks[key] = lock
            self._locks.move_to_end(key)
        return lock.acquire(blocking=False)

    def unlock(self, key):
        """释放锁。不做所有权校验，调用方必须保证只对自己成功 lock() 的
        key 调用 unlock()。这与 Redis 后端语义一致（Redis 端同样不用
        token 做 CAS 校验），且是 Cache 门面 SWR 跨线程释放锁所必需——
        主线程 lock、后台线程完成刷新后 unlock，若加线程绑定的所有权
        校验会直接破坏这个流程。
        """
        with self._guard:
            lock = self._locks.get(key)
        if lock is not None and lock.locked():
            lock.release()

    def _evict_idle_locks(self):
        """淘汰未被持有的空闲锁。必须在 _guard 保护下调用。
        持有中的锁一律跳过——淘汰它会让互斥失效。
        """
        if len(self._locks) <= self._maxsize:
            return
        for k in list(self._locks.keys()):
            if len(self._locks) <= self._maxsize:
                break
            if not self._locks[k].locked():
                del self._locks[k]

    def clear(self):
        """仅重置缓存数据，不清 _locks。

        _locks 可能仍被其他线程持有；若在此清空映射，持有者手中的
        Lock 对象会与 key 失去关联，后续同 key 的 lock() 会创建一把
        全新的、未持有的 Lock，从而在互斥仍应生效期间悄悄放行——
        这是比"锁字典多占一点内存"更危险的正确性问题，因此不清。
        """
        with self._guard:
            self._data.clear()
