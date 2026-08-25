import threading
import time
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

    def unlock(self, key: str) -> None: ...


class MemoryBackend:
    """进程内后端。用作 L1，也是测试时的 L2 替身。"""

    def __init__(self):
        self._data = {}
        self._locks = {}
        self._guard = threading.Lock()

    def get(self, key):
        with self._guard:
            return self._data.get(key)

    def set(self, key, value, ttl, now=None):
        now = time.time() if now is None else now
        with self._guard:
            self._data[key] = Entry(value=value, stored_at=now, ttl=ttl)

    def delete(self, key):
        with self._guard:
            self._data.pop(key, None)

    def lock(self, key, timeout):
        """非阻塞获取。timeout 表示锁的最长持有时间（与 Redis 后端语义一致），
        不是等待时间 —— 拿不到立即返回 False，由调用方决定等待策略。
        """
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        return lock.acquire(blocking=False)

    def unlock(self, key):
        with self._guard:
            lock = self._locks.get(key)
        if lock is not None and lock.locked():
            lock.release()

    def clear(self):
        with self._guard:
            self._data.clear()
