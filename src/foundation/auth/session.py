"""服务端会话：签发、认领、撤销。

**存储由调用方注入**：`backend` 是 `foundation.cache` 的
`CacheBackend`（`get` / `set(ttl)` / `delete` 三个方法正好够用），
生产态传 `RedisBackend`——会话因此跨进程重启保留，且撤销是真的撤销。

**随机与时钟也注入**：会话 id 必须不可预测，测试又要可复现；
两个需求只有靠注入才能同时满足（判据 16）。

降级为 `MemoryBackend` 时会话不跨重启，且受其 `maxsize` 淘汰限制——
对个位数用户无影响，但运维要知道"重启后所有人被登出"是预期的。
"""

import secrets
import time
from typing import Optional

_DEFAULT_TTL_SECONDS = 7 * 24 * 3600
_SESSION_ID_BYTES = 32


class SessionManager:
    """会话的生命周期。一个会话就是「id → 用户名」的一条带 TTL 的记录。"""

    def __init__(self, backend, ttl: int = _DEFAULT_TTL_SECONDS,
                 rng=None, now=None, prefix: str = 'session:'):
        if ttl <= 0:
            raise ValueError('ttl must be > 0, got %r' % (ttl,))
        self.backend = backend
        self.ttl = ttl
        self.prefix = prefix
        self._rng = rng or secrets.token_urlsafe
        self._now = now or time.time

    def create(self, user: str) -> str:
        """签发一个新会话，返回会话 id。

        id 用 `secrets.token_urlsafe(32)`——256 位熵，猜不出来。
        **不复用已有会话**：同一用户多处登录各拿各的 id，
        这样单独踢掉一处不会波及其它。
        """
        session_id = self._rng(_SESSION_ID_BYTES)
        self.backend.set(self._key(session_id), user, ttl=self.ttl, now=self._now())
        return session_id

    def resolve(self, session_id: Optional[str]) -> Optional[str]:
        """认领会话，返回用户名；无效或已过期返回 None。"""
        if not session_id or not isinstance(session_id, str):
            return None
        entry = self.backend.get(self._key(session_id))
        if entry is None:
            return None
        # backend.get 返回的是 Entry（带 value/expires_at），过期与否由它判断
        if not entry.is_fresh(now=self._now()):
            return None
        user = entry.value
        return user if isinstance(user, str) and user else None

    def revoke(self, session_id: Optional[str]) -> None:
        """撤销会话。**对不存在的 id 也静默成功**——登出要是幂等的。"""
        if session_id and isinstance(session_id, str):
            self.backend.delete(self._key(session_id))

    def _key(self, session_id: str) -> str:
        return f'{self.prefix}{session_id}'
