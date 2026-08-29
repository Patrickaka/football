"""入站限流：按客户端分桶的令牌桶。

单桶直接复用 `foundation.fetch.rate_limit.RateLimiter`（出站抓取用的那个）
——令牌桶就是令牌桶，它已经处理过 NTP 校时倒退，没必要再写一遍（判据 11）。

**但桶的集合不能复用 `DomainRateLimiters`**：那个字典永不淘汰。
出站场景的 key 是域名，就那么几个；入站场景的 key 是客户端 IP，
**无上限**——照搬过来等于给了任何人一条用随机源 IP 把内存撑爆的路。
所以这里的集合带容量上限与淘汰。
"""

import logging
import threading
from collections import OrderedDict
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from src.foundation.fetch.rate_limit import RateLimiter

log = logging.getLogger('api.rate_limit')

#: 限流也豁免健康探针——监控探针被限流会把"服务忙"误报成"服务挂了"。
EXEMPT_PATHS = frozenset({'/healthz'})


class ClientRateLimiters:
    """按客户端分桶，**带容量上限的 LRU 淘汰**。

    淘汰一个桶等于把那个客户端的配额重置。这是有意的取舍：被淘汰的必然是
    最久没来的客户端，重置它的配额没有安全意义；而不淘汰的代价是内存
    无上限增长。`maxsize` 要显著大于正常并发客户端数，否则活跃用户会
    互相把对方挤掉、配额被反复重置——那样限流就形同虚设。
    """

    def __init__(self, rate_per_sec: float, burst: int = 10, maxsize: int = 4096):
        if rate_per_sec <= 0:
            raise ValueError('rate_per_sec must be > 0, got %r' % (rate_per_sec,))
        if maxsize <= 0:
            raise ValueError('maxsize must be > 0, got %r' % (maxsize,))
        self.rate_per_sec = rate_per_sec
        self.burst = burst
        self.maxsize = maxsize
        self._buckets: OrderedDict = OrderedDict()
        self._guard = threading.Lock()

    def acquire(self, client: str, now: Optional[float] = None) -> float:
        """取一个令牌，返回调用方需要等待的秒数（0 表示放行）。"""
        with self._guard:
            limiter = self._buckets.get(client)
            if limiter is None:
                limiter = RateLimiter(rate_per_sec=self.rate_per_sec, burst=self.burst)
                self._buckets[client] = limiter
                if len(self._buckets) > self.maxsize:
                    evicted, _ = self._buckets.popitem(last=False)
                    log.debug('限流桶淘汰: %s', evicted)
            else:
                self._buckets.move_to_end(client)
        return limiter.acquire(now=now)

    def bucket_count(self) -> int:
        with self._guard:
            return len(self._buckets)


def client_key(request: Request) -> str:
    """限流的分组键：客户端 IP。

    **只认反代加的 `X-Real-IP`，不解析 `X-Forwarded-For`。**
    后者是一条谁都能往里塞的链——取第一个等于让客户端自己决定自己是谁，
    限流一秒钟就被绕过去了。线上 openresty 的
    `proxy_set_header X-Real-IP $remote_addr` 是它自己写的、覆盖式的，
    伪造不了。

    没有反代时（本地直连）退回 TCP 层的对端地址。
    """
    real_ip = request.headers.get('x-real-ip')
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else 'unknown'


def install_rate_limit(app, limiters: Optional[ClientRateLimiters], now=None):
    """挂上限流中间件。`limiters` 为 None 表示不限流。"""
    if limiters is None:
        return app

    @app.middleware('http')
    async def _limit(request: Request, call_next):
        if request.url.path.rstrip('/') in EXEMPT_PATHS or request.method == 'OPTIONS':
            return await call_next(request)

        wait = limiters.acquire(client_key(request), now=now() if now else None)
        if wait > 0:
            log.warning('限流触发 %s %s（来自 %s，建议 %.2fs 后重试）',
                        request.method, request.url.path, client_key(request), wait)
            return JSONResponse(
                {'detail': '请求过于频繁，请稍后再试'},
                status_code=429,
                headers={'Retry-After': str(max(1, int(wait + 0.999)))},
            )
        return await call_next(request)

    return app
