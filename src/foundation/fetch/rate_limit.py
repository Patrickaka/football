import threading
import time


class RateLimiter:
    """令牌桶。acquire 返回调用方需 sleep 的秒数，不自行阻塞。"""

    def __init__(self, rate_per_sec, burst=1):
        self.rate_per_sec = rate_per_sec
        self.burst = max(1, burst)
        self._tokens = float(self.burst)
        self._last = None
        self._guard = threading.Lock()

    def acquire(self, now=None):
        now = time.time() if now is None else now
        with self._guard:
            if self._last is None:
                self._last = now
            elapsed = max(0.0, now - self._last)
            self._last = now
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate_per_sec)
            if self._tokens >= 1:
                self._tokens -= 1
                return 0
            deficit = 1 - self._tokens
            self._tokens = 0
            return deficit / self.rate_per_sec


class DomainRateLimiters:
    """按域名隔离的限速器集合。"""

    def __init__(self, default_rate=1, burst=1, overrides=None):
        self.default_rate = default_rate
        self.burst = burst
        self.overrides = overrides or {}
        self._limiters = {}
        self._guard = threading.Lock()

    def for_domain(self, domain):
        with self._guard:
            limiter = self._limiters.get(domain)
            if limiter is None:
                rate = self.overrides.get(domain, self.default_rate)
                limiter = RateLimiter(rate_per_sec=rate, burst=self.burst)
                self._limiters[domain] = limiter
            return limiter
