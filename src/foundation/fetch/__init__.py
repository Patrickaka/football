"""外部抓取层：统一限速、退避重试、熔断与响应快照。

所有源站抓取必须走本层。旧实现无限速，直接导致 500.com 返回 503。
"""
from .circuit import CircuitBreaker
from .client import FetchClient, FetchError
from .rate_limit import DomainRateLimiters, RateLimiter
from .snapshot import SnapshotStore

__all__ = [
    'CircuitBreaker',
    'DomainRateLimiters',
    'FetchClient',
    'FetchError',
    'RateLimiter',
    'SnapshotStore',
]
