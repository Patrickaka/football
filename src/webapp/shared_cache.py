"""旧入口里的进程级共享缓存。

`server.py` 是多线程 HTTP server，Redis 连接与 L1 内存都属于进程级资源，
必须只建一次。各业务共用同一个 `Cache` 实例——key 里各自带业务前缀，
分开建实例只会多出几份互不相通的 L1。

Redis 不可用时降级为纯进程内存：缓存不该成为可用性的单点故障，但降级态下
它不跨重启保留，这恰恰是本模块要解决的问题的反面。
"""
import logging
import threading

log = logging.getLogger('webapp.cache')

_lock = threading.Lock()
_cache = None


def _build():
    try:
        from src.api.deps import Settings, build_cache

        return build_cache(Settings.from_env())
    except Exception as exc:
        log.warning('共享缓存不可用，退化为逐次计算: %s', exc)
        return None


def get_cache():
    global _cache
    if _cache is None:
        with _lock:
            if _cache is None:
                _cache = _build()
    return _cache


def reset():
    """测试用。"""
    global _cache
    with _lock:
        _cache = None
