import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from fastapi import Request

from src.foundation.cache import Cache, MemoryBackend, RedisBackend
from src.foundation.store import Database, make_engine, mysql_url_from_env

log = logging.getLogger('api.deps')

_executor: Optional[ThreadPoolExecutor] = None


@dataclass
class Settings:
    redis_url: Optional[str] = 'redis://127.0.0.1:6379/0'
    mysql_url: Optional[str] = None
    cache_default_ttl: int = 300
    max_task_workers: int = 2
    executor_workers: int = 4
    #: 每客户端每秒允许的请求数；<= 0 表示不限流。
    rate_limit_per_sec: float = 0.0
    #: 突发额度——正常网页一次打开会并发打好几个接口，
    #: burst 设得太小会把正常使用限死。
    rate_limit_burst: int = 20
    #: 限流桶的容量上限，超出按最久未用淘汰。
    rate_limit_clients: int = 4096

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        return cls(
            redis_url=env.get('REDIS_URL', 'redis://127.0.0.1:6379/0'),
            mysql_url=env.get('MYSQL_URL') or None,
            cache_default_ttl=int(env.get('CACHE_DEFAULT_TTL', '300')),
            max_task_workers=int(env.get('MAX_TASK_WORKERS', '2')),
            executor_workers=int(env.get('EXECUTOR_WORKERS', '4')),
            rate_limit_per_sec=float(env.get('RATE_LIMIT_PER_SEC', '0')),
            rate_limit_burst=int(env.get('RATE_LIMIT_BURST', '20')),
            rate_limit_clients=int(env.get('RATE_LIMIT_CLIENTS', '4096')),
        )


def build_cache(settings):
    """建两层缓存。Redis 不可用时降级为纯内存，不阻止服务启动。

    已知限制（降级态）：`Cache` 的单飞锁与 SWR 后台刷新锁只在 L2 上加锁。
    生产态 L2 是 `RedisBackend`，其锁通过 `SET NX EX` 实现，带物理过期，
    即便持锁方异常退出，锁也会在 TTL 后自动释放。降级为 `MemoryBackend`
    后，锁变成一把普通的 `threading.Lock`，`lock(key, timeout)` 的
    `timeout` 只是"意图上的持有上限"，没有任何强制机制会真正释放它。
    这意味着：如果某个持锁路径绕过了 `try/finally`（例如进程被信号
    杀死、C 扩展段错误等 Python 无法拦截的场景），该 key 会被永久锁死，
    直到进程重启——不像 Redis 锁那样能在 TTL 后自愈。

    之所以仍保留降级而不是让 Redis 不可用直接阻止启动：Redis 只是缓存，
    不应该成为服务可用性的单点故障；且 `Cache` 内部所有加锁路径都配对在
    `try/finally` 中，Python 无法强杀线程、`finally` 必然执行，实际触发
    上述限制的概率极低。运维需知悉这一点：降级态下缓存不跨重启保留，
    且单飞锁失去了物理过期兜底。
    """
    l1 = MemoryBackend()
    l2 = MemoryBackend()
    if settings.redis_url:
        try:
            import redis

            client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
            client.ping()
            l2 = RedisBackend(client)
        except Exception as exc:
            log.warning(
                '缓存降级为纯内存（Redis 不可用: %s）。降级态下缓存不跨进程重启保留；'
                '且单飞/SWR 锁失去 Redis 侧的物理过期兜底——若持锁路径绕过 try/finally '
                '异常退出，该 key 将被永久锁死直至进程重启。',
                exc,
            )
            l2 = MemoryBackend()
    return Cache(l1=l1, l2=l2, default_ttl=settings.cache_default_ttl)


def build_database(settings):
    url = settings.mysql_url or mysql_url_from_env()
    return Database(make_engine(url))


def get_cache(request: Request) -> Cache:
    """FastAPI 依赖：从 app.state 取出装配好的 Cache 单例。

    路由用 `Depends(get_cache)` 注入；测试用
    `app.dependency_overrides[get_cache] = lambda: fake_cache` 替换。
    """
    return request.app.state.cache


def get_db(request: Request) -> Database:
    """FastAPI 依赖：从 app.state 取出装配好的 Database 单例。

    路由用 `Depends(get_db)` 注入；测试用
    `app.dependency_overrides[get_db] = lambda: fake_db` 替换。
    """
    return request.app.state.db


def get_executor(workers=4):
    """同步重活专用线程池。"""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='Blocking')
    return _executor


async def run_blocking(fn, *args, **kwargs):
    """同步重活的唯一正确入口。

    这个项目里的"重活"全是同步的：ML 训练（约 22 秒）、回测、抓取，
    没有一个是原生异步实现。直接把它们写在 `async def` 里会在事件循环
    的单个线程上同步阻塞执行，冻住整个进程的所有并发请求——这比旧的
    `ThreadingHTTPServer`（每请求一个线程，互不阻塞）还要糟。因此：
    任何同步重活必须经 `run_blocking` 派发到 `get_executor()` 的线程池，
    严禁直接在 `async def` 路由/依赖里调用。
    """
    loop = asyncio.get_running_loop()
    if kwargs:
        from functools import partial

        return await loop.run_in_executor(get_executor(), partial(fn, *args, **kwargs))
    return await loop.run_in_executor(get_executor(), fn, *args)


def shutdown_executor():
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None
