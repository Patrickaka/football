"""篮球领域服务在旧入口里的装配点。

`server.py` 是个多线程 HTTP server，Database 连接池、Cache 与抓取客户端的
熔断状态都属于进程级资源，必须只建一次——每请求重建会让连接池反复开合，
也会让熔断器永远回到初始状态、形同虚设。

**一切失败都降级，不上抛**：迁移前 kv_store 在 MySQL 连不上时会退回 JSON
文件，接口照常出结果，只是少了 Elo 与历史。这条降级路径必须保留，否则
一次数据库抖动会把五个端点一起打掉。
"""
import logging
import threading

from src.domain.sports.basketball import factory
from src.api.runtime import background

log = logging.getLogger('api.runtime.basketball')

# 赔率采样间隔。快于 15 分钟意义不大——盘口本身没那么频繁地动，
# 而 okooo 与 500 都有限速，采太密只是白白挤占抓取配额。
ODDS_TRACKING_INTERVAL_MINUTES = 15

_lock = threading.Lock()
_context = None


class _Context:
    def __init__(self, db, cache, transport):
        self.db = db
        self.cache = cache
        self.transport = transport
        self.recorder = factory.build_recorder(db)
        self.prediction = factory.build_prediction_service(
            db, cache=cache, transport=transport, recorder=self.recorder)
        self.tracker = factory.build_odds_tracker(db, transport=transport)
        self.history = factory.build_odds_history_store(db)


def _build_context():
    return _Context(db=_build_database(), cache=_build_cache(),
                    transport=factory.build_transport())


def _build_database():
    try:
        from src.foundation.store import Database, make_engine, mysql_url_from_env

        return Database(make_engine(mysql_url_from_env()))
    except Exception as exc:
        log.warning('篮球领域服务未连上数据库，退化为纯市场价格: %s', exc)
        return None


def _build_cache():
    """Redis 不可用时降级为纯进程内存。

    降级态下缓存不跨重启保留，单飞锁也失去物理过期兜底，但缓存不该成为
    可用性的单点故障。
    """
    try:
        from src.api.deps import Settings, build_cache

        return build_cache(Settings.from_env())
    except Exception as exc:
        log.warning('篮球缓存不可用，逐次计算: %s', exc)
        return None


def get_context():
    """进程级单例。双重检查是因为 server.py 每个请求一个线程。"""
    global _context
    if _context is None:
        with _lock:
            if _context is None:
                _context = _build_context()
    return _context


def register_odds_tracking(interval_minutes=ODDS_TRACKING_INTERVAL_MINUTES):
    """把赔率采样登记到进程级后台调度器。

    走势要靠一天里反复采样攒出来，只在有人请求时才采是不够的——夜里没人
    看的时候盘口照样在动，而那段变化恰恰是开盘到临场的主要部分。

    没有数据库时不登记——采集的全部意义就是落盘。

    **装配必须在取锁之前完成。** `_lock` 不可重入，而 `get_context()` 自己
    也要拿它；在锁内调用它会持锁死锁，此后每个请求都会一起卡住。这个错误
    上线过一次，表现是五个端点全部 120 秒超时。
    """
    tracker = get_context().tracker
    if tracker is None:
        log.warning('赔率采样未登记：数据库不可用')
        return False
    return background.submit_periodic(
        'basketball_odds_tracking', lambda: tracker.track(None),
        interval_seconds=max(60, int(interval_minutes) * 60))


def is_odds_tracking_running():
    """给健康检查用：真的在跑，而不只是对象存在。"""
    return background.is_running()


def reset():
    """测试用：丢弃已建好的单例。后台调度器由 background.reset() 负责。"""
    global _context
    with _lock:
        _context = None
