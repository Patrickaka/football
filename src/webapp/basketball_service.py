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
from src.foundation.tasks import TaskScheduler

log = logging.getLogger('webapp.basketball')

# 赔率采样间隔。快于 15 分钟意义不大——盘口本身没那么频繁地动，
# 而 okooo 与 500 都有限速，采太密只是白白挤占抓取配额。
ODDS_TRACKING_INTERVAL_MINUTES = 15

_lock = threading.Lock()
_context = None
_scheduler = None


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


def start_odds_tracking(interval_minutes=ODDS_TRACKING_INTERVAL_MINUTES):
    """启动赔率快照的周期采样。

    走势要靠一天里反复采样攒出来，只在有人请求时才采是不够的——夜里没人
    看的时候盘口照样在动，而那段变化恰恰是开盘到临场的主要部分。

    重复调用只启动一次。采集失败不会终止后续周期（TaskScheduler 保证），
    没有数据库时直接不启动——采集的全部意义就是落盘。

    **装配必须在取锁之前完成。** `_lock` 是普通 Lock、不可重入，而
    `get_context()` 自己也要拿它——在锁内调用它会死锁，且是持锁死锁：
    此后每个走 `get_context()` 的请求都会一起卡住。这个错误上线过一次，
    表现是五个端点全部 120 秒超时。
    """
    global _scheduler
    tracker = get_context().tracker
    with _lock:
        if _scheduler is not None:
            return _scheduler
        if tracker is None:
            log.warning('赔率采样未启动：数据库不可用')
            return None
        # 留 2 个 worker：周期任务会长期占住一个，只给 1 个的话调度器会
        # 在启动时告警「一次性任务可能长时间得不到执行」——虽然这里目前
        # 没有一次性任务，但那条告警是对的，不该靠「反正没有」把它压下去。
        scheduler = TaskScheduler(max_workers=2)
        scheduler.submit_periodic(
            'basketball_odds_tracking',
            lambda: tracker.track(None),
            interval_seconds=max(60, int(interval_minutes) * 60))
        scheduler.start()
        _scheduler = scheduler
        return scheduler


def is_odds_tracking_running():
    """给健康检查用：真的在跑，而不只是对象存在。"""
    return _scheduler is not None and _scheduler.is_running()


def reset():
    """测试用：丢弃已建好的单例并停掉周期任务。"""
    global _context, _scheduler
    with _lock:
        if _scheduler is not None:
            _scheduler.shutdown(wait=False)
            _scheduler = None
        _context = None
