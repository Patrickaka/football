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

log = logging.getLogger('webapp.basketball')

_lock = threading.Lock()
_context = None


class _Context:
    def __init__(self, db, cache, transport):
        self.db = db
        self.cache = cache
        self.transport = transport
        self.prediction = factory.build_prediction_service(
            db, cache=cache, transport=transport, recorder=_Recorder())
        self.tracker = factory.build_odds_tracker(db, transport=transport)
        self.history = factory.build_odds_history_store(db)


class _Recorder:
    """预测记录仍落在旧的 kv_store 上。

    领域层只认 save / stats 两个方法，所以这里做一层薄适配；等 records.py
    迁完，把它换成真正的实现即可，领域层一行不用改。写失败只告警——
    记录是旁路，不该影响推荐本身。
    """

    def save(self, date, results, version):
        from src.basketball.records import save_predictions

        save_predictions(date, results, version)

    def stats(self):
        from src.basketball.records import get_prediction_stats

        return get_prediction_stats()


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


def reset():
    """测试用：丢弃已建好的单例。"""
    global _context
    with _lock:
        _context = None
