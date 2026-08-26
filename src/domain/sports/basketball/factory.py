"""把 basketball 领域的各个部件装配成可用的服务。

这里是**唯一**知道「谁依赖谁」的地方。各个模块自己只声明需要什么端口，
装配集中在一处，好处是接线错误只会出现在一个文件里，而不是散落在每个
调用点上。

三个外部资源由调用方提供，因为它们的生命周期属于进程而不是领域：
- `db`：foundation/store 的 Database
- `cache`：foundation/cache 的 Cache，可为 None（不缓存，逐次计算）
- `transport`：发真实网络请求的函数，可为 None（用默认的两套实现）

`recorder` 也由调用方提供：预测记录目前还落在旧的 kv_store 上，等它迁完
再把默认实现挪进来。
"""
import logging

from src.domain.sports.basketball import fetching, okooo_parsing, parsing
from src.domain.sports.basketball.analysis import BasketballAnalyzer
from src.domain.sports.basketball.calibration import BasketballCalibrator
from src.domain.sports.basketball.calibration_store import CalibrationStore
from src.domain.sports.basketball.elo import BasketballELORatingSystem
from src.domain.sports.basketball.elo_store import EloStore
from src.domain.sports.basketball.movement_map import MovementMapBuilder
from src.domain.sports.basketball.odds_history import OddsHistoryStore, OddsTracker
from src.domain.sports.basketball.prediction import PredictionService

log = logging.getLogger('domain.basketball.factory')


def build_transport(snapshots_root=None):
    """建带限速、重试、熔断的抓取通道。

    两套实现按主机名分派：okooo 需要 Session 预热与 gb2312 解码，
    500.com 用普通 urllib 即可。
    """
    client = fetching.build_fetch_client(
        transport=fetching.dispatch_transport(
            okooo=fetching.OkoooTransport(),
            default=fetching.urllib_get),
        snapshots_root=snapshots_root)
    return client.get


def build_analyzer(db):
    """Elo 与校准器都要读库。

    `db` 允许为 None：数据库不可用时退化为纯市场价格，而不是让整个端点
    失败。迁移前 kv_store 在 MySQL 连不上时会降级到 JSON 文件，同样是
    「少一点信息，但仍然出结果」——这条降级路径不能在迁移中丢掉。
    """
    if db is None:
        log.warning('数据库不可用，篮球分析退化为纯市场价格')
        return BasketballAnalyzer(elo=None, calibrator=None)
    return BasketballAnalyzer(
        elo=BasketballELORatingSystem(store=EloStore(db)),
        calibrator=BasketballCalibrator(store=CalibrationStore(db)))


def build_schedule_sources(transport):
    """两个数据源的赛程抓取。键名与请求参数 `source` 一致。"""
    return {
        '500': parsing.ScheduleFetcher(transport=transport).fetch,
        'okooo': okooo_parsing.OkoooScheduleFetcher(transport=transport).fetch,
    }


def build_movement_provider(db, transport, now_fn=None):
    """走势映射。它自己会在 500 源上先采一轮快照再算。

    `db` 为 None 时快照那一路整体缺席，只剩澳客赛程页自带的盘路。
    """
    store = OddsHistoryStore(db) if db is not None else None
    schedules = build_schedule_sources(transport)
    tracker = (OddsTracker(schedule_fetcher=schedules['500'], store=store,
                           now_fn=now_fn) if store is not None else None)
    return MovementMapBuilder(
        history_store=store,
        tracker=tracker,
        okooo_schedule=schedules['okooo'],
        bundle_fetcher=okooo_parsing.MarketBundleFetcher(transport=transport),
        now_fn=now_fn)


def build_prediction_service(db, cache=None, transport=None, recorder=None,
                             ttl=None, now_fn=None, today_fn=None):
    transport = transport or build_transport()
    kwargs = {} if ttl is None else {'ttl': ttl}
    return PredictionService(
        analyzer=build_analyzer(db),
        schedule_sources=build_schedule_sources(transport),
        movement_provider=build_movement_provider(db, transport, now_fn=now_fn),
        recorder=recorder,
        cache=cache,
        today_fn=today_fn,
        **kwargs)


def build_odds_tracker(db, transport=None, now_fn=None):
    """单独暴露采集器：/api/basketball/track 端点直接触发它。

    没有库就没有采集器——采集的全部意义就是落盘，给它一个空仓储只会
    在真正写入时才炸，而那时错误信息离原因已经很远了。
    """
    store = build_odds_history_store(db)
    if store is None:
        return None
    return OddsTracker(
        schedule_fetcher=parsing.ScheduleFetcher(
            transport=transport or build_transport()).fetch,
        store=store, now_fn=now_fn)


def build_odds_history_store(db):
    return OddsHistoryStore(db) if db is not None else None
