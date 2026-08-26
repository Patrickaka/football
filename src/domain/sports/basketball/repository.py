"""basketball 的表与仓储。

表结构依据 2026-08-26 从线上 kv_store 读到的**真实数据结构**设计，不是按
迁移前的猜测。三处与直觉不同的地方：

- Elo 的 ratings 只有评分数值，没有对局数（games_played 由 history 中
  非 initialized 的条目数算出）；history 是独立的变更流水
- recent_form 是近 N 场胜负的数值列表，无时间戳，故用位置索引 seq 保序；
  它是截断列表而非追加流水，写入时须先清空该队旧条目再整体写入
- 盘口快照有三类（spf 胜负 / rqspf 让分 / dx 大小分），字段各不相同，
  同一条快照里三类同时存在，故放在一张宽表而非拆三张
- 时间列一律存 ISO 8601 字符串而非数据库 DATETIME：跨 SQLite（测试）与
  MySQL（生产）行为一致，也与领域对象 to_dict() 的契约统一

- 预测记录的三个玩法与赛果存 JSON 原文而非拆列：字段随模型版本演进
  （线上同时存在两种形状），拆列会在每次模型迭代时逼着改表，而它们没有
  任何查询需求

`basketball_match_results` 与 `basketball_prediction_history` 两个 key 在线上
不存在，也无对应活代码，不建表。

`bb_calibration` 线上同样无数据，但 calibration.py 是活代码（分析流程在调用），
迁移它就需要落点，故按其 stats 的实际结构建表——不是因为有数据要迁。
"""
from sqlalchemy import Column, Float, Integer, MetaData, String, Table, Text

from src.foundation.store import Repository

METADATA = MetaData()

BB_ELO_RATING = Table(
    'bb_elo_rating', METADATA,
    Column('team', String(64), primary_key=True),
    Column('rating', Float, nullable=False),
    Column('updated_at', String(32), nullable=False),
)

BB_ELO_HISTORY = Table(
    'bb_elo_history', METADATA,
    Column('team', String(64), primary_key=True),
    Column('recorded_at', String(32), primary_key=True),
    Column('rating', Float, nullable=False),
    Column('event', String(64), nullable=False, default=''),
)

BB_ELO_RECENT_FORM = Table(
    'bb_elo_recent_form', METADATA,
    Column('team', String(64), primary_key=True),
    # 源数据是无时间戳的数值列表（近 N 场胜负），故用位置索引保序。
    Column('seq', Integer, primary_key=True),
    Column('result', Float, nullable=False),
)

BB_CALIBRATION = Table(
    'bb_calibration', METADATA,
    # bucket 形如 'spf|NBA|medium'，三段式：bet_type|league|confidence，
    # '*' 表通配（level 2 通配 league、level 3 再通配 confidence）。
    # 不拆成三列：调用方 get_stats 是全量加载后在内存里 split 过滤，
    # 不走数据库查询，拆开只增加写入成本。
    Column('bucket', String(128), primary_key=True),
    Column('count', Integer, nullable=False, default=0),
    Column('weighted_count', Float, nullable=False, default=0.0),
    Column('success', Integer, nullable=False, default=0),
    Column('weighted_success', Float, nullable=False, default=0.0),
    Column('predicted_sum', Float, nullable=False, default=0.0),
    Column('weighted_predicted_sum', Float, nullable=False, default=0.0),
)

BB_ODDS_SNAPSHOT = Table(
    'bb_odds_snapshot', METADATA,
    # match_key 形如 '2026-07-23_水星_火花'（日期_主队_客队），是 kv_store 里的原始键
    Column('match_key', String(128), primary_key=True),
    # 源结构是**列表**，用位置索引保序而不是拿时间戳当主键——与
    # bb_elo_recent_form 同样的理由。最初用的是 (match_key, captured_at)，
    # 那等于假设「同一场的两条快照时间戳必不相同」；真实数据里确实如此，
    # 但这个假设一旦不成立就是静默丢行，而不是报错。列表本来就允许重复。
    Column('seq', Integer, primary_key=True),
    Column('captured_at', String(32), nullable=False),
    # 盘口未变时不追加新快照，只把这一列往前推——它记的是「最后一次确认
    # 仍然没变」的时刻。首次采集的快照没有它，故可空；不能与 captured_at
    # 合并，两者语义不同：一个是变化发生的时刻，一个是最后确认的时刻。
    Column('observed_ts', String(32)),
    Column('spf_home', Float),
    Column('spf_away', Float),
    Column('rqspf_home', Float),
    Column('rqspf_away', Float),
    Column('dx_over', Float),
    Column('dx_under', Float),
    # handicap 在源数据里是字符串（如 '-1.5'），保持原样不转数值：
    # 让分盘可能出现 '受让+1.5' 之类的非纯数值写法，转换会丢信息。
    Column('handicap', String(16)),
    Column('total_line', Float),
)


BB_PREDICTION_RECORD = Table(
    'bb_prediction_record', METADATA,
    # 源结构是一个有序列表，`get_predictions(limit)` 取的是末尾 N 条、
    # 超过上限时截掉开头——两者都依赖插入顺序，故用位置索引保序。
    # 业务上的唯一键是 (date, match_id)（save_predictions 正是按它 upsert），
    # 但 match_id 本身不唯一：线上 73 条记录只有 43 个不同的 match_id，
    # 同一场比赛在不同日期、不同版本下会各留一条。
    # autoincrement=False 是必需的，不是风格问题：SQLAlchemy 会把**单列
    # 整数主键**默认当成自增列，MySQL 于是把 seq=0 视为「给我下一个自增值」
    # 而不是字面量 0。第一行 seq=0 被改写成 1，随后真正的 seq=1 那行走
    # upsert 的 update 分支把它覆盖掉——45 条源数据静默变成 44 条。
    # SQLite 没有这个行为，所以测试全绿、只有生产会丢数据。
    # 其余几张表是复合主键，不触发这条默认规则。
    Column('seq', Integer, primary_key=True, autoincrement=False),
    Column('date', String(16), nullable=False),
    Column('match_id', String(128), nullable=False),
    Column('num', String(32)),
    Column('league', String(64)),
    Column('home', String(64)),
    Column('away', String(64)),
    Column('time', String(32)),
    Column('version', String(64)),
    Column('created_at', String(32)),
    # 三个玩法与赛果都是嵌套字典，且字段随模型版本演进（线上同时存在
    # 2026-07-13-v2 与 2026-08-20-water-reverse-v3 两种形状）。拆成列会在
    # 每次模型迭代时逼着改表，而这些字段没有任何查询需求——查询只按
    # date 和「result 是否为空」。存 JSON 原文，一个字段都不丢。
    Column('spf', Text),
    Column('rqspf', Text),
    Column('dx', Text),
    Column('result', Text),
)


def create_all(db):
    """建表。生产环境首次部署与测试的 SQLite 内存库都用它。"""
    METADATA.create_all(db.engine)


class EloRatingRepository(Repository):
    table = BB_ELO_RATING


class EloHistoryRepository(Repository):
    table = BB_ELO_HISTORY


class EloRecentFormRepository(Repository):
    table = BB_ELO_RECENT_FORM


class CalibrationRepository(Repository):
    table = BB_CALIBRATION


class OddsSnapshotRepository(Repository):
    table = BB_ODDS_SNAPSHOT


class PredictionRecordRepository(Repository):
    table = BB_PREDICTION_RECORD
