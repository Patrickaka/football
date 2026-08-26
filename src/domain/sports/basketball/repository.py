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

**尚未迁移**：`basketball_prediction_records`（41 条，match_id 不唯一、
三类盘口字段不同、result 全空）的表结构依赖还不确定的查询模式，留到迁移
records.py 时一并设计。`basketball_calibration_db`、`basketball_match_results`、
`basketball_prediction_history` 三个 key 在线上不存在，不预先建表。
"""
from sqlalchemy import Column, Float, Integer, MetaData, String, Table

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

BB_ODDS_SNAPSHOT = Table(
    'bb_odds_snapshot', METADATA,
    # match_key 形如 '2026-07-23_水星_火花'（日期_主队_客队），是 kv_store 里的原始键
    Column('match_key', String(128), primary_key=True),
    Column('captured_at', String(32), primary_key=True),
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


def create_all(db):
    """建表。生产环境首次部署与测试的 SQLite 内存库都用它。"""
    METADATA.create_all(db.engine)


class EloRatingRepository(Repository):
    table = BB_ELO_RATING


class EloHistoryRepository(Repository):
    table = BB_ELO_HISTORY


class EloRecentFormRepository(Repository):
    table = BB_ELO_RECENT_FORM


class OddsSnapshotRepository(Repository):
    table = BB_ODDS_SNAPSHOT
