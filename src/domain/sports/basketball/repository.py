"""basketball 的六张表与仓储。

对应迁移前 common/kv_store 中的 6 个 key。除 bb_prediction_history 外都拆成
真正的列与索引；那一张的 payload 保留 JSON 字符串——它是历史归档，只整体
读写、不按字段查询，拆列没有收益。

时间列一律存 ISO 8601 字符串而非数据库 DATETIME：跨 SQLite（测试）与 MySQL
（生产）行为一致，也与领域对象 to_dict() 的契约统一。
"""
from sqlalchemy import Column, Float, Integer, MetaData, String, Table, Text

from src.foundation.store import Repository

METADATA = MetaData()

BB_ELO = Table(
    'bb_elo', METADATA,
    Column('team', String(64), primary_key=True),
    Column('rating', Float, nullable=False),
    Column('games', Integer, nullable=False, default=0),
    Column('updated_at', String(32), nullable=False),
)

BB_CALIBRATION = Table(
    'bb_calibration', METADATA,
    Column('bucket', String(64), primary_key=True),
    Column('bet_type', String(32), nullable=False),
    Column('hits', Integer, nullable=False, default=0),
    Column('total', Integer, nullable=False, default=0),
    Column('updated_at', String(32), nullable=False),
)

BB_ODDS_HISTORY = Table(
    'bb_odds_history', METADATA,
    Column('match_id', String(64), primary_key=True),
    Column('captured_at', String(32), primary_key=True),
    Column('home_odds', Float, nullable=False),
    Column('away_odds', Float, nullable=False),
    Column('source', String(32), nullable=False),
)

BB_PREDICTION_RECORD = Table(
    'bb_prediction_record', METADATA,
    Column('match_id', String(64), primary_key=True),
    Column('bet_type', String(32), primary_key=True),
    Column('pick', String(32), nullable=False),
    Column('prob', Float, nullable=False),
    Column('confidence', String(16), nullable=False),
    Column('created_at', String(32), nullable=False),
)

BB_MATCH_RESULT = Table(
    'bb_match_result', METADATA,
    Column('match_id', String(64), primary_key=True),
    Column('home_score', Integer, nullable=False),
    Column('away_score', Integer, nullable=False),
    Column('settled_at', String(32), nullable=False),
)

BB_PREDICTION_HISTORY = Table(
    'bb_prediction_history', METADATA,
    Column('match_id', String(64), primary_key=True),
    Column('predicted_at', String(32), primary_key=True),
    Column('payload', Text, nullable=False),
    Column('league', String(64), nullable=False, default=''),
)


def create_all(db):
    """建表。生产环境首次部署与测试的 SQLite 内存库都用它。"""
    METADATA.create_all(db.engine)


class EloRepository(Repository):
    table = BB_ELO


class CalibrationRepository(Repository):
    table = BB_CALIBRATION


class OddsHistoryRepository(Repository):
    table = BB_ODDS_HISTORY


class PredictionRecordRepository(Repository):
    table = BB_PREDICTION_RECORD


class MatchResultRepository(Repository):
    table = BB_MATCH_RESULT


class PredictionHistoryRepository(Repository):
    table = BB_PREDICTION_HISTORY
