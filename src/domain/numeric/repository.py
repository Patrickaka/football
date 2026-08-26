"""数字彩票的表与仓储。

表结构依据 2026-08-26 实读线上数据设计，不按命名推测（判据 4）。

**一张表装多种玩法**，靠 `game` 列隔离。分表的话每加一种彩票就要建一张
结构完全相同的表——不同的只是号码范围与个数，那是领域规则，不是存储结构。
不同彩种的期号会撞（各自编号），所以主键是 (game, issue) 而不是 issue。
"""
from sqlalchemy import Column, MetaData, String, Table, Text

from src.foundation.store import Repository

METADATA = MetaData()

NUMERIC_DRAW = Table(
    'numeric_draw', METADATA,
    Column('game', String(16), primary_key=True),
    Column('issue', String(32), primary_key=True),
    # 号码是集合，整体读写，没有「查哪些期开出过 37 号」这类查询——
    # 分析器一次性把 2048 期全加载进内存。存 JSON 比拆 20 列简单得多，
    # 也让号码个数不同的彩种共用一张表。
    Column('numbers', Text, nullable=False),
    Column('date', String(16)),
    # 溯源三件套。开奖结果出错时，判断该信哪一边全靠它们。
    Column('source', String(32)),
    Column('fetched_at', String(32)),
    Column('checksum', String(32)),
)


def create_all(db):
    METADATA.create_all(db.engine)


class DrawRepository(Repository):
    table = NUMERIC_DRAW
