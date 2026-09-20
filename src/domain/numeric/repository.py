"""数字彩票的表与仓储。

表结构依据 2026-08-26 实读线上数据设计，不按命名推测（判据 4）。

**一张表装多种玩法**，靠 `game` 列隔离。分表的话每加一种彩票就要建一张
结构完全相同的表——不同的只是号码范围与个数，那是领域规则，不是存储结构。
不同彩种的期号会撞（各自编号），所以主键是 (game, issue) 而不是 issue。
"""
from sqlalchemy import (
    Boolean, Column, Float, Integer, MetaData, String, Table, Text,
)

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


NUMERIC_STRATEGY_TRIAL = Table(
    'numeric_strategy_trial', METADATA,
    # 四元主键与旧实现的去重键一字不差：strategy_id + play_type +
    # tournament_round + tested_at。线上 23564 条里这个组合零重复。
    Column('game', String(16), primary_key=True),
    Column('strategy_id', String(64), primary_key=True),
    Column('play_type', String(16), primary_key=True),
    Column('tournament_round', String(32), primary_key=True),
    # 通常是 19 字符的 ISO 时间戳，但 holdout 预留那条把它赋成 issues_sha256
    # （`holdout.py` 里 `reservation['tested_at'] = reservation['issues_sha256']`），
    # 那是 64 字符。线上 sql_mode 带 STRICT_TRANS_TABLES，短了会直接插入失败
    # 而不是截断——现有 33839 条全是时间戳，所以这条路至今没被走到过。
    Column('tested_at', String(64), primary_key=True),
    # 权重字典的键随版本增删——线上一共出现过 13 种特征名，而单条记录只带
    # 其中几个。拆成列的话每加一个特征就要改表，而它们没有任何查询需求。
    Column('feature_weights', Text),
    Column('model_weights', Text),
    Column('window_size', Integer),
    Column('repeat_direction', String(16)),
    Column('raw_p_value', Float),
    Column('fdr_adjusted_p', Float),
    Column('validation_lift', Float),
    Column('n_permutations', Integer),
    # 以下五个字段是后加的，老记录没有。「没有这个键」与「值为空」是两件事，
    # 靠 _present 列区分——补默认值等于凭空造出结论。
    Column('pool_diversify', Boolean),
    Column('pool_max_last_numbers', Integer),
    Column('frequency_mode', String(32)),
    Column('final_selection_mode', String(32)),
    Column('practical_score', Float),
    Column('optional_present', Text),
    # 没有建模的字段整体存这里，不逐个开列。记录里除上面这些还带
    # trial_id / evidence_schema / validation_family / main_play_p_values /
    # main_play_validation，且随版本增删；漏掉一个不会报错，只会让读回来的
    # 记录少一个键——`holdout.py` 正是用 `evidence_schema != 2` 分辨新旧证据，
    # 少了它每条都会被判成 legacy。它们没有按列查询的需求，所以整体存 JSON。
    Column('extra', Text),
)


def create_all(db):
    METADATA.create_all(db.engine)


class DrawRepository(Repository):
    table = NUMERIC_DRAW


class StrategyTrialRepository(Repository):
    table = NUMERIC_STRATEGY_TRIAL
