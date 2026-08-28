# -*- coding: utf-8 -*-
"""北单赛果提取、历史校准与历史记录读写"""

from ..common.logger import setup_logger
from ..common import kv_store

log = setup_logger('beidan')

from .config import (
    BEIDAN_HISTORY_KEY, BEIDAN_HISTORY_LIMIT, MAX_GOALS,
)

# ─── 领域层适配 ───
#
# 赛果判定与校准算术在 `src/domain/sports/beidan/settlement.py`。
# 这里只做三件领域层不该碰的事：读写 kv、解析页面来的盘口文本、
# 按玩法名挑一个提取器。旧名字全部保住——`__init__.py` 导出了它们，
# `recommending` 按名字导入。

from src.domain.sports.beidan import handicap as _handicap
from src.domain.sports.beidan import settlement as _settlement

# 历史校准的默认门槛。**迁移前它们是函数签名里的裸默认值**：
# 至少 8 个加权样本才动手（少于这个数，因子基本由先验决定，动了也是噪声），
# 只看最近 200 条（更早的记录来自改过的模型，拿它们校准当前模型是刻舟求剑）。
BEIDAN_CALIBRATION_MIN_SAMPLES = 8
BEIDAN_CALIBRATION_LIMIT = 200


def calculate_implied_probability(odds_dict):
    """欧赔 → 去水后的隐含概率。"""
    return _settlement.implied_probability(odds_dict)


def _actual_spf_from_record(record):
    return _settlement.actual_spf(record)


def _actual_zjq_from_record(record):
    return _settlement.actual_zjq(record, max_goals=MAX_GOALS)


def _actual_bifen_from_record(record):
    """从已结算快照解析实际比分字符串 'h-a'（用于比分历史校准）"""
    return _settlement.actual_bifen(record)


def _actual_rqspf_from_record(record):
    """从已结算快照的实际比分 + 让球值，推导让球胜平负实际结果。

    **盘口在历史记录里存的是页面文本**：线上 500 条里 143 条有值，形如
    `'(-1)'`、`'(+1)'`、`'(-2)'`、`'(+2)'`，其余为 `None`。迁移前这里写的是
    `float(hc)`，对这四种形态**无一例外抛 ValueError**，被 except 吞成
    「按平手盘算」——分盘全部退化成不让球，`让平` 因此被系统性地误判成
    让胜或让负。同一个包里另外两处（`markets.py`、`recommending.py`）
    读的是同一个字段，用的都是 `parse_beidan_handicap`。

    改成同一个解析器。**这不改变任何线上行为**：这个函数的两个调用方
    （历史校准、`summarize_beidan_history`）都以 `settled` 为前提，
    而线上 500 条 `settled` 全是 False，仓库里也没有任何一处会把它置 True。
    """
    return _settlement.actual_rqspf(record, _handicap.parse(record.get('handicap')))


# 玩法 → 实际结果提取器。迁移前是函数体里一串 if/elif，未知玩法落到
# `actual = None`，于是一条样本都攒不上——保住这个语义：查不到就给个
# 恒返回 None 的提取器，而不是抛错。
_ACTUAL_EXTRACTORS = {
    'spf': _actual_spf_from_record,
    'zjq': _actual_zjq_from_record,
    'bifen': _actual_bifen_from_record,
    'rqspf': _actual_rqspf_from_record,
}


def apply_beidan_history_calibration(probabilities, bet_type, league=None,
                                     min_samples=BEIDAN_CALIBRATION_MIN_SAMPLES,
                                     limit=BEIDAN_CALIBRATION_LIMIT):
    """Use settled Beidan snapshots as a conservative reliability correction."""
    # 概率为空时不读存储：迁移前那道守卫排在 kv 读取之前。比分那一路
    # （`recommending.py:852`）传进来的矩阵确实可能是空的，顺序保住。
    records = _load_beidan_history() if probabilities else []
    return _settlement.apply_history_calibration(
        probabilities, records,
        _ACTUAL_EXTRACTORS.get(bet_type, lambda record: None), bet_type,
        league=league, min_samples=min_samples, limit=limit)


def _beidan_record_key(match):
    return _settlement.record_key(match)


def _load_beidan_history():
    data = kv_store.load(BEIDAN_HISTORY_KEY, [])
    return data if isinstance(data, list) else []


def _save_beidan_history(records):
    records = sorted(records, key=lambda r: r.get('created_at', ''), reverse=True)
    return kv_store.save(BEIDAN_HISTORY_KEY, records[:BEIDAN_HISTORY_LIMIT])
