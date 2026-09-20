# -*- coding: utf-8 -*-
"""把 kl8 的策略试验记录接到 foundation/store。

以前 33839 条试验全量常驻 `_cfg.STRATEGY_TRIAL_RESULTS`（约 57 MB），
而且每新增**一条**就要把 26.7 MB 的 JSON 整文件重写一遍——一次策略验证
产生成百上千条试验，这个文件就被重写成百上千次。改成按需查库之后，内存里
不再留全量，写入也变成单条追加。

本模块是 kl8（旧代码）对领域层的适配。方向是**旧依赖新**，反过来不行。
"""
import logging
import threading

from src.domain.numeric.trial_store import KEY_FIELDS, TrialStore

from .main_play_validation import MAIN_PLAYS, is_main_play

log = logging.getLogger('kl8.trial_sync')

GAME = 'kl8'
EXPOSURE_ROUND = 'holdout_exposure'

_store = None
_store_lock = threading.Lock()


def _open_store():
    """懒建并复用连接。

    与 `store_sync` 那边「每次新建」的做法不同：抓取路径一天跑几次，而 FDR
    在一轮验证里会被调用上百次，每次重建连接池的代价摊不平。
    """
    global _store
    with _store_lock:
        if _store is None:
            from src.foundation.store import (Database, make_engine,
                                              database_url_from_env)
            _store = TrialStore(Database(make_engine(database_url_from_env())),
                                game=GAME)
        return _store


def reset_store():
    """丢弃缓存的连接。测试替换数据库后调用。"""
    global _store
    with _store_lock:
        _store = None


def set_store(store):
    """直接注入一个 store（测试用）。"""
    global _store
    with _store_lock:
        _store = store


def family_play_types(play_type):
    """FDR 的检验族：`MAIN_PLAYS` 的几种玩法合成一族，其余各自成族。"""
    return MAIN_PLAYS if is_main_play(play_type) else (play_type,)


def record_trial(trial):
    """追加一条试验记录，返回是否落库成功。

    失败只记日志不外抛：漏记一条会让它不参与后续 FDR，影响统计精度，
    但让整轮验证因为一次写入失败而中断更糟。**holdout 预留是例外**——
    它靠这个返回值决定要不要往下走，没落库就不能当作已曝光。
    """
    try:
        _open_store().append(trial)
        return True
    except Exception as exc:
        log.error('策略试验记录入库失败（该条不会参与后续 FDR）: %s', exc)
        return False


def family_trials(play_type, include_exposures=False):
    """取同族的全部试验，按 tested_at 排序。

    **读失败必须抛，不能降级成空列表。** 读不到历史试验时 FDR 只剩当前
    这一条，校正等于不校正，p 值偏松——那会把一个无效策略判成显著，而且
    一声不响。宁可让这轮验证失败，也不要一个错误的结论。
    """
    exclude = () if include_exposures else (EXPOSURE_ROUND,)
    return _open_store().family_trials(family_play_types(play_type),
                                       exclude_rounds=exclude)


def update_fdr(trial, adjusted_p):
    """回写单条的 fdr_adjusted_p，返回是否命中。

    只写当前这一条。历史条的这个值是纯派生量（由 raw_p_value 重算即可），
    全仓没有任何地方读它来做决策，而整族批量回写意味着每次 FDR 都要更新
    几千行。
    """
    try:
        return _open_store().update_fields(trial, fdr_adjusted_p=adjusted_p)
    except Exception as exc:
        log.error('回写 fdr_adjusted_p 失败: %s', exc)
        return False


def trial_key(trial):
    """四元键，与库里的主键一致。"""
    return tuple(str(trial.get(field, '')) for field in KEY_FIELDS)


def index_of(trials, target):
    """按四元键找出 target 的下标，找不到返回 None。

    以前这里用 `trial is trial_record`——同一个列表里的同一个对象。查库
    返回的是新对象，身份比较不再成立，而 FDR 是**按位置**取校正值的，
    定位错了就会把校正结果安静地对应到另一条策略上。
    """
    key = trial_key(target)
    for index, trial in enumerate(trials):
        if trial_key(trial) == key:
            return index
    return None


def family_with_current(play_type, trial):
    """返回（同族试验列表, 当前这条的下标）。

    当前这条一定在列表里：落库失败时也要让它参与本次校正，否则 FDR 会按
    少一条的族去算，比不校正还难察觉。
    """
    trials = family_trials(play_type)
    index = index_of(trials, trial)
    if index is None:
        trials.append(trial)
        index = len(trials) - 1
    return trials, index


def update_fdr_by_key(key, adjusted_p):
    """按四元键回写 fdr_adjusted_p。

    调用方手里往往只剩这条记录的键（原来存的是 `id(trial)`），
    而键本身已经够定位了。
    """
    return update_fdr(dict(zip(KEY_FIELDS, key)), adjusted_p)
