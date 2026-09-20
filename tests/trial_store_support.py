# -*- coding: utf-8 -*-
"""给 kl8 的策略试验记录挂一个内存库。

试验记录从「全量常驻 `config.STRATEGY_TRIAL_RESULTS` + 整表重写 JSON」改成
按需查库之后，测试不能再 patch 那个列表和 `_persist_trial_results`——
生产代码已经不走它们了，patch 上去只会让测试测了个寂寞。
"""
from contextlib import contextmanager
from copy import deepcopy

from src.domain.numeric.repository import create_all
from src.domain.numeric.trial_store import TrialStore
from src.foundation.store import Database, make_engine
from src.kl8 import trial_sync


@contextmanager
def trial_store(history_trials=()):
    """挂一个预置了历史试验的内存库，退出时还原。"""
    db = Database(make_engine('sqlite+pysqlite:///:memory:'))
    create_all(db)
    store = TrialStore(db, game='kl8')
    store.append_many(deepcopy(list(history_trials)))
    trial_sync.set_store(store)
    try:
        yield store
    finally:
        trial_sync.reset_store()
