import os
import shutil
import tempfile

import pytest

_POLLUTING_FILES = [
    'data/kl8_active_strategies.json',
    'data/kl8_strategy_trials.json',
]


@pytest.fixture(scope='session', autouse=True)
def isolate_kl8_state():
    """kl8 测试的落盘产物会污染 reference 策略断言，测试期间移开，结束后还原。"""
    stash = tempfile.mkdtemp(prefix='kl8-stash-')
    moved = []
    for rel in _POLLUTING_FILES:
        if os.path.exists(rel):
            dst = os.path.join(stash, os.path.basename(rel))
            shutil.move(rel, dst)
            moved.append((rel, dst))
    yield
    for rel, dst in moved:
        os.makedirs(os.path.dirname(rel), exist_ok=True)
        shutil.move(dst, rel)
    shutil.rmtree(stash, ignore_errors=True)
