import os
import shutil
import tempfile

import pytest

_POLLUTING_FILES = [
    'data/kl8_active_strategies.json',
    'data/kl8_strategy_trials.json',
]


@pytest.fixture(scope='session', autouse=True)
def skip_startup_orchestration():
    """测试里不执行启动编排。

    编排会抓网络、起四个预热线程（3D 那个要跑 ML）、登记周期任务。
    几百个用例里每个 `TestClient(create_app())` 都做一遍，套件会从三分钟
    涨到十分钟以上——**这不是慢，是每个用例都在打外网**。

    需要验证编排本身的用例（`tests/api/test_startup.py`）显式打开。
    """
    os.environ.setdefault('RUN_STARTUP_TASKS', '0')
    yield


@pytest.fixture(scope='session', autouse=True)
def isolate_kl8_state():
    """kl8 测试的落盘产物会污染 reference 策略断言，测试期间移开，结束后还原。

    这两个文件是线上生产运行时数据，移动/还原任一步失败都不能静默丢失：
    - 移开阶段（setup）中途失败时，把已移走的文件立即移回原位再抛出异常，
      不让它们滞留在临时目录里。
    - 还原阶段（teardown）无论测试本身是否异常都会尝试执行；只有全部文件
      都成功移回后才清理临时目录，还原失败时保留临时目录并把异常继续抛出，
      避免把仅存的备份一并删除。
    """
    stash = tempfile.mkdtemp(prefix='kl8-stash-')
    moved = []
    try:
        for rel in _POLLUTING_FILES:
            if os.path.exists(rel):
                dst = os.path.join(stash, os.path.basename(rel))
                shutil.move(rel, dst)
                moved.append((rel, dst))
    except Exception:
        for rel, dst in moved:
            shutil.move(dst, rel)
        raise

    try:
        yield
    finally:
        try:
            for rel, dst in moved:
                os.makedirs(os.path.dirname(rel), exist_ok=True)
                shutil.move(dst, rel)
        except Exception:
            raise
        else:
            shutil.rmtree(stash, ignore_errors=True)
