"""进程唯一的后台调度器。

迁移前后台任务分散在三处：kl8 用 APScheduler、篮球赔率采样用自己新建的
`TaskScheduler`、还有几个裸线程做缓存预热。三套机制并存的代价不是「多写了
几行」，而是**没有任何一个地方能回答「现在后台在跑什么」**——健康检查只能
看到其中一套，排查时得挨个去翻。

现在统一到一个 `TaskScheduler`：各业务只负责登记任务，`server.py` 在全部
登记完之后启动一次。

**登记必须发生在启动之前**：`TaskScheduler.start()` 之后不再接受新任务。
这不是限制而是设计——任务集合在启动那一刻就固定下来，才谈得上「后台在跑
什么」有一个确定的答案。
"""
import logging
import threading

from src.foundation.tasks import TaskScheduler

log = logging.getLogger('webapp.background')

# 每个周期任务会长期占住一个 worker，所以这个数必须大于周期任务总数，
# 否则一次性任务永远排不上队。当前四个周期任务（kl8 三个 + 篮球采样一个）。
MAX_WORKERS = 6

_lock = threading.Lock()
_scheduler = None
_started = False


def scheduler():
    """取得进程级调度器。尚未启动，可继续登记任务。"""
    global _scheduler
    if _scheduler is None:
        with _lock:
            if _scheduler is None:
                _scheduler = TaskScheduler(max_workers=MAX_WORKERS)
    return _scheduler


def submit_periodic(name, fn, interval_seconds):
    """登记一个周期任务。重名会被拒绝——`results()` 以任务名为键。"""
    try:
        scheduler().submit_periodic(name, fn, interval_seconds=interval_seconds)
        return True
    except Exception as exc:
        log.warning('后台任务登记失败: name=%s %s', name, exc)
        return False


def start():
    """启动一次。重复调用无害。"""
    global _started
    with _lock:
        if _started:
            return _scheduler
        current = scheduler()
        current.start()
        _started = True
        log.info('后台调度器已启动: %d 个任务', current.task_count())
        return current


def is_running():
    return _scheduler is not None and _scheduler.is_running()


def task_count():
    return _scheduler.task_count() if _scheduler is not None else 0


def reset():
    """测试用：停掉并丢弃。"""
    global _scheduler, _started
    with _lock:
        if _scheduler is not None:
            _scheduler.shutdown(wait=False)
        _scheduler = None
        _started = False
