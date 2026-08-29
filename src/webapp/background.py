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
import os
import threading

from src.foundation.tasks import TaskScheduler

log = logging.getLogger('webapp.background')

# 每个周期任务会长期占住一个 worker，所以这个数必须大于周期任务总数，
# 否则一次性任务永远排不上队。当前六个：kl8 三个 + 篮球采样一个
# + 足球的赛后回填与时间分层扫描。加足球那两个之前这里是 6，正好等于任务数
# ——`TaskScheduler.start()` 只会告警、不会拒绝，一次性任务从此排不上队。
MAX_WORKERS = 8

# 周期任务首轮之间错开多久。**这台机器上冻死过两次**，两次都紧跟在
# `systemctl restart` 之后：重启把所有缓存清零，六个周期任务同时开跑，
# 内存被吃穿到连 sshd 都 fork 不出来（云监控上 Available 归零、
# 监控 agent 自己也断了上报）。
#
# 60 秒 × 五个间隔 = 最后一个任务晚五分钟才跑第一轮。这个代价是可接受的
# ——周期最短的那个是十分钟，晚五分钟不改变任何一天的结果；
# 而同时开跑的代价是整台机器失联半小时。
STARTUP_STAGGER_SECONDS = int(os.getenv('TASK_STARTUP_STAGGER', '60'))

_lock = threading.Lock()
_scheduler = None
_started = False


def scheduler():
    """取得进程级调度器。尚未启动，可继续登记任务。"""
    global _scheduler
    if _scheduler is None:
        with _lock:
            if _scheduler is None:
                _scheduler = TaskScheduler(
                    max_workers=MAX_WORKERS,
                    startup_stagger_seconds=STARTUP_STAGGER_SECONDS)
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


def shutdown(wait=True):
    """停掉调度器，保留单例。

    与 `reset()` 的区别：`reset` 会把单例丢掉（测试用，好让下一个用例
    重新登记任务），这里只是停——进程关闭时调用，之后不会再有人登记。
    """
    with _lock:
        if _scheduler is not None:
            _scheduler.shutdown(wait=wait)


def reset():
    """测试用：停掉并丢弃。"""
    global _scheduler, _started
    with _lock:
        if _scheduler is not None:
            _scheduler.shutdown(wait=False)
        _scheduler = None
        _started = False
