import logging
import threading
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger('foundation.tasks')


class TaskScheduler:
    """并发受限的任务调度器，支持一次性任务与周期任务。

    一次性任务按 priority 升序执行（数值小者先跑），并发数受 max_workers 限制。
    周期任务在 start() 后按固定间隔重复执行，直到 shutdown()。
    单个任务异常不影响其余任务，周期任务的单次失败也不终止后续周期。

    注意每个周期任务会长期占用一个 worker，配置 max_workers 时要把它们算进去。
    """

    def __init__(self, max_workers=2):
        if max_workers <= 0:
            raise ValueError('max_workers must be > 0, got %r' % (max_workers,))
        self.max_workers = max_workers
        self._pending = []
        self._pending_names = set()
        self._periodic = []
        self._results = {}
        self._executor = None
        self._started = False
        # start() 会清空待跑列表，所以登记数要单独留一份——健康检查要区分
        # 「没有任务要跑」和「有任务却没在跑」，前者是正常的空闲。
        self._task_count = 0
        self._stop = threading.Event()
        self._guard = threading.Lock()

    def submit(self, name, fn, priority=5):
        with self._guard:
            if self._started:
                raise RuntimeError('调度器已启动，不接受新任务')
            if name in self._pending_names:
                raise ValueError('任务名重复: %r' % (name,))
            self._pending_names.add(name)
            self._pending.append((priority, len(self._pending), name, fn))
            self._task_count += 1

    def submit_periodic(self, name, fn, interval_seconds, priority=5):
        """登记周期任务：start() 后按间隔重复执行，shutdown() 时停止。

        与一次性任务共用同一套重名检查——results() 以任务名为键，
        重名会让先完成的那个结果被静默覆盖。
        """
        if interval_seconds <= 0:
            raise ValueError(
                'interval_seconds must be > 0, got %r' % (interval_seconds,))
        with self._guard:
            if self._started:
                raise RuntimeError('调度器已启动，不接受新任务')
            if name in self._pending_names:
                raise ValueError('任务名重复: %r' % (name,))
            self._pending_names.add(name)
            self._periodic.append((name, fn, interval_seconds, priority))
            self._task_count += 1

    def start(self):
        with self._guard:
            if self._started:
                return
            self._started = True
            tasks = sorted(self._pending)
            periodic = list(self._periodic)
            self._pending = []
            self._periodic = []
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix='PlatformTask'
        )
        if periodic and len(periodic) >= self.max_workers:
            log.warning(
                '周期任务数(%d) >= max_workers(%d)，一次性任务可能长时间得不到执行',
                len(periodic), self.max_workers,
            )
        for name, fn, interval, _ in periodic:
            self._executor.submit(self._run_periodic, name, fn, interval)
        for _, _, name, fn in tasks:
            self._executor.submit(self._run, name, fn)

    def is_running(self):
        """已 start() 且尚未 shutdown()。

        健康检查要问的是「后台任务真的在跑吗」，而不是「调度器对象存在吗」。
        后者永远为真，把它当健康信号等于没有信号。
        """
        return self._started and not self._stop.is_set()

    def task_count(self):
        """登记过的任务数（含已启动的）。

        健康检查靠它区分「没有任务要跑」与「有任务却没在跑」——前者是正常的
        空闲，后者才是故障。不区分的话，一个还没接后台任务的服务会天天报
        degraded，久而久之这个信号就没人看了。
        """
        return self._task_count

    def shutdown(self, wait=True):
        self._stop.set()
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None

    def results(self):
        with self._guard:
            return dict(self._results)

    def _run(self, name, fn):
        try:
            value = fn()
        except Exception as exc:
            log.exception('后台任务失败: name=%s', name)
            with self._guard:
                self._results[name] = {'status': 'error', 'error': str(exc)}
            return
        with self._guard:
            self._results[name] = {'status': 'ok', 'value': value}

    def _run_periodic(self, name, fn, interval):
        """周期执行直到 shutdown。

        单次失败只记录不终止循环——一次瞬时故障（源站抖动、网络超时）
        不该让这个任务永久停摆。results() 里记录最近一次的结果与累计执行次数。
        """
        runs = 0
        while not self._stop.is_set():
            runs += 1
            try:
                value = fn()
            except Exception as exc:
                log.exception('周期任务失败: name=%s run=%d', name, runs)
                with self._guard:
                    self._results[name] = {
                        'status': 'error', 'error': str(exc), 'runs': runs}
            else:
                with self._guard:
                    self._results[name] = {
                        'status': 'ok', 'value': value, 'runs': runs}
            if self._stop.wait(interval):
                return
