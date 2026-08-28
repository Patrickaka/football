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

    def __init__(self, max_workers=2, startup_stagger_seconds=0):
        """`startup_stagger_seconds`：周期任务首轮之间的错开间隔。

        第 i 个周期任务等 `i × 间隔` 之后才跑第一轮，此后各按各的周期。
        **默认 0（不错开）**：错开多少是部署形态的事，取决于机器余量与
        任务本身有多重，不该由调度器替调用方决定。

        为什么需要它：进程刚起来时所有缓存都是空的，六个周期任务同时
        开跑等于把一天里最重的一次计算全挤在同一分钟。这台机器上因此
        整机冻死过两次——内存耗尽时连 sshd 都 fork 不出来。
        """
        if max_workers <= 0:
            raise ValueError('max_workers must be > 0, got %r' % (max_workers,))
        if startup_stagger_seconds < 0:
            raise ValueError('startup_stagger_seconds must be >= 0, got %r'
                             % (startup_stagger_seconds,))
        self.max_workers = max_workers
        self.startup_stagger_seconds = startup_stagger_seconds
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
        for index, (name, fn, interval, _) in enumerate(periodic):
            # 按登记顺序错开首轮。**顺序即优先级**：先登记的先跑，
            # 所以要把最该早出结果的任务排在前面。
            self._executor.submit(self._run_periodic, name, fn, interval,
                                  index * self.startup_stagger_seconds)
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

    def _run_periodic(self, name, fn, interval, start_delay):
        """周期执行直到 shutdown。

        单次失败只记录不终止循环——一次瞬时故障（源站抖动、网络超时）
        不该让这个任务永久停摆。results() 里记录最近一次的结果与累计执行次数。

        `start_delay` 只推迟**第一轮**，之后的间隔不受影响。等待期间也响应
        `shutdown()`——否则一个错开了五分钟的任务会把停机拖住五分钟。

        **没有默认值是有意的**：`start()` 总是显式传，给它配一个默认值等于
        写下一条任何调用方都走不到的分支（判据 9 第一类）。变异验证里把默认值
        改成 60 一样全绿，正是这个原因。
        """
        if start_delay and self._stop.wait(start_delay):
            return
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
