import logging
import threading
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger('foundation.tasks')


class TaskScheduler:
    """并发受限的一次性任务调度器。

    任务按 priority 升序执行（数值小者先跑），并发数受 max_workers 限制。
    单个任务异常不影响其余任务。
    """

    def __init__(self, max_workers=2):
        self.max_workers = max_workers
        self._pending = []
        self._results = {}
        self._executor = None
        self._started = False
        self._guard = threading.Lock()

    def submit(self, name, fn, priority=5):
        with self._guard:
            if self._started:
                raise RuntimeError('调度器已启动，不接受新任务')
            self._pending.append((priority, len(self._pending), name, fn))

    def start(self):
        with self._guard:
            if self._started:
                return
            self._started = True
            tasks = sorted(self._pending)
            self._pending = []
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix='PlatformTask'
        )
        for _, _, name, fn in tasks:
            self._executor.submit(self._run, name, fn)

    def shutdown(self, wait=True):
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
