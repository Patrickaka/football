import threading
import time
import unittest

from src.foundation.tasks.scheduler import TaskScheduler


ASYNC_TEST_TIMEOUT_SECONDS = 5


class TaskSchedulerTests(unittest.TestCase):
    def test_runs_submitted_task(self):
        done = []
        scheduler = TaskScheduler(max_workers=1)
        scheduler.submit('t', lambda: done.append('ran'))
        scheduler.start()
        scheduler.shutdown(wait=True)
        self.assertEqual(done, ['ran'])

    def test_runs_all_tasks(self):
        done = []
        scheduler = TaskScheduler(max_workers=2)
        for i in range(5):
            scheduler.submit(f't{i}', lambda i=i: done.append(i))
        scheduler.start()
        scheduler.shutdown(wait=True)
        self.assertEqual(sorted(done), [0, 1, 2, 3, 4])

    def test_respects_max_workers(self):
        """P4：启动期不允许所有预热任务同时抢 CPU。"""
        concurrent = []
        peak = [0]
        guard = threading.Lock()
        two_started = threading.Event()
        release = threading.Event()

        def work():
            with guard:
                concurrent.append(1)
                peak[0] = max(peak[0], len(concurrent))
                if len(concurrent) == 2:
                    two_started.set()
            release.wait()
            with guard:
                concurrent.pop()

        scheduler = TaskScheduler(max_workers=2)
        for i in range(6):
            scheduler.submit(f't{i}', work)
        scheduler.start()
        try:
            self.assertTrue(
                two_started.wait(timeout=ASYNC_TEST_TIMEOUT_SECONDS),
                '两个 worker 未能同时启动',
            )
            self.assertEqual(peak[0], 2)
        finally:
            release.set()
            scheduler.shutdown(wait=True)
        self.assertLessEqual(peak[0], 2)

    def test_higher_priority_runs_first(self):
        order = []
        scheduler = TaskScheduler(max_workers=1)
        scheduler.submit('low', lambda: order.append('low'), priority=9)
        scheduler.submit('high', lambda: order.append('high'), priority=1)
        scheduler.start()
        scheduler.shutdown(wait=True)
        self.assertEqual(order[0], 'high')

    def test_failing_task_does_not_stop_others(self):
        done = []

        def boom():
            raise RuntimeError('task failed')

        scheduler = TaskScheduler(max_workers=1)
        scheduler.submit('bad', boom, priority=1)
        scheduler.submit('good', lambda: done.append('ok'), priority=2)
        scheduler.start()
        scheduler.shutdown(wait=True)
        self.assertEqual(done, ['ok'])

    def test_results_record_success_and_failure(self):
        def boom():
            raise RuntimeError('task failed')

        scheduler = TaskScheduler(max_workers=1)
        scheduler.submit('bad', boom)
        scheduler.submit('good', lambda: 'value')
        scheduler.start()
        scheduler.shutdown(wait=True)
        results = scheduler.results()
        self.assertEqual(results['good']['status'], 'ok')
        self.assertEqual(results['bad']['status'], 'error')
        self.assertIn('task failed', results['bad']['error'])

    def test_submit_after_start_is_rejected(self):
        scheduler = TaskScheduler(max_workers=1)
        scheduler.start()
        with self.assertRaises(RuntimeError):
            scheduler.submit('late', lambda: None)
        scheduler.shutdown(wait=True)

    def test_zero_max_workers_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            TaskScheduler(max_workers=0)

    def test_negative_max_workers_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            TaskScheduler(max_workers=-1)

    def test_duplicate_task_name_rejected(self):
        scheduler = TaskScheduler(max_workers=1)
        scheduler.submit('dup', lambda: 'first')
        with self.assertRaises(ValueError) as ctx:
            scheduler.submit('dup', lambda: 'second')
        self.assertIn('dup', str(ctx.exception))
        scheduler.start()
        scheduler.shutdown(wait=True)
        self.assertEqual(scheduler.results()['dup']['value'], 'first')


class IsRunningTests(unittest.TestCase):
    """健康检查要问的是「后台任务真的在跑吗」。

    只判断「调度器对象存在」的话，那个判断永远为真——它不是信号，
    是会让人误以为一切正常的那种噪声。
    """

    def test_not_running_before_start(self):
        self.assertFalse(TaskScheduler().is_running())

    def test_running_after_start(self):
        scheduler = TaskScheduler()
        scheduler.start()
        self.addCleanup(scheduler.shutdown, wait=False)
        self.assertTrue(scheduler.is_running())

    def test_not_running_after_shutdown(self):
        scheduler = TaskScheduler()
        scheduler.start()
        scheduler.shutdown(wait=False)
        self.assertFalse(scheduler.is_running())


class TaskCountTests(unittest.TestCase):
    """区分「没有任务要跑」和「有任务却没在跑」。"""

    def test_starts_at_zero(self):
        self.assertEqual(TaskScheduler().task_count(), 0)

    def test_counts_both_kinds(self):
        scheduler = TaskScheduler()
        scheduler.submit('once', lambda: None)
        scheduler.submit_periodic('loop', lambda: None, interval_seconds=60)
        self.assertEqual(scheduler.task_count(), 2)

    def test_survives_start(self):
        """start() 会清空待跑列表，登记数不能跟着清零——否则运行中的调度器
        看起来就像「没有任务」。"""
        scheduler = TaskScheduler()
        scheduler.submit_periodic('loop', lambda: None, interval_seconds=60)
        scheduler.start()
        self.addCleanup(scheduler.shutdown, wait=False)
        self.assertEqual(scheduler.task_count(), 1)


class PeriodicTaskTests(unittest.TestCase):
    def test_periodic_task_runs_repeatedly(self):
        runs = []
        third_run = threading.Event()

        def tick():
            runs.append(1)
            if len(runs) >= 3:
                third_run.set()

        scheduler = TaskScheduler(max_workers=2)
        scheduler.submit_periodic('tick', tick, interval_seconds=0.05)
        scheduler.start()
        try:
            self.assertTrue(
                third_run.wait(timeout=ASYNC_TEST_TIMEOUT_SECONDS),
                f'{ASYNC_TEST_TIMEOUT_SECONDS}s 内至少应跑 3 次，实际 {len(runs)}',
            )
        finally:
            scheduler.shutdown(wait=True)
        self.assertGreaterEqual(len(runs), 3)

    def test_periodic_task_stops_after_shutdown(self):
        runs = []
        first_run = threading.Event()

        def tick():
            runs.append(1)
            first_run.set()

        scheduler = TaskScheduler(max_workers=2)
        scheduler.submit_periodic('tick', tick, interval_seconds=0.05)
        scheduler.start()
        try:
            self.assertTrue(
                first_run.wait(timeout=ASYNC_TEST_TIMEOUT_SECONDS),
                f'周期任务在 {ASYNC_TEST_TIMEOUT_SECONDS}s 内未启动',
            )
        finally:
            scheduler.shutdown(wait=True)
        settled = len(runs)
        time.sleep(0.1)
        self.assertEqual(len(runs), settled, 'shutdown 后不得继续执行')

    def test_periodic_failure_does_not_stop_subsequent_runs(self):
        runs = []
        third_run = threading.Event()

        def flaky():
            runs.append(1)
            if len(runs) >= 3:
                third_run.set()
            if len(runs) == 1:
                raise RuntimeError('first run fails')

        scheduler = TaskScheduler(max_workers=2)
        scheduler.submit_periodic('flaky', flaky, interval_seconds=0.05)
        scheduler.start()
        try:
            self.assertTrue(
                third_run.wait(timeout=ASYNC_TEST_TIMEOUT_SECONDS),
                f'首次失败后 {ASYNC_TEST_TIMEOUT_SECONDS}s 内至少应跑 3 次，'
                f'实际 {len(runs)}',
            )
        finally:
            scheduler.shutdown(wait=True)
        self.assertGreaterEqual(len(runs), 3, '单次失败不得终止后续周期')
        self.assertEqual(scheduler.results()['flaky']['status'], 'ok',
                         '最近一次成功后状态应为 ok')

    def test_periodic_rejects_non_positive_interval(self):
        scheduler = TaskScheduler(max_workers=1)
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                scheduler.submit_periodic('t', lambda: None, interval_seconds=bad)

    def test_periodic_shares_duplicate_name_check_with_submit(self):
        scheduler = TaskScheduler(max_workers=1)
        scheduler.submit('dup', lambda: None)
        with self.assertRaises(ValueError):
            scheduler.submit_periodic('dup', lambda: None, interval_seconds=1)

    def test_results_records_run_count(self):
        second_run = threading.Event()
        calls = []

        def tick():
            calls.append(1)
            if len(calls) >= 2:
                second_run.set()

        scheduler = TaskScheduler(max_workers=2)
        scheduler.submit_periodic('tick', tick, interval_seconds=0.05)
        scheduler.start()
        try:
            self.assertTrue(
                second_run.wait(timeout=ASYNC_TEST_TIMEOUT_SECONDS),
                f'周期任务在 {ASYNC_TEST_TIMEOUT_SECONDS}s 内只执行了 '
                f'{len(calls)} 次',
            )
        finally:
            scheduler.shutdown(wait=True)
        self.assertGreaterEqual(scheduler.results()['tick']['runs'], 2)


class StartupStaggerTests(unittest.TestCase):
    """周期任务首轮错开。

    **这台机器上因此整机冻死过两次**：重启把所有缓存清零，六个周期任务同时
    开跑，内存被吃穿到连 sshd 都 fork 不出来——TCP 连得上、ping 通、
    SSH 在 banner 交换阶段超时，而内核忙到没机会跑 OOM killer。
    """

    def test_zero_stagger_keeps_the_old_behaviour(self):
        """**默认不错开**：错开多少是部署形态的事，不该由调度器决定。"""
        scheduler = TaskScheduler(max_workers=4)
        self.assertEqual(scheduler.startup_stagger_seconds, 0)

    def test_a_negative_stagger_is_refused(self):
        with self.assertRaises(ValueError):
            TaskScheduler(max_workers=2, startup_stagger_seconds=-1)

    def test_the_first_round_is_delayed_by_position(self):
        """第 i 个任务等 `i × 间隔`。**要断言每一个的延迟**，
        只看「有没有延迟」的话，把它写成常数也一样通过（判据 5）。"""
        delays = []
        scheduler = TaskScheduler(max_workers=4, startup_stagger_seconds=7)
        original = scheduler._run_periodic

        def record(name, fn, interval, start_delay=0):
            delays.append((name, start_delay))

        scheduler._run_periodic = record
        for name in ('a', 'b', 'c'):
            scheduler.submit_periodic(name, lambda: None, interval_seconds=60)
        scheduler.start()
        scheduler.shutdown()
        self.assertEqual(sorted(delays), [('a', 0), ('b', 7), ('c', 14)])

    def test_the_delay_only_postpones_the_first_round(self):
        """错开只推迟第一轮，之后的间隔不受影响——否则周期就被改掉了。"""
        runs = []
        third_run = threading.Event()

        def tick():
            runs.append(1)
            if len(runs) >= 3:
                third_run.set()

        scheduler = TaskScheduler(max_workers=2, startup_stagger_seconds=0)
        scheduler.submit_periodic('t', tick, interval_seconds=0.01)
        scheduler.start()
        try:
            self.assertTrue(
                third_run.wait(timeout=ASYNC_TEST_TIMEOUT_SECONDS),
                f'周期任务只执行了 {len(runs)} 次',
            )
        finally:
            scheduler.shutdown()
        self.assertGreaterEqual(len(runs), 3)

    def test_shutdown_interrupts_the_stagger_wait(self):
        """**等待期间也要响应停机**：一个错开五分钟的任务不该把停机拖住五分钟。

        要看的那个任务必须排在第二位——**第一个的错开量是 0**，
        它会立刻开跑，拿它测等于什么也没测（判据 23）。
        """
        started = threading.Event()
        scheduler = TaskScheduler(max_workers=3, startup_stagger_seconds=30)
        scheduler.submit_periodic('first', lambda: None, interval_seconds=60)
        scheduler.submit_periodic('slow', lambda: started.set(),
                                  interval_seconds=60)
        scheduler.start()
        shutdown_done = threading.Event()

        def stop():
            scheduler.shutdown()
            shutdown_done.set()

        stopper = threading.Thread(target=stop, daemon=True)
        stopper.start()
        self.assertTrue(
            shutdown_done.wait(timeout=ASYNC_TEST_TIMEOUT_SECONDS),
            '停机被错开的等待拖住了',
        )
        stopper.join()
        self.assertFalse(started.is_set(), '错开期间不该已经跑过一轮')

    def test_a_staggered_task_has_not_run_yet(self):
        """错开生效的正面证据：等一小会儿，第二个任务还没动。"""
        first_ran = threading.Event()
        second_ran = threading.Event()
        scheduler = TaskScheduler(max_workers=4, startup_stagger_seconds=30)
        scheduler.submit_periodic('first', first_ran.set,
                                  interval_seconds=60)
        scheduler.submit_periodic('second', second_ran.set,
                                  interval_seconds=60)
        scheduler.start()
        try:
            self.assertTrue(
                first_ran.wait(timeout=ASYNC_TEST_TIMEOUT_SECONDS),
                '第一个任务不该被错开',
            )
        finally:
            scheduler.shutdown()
        self.assertFalse(second_ran.is_set(), '第二个任务应当还在等')


if __name__ == '__main__':
    unittest.main()
