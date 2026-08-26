import threading
import time
import unittest

from src.foundation.tasks.scheduler import TaskScheduler


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

        def work():
            with guard:
                concurrent.append(1)
                peak[0] = max(peak[0], len(concurrent))
            time.sleep(0.05)
            with guard:
                concurrent.pop()

        scheduler = TaskScheduler(max_workers=2)
        for i in range(6):
            scheduler.submit(f't{i}', work)
        scheduler.start()
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


if __name__ == '__main__':
    unittest.main()


class PeriodicTaskTests(unittest.TestCase):
    def test_periodic_task_runs_repeatedly(self):
        runs = []
        scheduler = TaskScheduler(max_workers=2)
        scheduler.submit_periodic('tick', lambda: runs.append(1), interval_seconds=0.05)
        scheduler.start()
        time.sleep(0.28)
        scheduler.shutdown(wait=True)
        self.assertGreaterEqual(len(runs), 3, f'0.28s 内至少应跑 3 次，实际 {len(runs)}')

    def test_periodic_task_stops_after_shutdown(self):
        runs = []
        scheduler = TaskScheduler(max_workers=2)
        scheduler.submit_periodic('tick', lambda: runs.append(1), interval_seconds=0.05)
        scheduler.start()
        time.sleep(0.15)
        scheduler.shutdown(wait=True)
        settled = len(runs)
        time.sleep(0.2)
        self.assertEqual(len(runs), settled, 'shutdown 后不得继续执行')

    def test_periodic_failure_does_not_stop_subsequent_runs(self):
        runs = []

        def flaky():
            runs.append(1)
            if len(runs) == 1:
                raise RuntimeError('first run fails')

        scheduler = TaskScheduler(max_workers=2)
        scheduler.submit_periodic('flaky', flaky, interval_seconds=0.05)
        scheduler.start()
        time.sleep(0.28)
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
        scheduler = TaskScheduler(max_workers=2)
        scheduler.submit_periodic('tick', lambda: None, interval_seconds=0.05)
        scheduler.start()
        time.sleep(0.18)
        scheduler.shutdown(wait=True)
        self.assertGreaterEqual(scheduler.results()['tick']['runs'], 2)
