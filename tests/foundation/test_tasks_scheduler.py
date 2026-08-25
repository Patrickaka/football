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


if __name__ == '__main__':
    unittest.main()
