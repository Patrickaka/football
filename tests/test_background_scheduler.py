"""进程唯一的后台调度器。

迁移前后台任务分散在三处：kl8 用 APScheduler、篮球采样自建 TaskScheduler、
另有裸线程做缓存预热。代价不是多写了几行，而是**没有任何一个地方能回答
「现在后台在跑什么」**——健康检查只看得到其中一套。
"""
import threading
import time
import unittest

from src.webapp import background


class _Base(unittest.TestCase):
    def setUp(self):
        background.reset()
        self.addCleanup(background.reset)


class RegistrationTests(_Base):
    def test_scheduler_is_a_process_singleton(self):
        self.assertIs(background.scheduler(), background.scheduler())

    def test_tasks_accumulate_before_start(self):
        background.submit_periodic('a', lambda: None, 60)
        background.submit_periodic('b', lambda: None, 60)
        self.assertEqual(background.task_count(), 2)

    def test_duplicate_names_are_rejected_not_fatal(self):
        """重名任务会让 results() 里先完成的那个被静默覆盖。拒绝它，
        但不能因此让启动流程崩掉——一个业务登记失败不该拖垮其余业务。"""
        self.assertTrue(background.submit_periodic('a', lambda: None, 60))
        self.assertFalse(background.submit_periodic('a', lambda: None, 60))
        self.assertEqual(background.task_count(), 1)

    def test_registration_after_start_is_rejected_not_fatal(self):
        """启动之后不再接受新任务——任务集合在启动那一刻固定下来，
        「后台在跑什么」才有一个确定的答案。"""
        background.submit_periodic('a', lambda: None, 60)
        background.start()
        self.assertFalse(background.submit_periodic('b', lambda: None, 60))


class StartTests(_Base):
    def test_not_running_before_start(self):
        background.submit_periodic('a', lambda: None, 60)
        self.assertFalse(background.is_running())

    def test_running_after_start(self):
        background.submit_periodic('a', lambda: None, 60)
        background.start()
        self.assertTrue(background.is_running())

    def test_start_is_idempotent(self):
        background.submit_periodic('a', lambda: None, 60)
        first = background.start()
        self.assertIs(background.start(), first)

    def test_registered_task_actually_runs(self):
        """登记了却不跑，是这类接线最典型的无声失败。"""
        ran = threading.Event()
        background.submit_periodic('a', ran.set, 60)
        background.start()
        self.assertTrue(ran.wait(timeout=5), '登记的任务没有被执行')

    def test_worker_budget_exceeds_periodic_tasks(self):
        """周期任务会长期占住 worker。预算不够的话一次性任务永远排不上队，
        而且只会表现为「那个任务好像没跑」，不会报错。"""
        self.assertGreater(background.MAX_WORKERS, 4)


class IsolationTests(_Base):
    def test_one_failing_task_does_not_stop_the_others(self):
        """一个任务抛异常不该让其余任务跟着停。"""
        ok = threading.Event()

        def boom():
            raise RuntimeError('炸了')

        background.submit_periodic('boom', boom, 60)
        background.submit_periodic('ok', ok.set, 60)
        background.start()
        self.assertTrue(ok.wait(timeout=5))

    def test_shutdown_stops_reporting_running(self):
        background.submit_periodic('a', lambda: None, 60)
        background.start()
        background.reset()
        self.assertFalse(background.is_running())
        self.assertEqual(background.task_count(), 0)


if __name__ == '__main__':
    unittest.main()


class Kl8RegistrationTests(_Base):
    """快乐8的三个周期任务登记到同一个调度器。

    迁移前它们用 APScheduler，另带一条「未安装则起三个裸线程」的降级分支。
    换成 foundation/tasks 之后两条路都不需要——它本来就是线程实现，
    没有可选依赖。
    """

    def _register(self, submit=None):
        from src.kl8.scheduler import register_kl8_tasks

        return register_kl8_tasks(submit or background.submit_periodic)

    def test_registers_three_tasks(self):
        names = self._register()
        self.assertEqual(len(names), 3)
        self.assertEqual(background.task_count(), 3)

    def test_task_names_are_stable(self):
        """任务名是 results() 的键，也是排查时的抓手，不该随手改。"""
        self.assertEqual(sorted(self._register()),
                         ['kl8_auto_refresh', 'kl8_history_backfill',
                          'kl8_strategy_verification'])

    def test_intervals_match_the_previous_schedule(self):
        """迁移不该顺手改变节奏：每小时更新、每 10 分钟补数、每 2 小时验证。"""
        calls = []
        self._register(lambda name, fn, interval: calls.append((name, interval)) or True)
        self.assertEqual(dict(calls), {
            'kl8_auto_refresh': 3600,
            'kl8_history_backfill': 600,
            'kl8_strategy_verification': 7200,
        })

    def test_registration_failure_is_reported_not_fatal(self):
        """一个任务登记失败不该拖垮其余任务。"""
        names = self._register(lambda name, fn, interval: name != 'kl8_auto_refresh')
        self.assertEqual(sorted(names),
                         ['kl8_history_backfill', 'kl8_strategy_verification'])

    def test_all_periodic_tasks_fit_the_worker_budget(self):
        """kl8 三个 + 篮球采样一个 = 四个周期任务，每个长期占一个 worker。"""
        self._register()
        background.submit_periodic('basketball_odds_tracking', lambda: None, 900)
        self.assertLess(background.task_count(), background.MAX_WORKERS,
                        '周期任务占满 worker，一次性任务将永远排不上队')
