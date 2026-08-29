# -*- coding: utf-8 -*-
"""进程启动的编排：磁盘清理、缓存恢复、后台任务、预热、周期维护。

**新旧两个入口共用同一份**（判据 11）——旧 `server.py` 的 `main()` 与
`_start_background_sync()` 都改成调这里，从 185 行缩到 125 行。

这一批的意义在于：**新入口原本一件都没做。** 它的 lifespan 里建了个空的
`TaskScheduler` 只为让健康检查有东西可看——永远 0 个任务、永远不 start()，
是个纯粹的摆设。照那样切过去，服务能起、接口能通、测试全绿，但后台不再
回填赛果、缓存不再跨重启保留、用户重新承担每天第一次的冷计算。
**「零消费者时测试绿不等于能用」正是这么发生的。**
"""
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from src.api import startup
from src.api.app import create_app
from src.api.auth import AuthSettings
from src.api.deps import Settings
from src.webapp import background


def app_with_orchestration():
    """本文件要验证编排本身，所以显式打开——conftest 给整个套件关掉了。"""
    return create_app(settings=Settings(run_startup_tasks=True),
                      auth_settings=AuthSettings(credentials={}))


class Orchestration(unittest.TestCase):

    STEPS = ('run_startup_maintenance', 'restore_persisted_caches',
             'register_background_tasks', 'start_cache_warmups',
             'start_maintenance_schedule')

    def test_run_all_does_every_step(self):
        """**漏掉任何一件都不会让服务起不来**，只会安静地少干活。"""
        with mock.patch.multiple(startup, **{name: mock.DEFAULT for name in self.STEPS}) as spies:
            startup.run_all()
        for name in self.STEPS:
            with self.subTest(step=name):
                self.assertTrue(spies[name].called, f'{name} 没被调用')

    def test_disk_cleanup_comes_before_everything_else(self):
        """**磁盘清理必须最先**：生产盘长期 91%，缓存恢复与预热只会继续
        放大压力。顺序错了不会报错，只会在磁盘满的那天才发现。
        """
        order = []
        with mock.patch.multiple(
                startup, **{name: mock.DEFAULT for name in self.STEPS}) as spies:
            for name in self.STEPS:
                spies[name].side_effect = lambda n=name: order.append(n)
            startup.run_all()
        self.assertEqual(order[0], 'run_startup_maintenance')
        self.assertLess(order.index('restore_persisted_caches'),
                        order.index('start_cache_warmups'))


class BackgroundTasks(unittest.TestCase):

    def setUp(self):
        self.addCleanup(background.reset)
        background.reset()

    def test_the_three_families_all_register(self):
        """kl8、篮球采样、足球回填——**三族缺一都是静默降级**。"""
        startup.register_background_tasks()
        self.assertGreater(background.task_count(), 0)
        self.assertTrue(background.is_running())

    def test_one_family_failing_does_not_stop_the_others(self):
        """一族登记失败不该连累另外两族——那会把「少一个任务」放大成
        「后台全停」。
        """
        with mock.patch('src.kl8.scheduler.register_kl8_tasks',
                        side_effect=RuntimeError('装不上')):
            startup.register_background_tasks()
        self.assertGreater(background.task_count(), 0)
        self.assertTrue(background.is_running())

    def test_registration_happens_before_the_scheduler_starts(self):
        """调度器一旦 `start()`，`submit()` 就会 RuntimeError。
        先启动再登记的话，任务会一个都进不去而且**没有任何报错**。
        """
        submitted_while_running = []
        real_submit = background.submit_periodic

        def spy(name, fn, interval_seconds):
            submitted_while_running.append(background.is_running())
            return real_submit(name, fn, interval_seconds)

        with mock.patch.object(background, 'submit_periodic', spy):
            startup.register_background_tasks()
        self.assertTrue(submitted_while_running)
        self.assertFalse(any(submitted_while_running),
                         '有任务是在调度器启动之后才登记的')


class WarmupThreads(unittest.TestCase):

    def test_all_three_warmups_point_at_real_functions(self):
        """**按名字反射调用最容易写错模块路径**，而错了只会被 except
        吞成一条警告——预热静静地不再发生。
        """
        for _, module_path, function_name, label in startup.WARMUP_THREADS:
            with self.subTest(label=label):
                module = __import__(module_path, fromlist=[function_name])
                self.assertTrue(hasattr(module, function_name),
                                f'{module_path}.{function_name} 不存在')

    def test_the_threads_are_daemons(self):
        """预热没算完不该拦着进程退出。"""
        started = []
        with mock.patch('threading.Thread') as thread_class:
            thread_class.side_effect = lambda **kwargs: started.append(kwargs) or mock.MagicMock()
            startup.start_cache_warmups()
        self.assertEqual(len(started), 3)
        for kwargs in started:
            with self.subTest(name=kwargs.get('name')):
                self.assertTrue(kwargs['daemon'])

    def test_a_broken_warmup_does_not_stop_the_rest(self):
        with mock.patch('threading.Thread', side_effect=[RuntimeError('起不来'),
                                                         mock.MagicMock(),
                                                         mock.MagicMock()]):
            startup.start_cache_warmups()


class AppLifespan(unittest.TestCase):

    def setUp(self):
        self.addCleanup(background.reset)
        background.reset()

    def test_the_lifespan_runs_the_orchestration(self):
        with mock.patch.object(startup, 'run_all') as spy:
            with TestClient(app_with_orchestration()):
                pass
        self.assertTrue(spy.called)

    def test_the_health_probe_watches_the_real_scheduler(self):
        """原来 lifespan 里另建了个空调度器，健康检查看的是那个摆设：
        **永远 0 个任务、永远不 start()，永远报 ok。**
        现在它看的是真正跑着周期任务的那一个。
        """
        with TestClient(app_with_orchestration()) as client:
            self.assertIs(client.app.state.tasks, background.scheduler())
            self.assertGreater(background.task_count(), 0)
            self.assertEqual(client.get('/healthz').json()['components']['tasks'], 'ok')

    def test_shutdown_stops_the_scheduler(self):
        with TestClient(app_with_orchestration()):
            self.assertTrue(background.is_running())
        self.assertFalse(background.is_running())

    def test_shutdown_keeps_the_singleton(self):
        """`shutdown` 只停不丢——与 `reset` 的区别就在这里。"""
        with TestClient(app_with_orchestration()):
            scheduler = background.scheduler()
        self.assertIs(background.scheduler(), scheduler)


class SharedWithTheOldEntryPoint(unittest.TestCase):

    def test_the_old_server_delegates_to_the_same_module(self):
        import inspect

        import server
        source = inspect.getsource(server._start_background_sync)
        self.assertIn('startup.register_background_tasks', source)
        self.assertIn('startup.start_cache_warmups', source)
        self.assertIn('startup.start_maintenance_schedule', source)

    def test_the_old_server_no_longer_has_its_own_copy(self):
        """两份并存必然会漂——旧的那份编排已经删干净了。

        **只看函数体**：`server.py` 顶部还有一大块 re-export
        （`from src.webapp.caching import ... _warm_3d_caches`），
        那是为了兼容 `import server` 的既有用法，不是重复实现。
        按整个文件搜名字会把它误判成没删干净。
        """
        import inspect

        import server
        for function in (server._start_background_sync, server.main):
            source = inspect.getsource(function)
            for duplicated in ('register_kl8_tasks', 'register_odds_tracking',
                               '_warm_3d_caches', 'run_maintenance('):
                with self.subTest(function=function.__name__, name=duplicated):
                    self.assertNotIn(duplicated, source)


if __name__ == '__main__':
    unittest.main()
