import unittest
from unittest import mock

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.deps import Settings, get_executor, shutdown_executor


class ExecutorWorkersWiringTests(unittest.TestCase):
    """回归测试：lifespan 必须用 settings.executor_workers 完成
    get_executor() 的首次初始化，否则该字段就是有声明无消费者的
    预留式设计（与决定 1 要求 max_task_workers 必须真正装配是同一
    类问题）。

    get_executor 是模块级全局单例，只有"首次调用"时传入的 workers
    才会生效——这意味着本测试必须在一个干净的单例上验证，否则测不出
    任何东西：真实 pytest 按文件名顺序执行时，
    tests/api/test_deps.py::RunBlockingTests 会先跑，其中的
    run_blocking() 隐式以默认 workers=4 创建了全局单例；如果不显式
    重置，等到这里再断言 workers=7 会因为单例早已存在而必然失败
    （或者更隐蔽地——如果测试本身也没重置，会静默验证到错误的默认值
    而非真正生效的配置值）。所以本测试 setUp/tearDown 都显式清理，
    确保断言的是"这次 lifespan 调用真的让配置生效"，不是"蒙对了默认值"。
    """

    def setUp(self):
        shutdown_executor()
        self.addCleanup(shutdown_executor)

    def test_lifespan_seeds_executor_with_configured_worker_count(self):
        settings = Settings(
            redis_url=None,
            mysql_url='sqlite+pysqlite:///:memory:',
            executor_workers=7,
        )
        app = create_app(settings)
        with TestClient(app):
            executor = get_executor()
            self.assertEqual(executor._max_workers, 7)


class LifespanShutdownOrderTests(unittest.TestCase):
    """回归测试：优雅停机必须先排空 SWR 刷新，再停消费者，最后释放消费者
    依赖的资源，顺序为
    cache.wait_for_refreshes → tasks.shutdown → shutdown_executor → db.dispose。

    SWR 刷新线程是 daemon，进程退出即被杀，`finally: self.l2.unlock(key)`
    得不到执行，会在 Redis 残留一把 TTL 最长 lock_timeout 秒的锁——不先
    排空就会在每次重启后的头 30 秒复现 P1 惊群。后两者的顺序也不能反：
    应先停消费者（executor/tasks），再释放它们依赖的资源（db）。
    用 mock 记录调用序列钉死顺序。
    """

    def setUp(self):
        shutdown_executor()
        self.addCleanup(shutdown_executor)

    def test_shutdown_sequence_matches_required_order(self):
        order = []

        fake_cache = mock.Mock()
        fake_cache.lock_timeout = 30
        fake_cache.wait_for_refreshes.side_effect = (
            lambda timeout=None: order.append('cache.wait_for_refreshes')
        )

        fake_db = mock.Mock()
        fake_db.dispose.side_effect = lambda: order.append('db.dispose')

        # 关闭的是**进程级的那一个调度器**，不是 lifespan 里另建的对象。
        # 原来这里建了个空的 TaskScheduler 只为让健康检查有东西可看——
        # 永远 0 个任务、永远不 start()，是个摆设。现在 app.state.tasks
        # 指向 `src.webapp.background` 的单例，关闭也走它。
        def recording_background_shutdown(wait=True):
            order.append('tasks.shutdown')

        def recording_shutdown_executor():
            order.append('shutdown_executor')

        settings = Settings(redis_url=None, mysql_url='sqlite+pysqlite:///:memory:')

        with mock.patch('src.api.app.build_cache', return_value=fake_cache), \
                mock.patch('src.api.app.build_database', return_value=fake_db), \
                mock.patch('src.api.app.background.shutdown',
                           side_effect=recording_background_shutdown), \
                mock.patch(
                    'src.api.app.shutdown_executor', side_effect=recording_shutdown_executor
                ):
            app = create_app(settings)
            with TestClient(app):
                pass  # 进入/退出触发 lifespan 的启动与关闭

        self.assertEqual(
            order,
            ['cache.wait_for_refreshes', 'tasks.shutdown', 'shutdown_executor', 'db.dispose'],
        )


if __name__ == '__main__':
    unittest.main()
