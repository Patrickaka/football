import unittest

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


if __name__ == '__main__':
    unittest.main()
