import contextlib
import unittest

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.deps import Settings


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(Settings(redis_url=None, mysql_url='sqlite+pysqlite:///:memory:'))
        # TestClient 只有在作为上下文管理器使用时才会触发 ASGI lifespan
        # （Starlette/FastAPI 标准行为），app.state 上的 cache/db/tasks 由
        # lifespan 装配；用 ExitStack + addCleanup 保证测试结束时退出
        # 上下文，从而触发 lifespan 的关闭路径（tasks.shutdown 等）。
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.client = stack.enter_context(TestClient(self.app))

    def test_healthz_returns_ok(self):
        response = self.client.get('/healthz')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'ok')

    def test_healthz_reports_components(self):
        payload = self.client.get('/healthz').json()
        self.assertIn('cache', payload['components'])
        self.assertIn('database', payload['components'])

    def test_healthz_reports_tasks_component_ok(self):
        """当前这个 app 没有登记任何后台任务——「没有任务要跑」是正常空闲。"""
        payload = self.client.get('/healthz').json()
        self.assertIn('tasks', payload['components'])
        self.assertEqual(payload['components']['tasks'], 'ok')
        self.assertEqual(self.app.state.tasks.task_count(), 0)

    def test_registered_but_unstarted_tasks_report_degraded(self):
        """有任务却没在跑才是故障。首版探针只判断对象存在，这种情况照报 ok。"""
        from src.api.routers.health import _probe_tasks
        from src.foundation.tasks import TaskScheduler

        idle = TaskScheduler()
        self.assertEqual(_probe_tasks(idle), 'ok')

        idle.submit_periodic('x', lambda: None, interval_seconds=60)
        self.assertEqual(_probe_tasks(idle), 'degraded')

        idle.start()
        self.addCleanup(idle.shutdown, wait=False)
        self.assertEqual(_probe_tasks(idle), 'ok')

    def test_healthz_reports_tasks_component_error_when_missing(self):
        """让 _probe_tasks 走 error 分支：只有 tasks 键真正为 None 时才会
        触发。硬编码 `return 'ok'` 的实现在这个用例下会返回错误的 'ok'，
        从而被本用例捕获——这是上一版遗漏的、能鉴别探测逻辑是否真的在
        工作的反例分支。
        """
        original_tasks = self.app.state.tasks
        # 必须在 stack.close()（setUp 里注册，触发 lifespan 关闭路径，
        # 其中会调用 app.state.tasks.shutdown()）之前把 tasks 恢复成真实
        # 调度器，否则关闭路径会因为 tasks 仍是 None 而报错。addCleanup
        # 按后进先出执行，这里比 setUp 里的 stack.close 后注册，会先跑。
        self.addCleanup(setattr, self.app.state, 'tasks', original_tasks)
        self.app.state.tasks = None
        payload = self.client.get('/healthz').json()
        self.assertEqual(payload['components']['tasks'], 'error')
        self.assertEqual(payload['status'], 'degraded')

    def test_openapi_schema_is_served(self):
        response = self.client.get('/openapi.json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('/healthz', response.json()['paths'])

    def test_unknown_path_returns_404(self):
        self.assertEqual(self.client.get('/api/not-registered').status_code, 404)


if __name__ == '__main__':
    unittest.main()
