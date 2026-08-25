import contextlib
import unittest

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.deps import Settings


class HealthTests(unittest.TestCase):
    def setUp(self):
        app = create_app(Settings(redis_url=None, mysql_url='sqlite+pysqlite:///:memory:'))
        # TestClient 只有在作为上下文管理器使用时才会触发 ASGI lifespan
        # （Starlette/FastAPI 标准行为），app.state 上的 cache/db/tasks 由
        # lifespan 装配；用 ExitStack + addCleanup 保证测试结束时退出
        # 上下文，从而触发 lifespan 的关闭路径（tasks.shutdown 等）。
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.client = stack.enter_context(TestClient(app))

    def test_healthz_returns_ok(self):
        response = self.client.get('/healthz')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'ok')

    def test_healthz_reports_components(self):
        payload = self.client.get('/healthz').json()
        self.assertIn('cache', payload['components'])
        self.assertIn('database', payload['components'])

    def test_healthz_reports_tasks_component_ok(self):
        payload = self.client.get('/healthz').json()
        self.assertIn('tasks', payload['components'])
        self.assertEqual(payload['components']['tasks'], 'ok')

    def test_openapi_schema_is_served(self):
        response = self.client.get('/openapi.json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('/healthz', response.json()['paths'])

    def test_unknown_path_returns_404(self):
        self.assertEqual(self.client.get('/api/not-registered').status_code, 404)


if __name__ == '__main__':
    unittest.main()
