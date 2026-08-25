import unittest

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.deps import Settings


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(Settings(redis_url=None, mysql_url='sqlite+pysqlite:///:memory:')))

    def test_healthz_returns_ok(self):
        response = self.client.get('/healthz')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'ok')

    def test_healthz_reports_components(self):
        payload = self.client.get('/healthz').json()
        self.assertIn('cache', payload['components'])
        self.assertIn('database', payload['components'])

    def test_openapi_schema_is_served(self):
        response = self.client.get('/openapi.json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('/healthz', response.json()['paths'])

    def test_unknown_path_returns_404(self):
        self.assertEqual(self.client.get('/api/not-registered').status_code, 404)


if __name__ == '__main__':
    unittest.main()
