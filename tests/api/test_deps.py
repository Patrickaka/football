import asyncio
import unittest

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.api.deps import Settings, build_cache, build_database, get_cache, get_db, run_blocking


class SettingsTests(unittest.TestCase):
    def test_defaults_are_usable_without_env(self):
        settings = Settings()
        self.assertGreater(settings.cache_default_ttl, 0)
        self.assertGreaterEqual(settings.max_task_workers, 1)

    def test_from_env_reads_overrides(self):
        settings = Settings.from_env({'CACHE_DEFAULT_TTL': '99', 'MAX_TASK_WORKERS': '3'})
        self.assertEqual(settings.cache_default_ttl, 99)
        self.assertEqual(settings.max_task_workers, 3)


class BuildCacheTests(unittest.TestCase):
    def test_falls_back_to_memory_when_redis_absent(self):
        """Redis 不可用时降级为纯内存，服务仍可启动。"""
        cache = build_cache(Settings(redis_url=None))
        cache.set('k', 'v')
        self.assertEqual(cache.get('k', lambda: 'recomputed'), 'v')


class GetCacheGetDbTests(unittest.TestCase):
    """get_cache / get_db 是给业务路由用 Depends() 注入、给测试用
    dependency_overrides 替换的标准入口——这两点都要测到，否则它们
    存在的理由本身就没被验证过。
    """

    def setUp(self):
        settings = Settings(redis_url=None, mysql_url='sqlite+pysqlite:///:memory:')
        self.cache = build_cache(settings)
        self.db = build_database(settings)

        app = FastAPI()
        app.state.cache = self.cache
        app.state.db = self.db

        @app.get('/cache-id')
        def _cache_id(cache=Depends(get_cache)):
            return {'id': id(cache)}

        @app.get('/db-id')
        def _db_id(db=Depends(get_db)):
            return {'id': id(db)}

        self.app = app
        self.client = TestClient(app)

    def test_get_cache_returns_app_state_cache(self):
        response = self.client.get('/cache-id')
        self.assertEqual(response.json()['id'], id(self.cache))

    def test_get_db_returns_app_state_db(self):
        response = self.client.get('/db-id')
        self.assertEqual(response.json()['id'], id(self.db))

    def test_dependency_overrides_replaces_cache(self):
        fake_cache = object()
        self.app.dependency_overrides[get_cache] = lambda: fake_cache
        try:
            response = self.client.get('/cache-id')
            self.assertEqual(response.json()['id'], id(fake_cache))
        finally:
            self.app.dependency_overrides.clear()

    def test_dependency_overrides_replaces_db(self):
        fake_db = object()
        self.app.dependency_overrides[get_db] = lambda: fake_db
        try:
            response = self.client.get('/db-id')
            self.assertEqual(response.json()['id'], id(fake_db))
        finally:
            self.app.dependency_overrides.clear()


class RunBlockingTests(unittest.TestCase):
    def test_runs_sync_function_off_event_loop(self):
        import threading

        loop_thread = threading.current_thread().name
        seen = {}

        def work():
            seen['thread'] = threading.current_thread().name
            return 'done'

        async def main():
            return await run_blocking(work)

        result = asyncio.run(main())
        self.assertEqual(result, 'done')
        self.assertNotEqual(seen['thread'], loop_thread)

    def test_propagates_exception(self):
        def boom():
            raise ValueError('failed')

        async def main():
            return await run_blocking(boom)

        with self.assertRaises(ValueError):
            asyncio.run(main())


if __name__ == '__main__':
    unittest.main()
