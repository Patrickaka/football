import asyncio
import unittest

from src.api.deps import Settings, build_cache, run_blocking


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
