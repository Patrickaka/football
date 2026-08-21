"""页面级抓取缓存/去重/限流的回归测试。

单场分析要打 4~6 个源站页面，其中 yazhi/daxiao 两页被平均盘与独赔重复请求。
这些测试锁住「重复 URL 只打一次、并发有上限、失败不入缓存」三条不变量。
"""

import threading
import time
import unittest

import src.football as fb


class FootballFetchCacheTests(unittest.TestCase):
    def setUp(self):
        self._original_raw = fb._fetch_raw
        fb.clear_fetch_cache()
        self.calls = []
        self.lock = threading.Lock()

    def tearDown(self):
        fb._fetch_raw = self._original_raw
        fb.clear_fetch_cache()

    def _stub(self, delay=0.0, body='BODY'):
        def _raw(url, encoding='gbk', referer=None):
            with self.lock:
                self.calls.append(url)
            if delay:
                time.sleep(delay)
            return f'{body}:{url}'
        return _raw

    def test_repeated_url_is_fetched_once_within_ttl(self):
        fb._fetch_raw = self._stub()
        url = f'{fb.BASE}/fenxi/yazhi-1.shtml'

        first = fb.fetch(url)
        second = fb.fetch(url)

        self.assertEqual(first, second)
        self.assertEqual(self.calls, [url])

    def test_concurrent_requests_for_same_url_hit_upstream_once(self):
        fb._fetch_raw = self._stub(delay=0.2)
        url = f'{fb.BASE}/fenxi/daxiao-1.shtml'
        threads = [threading.Thread(target=lambda: fb.fetch(url)) for _ in range(6)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(self.calls), 1)

    def test_upstream_concurrency_is_capped(self):
        state = {'now': 0, 'peak': 0}

        def _raw(url, encoding='gbk', referer=None):
            with self.lock:
                state['now'] += 1
                state['peak'] = max(state['peak'], state['now'])
            time.sleep(0.05)
            with self.lock:
                state['now'] -= 1
            return 'BODY'

        fb._fetch_raw = _raw
        threads = [
            threading.Thread(target=lambda i=i: fb.fetch(f'{fb.BASE}/fenxi/yazhi-{i}.shtml'))
            for i in range(fb.FETCH_MAX_CONCURRENCY * 3)
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertLessEqual(state['peak'], fb.FETCH_MAX_CONCURRENCY)

    def test_failures_are_not_cached(self):
        attempts = []

        def _raw(url, encoding='gbk', referer=None):
            attempts.append(url)
            raise OSError('upstream down')

        fb._fetch_raw = _raw
        for _ in range(2):
            with self.assertRaises(OSError):
                fb.fetch(f'{fb.BASE}/fenxi/yazhi-1.shtml')

        self.assertEqual(len(attempts), 2)

    def test_clear_fetch_cache_forces_refetch(self):
        fb._fetch_raw = self._stub()
        url = f'{fb.BASE}/fenxi/shuju-1.shtml'

        fb.fetch(url)
        fb.clear_fetch_cache()
        fb.fetch(url)

        self.assertEqual(len(self.calls), 2)


class FootballAnalyzeParallelFetchTests(unittest.TestCase):
    """analyze_match 的五组抓取必须真正并发，否则单场耗时是往返之和。"""

    def test_odds_pages_are_fetched_concurrently(self):
        barrier = threading.Barrier(5, timeout=5)
        original = {
            name: getattr(fb, name)
            for name in (
                'fetch_yazhi', 'fetch_ouzhi', 'fetch_daxiao',
                'fetch_team_strength', 'fetch_single_company_odds',
            )
        }

        def _blocking(*_args, **_kwargs):
            # 五个抓取都到齐才能放行；串行执行时这里会 BrokenBarrierError
            barrier.wait()
            raise RuntimeError('stubbed')

        for name in original:
            setattr(fb, name, _blocking)
        try:
            with self.assertRaises(ValueError) as ctx:
                fb.analyze_match({
                    'match_id': 'parallel-1',
                    'home': '主队', 'away': '客队', 'league': '英超', 'time': '',
                }, force_refresh=True)
        finally:
            for name, func in original.items():
                setattr(fb, name, func)

        # 只有五个抓取同时在飞，屏障才会放行并抛出 stubbed；
        # 串行执行时屏障超时崩溃，这里就看不到 stubbed。
        self.assertIn('亚盘数据获取失败', str(ctx.exception))
        self.assertIn('stubbed', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
