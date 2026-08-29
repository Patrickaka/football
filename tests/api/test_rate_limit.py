# -*- coding: utf-8 -*-
"""入站限流：按客户端分桶的令牌桶。

单桶复用出站抓取用的 `RateLimiter`（判据 11），**集合没有复用
`DomainRateLimiters`**——那个字典永不淘汰。出站的 key 是域名、就那么几个；
入站的 key 是客户端 IP、无上限，照搬过来等于给了任何人一条用随机源 IP
把内存撑爆的路。所以这里的集合带容量上限与 LRU 淘汰。

**默认不启用**（`RATE_LIMIT_PER_SEC` 未配置）：一个限错的限流器比没有
更糟，它会把正常用户挡在外面而且没人知道为什么。要开就得明确配。
"""
import unittest

from fastapi.testclient import TestClient

from src.api.app import build_rate_limiters, create_app
from src.api.auth import AuthSettings
from src.api.deps import Settings
from src.api.rate_limit import EXEMPT_PATHS, ClientRateLimiters


class Buckets(unittest.TestCase):

    def test_the_burst_is_spent_before_throttling_starts(self):
        limiters = ClientRateLimiters(rate_per_sec=1, burst=3)
        waits = [limiters.acquire('1.1.1.1', now=1000.0) for _ in range(5)]
        self.assertEqual(waits[:3], [0, 0, 0])
        self.assertTrue(all(w > 0 for w in waits[3:]))

    def test_tokens_come_back_over_time(self):
        limiters = ClientRateLimiters(rate_per_sec=2, burst=1)
        self.assertEqual(limiters.acquire('1.1.1.1', now=1000.0), 0)
        self.assertGreater(limiters.acquire('1.1.1.1', now=1000.0), 0)
        self.assertEqual(limiters.acquire('1.1.1.1', now=1000.5), 0)

    def test_a_faster_rate_refills_sooner(self):
        """**反方向**：速率是真的在起作用，不是摆设。"""
        slow = ClientRateLimiters(rate_per_sec=1, burst=1)
        fast = ClientRateLimiters(rate_per_sec=10, burst=1)
        for limiters in (slow, fast):
            limiters.acquire('1.1.1.1', now=1000.0)
        self.assertGreater(slow.acquire('1.1.1.1', now=1000.2), 0)
        self.assertEqual(fast.acquire('1.1.1.1', now=1000.2), 0)

    def test_clients_do_not_share_a_bucket(self):
        """一个客户端把自己限死了，不该连累别人。"""
        limiters = ClientRateLimiters(rate_per_sec=1, burst=1)
        limiters.acquire('1.1.1.1', now=1000.0)
        self.assertGreater(limiters.acquire('1.1.1.1', now=1000.0), 0)
        self.assertEqual(limiters.acquire('2.2.2.2', now=1000.0), 0)

    def test_the_bucket_count_is_capped(self):
        """**桶数必须有上限**——key 是 IP，不封顶就是一条撑爆内存的路。"""
        limiters = ClientRateLimiters(rate_per_sec=1, maxsize=8)
        for i in range(200):
            limiters.acquire(f'10.0.0.{i}', now=1000.0)
        self.assertEqual(limiters.bucket_count(), 8)

    def test_eviction_takes_the_least_recently_used(self):
        """淘汰掉的必须是最久没来的那个，不能把活跃客户端挤走。"""
        limiters = ClientRateLimiters(rate_per_sec=1, burst=1, maxsize=2)
        limiters.acquire('活跃', now=1000.0)
        limiters.acquire('沉默', now=1000.0)
        limiters.acquire('活跃', now=1000.0)      # 活跃的又来了一次
        limiters.acquire('新来的', now=1000.0)     # 触发淘汰

        # 活跃的桶还在（令牌已耗尽），沉默的被淘汰（配额重置）
        self.assertGreater(limiters.acquire('活跃', now=1000.0), 0)
        self.assertEqual(limiters.acquire('沉默', now=1000.0), 0)

    def test_a_non_positive_rate_is_refused(self):
        """速率配成 0 或负数是配置错误，不能默默变成"不限"或"全限"。"""
        for bad in (0, -1):
            with self.subTest(rate=bad):
                with self.assertRaises(ValueError):
                    ClientRateLimiters(rate_per_sec=bad)

    def test_a_non_positive_maxsize_is_refused(self):
        with self.assertRaises(ValueError):
            ClientRateLimiters(rate_per_sec=1, maxsize=0)


class Wiring(unittest.TestCase):

    def test_it_is_off_unless_configured(self):
        self.assertIsNone(build_rate_limiters(Settings()))
        self.assertIsNone(build_rate_limiters(Settings(rate_limit_per_sec=0)))

    def test_a_configured_rate_turns_it_on(self):
        limiters = build_rate_limiters(Settings(rate_limit_per_sec=5, rate_limit_burst=7))
        self.assertIsNotNone(limiters)
        self.assertEqual(limiters.rate_per_sec, 5)
        self.assertEqual(limiters.burst, 7)

    def test_settings_read_the_documented_environment_variables(self):
        settings = Settings.from_env({
            'RATE_LIMIT_PER_SEC': '3.5',
            'RATE_LIMIT_BURST': '9',
            'RATE_LIMIT_CLIENTS': '128',
        })
        self.assertEqual(settings.rate_limit_per_sec, 3.5)
        self.assertEqual(settings.rate_limit_burst, 9)
        self.assertEqual(settings.rate_limit_clients, 128)


def make_client(rate=1.0, burst=2, credentials=None):
    settings = Settings(rate_limit_per_sec=rate, rate_limit_burst=burst)
    auth = AuthSettings(credentials=credentials if credentials is not None else {'a': 'b'})
    return TestClient(create_app(settings=settings, auth_settings=auth))


class Middleware(unittest.TestCase):

    def test_excess_requests_get_429_with_retry_after(self):
        with make_client() as client:
            codes = [client.get('/auth/me').status_code for _ in range(5)]
            self.assertEqual(codes[:2], [200, 200])
            self.assertEqual(codes[2:], [429, 429, 429])

    def test_the_429_tells_the_caller_when_to_come_back(self):
        with make_client() as client:
            for _ in range(3):
                response = client.get('/auth/me')
            self.assertEqual(response.status_code, 429)
            self.assertGreaterEqual(int(response.headers['Retry-After']), 1)

    def test_the_health_probe_is_never_throttled(self):
        """**监控探针被限流会把"服务忙"误报成"服务挂了"。**"""
        with make_client() as client:
            self.assertEqual([client.get('/healthz').status_code for _ in range(6)],
                             [200] * 6)

    def test_the_exempt_list_is_exactly_the_health_probe(self):
        self.assertEqual(EXEMPT_PATHS, frozenset({'/healthz'}))

    def test_cors_preflight_is_not_throttled(self):
        with make_client() as client:
            codes = [client.options('/anything').status_code for _ in range(6)]
            self.assertNotIn(429, codes)

    def test_throttling_happens_before_authentication(self):
        """**顺序要紧**：未登录的洪水请求应该被限流直接挡掉，
        而不是先去查一遍会话（打 Redis）再返回 401——那正好是被攻击时
        最不该做的事。
        """
        with make_client() as client:
            codes = [client.get('/protected', headers={'accept': 'application/json'}
                                ).status_code for _ in range(5)]
            self.assertEqual(codes[:2], [401, 401])
            self.assertEqual(codes[2:], [429, 429, 429])

    def test_nothing_is_throttled_when_it_is_off(self):
        settings = Settings(rate_limit_per_sec=0)
        app = create_app(settings=settings, auth_settings=AuthSettings(credentials={}))
        with TestClient(app) as client:
            self.assertEqual([client.get('/healthz').status_code for _ in range(20)],
                             [200] * 20)


class ClientIdentification(unittest.TestCase):

    def test_the_forwarded_for_chain_is_ignored(self):
        """**`X-Forwarded-For` 是客户端能随便塞的**——认它等于让人自己
        决定自己是谁，限流一秒钟就被绕过去。只认反代覆盖式写入的
        `X-Real-IP`（openresty 的 `proxy_set_header X-Real-IP $remote_addr`）。
        """
        with make_client() as client:
            codes = [client.get('/auth/me',
                                headers={'X-Forwarded-For': f'9.9.9.{i}'}).status_code
                     for i in range(5)]
            self.assertEqual(codes[2:], [429, 429, 429])

    def test_the_real_ip_header_does_separate_clients(self):
        """**反方向**：反代写的 `X-Real-IP` 必须真的分开不同客户端，
        否则所有请求都算成一个（都来自反代），限流会误伤全体。
        """
        with make_client() as client:
            first = [client.get('/auth/me', headers={'X-Real-IP': '1.1.1.1'}).status_code
                     for _ in range(3)]
            second = client.get('/auth/me', headers={'X-Real-IP': '2.2.2.2'}).status_code
            self.assertEqual(first, [200, 200, 429])
            self.assertEqual(second, 200)


if __name__ == '__main__':
    unittest.main()
