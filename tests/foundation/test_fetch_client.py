import tempfile
import threading
import time
import unittest
from unittest import mock
from urllib.parse import urlparse

from src.foundation.fetch.circuit import CircuitBreaker
from src.foundation.fetch.client import (
    FetchClient, FetchError, PermanentFetchError,
)
from src.foundation.fetch.rate_limit import DomainRateLimiters
from src.foundation.fetch.snapshot import SnapshotStore


class RecordingTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, timeout):
        self.calls.append(url)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FetchClientTests(unittest.TestCase):
    def setUp(self):
        self.slept = []
        self.limiters = DomainRateLimiters(default_rate=1000, burst=1000)

    def make_client(self, responses, **kwargs):
        self.transport = RecordingTransport(responses)
        return FetchClient(
            transport=self.transport,
            limiters=self.limiters,
            sleep_fn=self.slept.append,
            **kwargs,
        )

    def test_successful_fetch_returns_body(self):
        client = self.make_client(['ok'])
        self.assertEqual(client.get('https://a.com/x'), 'ok')

    def test_retries_on_failure_then_succeeds(self):
        client = self.make_client([IOError('boom'), 'ok'])
        self.assertEqual(client.get('https://a.com/x'), 'ok')
        self.assertEqual(len(self.transport.calls), 2)

    def test_backoff_grows_between_retries(self):
        client = self.make_client([IOError('a'), IOError('b'), 'ok'])
        client.get('https://a.com/x')
        self.assertEqual(len(self.slept), 2)
        self.assertGreater(self.slept[1], self.slept[0])

    def test_raises_after_max_retries(self):
        client = self.make_client([IOError('a'), IOError('b'), IOError('c')], max_retries=3)
        with self.assertRaises(FetchError):
            client.get('https://a.com/x')

    def test_open_circuit_blocks_request(self):
        client = self.make_client([IOError('a')] * 6, max_retries=1, failure_threshold=2)
        for _ in range(2):
            with self.assertRaises(FetchError):
                client.get('https://a.com/x')
        before = len(self.transport.calls)
        with self.assertRaises(FetchError):
            client.get('https://a.com/x')
        self.assertEqual(len(self.transport.calls), before)

    def test_circuit_is_per_domain(self):
        client = self.make_client([IOError('a'), IOError('a'), 'ok'], max_retries=1, failure_threshold=2)
        for _ in range(2):
            with self.assertRaises(FetchError):
                client.get('https://bad.com/x')
        self.assertEqual(client.get('https://good.com/x'), 'ok')

    def test_snapshot_saved_on_success(self):
        root = tempfile.mkdtemp(prefix='snap-')
        client = self.make_client(['body'], snapshots=SnapshotStore(root))
        client.get('https://a.com/x')
        self.assertEqual(SnapshotStore(root).load('https://a.com/x'), 'body')

    def test_snapshot_used_as_fallback_on_total_failure(self):
        root = tempfile.mkdtemp(prefix='snap-')
        SnapshotStore(root).save('https://a.com/x', 'cached-body')
        client = self.make_client([IOError('a')] * 3, max_retries=3, snapshots=SnapshotStore(root))
        self.assertEqual(client.get('https://a.com/x'), 'cached-body')

    def test_rate_limiter_wait_is_slept(self):
        limiters = DomainRateLimiters(default_rate=1, burst=1)
        self.transport = RecordingTransport(['a', 'b'])
        client = FetchClient(
            transport=self.transport, limiters=limiters, sleep_fn=self.slept.append
        )
        client.get('https://a.com/x')
        client.get('https://a.com/x')
        self.assertTrue(any(s > 0 for s in self.slept))


class PermanentFailureTests(unittest.TestCase):
    """确定性失败不重试。

    退避重试假设失败是暂时的。有一类失败重试多少次都是同样结果——被 WAF
    拦、404、鉴权失败。对它们重试只是把一次失败的代价乘以重试次数，
    在限速的域名上尤其昂贵：篮球的 okooo 限到 0.4 rps，一次无谓的三连重试
    就要多花好几秒。
    """

    def setUp(self):
        self.slept = []
        self.limiters = DomainRateLimiters(default_rate=1000, burst=1000)

    def _client(self, responses, **kwargs):
        self.transport = RecordingTransport(responses)
        return FetchClient(transport=self.transport, limiters=self.limiters,
                           sleep_fn=self.slept.append, **kwargs)

    def test_permanent_error_is_not_retried(self):
        client = self._client([PermanentFetchError('WAF'), 'ok'], max_retries=3)
        with self.assertRaises(FetchError):
            client.get('https://a.com/x')
        self.assertEqual(len(self.transport.calls), 1, '确定性失败被重试了')

    def test_permanent_error_does_not_sleep_for_backoff(self):
        client = self._client([PermanentFetchError('WAF')], max_retries=3)
        with self.assertRaises(FetchError):
            client.get('https://a.com/x')
        self.assertEqual(self.slept, [], '为一次注定失败的请求白等了退避')

    def test_transient_error_is_still_retried(self):
        """只对确定性失败短路，普通故障的重试不受影响。"""
        client = self._client([IOError('抖了一下'), 'ok'], max_retries=3)
        self.assertEqual(client.get('https://a.com/x'), 'ok')
        self.assertEqual(len(self.transport.calls), 2)

    def test_permanent_error_still_counts_towards_the_breaker(self):
        """不重试不等于不计数——这类失败最该让熔断尽快开路。"""
        breaker = SpyBreaker()
        client = self._client([PermanentFetchError('WAF')], max_retries=3)
        client._breakers['a.com'] = breaker
        with self.assertRaises(FetchError):
            client.get('https://a.com/x')
        self.assertEqual(breaker.failure_calls, 1)
        self.assertEqual(breaker.success_calls, 0)

    def test_subclasses_are_treated_as_permanent(self):
        class Blocked(PermanentFetchError):
            pass

        client = self._client([Blocked('WAF'), 'ok'], max_retries=3)
        with self.assertRaises(FetchError):
            client.get('https://a.com/x')
        self.assertEqual(len(self.transport.calls), 1)

    def test_permanent_error_still_falls_back_to_a_snapshot(self):
        import tempfile

        from src.foundation.fetch import SnapshotStore

        with tempfile.TemporaryDirectory() as root:
            snapshots = SnapshotStore(root)
            snapshots.save('https://a.com/x', '旧的一份')
            client = FetchClient(transport=RecordingTransport(
                [PermanentFetchError('WAF')]), limiters=self.limiters,
                snapshots=snapshots, sleep_fn=self.slept.append)
            self.assertEqual(client.get('https://a.com/x'), '旧的一份')


class SpyBreaker:
    """回归测试专用：只统计调用次数，allow() 恒放行，不做真实熔断判定。"""

    def __init__(self):
        self.allow_calls = 0
        self.success_calls = 0
        self.failure_calls = 0

    def allow(self, now=None):
        self.allow_calls += 1
        return True

    def record_success(self):
        self.success_calls += 1

    def record_failure(self, now=None):
        self.failure_calls += 1


class FetchClientBreakerReportPairingTests(unittest.TestCase):
    """回归测试：钉死"一次 allow() 放行必须恰好配对一次上报"这条契约。

    fix round 1 之前的实现在重试循环内逐次调用 record_failure，会让一次
    allow() 放行对应多次上报；本用例把 breaker 换成 SpyBreaker 直接统计
    调用次数，可稳定复现该退化。
    """

    def setUp(self):
        self.slept = []
        self.limiters = DomainRateLimiters(default_rate=1000, burst=1000)

    def make_client_with_spy(self, responses, **kwargs):
        transport = RecordingTransport(responses)
        client = FetchClient(
            transport=transport,
            limiters=self.limiters,
            sleep_fn=self.slept.append,
            **kwargs,
        )
        spy = SpyBreaker()
        client._breakers[urlparse('https://a.com/x').netloc] = spy
        return client, spy

    def test_retry_then_success_reports_exactly_one_success_zero_failure(self):
        client, spy = self.make_client_with_spy([IOError('a'), IOError('b'), 'ok'])
        self.assertEqual(client.get('https://a.com/x'), 'ok')
        self.assertEqual(spy.allow_calls, 1)
        self.assertEqual(spy.success_calls, 1)
        self.assertEqual(spy.failure_calls, 0)

    def test_retries_exhausted_reports_exactly_one_failure_zero_success(self):
        client, spy = self.make_client_with_spy(
            [IOError('a'), IOError('b'), IOError('c')], max_retries=3
        )
        with self.assertRaises(FetchError):
            client.get('https://a.com/x')
        self.assertEqual(spy.allow_calls, 1)
        self.assertEqual(spy.success_calls, 0)
        self.assertEqual(spy.failure_calls, 1)


class FetchClientConstructorValidationTests(unittest.TestCase):
    """回归测试：构造参数非法值必须在 __init__ 时立即拒绝，
    不能悄悄放行然后在运行期产生诡异行为（如 max_retries=0 从不
    调用 transport 却仍上报一次 record_failure）。
    """

    def make(self, **overrides):
        kwargs = dict(
            transport=lambda url, timeout: 'ok',
            limiters=DomainRateLimiters(default_rate=1000, burst=1000),
        )
        kwargs.update(overrides)
        return FetchClient(**kwargs)

    def test_max_retries_zero_rejected(self):
        with self.assertRaises(ValueError):
            self.make(max_retries=0)

    def test_max_retries_negative_rejected(self):
        with self.assertRaises(ValueError):
            self.make(max_retries=-1)

    def test_base_backoff_negative_rejected(self):
        with self.assertRaises(ValueError):
            self.make(base_backoff=-1)

    def test_base_backoff_zero_accepted(self):
        self.make(base_backoff=0)

    def test_failure_threshold_zero_rejected(self):
        with self.assertRaises(ValueError):
            self.make(failure_threshold=0)

    def test_failure_threshold_negative_rejected(self):
        with self.assertRaises(ValueError):
            self.make(failure_threshold=-1)

    def test_recovery_timeout_zero_rejected(self):
        with self.assertRaises(ValueError):
            self.make(recovery_timeout=0)

    def test_recovery_timeout_negative_rejected(self):
        with self.assertRaises(ValueError):
            self.make(recovery_timeout=-1)

    def test_valid_construction_does_not_raise(self):
        client = self.make(
            max_retries=1, base_backoff=0, failure_threshold=1, recovery_timeout=1
        )
        self.assertEqual(client.get('https://a.com/x'), 'ok')

    def test_rejected_construction_raises_fetcherror_subtype_not_bare_valueerror_downstream(self):
        """构造期抛的是裸 ValueError（非 FetchError），调用方在构造阶段
        就能感知配置错误，而不是让它穿透到运行期的 except FetchError。
        """
        with self.assertRaises(ValueError):
            self.make(base_backoff=-1)
        # 明确不是 FetchError 的实例——构造期失败与运行期抓取失败是两类问题。
        try:
            self.make(base_backoff=-1)
        except ValueError as exc:
            self.assertNotIsInstance(exc, FetchError)


class FetchClientBreakerConcurrencyTests(unittest.TestCase):
    """回归测试：并发首次访问同一域名必须拿到同一把熔断器实例，
    否则失败计数会被多把熔断器摊薄，形同虚设。

    单纯用 Barrier 同步线程起跑不足以稳定复现竞态——check-then-act
    的窗口太窄（几微秒），GIL 的默认切换间隔（5ms）经常让整个
    `_breaker` 调用在一次调度片内跑完，race 撞不上。因此用一个
    "变慢的 CircuitBreaker 构造函数"人为撑大窗口：所有线程先在
    barrier 处对齐同时起跑，再各自调用 `_breaker(domain)`；构造过程
    人为 sleep 一小段时间。若无锁保护，多个线程会在这段 sleep 期间
    都判定"当前无实例"从而各自构造一把，产生多把不同实例；有锁保护
    时，同一时刻只有一个线程能进入构造路径，其余线程会阻塞在锁上，
    待其释放后直接复用已写入 dict 的实例。
    """

    def test_concurrent_first_access_shares_single_breaker_instance(self):
        client = FetchClient(
            transport=lambda url, timeout: 'ok',
            limiters=DomainRateLimiters(default_rate=1000, burst=1000),
        )
        domain = 'concurrent.example.com'
        thread_count = 16
        start_barrier = threading.Barrier(thread_count)

        from src.foundation.fetch.circuit import CircuitBreaker as RealCircuitBreaker

        class SlowCircuitBreaker(RealCircuitBreaker):
            def __init__(self, *args, **kwargs):
                time.sleep(0.15)
                super().__init__(*args, **kwargs)

        breakers = []
        breakers_lock = threading.Lock()

        def worker():
            start_barrier.wait(timeout=5)
            breaker = client._breaker(domain)
            with breakers_lock:
                breakers.append(breaker)

        with mock.patch('src.foundation.fetch.client.CircuitBreaker', SlowCircuitBreaker):
            threads = [threading.Thread(target=worker) for _ in range(thread_count)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

        self.assertEqual(len(breakers), thread_count)
        self.assertEqual(len(set(id(b) for b in breakers)), 1)


if __name__ == '__main__':
    unittest.main()
