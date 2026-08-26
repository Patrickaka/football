import threading
import unittest

from src.foundation.fetch.circuit import CircuitBreaker


class CircuitBreakerTests(unittest.TestCase):
    def setUp(self):
        self.cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

    def test_starts_closed_and_allows(self):
        self.assertEqual(self.cb.state, 'closed')
        self.assertTrue(self.cb.allow(now=100.0))

    def test_stays_closed_below_threshold(self):
        self.cb.record_failure(now=100.0)
        self.cb.record_failure(now=101.0)
        self.assertEqual(self.cb.state, 'closed')
        self.assertTrue(self.cb.allow(now=102.0))

    def test_opens_at_threshold(self):
        for i in range(3):
            self.cb.record_failure(now=100.0 + i)
        self.assertEqual(self.cb.state, 'open')
        self.assertFalse(self.cb.allow(now=104.0))

    def test_success_resets_failure_count(self):
        self.cb.record_failure(now=100.0)
        self.cb.record_failure(now=101.0)
        self.cb.record_success()
        self.cb.record_failure(now=102.0)
        self.assertEqual(self.cb.state, 'closed')

    def test_half_opens_after_recovery_timeout(self):
        for i in range(3):
            self.cb.record_failure(now=100.0 + i)
        self.assertTrue(self.cb.allow(now=200.0))
        self.assertEqual(self.cb.state, 'half_open')

    def test_half_open_success_closes_circuit(self):
        for i in range(3):
            self.cb.record_failure(now=100.0 + i)
        self.cb.allow(now=200.0)
        self.cb.record_success()
        self.assertEqual(self.cb.state, 'closed')
        self.assertTrue(self.cb.allow(now=201.0))

    def test_half_open_failure_reopens_circuit(self):
        for i in range(3):
            self.cb.record_failure(now=100.0 + i)
        self.cb.allow(now=200.0)
        self.cb.record_failure(now=201.0)
        self.assertEqual(self.cb.state, 'open')
        self.assertFalse(self.cb.allow(now=202.0))

    def test_half_open_second_allow_call_is_rejected(self):
        for i in range(3):
            self.cb.record_failure(now=100.0 + i)
        self.assertTrue(self.cb.allow(now=200.0))
        self.assertEqual(self.cb.state, 'half_open')
        # 探测名额已被占用，同一 half_open 期间的后续调用必须被拒绝，
        # 直到 record_success/record_failure 让状态离开 half_open。
        self.assertFalse(self.cb.allow(now=200.1))
        self.assertFalse(self.cb.allow(now=200.2))

    def test_half_open_allows_exactly_one_concurrent_probe(self):
        for i in range(3):
            self.cb.record_failure(now=100.0 + i)
        n_threads = 50
        barrier = threading.Barrier(n_threads)
        allowed = []
        lock = threading.Lock()

        def worker():
            barrier.wait()
            result = self.cb.allow(now=200.0)
            with lock:
                allowed.append(result)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(sum(1 for r in allowed if r), 1)
        self.assertEqual(self.cb.state, 'half_open')

    def test_probe_lease_expires_and_self_heals(self):
        for i in range(3):
            self.cb.record_failure(now=100.0 + i)
        self.assertTrue(self.cb.allow(now=200.0))  # 探针拿到名额，未上报
        self.assertEqual(self.cb.state, 'half_open')
        # 租约到期（默认 probe_timeout == recovery_timeout == 60）之前仍应拒绝
        self.assertFalse(self.cb.allow(now=259.9))
        # 租约到期后应能自愈：放行下一次探测
        self.assertTrue(self.cb.allow(now=260.0))
        self.assertEqual(self.cb.state, 'half_open')

    def test_probe_lease_not_expired_still_rejects(self):
        for i in range(3):
            self.cb.record_failure(now=100.0 + i)
        self.assertTrue(self.cb.allow(now=200.0))
        for t in (200.1, 210.0, 230.0, 259.999):
            self.assertFalse(self.cb.allow(now=t))
        self.assertEqual(self.cb.state, 'half_open')

    def test_probe_timeout_defaults_to_recovery_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=30)
        self.assertEqual(cb.probe_timeout, 30)
        cb.record_failure(now=100.0)
        cb.allow(now=200.0)  # 拿到探针，未上报
        self.assertFalse(cb.allow(now=229.9))
        self.assertTrue(cb.allow(now=230.0))

    def test_explicit_probe_timeout_overrides_default(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60, probe_timeout=5)
        self.assertEqual(cb.probe_timeout, 5)
        cb.record_failure(now=100.0)
        cb.allow(now=200.0)  # 拿到探针，未上报
        self.assertFalse(cb.allow(now=204.9))
        self.assertTrue(cb.allow(now=205.0))

    def test_probe_lease_reset_after_record_success(self):
        for i in range(3):
            self.cb.record_failure(now=100.0 + i)
        self.cb.allow(now=200.0)
        self.cb.record_success()
        # 成功上报后租约状态应清空，closed 状态下 allow 恒为 True，与 probe 无关
        self.assertTrue(self.cb.allow(now=200.01))
        self.assertEqual(self.cb.state, 'closed')

    def test_probe_lease_reset_after_record_failure(self):
        for i in range(3):
            self.cb.record_failure(now=100.0 + i)
        self.cb.allow(now=200.0)
        self.cb.record_failure(now=200.5)
        self.assertEqual(self.cb.state, 'open')
        # 重新进入 open，需等待新一轮 recovery_timeout，不受旧探针租约影响
        self.assertFalse(self.cb.allow(now=200.6))
        self.assertTrue(self.cb.allow(now=260.5))

    def test_half_open_allows_exactly_one_concurrent_probe_still_holds_with_lease(self):
        for i in range(3):
            self.cb.record_failure(now=100.0 + i)
        n_threads = 50
        barrier = threading.Barrier(n_threads)
        allowed = []
        lock = threading.Lock()

        def worker():
            barrier.wait()
            result = self.cb.allow(now=200.0)
            with lock:
                allowed.append(result)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(sum(1 for r in allowed if r), 1)
        self.assertEqual(self.cb.state, 'half_open')

    def test_zero_probe_timeout_rejected(self):
        with self.assertRaisesRegex(ValueError, 'probe_timeout'):
            CircuitBreaker(failure_threshold=1, recovery_timeout=60, probe_timeout=0)

    def test_negative_probe_timeout_rejected(self):
        with self.assertRaisesRegex(ValueError, 'probe_timeout'):
            CircuitBreaker(failure_threshold=1, recovery_timeout=60, probe_timeout=-1)

    def test_zero_or_negative_failure_threshold_rejected(self):
        with self.assertRaisesRegex(ValueError, 'failure_threshold'):
            CircuitBreaker(failure_threshold=0)
        with self.assertRaisesRegex(ValueError, 'failure_threshold'):
            CircuitBreaker(failure_threshold=-1)

    def test_zero_or_negative_recovery_timeout_rejected(self):
        with self.assertRaisesRegex(ValueError, 'recovery_timeout'):
            CircuitBreaker(recovery_timeout=0)
        with self.assertRaisesRegex(ValueError, 'recovery_timeout'):
            CircuitBreaker(recovery_timeout=-1)

    def test_valid_construction_unaffected_by_new_validation(self):
        CircuitBreaker()  # 全部使用默认值
        CircuitBreaker(failure_threshold=1, recovery_timeout=1, probe_timeout=1)
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=30)
        self.assertEqual(cb.probe_timeout, 30)  # 未传时仍默认等于 recovery_timeout


class TripTests(unittest.TestCase):
    """trip()：立即开路，不等失败次数攒够。"""

    def test_trip_opens_immediately(self):
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        breaker.trip(now=1000)
        self.assertEqual(breaker.state, 'open')
        self.assertFalse(breaker.allow(now=1000))

    def test_tripped_breaker_still_recovers_on_schedule(self):
        """开路不是永久的——出口 IP 会变、封锁会撤，恢复窗口照常走。"""
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        breaker.trip(now=1000)
        self.assertFalse(breaker.allow(now=1059))
        self.assertTrue(breaker.allow(now=1060))
        self.assertEqual(breaker.state, 'half_open')

    def test_success_after_trip_closes_again(self):
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        breaker.trip(now=1000)
        breaker.allow(now=1060)
        breaker.record_success()
        self.assertEqual(breaker.state, 'closed')
        self.assertTrue(breaker.allow(now=1061))


if __name__ == '__main__':
    unittest.main()
