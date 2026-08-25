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


if __name__ == '__main__':
    unittest.main()
