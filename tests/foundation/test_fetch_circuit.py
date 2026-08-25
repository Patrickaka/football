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


if __name__ == '__main__':
    unittest.main()
