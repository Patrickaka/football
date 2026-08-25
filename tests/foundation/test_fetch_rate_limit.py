import unittest

from src.foundation.fetch.rate_limit import DomainRateLimiters, RateLimiter


class RateLimiterTests(unittest.TestCase):
    def test_first_call_needs_no_wait(self):
        limiter = RateLimiter(rate_per_sec=1, burst=1)
        self.assertEqual(limiter.acquire(now=100.0), 0)

    def test_burst_allows_configured_count(self):
        limiter = RateLimiter(rate_per_sec=1, burst=3)
        waits = [limiter.acquire(now=100.0) for _ in range(3)]
        self.assertEqual(waits, [0, 0, 0])

    def test_exceeding_burst_requires_wait(self):
        limiter = RateLimiter(rate_per_sec=1, burst=1)
        limiter.acquire(now=100.0)
        self.assertGreater(limiter.acquire(now=100.0), 0)

    def test_tokens_refill_over_time(self):
        limiter = RateLimiter(rate_per_sec=2, burst=1)
        limiter.acquire(now=100.0)
        self.assertEqual(limiter.acquire(now=100.5), 0)

    def test_wait_time_matches_deficit(self):
        limiter = RateLimiter(rate_per_sec=2, burst=1)
        limiter.acquire(now=100.0)
        self.assertAlmostEqual(limiter.acquire(now=100.0), 0.5, places=3)


class DomainRateLimitersTests(unittest.TestCase):
    def test_same_domain_returns_same_limiter(self):
        limiters = DomainRateLimiters(default_rate=1)
        self.assertIs(limiters.for_domain('a.com'), limiters.for_domain('a.com'))

    def test_different_domains_are_independent(self):
        limiters = DomainRateLimiters(default_rate=1)
        self.assertIsNot(limiters.for_domain('a.com'), limiters.for_domain('b.com'))

    def test_override_applies_to_named_domain(self):
        limiters = DomainRateLimiters(default_rate=10, overrides={'slow.com': 0.5})
        self.assertEqual(limiters.for_domain('slow.com').rate_per_sec, 0.5)
        self.assertEqual(limiters.for_domain('other.com').rate_per_sec, 10)


if __name__ == '__main__':
    unittest.main()
