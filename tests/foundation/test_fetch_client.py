import tempfile
import unittest

from src.foundation.fetch.circuit import CircuitBreaker
from src.foundation.fetch.client import FetchClient, FetchError
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


if __name__ == '__main__':
    unittest.main()
