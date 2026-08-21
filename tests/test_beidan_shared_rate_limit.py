"""北单与足球共用 odds.500.com 限速预算：领号、退避上报与域名隔离"""

import unittest
import urllib.error
from unittest.mock import patch

import src.beidan.fetching as bf


class SharedRateBudgetHostTests(unittest.TestCase):
    def test_only_the_shared_host_participates(self):
        self.assertTrue(bf._shares_football_rate_budget('https://odds.500.com/fenxi/yazhi-1.shtml'))
        self.assertFalse(bf._shares_football_rate_budget('https://www.okooo.com/danchang/'))
        self.assertFalse(bf._shares_football_rate_budget(''))
        self.assertFalse(bf._shares_football_rate_budget(None))


class SharedRateBudgetFetchTests(unittest.TestCase):
    def _fetch(self, url, opener):
        with patch.object(bf, '_await_fetch_throttle') as wait, \
             patch.object(bf, '_await_rate_slot') as slot, \
             patch.object(bf, '_enter_fetch_throttle') as throttle, \
             patch.object(bf.urllib.request, 'urlopen', opener):
            bf.fetch(url)
        return wait, slot, throttle

    @staticmethod
    def _ok_opener(*args, **kwargs):
        class _Resp:
            def read(self):
                return b'<html></html>'
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
        return _Resp()

    @staticmethod
    def _rate_limited_opener(*args, **kwargs):
        raise urllib.error.HTTPError('u', 429, 'Too Many Requests', {}, None)

    def test_takes_a_rate_slot_for_shared_host(self):
        wait, slot, _ = self._fetch('https://odds.500.com/x.shtml', self._ok_opener)
        wait.assert_called_once()
        slot.assert_called_once()

    def test_skips_rate_limiter_for_other_hosts(self):
        """okooo 是另一个域名，不该消耗 500.com 的配额"""
        wait, slot, _ = self._fetch('https://www.okooo.com/danchang/', self._ok_opener)
        wait.assert_not_called()
        slot.assert_not_called()

    def test_reports_rate_limit_into_global_throttle(self):
        """撞 429 要写进全局冷却，足球侧才会跟着一起退避"""
        _, _, throttle = self._fetch('https://odds.500.com/x.shtml', self._rate_limited_opener)
        throttle.assert_called_once()
        self.assertGreater(throttle.call_args[0][0], 0)

    def test_other_host_rate_limit_does_not_throttle_shared_budget(self):
        _, _, throttle = self._fetch('https://www.okooo.com/x', self._rate_limited_opener)
        throttle.assert_not_called()

    def test_non_rate_limit_error_does_not_throttle(self):
        def not_found(*args, **kwargs):
            raise urllib.error.HTTPError('u', 404, 'Not Found', {}, None)

        _, _, throttle = self._fetch('https://odds.500.com/x.shtml', not_found)
        throttle.assert_not_called()


if __name__ == '__main__':
    unittest.main()
