# -*- coding: utf-8 -*-
"""分析失败时不得留下仍在抓取的线程。

`analyze_match` 并发发起五组抓取，亚盘/欧赔/大小球任一失败就立刻向上抛。
若此时线程池没排空，`fetch_team_strength` 与 `fetch_single_company_odds`
会继续向源站发请求——这一场已经分析失败了，那几个请求纯属白打，而且失败
往往正是源站在限流，继续打只会加重。
"""

import threading
import time
import unittest
from unittest import mock

from src.football import analyze_match
from src.football import parsing as fb_parsing


def _alive_odds_threads():
    return [t for t in threading.enumerate()
            if t.name.startswith('FootballOdds') and t.is_alive()]


class FailedAnalysisLeavesNoLiveFetches(unittest.TestCase):

    def setUp(self):
        self.finished = threading.Event()

    def _slow_fetch(self, *args, **kwargs):
        time.sleep(0.4)
        self.finished.set()
        return {}

    def test_pool_is_drained_before_the_error_propagates(self):
        match = {'match_id': '1430017', 'home': '主队', 'away': '客队',
                 'league': '英超', 'time': '08-28 20:00'}

        with mock.patch.object(fb_parsing, 'fetch_yazhi',
                               side_effect=ValueError('亚盘页 404')), \
             mock.patch.object(fb_parsing, 'fetch_ouzhi', return_value={}), \
             mock.patch.object(fb_parsing, 'fetch_daxiao', return_value={}), \
             mock.patch.object(fb_parsing, 'fetch_team_strength',
                               side_effect=self._slow_fetch), \
             mock.patch.object(fb_parsing, 'fetch_single_company_odds',
                               side_effect=self._slow_fetch):
            with self.assertRaises(ValueError):
                analyze_match(match, force_refresh=True)

            alive = _alive_odds_threads()

        self.assertEqual(alive, [], f'异常传出时仍有抓取线程在跑: {alive}')
        self.assertTrue(
            self.finished.is_set(),
            '慢抓取还没跑完就判定为已排空，说明断言本身没生效',
        )
