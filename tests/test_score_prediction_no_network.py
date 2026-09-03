# -*- coding: utf-8 -*-
"""分析用例不许把真实网络请求漏到 mock 窗口之外。

`analyze_match` 用线程池并发抓五组数据，且 `pool.shutdown(wait=False)`。
亚盘/欧赔/大小球任一失败就立刻向上抛，此时 `fetch_team_strength` 与
`fetch_single_company_odds` 两个线程还在跑——调用方的 `mock.patch` 窗口
已经关闭，它们于是打到真的 500.com。

后果不只是"测试碰了网络"：泄漏出去的请求把源站打到限流，下一条用例的真实
抓取就更容易失败，于是自激。表现是全量并行下随机冒出
`ValueError: 亚盘数据获取失败: expected string or bytes-like object, got 'NoneType'`，
而单跑该文件永远是绿的。
"""

import threading
import time
import unittest
import urllib.request
from unittest import mock

import tests.test_score_prediction as suite


def _wait_for_odds_threads(timeout=15.0):
    """等线程池里那几个 FootballOdds 线程跑完，否则断言等于没等。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not any(t.name.startswith('FootballOdds') and t.is_alive()
                   for t in threading.enumerate()):
            return True
        time.sleep(0.05)
    return False


class AnalysisNeverEscapesToTheRealNetwork(unittest.TestCase):

    def test_failed_analysis_does_not_leak_live_requests(self):
        calls = []

        def _refuse(*args, **kwargs):
            target = args[0] if args else kwargs.get('url')
            calls.append(getattr(target, 'full_url', target))
            raise AssertionError('测试期间发起了真实网络请求')

        with mock.patch.object(urllib.request, 'urlopen', side_effect=_refuse):
            # 夹具里没有这个 match_id，亚盘那一环必然失败，走的正是泄漏路径。
            with self.assertRaises(Exception):
                suite._analyze_offline('9999999')
            self.assertTrue(_wait_for_odds_threads(), '线程池线程未在超时内结束')

        self.assertEqual(calls, [], f'泄漏了真实请求: {calls[:5]}')
