# -*- coding: utf-8 -*-
"""首屏聚合（BFF）。

一次请求返回该屏全部数据。足球首屏原来是 2 次元数据 + 5 批预测。

**只读缓存、绝不在请求线程做冷计算**是这一层的硬约束，不是优化：
一场比赛的完整分析要十几秒，首屏若代跑，用户在白屏上等，请求还会穿透
到第三方源站。冷数据由后台预热任务算好，这里只把算好的取出来。
"""
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import AuthSettings
from src.api.services import bff as service
from src.api.services import football as football_service

MATCHES = [
    {'match_id': 'm1', 'home': '主一', 'away': '客一', 'time': '2026-08-30 20:00'},
    {'match_id': 'm2', 'home': '主二', 'away': '客二', 'time': '2026-08-30 21:00'},
]
FRESH = {'lottery': {}, 'model': {}, 'prediction_logic_version': None}


def make_client():
    return TestClient(create_app(auth_settings=AuthSettings(credentials={})))


class ColdComputationIsForbidden(unittest.TestCase):
    """**这一族是 BFF 存在的理由。** 破了它，首屏就退回十几秒白屏。"""

    def test_it_never_calls_analyze_match(self):
        with mock.patch.object(football_service, 'matches_payload',
                               return_value={'matches': list(MATCHES)}), \
             mock.patch.object(service, '_professional_status', return_value={}), \
             mock.patch('src.football.config.get_cache', return_value=None), \
             mock.patch('src.football.pipeline.analyze_match') as analyze:
            service.football_home_payload()
        self.assertFalse(analyze.called, 'BFF 不能在请求线程里算分析')

    def test_a_cache_miss_is_reported_as_pending_not_computed(self):
        """缓存没有就如实说"还没算好"，**不是现算**。"""
        with mock.patch.object(football_service, 'matches_payload',
                               return_value={'matches': list(MATCHES)}), \
             mock.patch.object(service, '_professional_status', return_value={}), \
             mock.patch('src.football.config.get_cache', return_value=None):
            payload = service.football_home_payload()
        self.assertEqual(payload['predictions']['ready'], [])
        self.assertEqual(payload['predictions']['pending'], ['m1', 'm2'])

    def test_pending_is_not_hidden(self):
        """**`pending` 不能藏起来。** 返回空列表会让前端以为数据齐了，
        那几场就永远不显示——一个看起来正常、实际少了内容的首屏。
        """
        with mock.patch.object(football_service, 'matches_payload',
                               return_value={'matches': list(MATCHES)}), \
             mock.patch.object(service, '_professional_status', return_value={}), \
             mock.patch('src.football.config.get_cache', return_value=None):
            payload = service.football_home_payload()
        self.assertEqual(payload['coverage'],
                         {'total': 2, 'ready': 0, 'pending': 2})


class CacheKeyMustMatchTheWriter(unittest.TestCase):
    """**key 算错不会报错，只会永远 miss**——那样 BFF 就是个永远返回
    "计算中"的空壳，看起来在工作、实际什么也没做。
    """

    def test_the_key_comes_from_the_shared_helper(self):
        from src.football.pipeline import analysis_cache_key
        self.assertEqual(analysis_cache_key(MATCHES[0]), 'm1_主一_客一')

    def test_analyze_match_uses_the_same_helper(self):
        """写入方也必须走这一份，否则两边各算各的。"""
        import inspect

        from src.football import pipeline
        source = inspect.getsource(pipeline.analyze_match)
        self.assertIn('cache_key = analysis_cache_key(match)', source)

    def test_a_cached_entry_is_looked_up_with_that_key(self):
        seen = []

        def spy(namespace, key, match_time):
            seen.append((namespace, key))
            return dict(FRESH)

        with mock.patch.object(football_service, 'matches_payload',
                               return_value={'matches': list(MATCHES)}), \
             mock.patch.object(service, '_professional_status', return_value={}), \
             mock.patch('src.football.config.get_cache', spy), \
             mock.patch('src.football.pipeline._is_prediction_cache_current',
                        return_value=True):
            service.football_home_payload()
        self.assertEqual(seen, [('match_analysis', 'm1_主一_客一'),
                                ('match_analysis', 'm2_主二_客二')])


class StaleCacheCountsAsMissing(unittest.TestCase):

    def test_an_out_of_date_logic_version_is_not_served(self):
        """**逻辑版本变了的缓存等于没有**——照 `analyze_match` 的规矩来。

        不然首屏会拿旧口径的结果去渲染，而用户点开详情重算后数字对不上。
        """
        with mock.patch.object(football_service, 'matches_payload',
                               return_value={'matches': list(MATCHES)}), \
             mock.patch.object(service, '_professional_status', return_value={}), \
             mock.patch('src.football.config.get_cache', return_value=dict(FRESH)), \
             mock.patch('src.football.pipeline._is_prediction_cache_current',
                        return_value=False):
            payload = service.football_home_payload()
        self.assertEqual(payload['predictions']['pending'], ['m1', 'm2'])

    def test_a_current_one_is_served(self):
        """**反方向**：版本对得上就得用，否则缓存等于白建。"""
        with mock.patch.object(football_service, 'matches_payload',
                               return_value={'matches': list(MATCHES)}), \
             mock.patch.object(service, '_professional_status', return_value={}), \
             mock.patch('src.football.config.get_cache', return_value=dict(FRESH)), \
             mock.patch('src.football.pipeline._is_prediction_cache_current',
                        return_value=True):
            payload = service.football_home_payload()
        self.assertEqual([entry['match_id'] for entry in payload['predictions']['ready']],
                         ['m1', 'm2'])
        self.assertEqual(payload['predictions']['pending'], [])


class Resilience(unittest.TestCase):

    def test_a_broken_professional_status_does_not_sink_the_screen(self):
        """它只是个角标，取不到不该让整个首屏失败。"""
        with mock.patch.object(football_service, 'matches_payload',
                               return_value={'matches': list(MATCHES)}), \
             mock.patch.object(football_service, 'football_professional_status_payload',
                               side_effect=RuntimeError('挂了')), \
             mock.patch('src.football.config.get_cache', return_value=None):
            payload = service.football_home_payload()
        self.assertIn('error', payload['professional_status'])
        self.assertEqual(payload['coverage']['total'], 2)

    def test_a_broken_cache_read_degrades_to_pending(self):
        with mock.patch.object(football_service, 'matches_payload',
                               return_value={'matches': list(MATCHES)}), \
             mock.patch.object(service, '_professional_status', return_value={}), \
             mock.patch('src.football.config.get_cache',
                        side_effect=RuntimeError('缓存挂了')):
            payload = service.football_home_payload()
        self.assertEqual(payload['predictions']['pending'], ['m1', 'm2'])

    def test_a_failed_match_list_is_reported_not_swallowed(self):
        """比赛列表拿不到就没有首屏可言，必须如实报错。"""
        with mock.patch.object(football_service, 'matches_payload',
                               return_value={'error': '赛程源不可用',
                                             'source_status': {'source': 'okooo'}}):
            payload = service.football_home_payload()
        self.assertEqual(payload['error'], '赛程源不可用')
        self.assertNotIn('predictions', payload)


class Endpoint(unittest.TestCase):

    def test_the_route_exists(self):
        app = create_app(auth_settings=AuthSettings(credentials={}))
        self.assertIn('/api/bff/football/home', app.openapi()['paths'])

    def test_it_returns_the_aggregate(self):
        payload = {'matches': [], 'predictions': {'ready': [], 'pending': []},
                   'coverage': {'total': 0, 'ready': 0, 'pending': 0}}
        with mock.patch.object(service, 'football_home_payload', return_value=payload):
            with make_client() as client:
                response = client.get('/api/bff/football/home')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), payload)

    def test_it_is_not_public(self):
        client = TestClient(create_app(auth_settings=AuthSettings(credentials={'a': 'b'})))
        with client:
            self.assertEqual(
                client.get('/api/bff/football/home',
                           headers={'accept': 'application/json'}).status_code, 401)

    def test_it_replaces_two_round_trips(self):
        """首屏原来要 `/api/matches` + `/api/football/professional-status` 两次。"""
        with mock.patch.object(football_service, 'matches_payload',
                               return_value={'matches': list(MATCHES)}) as matches, \
             mock.patch.object(football_service, 'football_professional_status_payload',
                               return_value={'ok': True}) as status, \
             mock.patch('src.football.config.get_cache', return_value=None):
            payload = service.football_home_payload()
        self.assertTrue(matches.called)
        self.assertTrue(status.called)
        self.assertIn('matches', payload)
        self.assertIn('professional_status', payload)


if __name__ == '__main__':
    unittest.main()
