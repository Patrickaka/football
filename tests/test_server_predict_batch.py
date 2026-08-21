"""批量预测接口：顺序、错误隔离、入参归一化与限额"""

import unittest
from unittest.mock import patch

import server
import src.webapp.football_api as football_api


def _handler():
    handler = server.Handler.__new__(server.Handler)
    handler._log = server.log
    return handler


def _match(match_id, **extra):
    base = {'match_id': match_id, 'home': f'主{match_id}', 'away': f'客{match_id}'}
    base.update(extra)
    return base


class PredictBatchPayloadTests(unittest.TestCase):
    def setUp(self):
        self.handler = _handler()

    def test_results_follow_input_order(self):
        matches = [_match(str(i)) for i in range(5)]
        with patch.object(football_api, 'analyze_match',
                          side_effect=lambda m, force_refresh=False: {'id': m['match_id']}):
            payload = self.handler._predict_batch_payload({'matches': matches})

        self.assertEqual([entry['match_id'] for entry in payload['results']],
                         ['0', '1', '2', '3', '4'])
        self.assertEqual([entry['result']['id'] for entry in payload['results']],
                         ['0', '1', '2', '3', '4'])

    def test_single_failure_does_not_abort_batch(self):
        def flaky(match, force_refresh=False):
            if match['match_id'] == '2':
                raise ValueError('亚盘数据获取失败')
            return {'id': match['match_id']}

        matches = [_match(str(i)) for i in range(4)]
        with patch.object(football_api, 'analyze_match', side_effect=flaky):
            results = self.handler._predict_batch_payload({'matches': matches})['results']

        self.assertEqual(results[2]['error'], '亚盘数据获取失败')
        self.assertNotIn('result', results[2])
        self.assertEqual([r['result']['id'] for r in results if 'result' in r], ['0', '1', '3'])

    def test_unexpected_exception_is_isolated_too(self):
        def boom(match, force_refresh=False):
            if match['match_id'] == '1':
                raise RuntimeError('数据库连接断开')
            return {'id': match['match_id']}

        with patch.object(football_api, 'analyze_match', side_effect=boom):
            results = self.handler._predict_batch_payload(
                {'matches': [_match('0'), _match('1')]})['results']

        self.assertIn('数据库连接断开', results[1]['error'])
        self.assertEqual(results[0]['result']['id'], '0')

    def test_force_refresh_is_propagated(self):
        seen = []
        with patch.object(football_api, 'analyze_match',
                          side_effect=lambda m, force_refresh=False: seen.append(force_refresh) or {}):
            self.handler._predict_batch_payload({'matches': [_match('1')], 'force_refresh': True})
        self.assertEqual(seen, [True])

    def test_batch_and_single_build_the_same_match(self):
        """批量的 JSON 入参与单场的查询参数须归一化成同一个 match 字典"""
        captured = []
        raw = _match('99', league='英超', time='08-21 22:00', num='周三201',
                     lottery_handicap=-1, lottery_source='available',
                     lottery_offer_matched=True, lottery_available_markets=['spf'],
                     lottery_spf_available=True, lottery_spf_odds={'h': 1.9},
                     okooo_id='ok9', schedule_source='500')
        with patch.object(football_api, 'analyze_match',
                          side_effect=lambda m, force_refresh=False: captured.append(m) or {}):
            self.handler._predict_batch_payload({'matches': [raw]})
            self.handler._predict_payload({
                'match_id': ['99'], 'home': ['主99'], 'away': ['客99'],
                'league': ['英超'], 'time': ['08-21 22:00'], 'num': ['周三201'],
                'lottery_handicap': ['-1'], 'lottery_source': ['available'],
                'lottery_offer_matched': ['true'], 'lottery_available_markets': ['spf'],
                'lottery_spf_available': ['true'], 'lottery_spf_odds': ['{"h": 1.9}'],
                'okooo_id': ['ok9'], 'schedule_source': ['500'],
            })

        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0], captured[1])

    def test_rejects_batch_over_limit(self):
        oversized = [_match(str(i)) for i in range(football_api.FOOTBALL_BATCH_LIMIT + 1)]
        payload = self.handler._predict_batch_payload({'matches': oversized})
        self.assertIn('单批最多', payload['error'])
        self.assertNotIn('results', payload)

    def test_rejects_malformed_body(self):
        for body in (None, [], 'x', {}, {'matches': []}, {'matches': 'abc'}, {'matches': ['x', 1]}):
            with self.subTest(body=body):
                payload = self.handler._predict_batch_payload(body)
                self.assertIn('error', payload)
                self.assertNotIn('results', payload)


if __name__ == '__main__':
    unittest.main()
