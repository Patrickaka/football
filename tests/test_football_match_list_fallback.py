import json
import os
import tempfile
import unittest
from unittest.mock import patch

import src.football as football
from src.football import config as fb_config
from src.football import fetching as fb_fetching


class FootballMatchListFallbackTests(unittest.TestCase):
    def test_remote_fetch_retries_http_when_https_fails(self):
        page = 'shuju-123.shtml title="A VSB数据分析"'
        with patch.object(fb_config, 'INDEX_URLS', ('https://primary', 'http://fallback')), \
             patch.object(fb_fetching, 'fetch', side_effect=[OSError('TLS'), page]) as mocked:
            matches = football._fetch_match_list_remote()
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(matches[0]['match_id'], '123')

    def test_successful_live_fetch_is_persisted(self):
        matches = [{'match_id': '1', 'home': 'A', 'away': 'B', 'time': '08-08 20:00'}]
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(fb_config, 'MATCH_LIST_CACHE_PATH', os.path.join(folder, 'matches.json')), \
             patch.object(fb_fetching, '_fetch_match_list_remote', return_value=matches):
            self.assertEqual(football.fetch_match_list(), matches)
            with open(fb_config.MATCH_LIST_CACHE_PATH, encoding='utf-8') as handle:
                self.assertEqual(json.load(handle)['matches'], matches)
            self.assertEqual(football.get_match_list_status()['source'], 'live')

    def test_network_failure_returns_last_successful_snapshot(self):
        cached = [{'match_id': '2', 'home': 'C', 'away': 'D', 'time': '08-08 21:00'}]
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(fb_config, 'MATCH_LIST_CACHE_PATH', os.path.join(folder, 'matches.json')):
            football._save_match_list_cache(cached)
            with patch.object(fb_fetching, '_fetch_match_list_remote', side_effect=OSError('TLS failed')), \
                 patch.object(fb_fetching, '_zgzcw_schedule_fallback', return_value=[]):
                self.assertEqual(football.fetch_match_list(), cached)
            status = football.get_match_list_status()
            self.assertEqual(status['source'], 'disk_cache')
            self.assertTrue(status['stale'])
            self.assertIn('TLS failed', status['error'])

    def test_500_failure_switches_to_zgzcw_before_disk_cache(self):
        offer = {
            'zgzcw_id': '9001', 'analysis_id': '8001',
            'num': '周五001', 'home': '主队', 'away': '客队',
            'time': '20:00', 'lottery_handicap': -1,
            'spf_available': True, 'rqspf_available': True,
            'available_markets': ['spf', 'rqspf'],
            'spf_odds': {'胜': 2.0, '平': 3.2, '负': 3.4},
            'rqspf_odds': {'让胜': 3.8, '让平': 3.6, '让负': 1.7},
        }
        cached = [{'match_id': '5001', 'num': '周五001', 'league': '测试联赛', 'time': '08-08 20:00'}]
        with patch('src.football.zgzcw_lottery.fetch_zgzcw_jczq_schedule', return_value=[offer]):
            result = football._zgzcw_schedule_fallback(cached)
        self.assertEqual(result[0]['match_id'], '5001')
        self.assertEqual(result[0]['schedule_source'], 'zgzcw')
        self.assertTrue(result[0]['analysis_source_id_available'])


if __name__ == '__main__':
    unittest.main()
