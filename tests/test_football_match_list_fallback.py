import json
import os
import tempfile
import unittest
from unittest.mock import patch

import src.football as football


class FootballMatchListFallbackTests(unittest.TestCase):
    def test_remote_fetch_retries_http_when_https_fails(self):
        page = 'shuju-123.shtml title="A VSB数据分析"'
        with patch.object(football, 'INDEX_URLS', ('https://primary', 'http://fallback')), \
             patch.object(football, 'fetch', side_effect=[OSError('TLS'), page]) as mocked:
            matches = football._fetch_match_list_remote()
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(matches[0]['match_id'], '123')

    def test_successful_live_fetch_is_persisted(self):
        matches = [{'match_id': '1', 'home': 'A', 'away': 'B', 'time': '08-08 20:00'}]
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(football, 'MATCH_LIST_CACHE_PATH', os.path.join(folder, 'matches.json')), \
             patch.object(football, '_fetch_match_list_remote', return_value=matches):
            self.assertEqual(football.fetch_match_list(), matches)
            with open(football.MATCH_LIST_CACHE_PATH, encoding='utf-8') as handle:
                self.assertEqual(json.load(handle)['matches'], matches)
            self.assertEqual(football.get_match_list_status()['source'], 'live')

    def test_network_failure_returns_last_successful_snapshot(self):
        cached = [{'match_id': '2', 'home': 'C', 'away': 'D', 'time': '08-08 21:00'}]
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(football, 'MATCH_LIST_CACHE_PATH', os.path.join(folder, 'matches.json')):
            football._save_match_list_cache(cached)
            with patch.object(football, '_fetch_match_list_remote', side_effect=OSError('TLS failed')):
                self.assertEqual(football.fetch_match_list(), cached)
            status = football.get_match_list_status()
            self.assertEqual(status['source'], 'disk_cache')
            self.assertTrue(status['stale'])
            self.assertIn('TLS failed', status['error'])


if __name__ == '__main__':
    unittest.main()
