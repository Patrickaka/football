# -*- coding: utf-8 -*-
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from src.football import config as fb_config
from src.football import fetching as fb_fetching
from src.football.pipeline import _lottery_only_market_inputs
from src.football.sporttery import parse_sporttery_calculator


def _market(code):
    return {'poolCode': code, 'poolStatus': 'Selling'}


class SportteryParserTests(unittest.TestCase):
    def test_both_markets_are_preserved(self):
        payload = {
            'success': True, 'errorCode': '0',
            'value': {'matchInfoList': [{'subMatchList': [{
                'matchId': 2041215, 'matchNumStr': '周二001',
                'matchDate': '2026-09-02', 'matchTime': '02:00:00',
                'leagueAbbName': '沙职', 'homeTeamAbbName': '利雅新月',
                'awayTeamAbbName': '吉达国民',
                'poolList': [_market('HAD'), _market('HHAD')],
                'had': {'h': '1.35', 'd': '4.60', 'a': '5.85'},
                'hhad': {'h': '2.05', 'd': '3.80', 'a': '2.65',
                         'goalLineValue': '-1.00'},
            }]}]},
        }
        match = parse_sporttery_calculator(payload)[0]
        self.assertEqual(match['match_id'], 'sporttery_2041215')
        self.assertEqual(match['num'], '周二001')
        self.assertEqual(match['lottery_available_markets'], ['spf', 'rqspf'])
        self.assertEqual(match['lottery_handicap'], -1)
        self.assertEqual(match['lottery_spf_odds']['胜'], 1.35)
        self.assertEqual(match['lottery_rqspf_odds']['让负'], 2.65)

    def test_bayern_only_rqspf_keeps_plus_three(self):
        payload = {
            'success': True, 'errorCode': 0,
            'value': {'matchInfoList': [{'subMatchList': [{
                'matchId': 2041242, 'matchNumStr': '周三010',
                'matchDate': '2026-09-03', 'matchTime': '02:45:00',
                'leagueAbbName': '德国杯', 'homeTeamAbbName': '奥斯纳',
                'awayTeamAbbName': '拜仁', 'poolList': [_market('HHAD')],
                'had': {},
                'hhad': {'h': '2.02', 'd': '4.70', 'a': '2.37',
                         'goalLineValue': '+3.00'},
            }]}]},
        }
        match = parse_sporttery_calculator(payload)[0]
        self.assertFalse(match['lottery_spf_available'])
        self.assertTrue(match['lottery_rqspf_available'])
        self.assertEqual(match['lottery_available_markets'], ['rqspf'])
        self.assertEqual(match['lottery_handicap'], 3)
        home, draw, away, total = _lottery_only_market_inputs(match)
        self.assertGreater(home, away, '主队受让3球表示未让球时拜仁应明显占优')
        self.assertGreater(total, 3.0)


class SportteryPriorityTests(unittest.TestCase):
    def test_official_schedule_is_used_before_500(self):
        matches = [{
            'match_id': 'sporttery_1', 'home': '主队', 'away': '客队',
            'time': '09-02 20:00', 'schedule_source': 'sporttery',
        }]
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(fb_config, 'MATCH_LIST_CACHE_PATH', os.path.join(folder, 'matches.json')), \
             patch.object(fb_fetching, '_sporttery_schedule', return_value=matches), \
             patch.object(fb_fetching, '_fetch_match_list_remote') as old_source:
            result = fb_fetching.fetch_match_list()
            with open(fb_config.MATCH_LIST_CACHE_PATH, encoding='utf-8') as handle:
                persisted = json.load(handle)['matches']
        self.assertEqual(result, matches)
        self.assertEqual(persisted, matches)
        self.assertFalse(old_source.called)
        self.assertEqual(fb_fetching.get_match_list_status()['source'], 'sporttery')


if __name__ == '__main__':
    unittest.main()
