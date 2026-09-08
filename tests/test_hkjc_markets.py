# -*- coding: utf-8 -*-
import unittest
from copy import deepcopy

from src.football.hkjc_markets import (
    enrich_with_hkjc_markets,
    parse_hkjc_matches,
)
from src.football.pipeline import _is_hkjc_cache_current, _lottery_only_market_inputs


def _line(condition, combinations, main=True):
    return {
        'condition': condition,
        'main': main,
        'status': 'AVAILABLE',
        'combinations': [
            {'str': key, 'currentOdds': str(value), 'status': 'AVAILABLE'}
            for key, value in combinations.items()
        ],
    }


def _raw_match(match_id='5001', home_en='West Ham', away_en='Wolves',
               home_ch='韋斯咸', away_ch='狼隊'):
    return {
        'id': match_id,
        'matchDate': '2026-09-02+08:00',
        'kickOffTime': '2026-09-02T02:45:00.000+08:00',
        'homeTeam': {'name_en': home_en, 'name_ch': home_ch},
        'awayTeam': {'name_en': away_en, 'name_ch': away_ch},
        'foPools': [
            {'oddsType': 'HAD', 'updateAt': '2026-09-01T10:00:00+08:00',
             'lines': [_line('0.0', {'H': 1.94, 'D': 3.50, 'A': 3.00})]},
            {'oddsType': 'HDC', 'updateAt': '2026-09-01T10:01:00+08:00',
             'lines': [_line('0.0/-0.5', {'H': 1.79, 'A': 2.02})]},
            {'oddsType': 'HIL', 'updateAt': '2026-09-01T10:02:00+08:00',
             'lines': [_line('2.5/3.0', {'H': 1.95, 'L': 1.75})]},
        ],
    }


class HkjcMarketTests(unittest.TestCase):
    def test_parses_quarter_asian_and_total_lines(self):
        payload = {'data': {'matches': [_raw_match()]}}
        match = parse_hkjc_matches(payload)[0]
        self.assertEqual(match['asian_current']['handicap'], 0.25)
        self.assertEqual(match['asian_current']['home_odds'], 1.79)
        self.assertEqual(match['asian_current']['odds_format'], 'decimal')
        self.assertEqual(match['total_current']['odds_format'], 'decimal')
        self.assertEqual(match['total_current']['line'], 2.75)
        self.assertEqual(match['had_odds']['平'], 3.50)
        self.assertEqual(match['updated_at'], '2026-09-01T10:02:00+08:00')

    def test_team_codes_disambiguate_same_kickoff(self):
        sporttery = [{
            'date': '2026-09-02', 'time': '09-02 02:45',
            'home': '斯旺西', 'away': '沃特福德',
            'home_full': '斯旺西', 'away_full': '沃特福德',
            'home_code': 'SWA', 'away_code': 'WAT',
            'lottery_spf_odds': {'胜': 1.75, '平': 3.45, '负': 3.70},
        }]
        wrong = parse_hkjc_matches({'data': {'matches': [
            _raw_match('wrong', 'Huddersfield', 'Oxford Utd', '哈特斯菲爾德', '牛津聯')
        ]}})[0]
        right = parse_hkjc_matches({'data': {'matches': [
            _raw_match('right', 'Swansea', 'Watford', '史雲斯', '屈福特')
        ]}})[0]
        enrich_with_hkjc_markets(sporttery, [wrong, right])
        self.assertEqual(sporttery[0]['hkjc_id'], 'right')
        self.assertTrue(sporttery[0]['asian_offer_matched'])

    def test_hkjc_had_and_total_are_blended_with_rqspf_only_match(self):
        match = {
            'lottery_handicap': 3,
            'lottery_spf_odds': None,
            'lottery_rqspf_odds': {'让胜': 2.02, '让平': 4.70, '让负': 2.37},
            'hkjc_had_odds': {'胜': 26.0, '平': 10.0, '负': 1.02},
            'total_current': {'line': 4.5, 'over_odds': 1.97, 'under_odds': 1.74},
        }
        home, draw, away, total = _lottery_only_market_inputs(match)
        self.assertGreater(home, draw)
        self.assertGreater(draw, away)
        self.assertEqual(total, 4.5)

    def test_market_update_invalidates_cached_analysis(self):
        match = {
            'hkjc_id': '5001', 'hkjc_updated_at': '2026-09-01T11:00:00+08:00',
            'asian_offer_matched': True,
            'asian_current': {'handicap': 0.5},
            'total_offer_matched': True,
            'total_current': {'line': 2.75},
        }
        cached = {
            'asian': {'source_matched': True, 'handicap': 0.25,
                      'updated_at': '2026-09-01T10:00:00+08:00'},
            'total': {'source_matched': True, 'close_line': 2.5,
                      'source_event_id': '5001',
                      'updated_at': '2026-09-01T10:00:00+08:00'},
        }
        self.assertFalse(_is_hkjc_cache_current(cached, match))

    def test_same_timestamp_and_line_still_compare_both_current_prices(self):
        match = {
            'hkjc_id': '5001', 'hkjc_updated_at': '2026-09-01T11:00:00+08:00',
            'total_offer_matched': True,
            'total_current': {'line': 2.75, 'over_odds': 1.95, 'under_odds': 1.75},
        }
        cached = {'total': {
            'source_matched': True, 'close_line': 2.75, 'source_event_id': '5001',
            'updated_at': match['hkjc_updated_at'],
            'close_water': {'over': 1.95, 'under': 1.75},
        }}
        self.assertTrue(_is_hkjc_cache_current(cached, match))
        for change in ({'over_odds': 1.85}, {'under_odds': 1.85},
                       {'over_odds': 1.85, 'under_odds': 1.85}):
            with self.subTest(change=change):
                current = deepcopy(match)
                current['total_current'].update(change)
                self.assertFalse(_is_hkjc_cache_current(cached, current))
        # Sources may encode an identical decimal price as text.
        current = deepcopy(match)
        current['total_current'].update(over_odds='1.950', under_odds='1.75')
        self.assertTrue(_is_hkjc_cache_current(cached, current))

    def test_missing_or_invalid_cached_price_cannot_claim_a_current_snapshot(self):
        match = {
            'hkjc_id': '5001', 'total_offer_matched': True,
            'total_current': {'line': 2.75, 'over_odds': 1.95, 'under_odds': 1.75},
        }
        for water in (None, {}, {'over': 1.95}, {'under': 1.75},
                      {'over': float('nan'), 'under': 1.75},
                      {'over': 1.95, 'under': float('inf')}):
            with self.subTest(water=water):
                cached = {'total': {'source_matched': True, 'close_line': 2.75,
                                    'source_event_id': '5001',
                                    'close_water': water}}
                self.assertFalse(_is_hkjc_cache_current(cached, match))

    def test_same_market_values_do_not_reuse_another_or_unidentified_hkjc_event(self):
        match = {
            'hkjc_id': '5001', 'hkjc_updated_at': '2026-09-01T11:00:00+08:00',
            'total_offer_matched': True,
            'total_current': {'line': 2.75, 'over_odds': 1.95, 'under_odds': 1.75},
        }
        cached = {'total': {
            'source_matched': True, 'source_event_id': '5001', 'close_line': 2.75,
            'updated_at': match['hkjc_updated_at'],
            'close_water': {'over': 1.95, 'under': 1.75},
        }}
        self.assertTrue(_is_hkjc_cache_current(cached, match))
        self.assertFalse(_is_hkjc_cache_current(cached, {**match, 'hkjc_id': '5002'}))
        for event_id in (None, '', '5002'):
            with self.subTest(event_id=event_id):
                previous = deepcopy(cached)
                previous['total']['source_event_id'] = event_id
                self.assertFalse(_is_hkjc_cache_current(previous, match))
        previous = deepcopy(cached)
        previous['total'].pop('source_event_id')
        self.assertFalse(_is_hkjc_cache_current(previous, match))


if __name__ == '__main__':
    unittest.main()
