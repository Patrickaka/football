import unittest
from unittest.mock import Mock, patch

from src.basketball import analyze_daxiao, analyze_rqspf, analyze_spf, find_value_bets
from src.basketball.records import save_predictions


class BasketballPredictionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.match = {
            'league': 'NBA', 'home': 'Home', 'away': 'Away',
            'spf_home': 1.55, 'spf_away': 2.35,
            'handicap': '-4.5', 'rqspf_home': 1.70, 'rqspf_away': 1.95,
            'total_line': 215.5, 'dx_over': 1.70, 'dx_under': 1.95,
        }

    @patch('src.basketball._elo_predictions')
    def test_cold_start_elo_does_not_move_market(self, elo_predictions):
        elo_predictions.return_value = (
            {'home_prob': 0.1}, {'expected_margin': -20},
            {'expected_total': 180}, 0.0,
        )
        with patch('src.basketball._calibrate_pick', side_effect=lambda _t, a, b, _l, _c: ((a, b), max(a, b))):
            result = analyze_spf(self.match)
        self.assertAlmostEqual(result['home_prob'], result['market_home_prob'], places=4)
        self.assertEqual(result['elo_trust'], 0.0)

    @patch('src.basketball._elo_predictions')
    def test_mature_elo_is_blended_into_all_markets(self, elo_predictions):
        elo_predictions.return_value = (
            {'home_prob': 0.8}, {'expected_margin': 10},
            {'expected_total': 235}, 1.0,
        )
        with patch('src.basketball._calibrate_pick', side_effect=lambda _t, a, b, _l, _c: ((a, b), max(a, b))):
            spf = analyze_spf(self.match)
            rq = analyze_rqspf(self.match)
            dx = analyze_daxiao(self.match)
        self.assertGreater(spf['home_prob'], spf['market_home_prob'])
        self.assertGreater(rq['home_prob'], rq['market_home_prob'])
        self.assertGreater(dx['over_prob'], dx['market_over_prob'])

    def test_value_list_excludes_non_official_opinions(self):
        rows = [{'match': self.match, 'spf': {
            'available': True, 'playable': False, 'home_prob': 0.8,
            'away_prob': 0.2, 'recommendation': '主胜',
        }, 'rqspf': None, 'dx': None}]
        self.assertEqual(find_value_bets(rows), [])

    @patch('src.basketball._elo_predictions')
    def test_spread_requires_fresh_confirming_movement(self, elo_predictions):
        elo_predictions.return_value = ({}, {}, {}, 0.0)
        strong_home = {**self.match, 'rqspf_home': 1.30, 'rqspf_away': 3.20}
        with patch('src.basketball._calibrate_pick', side_effect=lambda _t, a, b, _l, _c: ((a, b), max(a, b))):
            missing = analyze_rqspf(strong_home)
            contrary = analyze_rqspf(strong_home, {
                'available': True, 'side': 'away', 'strength': .5,
                'samples': 4, 'stale': False, 'steam': False,
            })
            confirmed = analyze_rqspf(strong_home, {
                'available': True, 'side': 'home', 'strength': .5,
                'samples': 4, 'stale': False, 'steam': False,
            })
        self.assertFalse(missing['official'])
        self.assertEqual(missing['skip_reason'], 'movement_unavailable')
        self.assertFalse(contrary['official'])
        self.assertEqual(contrary['skip_reason'], 'movement_conflicts_with_model')
        self.assertTrue(confirmed['official'])

    @patch('src.basketball._elo_predictions')
    def test_totals_requires_fresh_confirming_movement(self, elo_predictions):
        elo_predictions.return_value = ({}, {}, {}, 0.0)
        with patch('src.basketball._calibrate_pick', side_effect=lambda _t, a, b, _l, _c: ((a, b), max(a, b))):
            result = analyze_daxiao(self.match, {
                'available': True, 'side': 'under', 'strength': .6,
                'samples': 5, 'stale': True, 'steam': False,
            })
        self.assertFalse(result['official'])
        self.assertEqual(result['skip_reason'], 'movement_stale')

    def test_refresh_preserves_settled_result_and_other_same_day_match(self):
        old = [
            {'date': '2026-07-20', 'match_id': 'm1', 'result': {'home_score': 100, 'away_score': 90}},
            {'date': '2026-07-20', 'match_id': 'm2', 'result': None},
        ]
        saved = []
        refreshed = [{'match': {'id': 'm1', 'league': 'NBA', 'home': 'H', 'away': 'A'},
                      'spf': {'available': True, 'recommendation': '主胜'}}]
        with patch('src.basketball.records.kv_store.load', return_value=old), \
                patch('src.basketball.records.kv_store.save', side_effect=lambda _k, rows: saved.extend(rows)):
            save_predictions('2026-07-20', refreshed, 'v2')
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[0]['result']['home_score'], 100)
        self.assertEqual(saved[1]['match_id'], 'm2')


if __name__ == '__main__':
    unittest.main()
