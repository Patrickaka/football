import unittest
from unittest.mock import patch

from src.beidan import (
    apply_beidan_history_calibration,
    analyze_spf,
    analyze_zjq,
    assess_recommendation_quality,
    assess_score_consistency,
    build_zjq_group_recommendation,
    enhance_scores_with_cs,
    save_beidan_prediction_snapshot,
)


class BeidanQualityTests(unittest.TestCase):
    def test_narrow_spf_call_is_marked_as_split(self):
        quality = assess_recommendation_quality({'胜': 0.36, '平': 0.34, '负': 0.30}, '胜')

        self.assertEqual(quality['level'], 'split')
        self.assertTrue(quality['avoid_single'])
        self.assertEqual([x['option'] for x in quality['top2']], ['胜', '平'])

    def test_score_conflict_downgrades_quality(self):
        scores = [
            {'score': '0-1', 'probability': 0.16, 'home_goals': 0, 'away_goals': 1},
            {'score': '1-1', 'probability': 0.12, 'home_goals': 1, 'away_goals': 1},
            {'score': '0-2', 'probability': 0.08, 'home_goals': 0, 'away_goals': 2},
        ]
        consistency = assess_score_consistency(scores, '胜')
        quality = assess_recommendation_quality(
            {'胜': 0.48, '平': 0.28, '负': 0.24},
            '胜',
            {'score_consistency': consistency}
        )

        self.assertTrue(consistency['conflict'])
        self.assertEqual(quality['level'], 'split')
        self.assertTrue(quality['avoid_single'])

    def test_analyze_spf_uses_adjusted_probabilities_for_quality(self):
        match = {
            'id': 'm1',
            'num': '001',
            'home': 'A',
            'away': 'B',
            'league': '',
            'time': '20:00',
            'handicap': 0,
        }
        asian_data = {
            'history': [
                {'home_odds': 0.82, 'away_odds': 0.96},
                {'home_odds': 0.78, 'away_odds': 1.00},
            ]
        }

        with patch('src.beidan.fetch_ouzhi_odds', return_value={'home': 2.7, 'draw': 3.0, 'away': 2.6}):
            result = analyze_spf(match, asian_data=asian_data)

        self.assertEqual(result['prediction'], '胜')
        self.assertIn('quality', result)
        self.assertIn(result['quality']['level'], {'medium', 'split', 'low', 'strong'})
        self.assertIn('raw_probabilities', result)
        self.assertIn('score_consistency', result)
        self.assertIn('history_calibration', result)

    def test_history_calibration_lifts_underestimated_settled_outcome(self):
        records = []
        for idx in range(10):
            records.append({
                'settled': True,
                'league': 'L',
                'actual': {'score': '1-0'},
                'spf': {'probabilities': {'胜': 0.30, '平': 0.35, '负': 0.35}},
            })

        with patch('src.beidan._load_beidan_history', return_value=records):
            adjusted, meta = apply_beidan_history_calibration(
                {'胜': 0.34, '平': 0.33, '负': 0.33},
                'spf',
                league='L'
            )

        self.assertTrue(meta['applied'])
        self.assertGreater(adjusted['胜'], 0.34)
        self.assertAlmostEqual(sum(adjusted.values()), 1.0, places=6)

    def test_history_calibration_waits_for_enough_samples(self):
        records = [{
            'settled': True,
            'actual': {'score': '0-1'},
            'spf': {'probabilities': {'胜': 0.40, '平': 0.30, '负': 0.30}},
        }]

        with patch('src.beidan._load_beidan_history', return_value=records):
            adjusted, meta = apply_beidan_history_calibration(
                {'胜': 0.34, '平': 0.33, '负': 0.33},
                'spf'
            )

        self.assertFalse(meta['applied'])
        self.assertEqual(adjusted['胜'], 0.34)

    def test_enhance_scores_with_cs_keeps_dict_shape(self):
        score_prediction = {
            'top3': [
                {'score': '1-0', 'probability': 0.12, 'home_goals': 1, 'away_goals': 0},
                {'score': '1-1', 'probability': 0.10, 'home_goals': 1, 'away_goals': 1},
                {'score': '0-0', 'probability': 0.08, 'home_goals': 0, 'away_goals': 0},
            ]
        }
        cs_history = [
            {'score': '2-1', 'odds': 6.0},
            {'score': '1-0', 'odds': 5.0},
        ]

        enhanced = enhance_scores_with_cs(score_prediction, cs_history)

        self.assertIsInstance(enhanced['top3'][0], dict)
        self.assertIn('score', enhanced['top3'][0])
        self.assertIn('probability', enhanced['top3'][0])

    def test_analyze_zjq_blends_market_goal_odds(self):
        match = {
            'id': 'm1',
            'num': '001',
            'home': 'A',
            'away': 'B',
            'league': '英超',
            'time': '20:00',
        }
        zjq_odds = {'m1': {'0': 15.0, '1': 7.0, '2': 3.0, '3': 3.2, '4': 5.0, '5': 9.0, '6': 18.0, '7+': 28.0}}

        with patch('src.beidan.fetch_ouzhi_odds', return_value={'home': 2.0, 'draw': 3.3, 'away': 3.6}):
            result = analyze_zjq(match, zjq_odds=zjq_odds)

        self.assertTrue(result['market_adjusted'])
        self.assertIn('odds', result)
        self.assertIn('quality', result)
        self.assertIn('goal_groups', result)
        self.assertIn(result['goal_groups']['primary']['key'], {'small', 'middle', 'big'})
        self.assertAlmostEqual(sum(result['probabilities'].values()), 1.0, places=6)

    def test_zjq_group_recommendation_prefers_best_band(self):
        groups = build_zjq_group_recommendation({
            '0': 0.02, '1': 0.08, '2': 0.16,
            '3': 0.24, '4': 0.22, '5': 0.14, '6': 0.08, '7+': 0.06,
        })

        self.assertEqual(groups['primary']['key'], 'big')
        self.assertEqual(groups['primary']['options'], ['3', '4', '5', '6', '7+'])

    def test_save_prediction_snapshot_upserts_by_match_key(self):
        saved_payloads = []
        result = {
            'source': 'okooo',
            'recommendations': [{
                'date': '2026-07-08',
                'num': '001',
                'time': '20:00',
                'league': 'L',
                'home': 'A',
                'away': 'B',
                'spf': {'prediction': '胜', 'confidence': 0.48, 'quality': {'level': 'strong'}},
                'zjq': {
                    'prediction': '3',
                    'confidence': 0.24,
                    'quality': {'level': 'medium'},
                    'goal_groups': {'primary': {'key': 'big'}},
                },
            }]
        }

        with patch('src.beidan._load_beidan_history', return_value=[]), \
                patch('src.beidan._save_beidan_history', side_effect=lambda rows: saved_payloads.append(rows)):
            summary = save_beidan_prediction_snapshot(result)

        self.assertEqual(summary['saved'], 1)
        self.assertEqual(saved_payloads[0][0]['key'], '2026-07-08|001|A|B')
        self.assertEqual(saved_payloads[0][0]['zjq']['goal_groups']['primary']['key'], 'big')


if __name__ == '__main__':
    unittest.main()
