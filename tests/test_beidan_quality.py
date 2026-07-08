import unittest
from unittest.mock import patch

from src.beidan import (
    analyze_spf,
    analyze_zjq,
    assess_recommendation_quality,
    enhance_scores_with_cs,
)


class BeidanQualityTests(unittest.TestCase):
    def test_narrow_spf_call_is_marked_as_split(self):
        quality = assess_recommendation_quality({'胜': 0.36, '平': 0.34, '负': 0.30}, '胜')

        self.assertEqual(quality['level'], 'split')
        self.assertTrue(quality['avoid_single'])
        self.assertEqual([x['option'] for x in quality['top2']], ['胜', '平'])

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
        self.assertAlmostEqual(sum(result['probabilities'].values()), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
