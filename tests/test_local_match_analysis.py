import unittest

from src.common.local_match_analysis import build_decision, normalize_probabilities
from src.basketball import assess_basketball_upset, build_basketball_analysis


class LocalMatchAnalysisTests(unittest.TestCase):
    def test_normalizes_invalid_probabilities(self):
        probs = normalize_probabilities({'a': 2, 'b': -1, 'c': None})
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertEqual(probs['b'], 0.0)

    def test_close_match_uses_double_selection(self):
        decision = build_decision({'胜': 0.38, '平': 0.33, '负': 0.29}, confidence='high')
        self.assertEqual(decision['action'], '双选')
        self.assertEqual(decision['secondary'], '平')

    def test_low_confidence_is_not_forced_to_single(self):
        decision = build_decision({'主胜': 0.57, '客胜': 0.43}, confidence='low')
        self.assertEqual(decision['action'], '观望')

    def test_basketball_scores_are_projections_not_fake_probabilities(self):
        result = build_basketball_analysis({
            'match': {'home': 'A', 'away': 'B'},
            'spf': {'home_prob': 0.7, 'away_prob': 0.3, 'confidence': 'high'},
            'rqspf': {'elo_margin': 6},
            'dx': {'total_line': 210, 'elo_total': 208,
                   'over_prob': 0.45, 'under_prob': 0.55,
                   'recommendation': '小分'},
        })
        self.assertEqual(result['decision']['action'], '单选')
        self.assertEqual(len(result['score_picks']), 2)
        self.assertTrue(all(p['projected'] for p in result['score_picks']))
        self.assertTrue(all(p['probability'] is None for p in result['score_picks']))

    def test_basketball_detects_model_market_upset_risk(self):
        upset = assess_basketball_upset({
            'home_prob': 0.62, 'away_prob': 0.38,
            'market_home_prob': 0.46, 'elo_trust': 0.8,
            'books_ml': {'trend': 'away_backing'},
        }, {'recommendation': '让负'})
        self.assertTrue(upset['alert'])
        self.assertEqual(upset['level'], 'high')
        self.assertIn('本地模型与市场胜负方向相反', upset['signals'])
        self.assertIn('赔率资金走势与热门方向相反', upset['signals'])

    def test_basketball_stable_favorite_has_no_alert(self):
        upset = assess_basketball_upset({
            'home_prob': 0.68, 'away_prob': 0.32,
            'market_home_prob': 0.64, 'elo_trust': 0.8,
            'books_ml': {'trend': 'home_backing'},
        }, {'recommendation': '让胜'})
        self.assertFalse(upset['alert'])


if __name__ == '__main__':
    unittest.main()
