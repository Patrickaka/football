import unittest

from src.football.contextual_fusion import apply_contextual_fusion


class ContextualFusionTests(unittest.TestCase):
    def test_qualified_h2h_and_motivation_adjust_distribution(self):
        candidates = [((1, 0), .4), ((1, 1), .35), ((0, 1), .25)]
        adjusted, meta = apply_contextual_fusion(candidates, {
            'h2h': {'games': 8, 'home_wins': 5, 'draws': 2, 'away_wins': 1,
                    'avg_goals': 3.1, 'quality_score': .8},
            'motivation': {'home': 1, 'away': 0, 'quality_score': .9,
                           'source': 'official'},
        })
        values = dict(adjusted)
        self.assertTrue(meta['applied'])
        self.assertGreater(values[(1, 0)], .4)
        self.assertAlmostEqual(sum(values.values()), 1.0)

    def test_free_text_or_low_quality_context_is_not_used(self):
        candidates = [((1, 0), .6), ((0, 1), .4)]
        adjusted, meta = apply_contextual_fusion(candidates, {
            'style_notes': '主队必须赢',
            'motivation': {'home': 1, 'away': 0, 'quality_score': .2},
        })
        self.assertFalse(meta['applied'])
        self.assertEqual(adjusted, candidates)


if __name__ == '__main__':
    unittest.main()
