import unittest

import src.football as football


class PredictionPostprocessTests(unittest.TestCase):
    def test_goal_distribution_anchor_moves_mean_toward_total_line(self):
        dist = {1: 0.50, 2: 0.30, 5: 0.20}
        before = sum(goals * prob for goals, prob in dist.items())

        adjusted, meta = football._anchor_goal_dist_to_total_line(dist, {'close_line': 3.0})
        after = sum(goals * prob for goals, prob in adjusted.items())

        self.assertTrue(meta['applied'])
        self.assertGreater(after, before)
        self.assertAlmostEqual(sum(adjusted.values()), 1.0)

    def test_goal_over_under_uses_actual_line(self):
        result = football._goal_over_under_from_line({2: 0.4, 3: 0.6}, {'close_line': 2.5})

        self.assertAlmostEqual(result['over'], 0.6)
        self.assertAlmostEqual(result['under'], 0.4)
        self.assertEqual(result['line'], 2.5)

    def test_score_total_line_factor_penalizes_low_score_on_high_line(self):
        low_score_factor = football._score_total_line_factor(0, 0, 3.25)
        aligned_factor = football._score_total_line_factor(2, 1, 3.25)

        self.assertLess(low_score_factor, aligned_factor)

    def test_common_score_overheat_factor_dampens_hot_common_score(self):
        factor = football._common_score_overheat_factor(1, 1, 0.24, 2.5)

        self.assertLess(factor, 1.0)


if __name__ == '__main__':
    unittest.main()
