import unittest

import src.football.ml as ml


class FootballMLFallbackTests(unittest.TestCase):
    def setUp(self):
        self._old_numpy_available = ml.NUMPY_AVAILABLE

    def tearDown(self):
        ml.NUMPY_AVAILABLE = self._old_numpy_available

    def test_goal_count_prediction_works_without_numpy(self):
        ml.NUMPY_AVAILABLE = False
        candidates = [((1, 1), 0.35), ((2, 1), 0.25), ((0, 0), 0.20), ((3, 1), 0.20)]

        result = ml.predict_goal_counts_from_candidates(candidates, max_goals=4)

        self.assertTrue(result['recommendations'])
        self.assertAlmostEqual(sum(result['distribution_dict'].values()), 1.0)
        self.assertIsInstance(result['matrix'], list)

    def test_dixon_coles_1x2_works_without_numpy(self):
        ml.NUMPY_AVAILABLE = False

        probs = ml.dixon_coles_1x2_prob(1.4, 1.1, max_goals=5, rho=0.08)

        self.assertGreater(probs['home'], 0)
        self.assertGreater(probs['draw'], 0)
        self.assertGreater(probs['away'], 0)
        self.assertAlmostEqual(sum(probs.values()), 1.0)


if __name__ == '__main__':
    unittest.main()
