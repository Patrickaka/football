import unittest

import src.football.dynamic_weights as dynamic_weights


class DynamicWeightsFallbackTests(unittest.TestCase):
    def setUp(self):
        self._old_numpy_available = dynamic_weights.NUMPY_AVAILABLE

    def tearDown(self):
        dynamic_weights.NUMPY_AVAILABLE = self._old_numpy_available

    def test_get_dynamic_weights_works_without_numpy(self):
        dynamic_weights.NUMPY_AVAILABLE = False

        weights = dynamic_weights.get_dynamic_weights(0.6)

        self.assertEqual(len(weights), 4)
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertGreater(weights[0], 0)

    def test_fuse_predictions_works_without_numpy(self):
        dynamic_weights.NUMPY_AVAILABLE = False

        fused = dynamic_weights.fuse_predictions(
            {'1-0': 0.6, '1-1': 0.4},
            {'1-0': 0.5, '0-1': 0.5},
            {'1-1': 0.7, '0-1': 0.3},
            confidence=0.6,
        )

        self.assertAlmostEqual(sum(fused.values()), 1.0)
        self.assertIn('1-0', fused)


if __name__ == '__main__':
    unittest.main()
