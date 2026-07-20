import unittest

from src.beidan import DC_RHO, OU_TOTAL_BLEND, SCORE_SPLIT, STRENGTH_SPLIT


class BeidanBacktestParameterTests(unittest.TestCase):
    def test_production_parameters_match_2744_match_backtest_winners(self):
        self.assertEqual(DC_RHO, -0.08)
        self.assertEqual(OU_TOTAL_BLEND, 0.6)
        self.assertEqual(SCORE_SPLIT, 0.45)
        self.assertEqual(STRENGTH_SPLIT, SCORE_SPLIT)


if __name__ == '__main__':
    unittest.main()
