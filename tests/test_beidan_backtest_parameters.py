import unittest

from src.beidan import DC_RHO, OU_TOTAL_BLEND, SCORE_SPLIT, STRENGTH_SPLIT


class BeidanBacktestParameterTests(unittest.TestCase):
    def test_production_parameters_match_2744_match_backtest_winners(self):
        # DC_RHO 由 -0.08 调整为 0.0：405 场真实线上样本证明负 rho 使模型把 1-1
        # 当作最看好比分的场次高达 58%（真实仅 13%），置 0 后去除人为低分抬高，
        # 线上 1-1 占比降至 41%、进球命中提升，五大联赛离线回测中性偏好。
        self.assertEqual(DC_RHO, 0.0)
        self.assertEqual(OU_TOTAL_BLEND, 0.6)
        self.assertEqual(SCORE_SPLIT, 0.45)
        self.assertEqual(STRENGTH_SPLIT, SCORE_SPLIT)


if __name__ == '__main__':
    unittest.main()
