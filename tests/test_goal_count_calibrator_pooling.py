#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GoalCountCalibrator 部分池化(partial pooling)行为测试。

验证：反馈闭环不再是“要么不动、要么满额”的硬阈值死代码，而是随样本量渐进生效：
  - 样本低于激活下限 → 不校准(空因子)
  - 少量样本 → 因子向 1.0 收缩(弱校准)
  - 大量样本 → 因子逼近观测频率比(强校准)
"""
import unittest

from src.football.goal_count_calibrator import (
    GoalCountCalibrator, MIN_ACTIVATION_SAMPLES, POOLING_K,
)


class PartialPoolingTests(unittest.TestCase):
    def _fresh(self):
        c = GoalCountCalibrator()
        c.clear()  # 隔离已有历史，保证测试确定性
        return c

    # 预测分布偏向 2 球，但实际总是 3 球 → 3 球的原始校准因子应 > 1
    PRED = {0: 0.1, 1: 0.3, 2: 0.4, 3: 0.2}

    def _record(self, c, n):
        for _ in range(n):
            c.record_result('测试联赛', 2.5, self.PRED,
                            actual_total_goals=3, expected_total_goals=2.0, asian=0.0)

    def test_below_floor_does_not_calibrate(self):
        c = self._fresh()
        self._record(c, MIN_ACTIVATION_SAMPLES - 1)
        factors = c.get_calibration_factors('测试联赛', 2.5, 2.0, 0.0)
        self.assertEqual(factors, {}, "样本低于激活下限时不应产生校准因子")

    def test_small_sample_shrinks_toward_one(self):
        c = self._fresh()
        n = 5
        self._record(c, n)
        factors = c.get_calibration_factors('测试联赛', 2.5, 2.0, 0.0)
        self.assertTrue(factors, "达到激活下限后应产生校准因子")
        # 3 球方向应被上调(>1)，但因样本少而向 1.0 收缩，不应到满额 2.0
        f3 = factors.get(3)
        self.assertIsNotNone(f3)
        self.assertGreater(f3, 1.0)
        self.assertLess(f3, 1.6, "少样本时因子应显著向 1.0 收缩")

    def test_large_sample_approaches_raw_ratio(self):
        c = self._fresh()
        self._record(c, 60)
        factors = c.get_calibration_factors('测试联赛', 2.5, 2.0, 0.0)
        f3 = factors.get(3)
        self.assertIsNotNone(f3)
        # 样本充足时应明显强于少样本档，逼近上限(原始比被夹取到 2.0)
        self.assertGreater(f3, 1.7)

    def test_monotonic_in_sample_size(self):
        # 同一偏差下，样本越多，校准强度(|factor-1|)应越大
        c_small = self._fresh(); self._record(c_small, 5)
        c_big = self._fresh(); self._record(c_big, 50)
        f_small = c_small.get_calibration_factors('测试联赛', 2.5, 2.0, 0.0).get(3, 1.0)
        f_big = c_big.get_calibration_factors('测试联赛', 2.5, 2.0, 0.0).get(3, 1.0)
        self.assertGreater(f_big - 1.0, f_small - 1.0)


if __name__ == '__main__':
    unittest.main()
