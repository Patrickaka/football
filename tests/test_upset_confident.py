#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""爆冷识别 + 稳胆档行为测试。"""
import unittest

from src.beidan import (
    assess_upset_risk,
    UPSET_CONFIDENT_FAV_MIN, UPSET_CONFIDENT_GAP_MIN,
)


class UpsetRiskTests(unittest.TestCase):
    def test_high_upset_alert(self):
        # 弱热门 + 胶着 + 非热门总概率高 → 高风险预警
        info = assess_upset_risk({'胜': 0.42, '平': 0.33, '负': 0.25})
        self.assertTrue(info['alert'])
        self.assertIn(info['level'], ('high', 'medium'))
        self.assertFalse(info['confident'])

    def test_confident_favorite(self):
        # 强热门 + 差距悬殊 → 稳胆(非预警、confident)
        info = assess_upset_risk({'胜': 0.62, '平': 0.24, '负': 0.14})
        self.assertFalse(info['alert'])
        self.assertTrue(info['confident'])
        self.assertEqual(info['label'], '热门稳胆')
        self.assertGreaterEqual(info['favorite_prob'], UPSET_CONFIDENT_FAV_MIN)
        self.assertGreaterEqual(info['gap'], UPSET_CONFIDENT_GAP_MIN)

    def test_weak_favorite_not_confident(self):
        # 中等热门、差距不够 → 既不预警也不稳胆(普通)
        info = assess_upset_risk({'胜': 0.50, '平': 0.28, '负': 0.22})
        self.assertFalse(info['alert'])
        self.assertFalse(info['confident'])

    def test_confident_requires_wide_gap(self):
        # 热门概率够高但差距太小(次热门也高) → 不算稳胆
        info = assess_upset_risk({'胜': 0.60, '平': 0.45, '负': 0.15})
        # 归一化后差距会被压缩，验证 gap 门槛生效
        self.assertFalse(info['confident'] and info['gap'] < UPSET_CONFIDENT_GAP_MIN)


if __name__ == '__main__':
    unittest.main()
