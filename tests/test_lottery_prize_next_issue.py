#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v4.5/v3.3：跨年期号推算 + 奖级结算 测试"""
import sys
import unittest

sys.path.insert(0, '.')

from src.lottery import _next_issue, dlt_prize_tier, LOTTERY_PREDICTOR_VERSION
from src.ssq import ssq_prize_tier, calculate_prediction_stats, SSQ_PREDICTION_VERSION


def _hist(*issues):
    return [{'issue': s, 'front': [1, 2, 3, 4, 5], 'back': [1, 2]} for s in issues]


class TestNextIssue(unittest.TestCase):
    def test_mid_year_increment(self):
        hist = _hist('2025150', '2025100', '2026082')
        self.assertEqual(_next_issue('2026082', hist), '2026083')
        self.assertEqual(_next_issue('2025100', hist), '2025101')

    def test_year_rollover_completed_year(self):
        # 2025 已完结（数据最后一年是 2026），当年最大期号 150 → 跨年
        hist = _hist('2025150', '2025149', '2026082')
        self.assertEqual(_next_issue('2025150', hist), '2026001')

    def test_year_rollover_last_year_threshold(self):
        # 最后一年且已开 >=145 期且到达当年最大值 → 跨年
        hist = _hist(*(f'2025{i:03d}' for i in range(1, 151)))
        self.assertEqual(_next_issue('2025150', hist), '2026001')
        # 最后一年但期数不足（数据未抓全）→ 不跨年
        hist2 = _hist(*(f'2026{i:03d}' for i in range(1, 83)))
        self.assertEqual(_next_issue('2026082', hist2), '2026083')

    def test_fallback_without_history(self):
        self.assertEqual(_next_issue('2025150', None), '2025151')
        self.assertEqual(_next_issue('2025155', None), '2026001')

    def test_version_bump(self):
        self.assertEqual(LOTTERY_PREDICTOR_VERSION, 'dlt-v4.5-next-issue')
        self.assertEqual(SSQ_PREDICTION_VERSION, 'ssq-v3.3-prize-stats')


class TestSsqPrizeTier(unittest.TestCase):
    def test_table(self):
        self.assertEqual(ssq_prize_tier(6, True), 1)
        self.assertEqual(ssq_prize_tier(6, False), 2)
        self.assertEqual(ssq_prize_tier(5, True), 3)
        self.assertEqual(ssq_prize_tier(5, False), 4)
        self.assertEqual(ssq_prize_tier(4, True), 4)
        self.assertEqual(ssq_prize_tier(4, False), 5)
        self.assertEqual(ssq_prize_tier(3, True), 5)
        self.assertEqual(ssq_prize_tier(3, False), 0)
        self.assertEqual(ssq_prize_tier(2, True), 6)
        self.assertEqual(ssq_prize_tier(1, True), 6)
        self.assertEqual(ssq_prize_tier(0, True), 6)
        self.assertEqual(ssq_prize_tier(2, False), 0)

    def test_stats_fallback_recompute(self):
        # 旧记录无 prize 字段 → 按命中数回算
        records = [{
            'period': '26001', 'settled': True,
            'results': [
                {'red_hits': 2, 'blue_hit': True},   # 六等
                {'red_hits': 3, 'blue_hit': False},  # 未中
            ],
        }]
        stats = calculate_prediction_stats(records)
        self.assertEqual(stats['prize_counts']['6'], 1)
        self.assertEqual(stats['prize_counts']['0'], 1)
        self.assertEqual(stats['any_prize_rate'], 0.5)
        self.assertIn('baseline', stats)


class TestDltPrizeTier(unittest.TestCase):
    def test_table(self):
        self.assertEqual(dlt_prize_tier(5, 2), 1)
        self.assertEqual(dlt_prize_tier(5, 1), 2)
        self.assertEqual(dlt_prize_tier(5, 0), 3)
        self.assertEqual(dlt_prize_tier(4, 2), 4)
        self.assertEqual(dlt_prize_tier(4, 1), 5)
        self.assertEqual(dlt_prize_tier(3, 2), 6)
        self.assertEqual(dlt_prize_tier(4, 0), 7)
        self.assertEqual(dlt_prize_tier(3, 1), 8)
        self.assertEqual(dlt_prize_tier(2, 2), 8)
        self.assertEqual(dlt_prize_tier(3, 0), 9)
        self.assertEqual(dlt_prize_tier(2, 1), 9)
        self.assertEqual(dlt_prize_tier(1, 2), 9)
        self.assertEqual(dlt_prize_tier(0, 2), 9)
        self.assertEqual(dlt_prize_tier(2, 0), 0)
        self.assertEqual(dlt_prize_tier(1, 1), 0)
        self.assertEqual(dlt_prize_tier(0, 0), 0)


if __name__ == '__main__':
    unittest.main()
