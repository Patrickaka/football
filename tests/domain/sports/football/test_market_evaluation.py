# -*- coding: utf-8 -*-
"""市场评估底座：用同一把尺子量任何概率来源。

football-data 一行同时带 Pinnacle、Bet365、市场均价的开盘与收盘赔率。
把每个来源都当成"一个预测者"评 log loss / Brier / 校准，再算
「尖锐价 × 软盘赔率 − 1」的 EV 策略回报——这才是后面所有改动值不值的判据。
"""

import math
import unittest

from src.domain.sports.football.market_evaluation import (
    expected_value_picks,
    implied_probabilities,
    probability_metrics,
    roi_summary,
    score_rows,
)


def _row(ftr, ps=(2.0, 3.5, 4.0), b365=None, **extra):
    row = {'Date': '15/08/2025', 'HomeTeam': 'H', 'AwayTeam': 'A', 'FTR': ftr,
           'PSCH': ps[0], 'PSCD': ps[1], 'PSCA': ps[2]}
    if b365:
        row.update({'B365CH': b365[0], 'B365CD': b365[1], 'B365CA': b365[2]})
    row.update(extra)
    return row


class ImpliedProbabilityTests(unittest.TestCase):

    def test_devigged_probabilities_sum_to_one_and_keep_order(self):
        probs = implied_probabilities(2.0, 3.5, 4.0)

        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertGreater(probs['H'], probs['D'])
        self.assertGreater(probs['D'], probs['A'])

    def test_invalid_odds_yield_nothing(self):
        self.assertIsNone(implied_probabilities(1.0, 3.5, 4.0))
        self.assertIsNone(implied_probabilities(None, 3.5, 4.0))


class PowerDevigTests(unittest.TestCase):
    """比例去水会高估冷门；幂法把水按赔率高低不均匀地扣，冷门被压回去。"""

    def test_power_devig_sums_to_one(self):
        probs = implied_probabilities(1.30, 5.50, 11.0, method='power')

        self.assertAlmostEqual(sum(probs.values()), 1.0, places=9)

    def test_power_devig_shrinks_the_longshot_relative_to_proportional(self):
        proportional = implied_probabilities(1.30, 5.50, 11.0)
        power = implied_probabilities(1.30, 5.50, 11.0, method='power')

        self.assertLess(power['A'], proportional['A'])
        self.assertGreater(power['H'], proportional['H'])

    def test_fair_odds_are_left_untouched(self):
        # 1/2 + 1/4 + 1/4 = 1，没有水，两种方法都该原样返回。
        power = implied_probabilities(2.0, 4.0, 4.0, method='power')

        self.assertAlmostEqual(power['H'], 0.5, places=9)
        self.assertAlmostEqual(power['D'], 0.25, places=9)

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            implied_probabilities(2.0, 3.5, 4.0, method='magic')

    def test_scoring_and_picks_accept_the_method(self):
        rows = [_row('H', ps=(1.30, 5.50, 11.0), b365=(1.33, 5.20, 12.0))]

        scored = score_rows(rows, 'pinnacle_close', devig='power')
        picks_prop = expected_value_picks(rows, 'pinnacle_close', 'b365_close', 0.0)
        picks_power = expected_value_picks(rows, 'pinnacle_close', 'b365_close', 0.0,
                                           devig='power')

        self.assertAlmostEqual(sum(scored[0]['probs'].values()), 1.0)
        # 比例去水下客胜 12.0 看起来有价值，幂法把这个假优势压掉。
        self.assertIn('A', [p['selection'] for p in picks_prop])
        self.assertNotIn('A', [p['selection'] for p in picks_power])


class ScoreRowsTests(unittest.TestCase):

    def test_rows_missing_the_source_columns_are_skipped(self):
        scored = score_rows([_row('H'), {'FTR': 'H'}], 'pinnacle_close')

        self.assertEqual(len(scored), 1)
        self.assertEqual(scored[0]['result'], 'H')
        self.assertAlmostEqual(sum(scored[0]['probs'].values()), 1.0)

    def test_unknown_source_is_rejected_loudly(self):
        with self.assertRaises(KeyError):
            score_rows([_row('H')], 'no_such_book')


class ProbabilityMetricTests(unittest.TestCase):

    def test_log_loss_and_brier_on_a_known_forecast(self):
        scored = [{'probs': {'H': 0.5, 'D': 0.3, 'A': 0.2}, 'result': 'H'},
                  {'probs': {'H': 0.5, 'D': 0.3, 'A': 0.2}, 'result': 'A'}]

        metrics = probability_metrics(scored)

        self.assertEqual(metrics['n'], 2)
        self.assertAlmostEqual(metrics['log_loss'],
                               -(math.log(0.5) + math.log(0.2)) / 2)
        brier_first = (0.5 - 1) ** 2 + 0.3 ** 2 + 0.2 ** 2
        brier_second = 0.5 ** 2 + 0.3 ** 2 + (0.2 - 1) ** 2
        self.assertAlmostEqual(metrics['brier'], (brier_first + brier_second) / 2)
        self.assertAlmostEqual(metrics['top1_hit_rate'], 0.5)

    def test_perfectly_calibrated_forecasts_have_near_zero_ece(self):
        # 十场都报 0.7 主胜，七场真的主胜：那一桶的频率恰好等于报的概率。
        scored = [{'probs': {'H': 0.7, 'D': 0.2, 'A': 0.1},
                   'result': 'H' if i < 7 else 'A'} for i in range(10)]

        self.assertAlmostEqual(probability_metrics(scored)['ece'], 0.0, places=6)

    def test_empty_input_reports_no_samples_instead_of_dividing_by_zero(self):
        self.assertEqual(probability_metrics([])['n'], 0)


class ExpectedValuePickTests(unittest.TestCase):

    def test_only_selections_above_the_threshold_are_picked(self):
        # Pinnacle 主胜 2.0/3.5/4.0 → 去水后主胜约 0.49；Bet365 主胜开 2.30，
        # EV = 0.49 × 2.30 − 1 ≈ +0.13，平/负在 Bet365 更低不值得下。
        rows = [_row('H', b365=(2.30, 3.30, 3.60))]

        picks = expected_value_picks(rows, 'pinnacle_close', 'b365_close', threshold=0.05)

        self.assertEqual([p['selection'] for p in picks], ['H'])
        self.assertGreater(picks[0]['ev'], 0.05)
        self.assertEqual(picks[0]['odds'], 2.30)
        self.assertTrue(picks[0]['won'])

    def test_no_edge_means_no_pick(self):
        rows = [_row('H', b365=(2.0, 3.5, 4.0))]

        self.assertEqual(expected_value_picks(rows, 'pinnacle_close', 'b365_close', 0.0), [])

    def test_losing_pick_is_marked_lost(self):
        rows = [_row('A', b365=(2.30, 3.30, 3.60))]

        picks = expected_value_picks(rows, 'pinnacle_close', 'b365_close', 0.05)

        self.assertFalse(picks[0]['won'])


class RoiSummaryTests(unittest.TestCase):

    def test_roi_is_profit_per_unit_staked(self):
        picks = [{'odds': 2.0, 'won': True}, {'odds': 2.0, 'won': False},
                 {'odds': 3.0, 'won': False}]

        summary = roi_summary(picks)

        self.assertEqual(summary['n'], 3)
        self.assertAlmostEqual(summary['roi'], (1.0 - 1.0 - 1.0) / 3)
        self.assertAlmostEqual(summary['hit_rate'], 1 / 3)

    def test_confidence_interval_narrows_with_more_bets(self):
        few = roi_summary([{'odds': 2.0, 'won': i % 2 == 0} for i in range(10)])
        many = roi_summary([{'odds': 2.0, 'won': i % 2 == 0} for i in range(1000)])

        width = lambda s: s['roi_ci95'][1] - s['roi_ci95'][0]
        self.assertLess(width(many), width(few))

    def test_no_picks_is_reported_not_crashed(self):
        self.assertEqual(roi_summary([])['n'], 0)
