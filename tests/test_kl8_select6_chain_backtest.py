import unittest

from scripts.backtest_kl8_select6_chain import (
    _one_chain,
    _paired_summary,
    _row_score,
    _score,
    _strategy_slate,
    _summarize,
)


class _FakeAnalyzer:
    history_data = []
    statistics = {'last_numbers': set()}

    def build_pool_by_strategy(self, strategy, pool_size=80):
        return {
            'candidates': [(number, float(81 - number)) for number in range(1, 81)],
        }


class KL8Select6ChainBacktestTests(unittest.TestCase):
    def test_chain_round_zero_is_primary_and_later_rounds_are_disjoint(self):
        strategy = {
            'pool_max_last_numbers': 3,
            'final_selection_mode': 'concentrated',
        }

        groups = _one_chain(_FakeAnalyzer(), strategy, rounds=5)

        self.assertEqual(groups[0], [1, 2, 3, 4, 5, 6])
        self.assertEqual(groups[1], [7, 8, 9, 10, 11, 12])
        self.assertEqual(len({number for group in groups for number in group}), 30)

    def test_summary_prioritizes_primary_and_first_early_hit_round(self):
        metrics = _summarize([
            [1, 0, 3, 2, 4],
            [2, 3, 1, 0, 1],
        ], rounds=5)

        self.assertEqual(metrics['primary_mean_hits'], 1.5)
        self.assertEqual(metrics['early_any_hit_3_rate'], 1.0)
        self.assertEqual(metrics['early_any_hit_4_rate'], 0.5)
        self.assertEqual(metrics['mean_first_hit_3_round'], 1.5)

    def test_paired_interval_keeps_zero_when_difference_is_uncertain(self):
        comparison = _paired_summary([1.0, -1.0, 1.0, -1.0])

        self.assertEqual(comparison['mean'], 0.0)
        self.assertLess(comparison['ci_95'][0], 0.0)
        self.assertGreater(comparison['ci_95'][1], 0.0)

    def test_primary_accuracy_outweighs_late_chain_coverage(self):
        primary_better = [3, 0, 0, 0, 0]
        coverage_better = [1, 4, 4, 4, 4]

        self.assertGreater(_row_score(primary_better), _row_score(coverage_better))

        primary_metrics = _summarize([primary_better], rounds=5)
        coverage_metrics = _summarize([coverage_better], rounds=5)
        self.assertGreater(_score(primary_metrics), _score(coverage_metrics))

    def test_slate_keeps_v10_8_and_v10_9_controls(self):
        slate = _strategy_slate()

        self.assertEqual(slate['previous_reference_v3']['window_size'], 100)
        self.assertEqual(
            slate['previous_reference_v3']['final_selection_mode'],
            'concentrated',
        )
        self.assertEqual(slate['previous_chain_v4']['window_size'], 150)
        self.assertEqual(
            slate['previous_chain_v4']['final_selection_mode'],
            'shape_balanced',
        )


if __name__ == '__main__':
    unittest.main()
