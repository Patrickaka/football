import unittest

from scripts.backtest_kl8_select6_chain import (
    _one_chain,
    _paired_summary,
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


if __name__ == '__main__':
    unittest.main()
