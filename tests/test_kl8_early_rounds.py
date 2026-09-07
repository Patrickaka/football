import unittest
from unittest.mock import patch

from scripts.backtest.backtest_kl8_early_rounds import live_groups, objective, summarize, run_slice, slate
from src.kl8 import KL8Analyzer


class EarlyRoundAuditTests(unittest.TestCase):
    def test_walk_forward_excludes_target_and_newer_draws(self):
        raw = [{'issue': str(10-i), 'numbers': list(range(1, 21))} for i in range(6)]
        seen = []

        def groups(analyzer, strategy):
            seen.append([r['issue'] for r in analyzer.history_data])
            return {'select_6': [[1, 2, 3, 4, 5, 6]] * 3,
                    'fu_shi_7': [[1, 2, 3, 4, 5, 6, 7]] * 3}

        with patch.object(KL8Analyzer, 'update_statistics'), \
             patch('scripts.backtest.backtest_kl8_early_rounds.live_groups', side_effect=groups):
            result = run_slice(raw, [2, 3], {'current': {}})
        self.assertEqual(seen, [['7', '6', '5'], ['6', '5']])
        self.assertEqual(len(result['current']), 2)

    def test_expanded_slate_is_opt_in_and_does_not_mutate_defaults(self):
        original = slate()
        expanded = slate(expanded=True)
        self.assertEqual(len(original), 4)
        self.assertEqual(len(expanded), 8)
        self.assertEqual(original, slate())
        self.assertFalse(expanded['select6_hot_balanced_150']['pool_diversify'])

    def test_objective_ignores_primary_and_does_not_sum_round_hits(self):
        self.assertEqual(objective({'select_6': [6, 2, 2], 'fu_shi_7': [7, 3, 3]}), 0)
        self.assertEqual(objective({'select_6': [0, 4, 5], 'fu_shi_7': [0, 4, 5]}), 1.5)

    def test_summary_separates_rounds_and_plays(self):
        summary = summarize([{'select_6': [1, 4, 2], 'fu_shi_7': [2, 5, 3]}])
        self.assertEqual(summary['select_6'][1]['hit_4_rate'], 1)
        self.assertEqual(summary['select_6'][1]['hit_5_rate'], 0)
        self.assertEqual(summary['fu_shi_7'][1]['hit_5_rate'], 1)

    def test_live_chain_is_disjoint_and_does_not_write_records(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [{'issue': '2026001', 'numbers': list(range(1, 21))}]
        analyzer.using_simulated_data = False
        analyzer.statistics = {'last_numbers': set()}
        strategy = {'pool_max_last_numbers': 3, 'final_selection_mode': 'concentrated'}
        with patch.object(analyzer, 'build_pool_by_strategy', return_value={
            'candidates': [(n, float(81-n)) for n in range(1, 81)],
        }):
            groups = live_groups(analyzer, strategy)
        for play, size in [('select_6', 6), ('fu_shi_7', 7)]:
            self.assertEqual([len(g) for g in groups[play]], [size] * 3)
            self.assertEqual(len(set(sum(groups[play], []))), size * 3)
        self.assertTrue(set(groups['select_6'][1]) <= set(groups['fu_shi_7'][1]))

    def test_exclusion_mode_override_is_used(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = []
        analyzer.statistics = {'last_numbers': set()}
        strategy = {'final_selection_mode': 'concentrated', 'exclusion_selection_mode': 'low_repeat'}
        with patch('src.kl8.strategies.resolve_play_strategy', return_value=strategy), \
             patch.object(analyzer, 'build_pool_by_strategy', return_value={
                 'candidates': [(n, float(81-n)) for n in range(1, 81)],
             }):
            result, _ = analyzer._calculate_select_recalculation('select_6', [1, 2, 3, 4, 5, 6])
        self.assertEqual(result['quality']['requested_selection_mode'], 'low_repeat')


if __name__ == '__main__':
    unittest.main()
