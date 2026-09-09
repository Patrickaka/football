import unittest
from unittest.mock import patch

from scripts.backtest.backtest_kl8_early_rounds import (
    live_groups, objective, summarize, run_slice, slate, first_round_slate,
    round_non_regression, promotion_checks, simplified_first_round_slate,
)
from src.kl8 import KL8Analyzer
from src.kl8.strategies import resolve_exclusion_strategy


class EarlyRoundAuditTests(unittest.TestCase):
    def test_joint_objective_cannot_promote_a_losing_play(self):
        baseline = [{'select_6': [1, 3, 3], 'fu_shi_7': [2, 4, 4]}] * 100
        candidate = [{'select_6': [1, 5, 5], 'fu_shi_7': [2, 3, 3]}] * 100
        self.assertGreater(objective(candidate[0]), objective(baseline[0]))
        checks = promotion_checks('candidate', candidate, baseline, {'ci_95': [0.4, 0.6]})
        self.assertTrue(checks['primary_mean_non_regression'])
        self.assertFalse(checks['promotion_supported'])

    def test_gain_in_first_round_cannot_pay_for_second_round_loss(self):
        baseline = [{'select_6': [1, 3, 4], 'fu_shi_7': [2, 3, 4]}] * 100
        candidate = [{'select_6': [1, 5, 3], 'fu_shi_7': [2, 5, 3]}] * 100
        checks = promotion_checks('candidate', candidate, baseline, {'ci_95': [0.4, 0.6]})
        self.assertTrue(checks['first_round_non_regression'])
        self.assertFalse(checks['second_round_non_regression'])
        self.assertFalse(checks['promotion_supported'])
        improving = [{'select_6': [1, 5, 4], 'fu_shi_7': [2, 5, 4]}] * 100
        self.assertTrue(promotion_checks('candidate', improving, baseline,
                                        {'ci_95': [0.4, 0.6]})['promotion_supported'])

    def test_simplified_candidates_are_opt_in_and_freeze_primary_configuration(self):
        candidates = simplified_first_round_slate()
        baseline = candidates['current']
        self.assertEqual(len(candidates), 3)
        for name, candidate in candidates.items():
            if name == 'current':
                continue
            self.assertEqual({k: v for k, v in candidate.items() if k != 'first_exclusion_strategy'},
                             baseline)
            self.assertFalse(candidate['is_validated'])
        self.assertNotIn('first_exclusion_strategy', slate()['current'])

    def test_promotion_guard_does_not_hide_compound_regression(self):
        baseline = [{'select_6': [1, 4, 4], 'fu_shi_7': [2, 4, 4]}]
        candidate = [{'select_6': [1, 5, 5], 'fu_shi_7': [2, 3, 3]}]
        self.assertFalse(round_non_regression(candidate, baseline, 1))
        self.assertFalse(round_non_regression(candidate, baseline, 2))
        self.assertTrue(round_non_regression(baseline, baseline, 1))

    def test_first_round_objective_does_not_reward_second_round(self):
        later_only = {'select_6': [6, 2, 5], 'fu_shi_7': [7, 3, 5]}
        early = {'select_6': [0, 4, 0], 'fu_shi_7': [0, 4, 0]}
        self.assertEqual(objective(later_only, first_round=True), 0)
        self.assertEqual(summarize([early], first_round=True)['objective'], 1)

    def test_first_round_override_leaves_primary_and_later_strategy_unchanged(self):
        baseline = first_round_slate()['first_window_75']
        baseline['is_validated'] = True
        result = resolve_exclusion_strategy(baseline, 'select_6', list(range(1, 7)))
        self.assertEqual(result['window_size'], 75)
        self.assertFalse(result['is_validated'])
        self.assertTrue(baseline['is_validated'])
        self.assertNotEqual(baseline['strategy_id'], result['strategy_id'])
        for play, excluded in [('select_6', []), ('select_6', list(range(1, 13))),
                               ('select_5', list(range(1, 7)))]:
            self.assertIs(resolve_exclusion_strategy(baseline, play, excluded), baseline)

    def test_first_round_reranking_keeps_six_seven_sizes_and_linked_first_round(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [{'issue': '2026001', 'numbers': list(range(1, 21))}]
        analyzer.using_simulated_data = False
        analyzer.statistics = {'last_numbers': set()}
        baseline = {'window_size': 100, 'pool_max_last_numbers': 3,
                    'final_selection_mode': 'concentrated'}
        candidate = {**baseline, 'first_exclusion_strategy': {
            'strategy_id': 'test_first', 'window_size': 75,
        }}

        def ranking(strategy, pool_size=80):
            order = range(80, 0, -1) if strategy['window_size'] == 75 else range(1, 81)
            return {'candidates': [(n, float(81-i)) for i, n in enumerate(order)]}

        with patch.object(analyzer, 'build_pool_by_strategy', side_effect=ranking):
            old_groups = live_groups(analyzer, baseline)
            new_groups = live_groups(analyzer, candidate)
        self.assertEqual(old_groups['select_6'][0], new_groups['select_6'][0])
        self.assertNotEqual(old_groups['select_6'][1], new_groups['select_6'][1])
        self.assertTrue(set(new_groups['select_6'][1]) <= set(new_groups['fu_shi_7'][1]))
        for play, size in [('select_6', 6), ('fu_shi_7', 7)]:
            self.assertEqual([len(g) for g in new_groups[play]], [size] * 3)
            self.assertEqual(len(set(sum(new_groups[play], []))), 3 * size)

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
        self.assertEqual(groups['fu_shi_7'][0], list(range(1, 8)))
        unused = set(groups['select_6'][1]) - set(groups['fu_shi_7'][0])
        self.assertTrue(unused <= set(groups['fu_shi_7'][1]))

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
