"""Legacy chain audits must follow live primary constraints and reranking."""
from contextlib import redirect_stderr
from copy import deepcopy
import io
import unittest
from unittest.mock import patch

from scripts.backtest import backtest_kl8_select6_chain as chain
from src.kl8 import KL8Analyzer
from src.kl8 import backtest as live_backtest
from src.kl8 import strategies


class RankedAnalyzer(KL8Analyzer):
    """Use real pure recalculation with controlled ranks and no storage setup."""
    def __init__(self, last_numbers=()):
        self.history_data = []
        self.statistics = {'last_numbers': set(last_numbers)}
        self.pool_calls = []

    def build_pool_by_strategy(self, strategy, pool_size=80):
        self.pool_calls.append((strategy.get('window_size'), pool_size))
        order = range(80, 0, -1) if strategy.get('window_size') == 150 else range(1, 81)
        return {'candidates': [(number, float(80-rank))
                               for rank, number in enumerate(order)][:pool_size]}


class LiveChainParityTests(unittest.TestCase):
    BASE = {'strategy_id': 'controlled', 'window_size': 100,
            'pool_max_last_numbers': 5, 'final_selection_mode': 'concentrated'}

    def test_primary_passes_live_adaptive_cap_to_its_pool_builder(self):
        analyzer = RankedAnalyzer(last_numbers=range(1, 6))
        with patch.object(live_backtest, '_adaptive_repeat_cap', return_value=2), \
             patch.object(live_backtest, '_select_final_candidate_pool',
                          wraps=live_backtest._select_final_candidate_pool) as select:
            chain._one_chain(analyzer, self.BASE, rounds=1)
        self.assertEqual(select.call_args.kwargs['max_last_numbers'], 2)

    def test_explicit_final_cap_overrides_the_adaptive_primary_default(self):
        analyzer = RankedAnalyzer(last_numbers=range(1, 6))
        with patch.object(live_backtest, '_adaptive_repeat_cap', return_value=2), \
             patch.object(live_backtest, '_select_final_candidate_pool',
                          wraps=live_backtest._select_final_candidate_pool) as select:
            chain._one_chain(analyzer, dict(self.BASE, final_max_last_numbers=4), rounds=1)
        self.assertEqual(select.call_args.kwargs['max_last_numbers'], 4)

    def test_primary_minimum_repeats_can_use_numbers_outside_top_twenty(self):
        analyzer = RankedAnalyzer(last_numbers=(70, 71))
        groups = chain._one_chain(analyzer, dict(self.BASE, final_min_last_numbers=2), rounds=1)
        self.assertEqual(groups, [[1, 2, 3, 4, 70, 71]])

    def test_first_round_override_reranks_once_and_later_round_keeps_all_exclusions(self):
        strategy = dict(self.BASE, first_exclusion_strategy={
            'strategy_id': 'first-only', 'window_size': 150,
        })
        original = deepcopy(strategy)
        analyzer = RankedAnalyzer()
        with patch.object(strategies, 'resolve_play_strategy', side_effect=AssertionError('global strategy read')):
            groups = chain._one_chain(analyzer, strategy, rounds=3)
        self.assertEqual(groups, [list(range(1, 7)), list(range(75, 81)), list(range(7, 13))])
        self.assertEqual(analyzer.pool_calls, [(100, 80), (150, 40), (100, 40)])
        self.assertEqual(strategy, original)

    def test_each_followup_calls_pure_production_calculation_with_cumulative_exclusion(self):
        analyzer = RankedAnalyzer()
        with patch.object(analyzer, '_calculate_select_recalculation',
                          wraps=analyzer._calculate_select_recalculation) as calculate, \
             patch.object(analyzer, '_save_exclude_recalculation',
                          side_effect=AssertionError('audit must not save records')):
            groups = chain._one_chain(analyzer, self.BASE, rounds=4)
        self.assertEqual(calculate.call_count, 3)
        for round_number, call in enumerate(calculate.call_args_list, 1):
            self.assertEqual(call.args, ('select_6', sorted({n for group in groups[:round_number] for n in group})))
            self.assertIs(call.kwargs['strategy'], self.BASE)

    def test_number_space_limits_the_chain_to_thirteen_full_disjoint_tickets(self):
        groups = chain._one_chain(RankedAnalyzer(), self.BASE, rounds=99)
        self.assertEqual(len(groups), 13)
        self.assertTrue(all(len(group) == 6 for group in groups))
        numbers = [number for group in groups for number in group]
        self.assertEqual(len(set(numbers)), 78)
        self.assertTrue(all(1 <= number <= 80 for number in numbers))

    def test_broken_duplicate_output_is_an_audit_error_instead_of_silent_bad_metrics(self):
        analyzer = RankedAnalyzer()
        with patch.object(analyzer, '_calculate_select_recalculation', return_value=(
                {'numbers': [1, 7, 8, 9, 10, 11]}, [])):
            with self.assertRaisesRegex(ValueError, 'previously unused'):
                chain._one_chain(analyzer, self.BASE, rounds=2)

    def test_cli_rejects_impossible_round_count_before_reading_history(self):
        with patch('sys.argv', ['backtest_kl8_select6_chain.py', '--rounds', '14',
                                '--history', 'must-not-open-this-file.json']), \
             redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as raised:
                chain.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('rounds must be 1..13', errors.getvalue())


if __name__ == '__main__':
    unittest.main()
