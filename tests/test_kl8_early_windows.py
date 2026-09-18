import unittest
from copy import deepcopy
from scripts.diagnose.compare_kl8_early_windows import candidates
from src.kl8.strategies import resolve_exclusion_strategy
from src.kl8.records import _strategy_fingerprint


class EarlyWindowTests(unittest.TestCase):
    def test_scope_and_primary_are_preserved(self):
        slate = candidates()
        for name, strategy in slate.items():
            if name == 'current':
                continue
            primary = deepcopy(strategy)
            primary.pop('early_exclusion_strategy')
            self.assertEqual(primary, slate['current'])
            self.assertNotEqual(_strategy_fingerprint(primary), _strategy_fingerprint(strategy))
            before = deepcopy(strategy)
            for count in (6, 12):
                resolved = resolve_exclusion_strategy(strategy, 'select_6', range(count))
                self.assertEqual(resolved['strategy_id'], strategy['early_exclusion_strategy']['strategy_id'])
                self.assertFalse(resolved['is_validated'])
            for count in (0, 5, 7, 13, 18, 24):
                self.assertIs(resolve_exclusion_strategy(strategy, 'select_6', range(count)), strategy)
            self.assertEqual(before, strategy)

    def test_first_round_override_keeps_precedence_and_old_configs_work(self):
        strategy = candidates()['window_250']
        strategy['first_exclusion_strategy'] = {'strategy_id': 'legacy', 'window_size': 75}
        self.assertEqual(resolve_exclusion_strategy(strategy, 'select_6', range(6))['window_size'], 75)
        self.assertEqual(resolve_exclusion_strategy(strategy, 'select_6', range(12))['window_size'], 250)
        strategy.pop('early_exclusion_strategy')
        self.assertIs(resolve_exclusion_strategy(strategy, 'select_6', range(12)), strategy)
