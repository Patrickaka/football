"""Cross-market strength must respect Asian refunds and split stakes.

Reference prices below enumerate a score matrix and settle each explicit stake.
They do not call the production settlement/Poisson helpers to build expectations.
"""
import ast
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from src.domain.sports.football import markets


def reference_prices(home_mean, away_mean, stakes):
    probabilities = dict(home=0.0, draw=0.0, away=0.0)
    wins = losses = outright_wins = 0.0
    for home in range(45):
        for away in range(45):
            mass = (math.exp(-home_mean) * home_mean ** home / math.factorial(home)
                    * math.exp(-away_mean) * away_mean ** away / math.factorial(away))
            difference = home - away
            result = 'home' if difference > 0 else 'away' if difference < 0 else 'draw'
            probabilities[result] += mass
            for line, stake in stakes:
                if difference > line:
                    wins += mass * stake
                elif difference < line:
                    losses += mass * stake
            if all(difference > line for line, _ in stakes):
                outright_wins += mass
    return probabilities, wins / (wins + losses), outright_wins


class FairAsianPriceTests(unittest.TestCase):
    # Explicit stake splits, including averaged (non-quarter-grid) market lines.
    CASES = [
        (-2.5, [(-2.5, 1)]), (-2.0, [(-2.0, 1)]),
        (-0.75, [(-1.0, .5), (-0.5, .5)]),
        (-0.5, [(-0.5, 1)]), (-0.25, [(-0.5, .5), (0.0, .5)]),
        (-0.18, [(-0.5, .36), (0.0, .64)]), (0.0, [(0.0, 1)]),
        (0.18, [(0.0, .64), (0.5, .36)]),
        (0.25, [(0.0, .5), (0.5, .5)]), (0.5, [(0.5, 1)]),
        (0.75, [(0.5, .5), (1.0, .5)]), (1.0, [(1.0, 1)]),
        (2.0, [(2.0, 1)]), (2.25, [(2.0, .5), (2.5, .5)]),
        (2.5, [(2.5, 1)]),
    ]

    def test_prices_match_independent_score_settlement(self):
        for home, away in ((1.7, 1.2), (3.5, .5), (.5, 3.5)):
            for line, stakes in self.CASES:
                with self.subTest(home=home, away=away, line=line):
                    euro, asian, _ = reference_prices(home, away, stakes)
                    actual = markets._poisson_market_quotes(home + away, home - away, line)
                    for key, expected in euro.items():
                        self.assertAlmostEqual(actual[key], expected, places=12)
                    self.assertAlmostEqual(actual['asian_home'], asian, places=12)

    def test_push_and_split_stakes_are_not_unconditional_cover(self):
        for line, stakes in ((1.0, [(1.0, 1)]), (1.25, [(1.0, .5), (1.5, .5)])):
            with self.subTest(line=line):
                _, fair_price, outright_win = reference_prices(2.2, 1.1, stakes)
                self.assertGreater(abs(fair_price - outright_win), .03)
                actual = markets._poisson_market_quotes(3.3, 1.1, line)
                self.assertAlmostEqual(actual['asian_home'], fair_price, places=12)

    def test_same_market_recovers_strength_for_integer_half_and_quarter_lines(self):
        for line, stakes in self.CASES:
            with self.subTest(line=line):
                euro, asian, _ = reference_prices(1.7, 1.2, stakes)
                result = markets.compute_euro_asian_deviation(
                    euro, line, total_goals=2.9,
                    asian_home_probability=asian, markets_observed=True)
                self.assertEqual(result['method'], 'poisson_fair_price')
                self.assertFalse(result['fit_failed'])
                self.assertAlmostEqual(result['euro_supremacy'], .5, places=6)
                self.assertAlmostEqual(result['asian_supremacy'], .5, places=6)
                self.assertEqual(result['abs_deviation'], 0.0)

    def test_consistent_deep_favorite_is_not_an_automatic_conflict(self):
        euro, asian, _ = reference_prices(3.5, .5, [(2.5, 1)])
        legacy = markets.compute_euro_asian_deviation(euro, 2.5)
        result = markets.compute_euro_asian_deviation(
            euro, 2.5, total_goals=4.0,
            asian_home_probability=asian, markets_observed=True)
        self.assertGreater(legacy['abs_deviation'], .50)
        self.assertEqual(result['abs_deviation'], 0.0)
        self.assertEqual(result['euro_supremacy'], 3.0)
        self.assertEqual(result['asian_supremacy'], 3.0)
        self.assertEqual(result['legacy_deviation'], legacy['deviation'])
        self.assertEqual(result['input_basis']['total_goals'], 4.0)

    def test_actual_strength_conflict_and_home_away_mirror(self):
        euro, _, _ = reference_prices(3.0, .5, [(.5, .5), (1.0, .5)])
        _, asian, _ = reference_prices(2.0, 1.5, [(.5, .5), (1.0, .5)])
        direct = markets.compute_euro_asian_deviation(
            euro, .75, total_goals=3.5,
            asian_home_probability=asian, markets_observed=True)
        mirrored = markets.compute_euro_asian_deviation(
            dict(home=euro['away'], draw=euro['draw'], away=euro['home']),
            -.75, total_goals=3.5,
            asian_home_probability=1-asian, markets_observed=True)
        self.assertEqual(direct['abs_deviation'], 2.0)
        self.assertGreaterEqual(direct['abs_deviation'], .50)
        self.assertEqual(mirrored['abs_deviation'], direct['abs_deviation'])
        for key in ('euro_supremacy', 'asian_supremacy', 'deviation'):
            self.assertEqual(mirrored[key], -direct[key])


class DeviationCompatibilityAndFailureTests(unittest.TestCase):
    EURO = dict(home=.6, draw=.25, away=.15)

    def test_original_three_argument_contract_retains_exact_four_fields(self):
        self.assertEqual(markets.compute_euro_asian_deviation(self.EURO, .75), {
            'implied_handicap': .81, 'actual_handicap': .75,
            'deviation': .06, 'abs_deviation': .06,
        })
        self.assertEqual(markets.compute_euro_asian_deviation(self.EURO, .75, 2.0), {
            'implied_handicap': .9, 'actual_handicap': .75,
            'deviation': .15, 'abs_deviation': .15,
        })

    def test_missing_or_proxy_inputs_keep_conservative_original_difference(self):
        old = markets.compute_euro_asian_deviation(self.EURO, 2.0)
        for options in (
            dict(markets_observed=False, total_goals=3.0, asian_home_probability=.5),
            dict(markets_observed=True, total_goals=3.0),
            dict(markets_observed=True, asian_home_probability=.5),
            dict(total_goals=3.0, asian_home_probability=.5),
        ):
            with self.subTest(options=options):
                result = markets.compute_euro_asian_deviation(self.EURO, 2.0, **options)
                self.assertEqual(result['method'], 'linear_probability_proxy')
                self.assertFalse(result['fit_failed'])
                self.assertTrue(result['fallback_reason'])
                for key, value in old.items():
                    self.assertEqual(result[key], value)

    def test_invalid_observed_inputs_fail_closed_and_are_strict_json_serializable(self):
        valid = dict(euro_probs=self.EURO, asian_handicap=.75,
                     total_goals=3.0, asian_home_probability=.5, markets_observed=True)
        invalid = [
            dict(total_goals=value) for value in (0, -1, float('nan'), float('inf'), 'bad')
        ] + [dict(asian_home_probability=value) for value in (
            0, 1, -1, float('nan'), float('inf'), 'bad')]
        invalid += [dict(asian_handicap=float('nan')),
                    dict(asian_handicap=float('inf')),
                    dict(euro_probs={'home': .6, 'away': .4}),
                    dict(euro_probs={'home': .6, 'draw': .5, 'away': .4}),
                    dict(euro_probs={'home': .6, 'draw': float('nan'), 'away': .4}),
                    dict(euro_probs=None)]
        for changes in invalid:
            with self.subTest(changes=changes):
                result = markets.compute_euro_asian_deviation(**dict(valid, **changes))
                self.assertTrue(result['fit_failed'])
                self.assertEqual(result['method'], 'linear_probability_proxy')
                self.assertTrue(result['fallback_reason'])
                json.dumps(result, allow_nan=False)

    def test_unreachable_poisson_target_is_reported_as_fit_failure(self):
        result = markets.compute_euro_asian_deviation(
            dict(home=.8, draw=.1, away=.1), .75,
            total_goals=.2, asian_home_probability=.5, markets_observed=True)
        self.assertTrue(result['fit_failed'])
        self.assertEqual(result['fit_error'], 'price_outside_poisson_support')

    def test_unexpected_fit_failure_remains_explicit_even_with_zero_legacy_difference(self):
        with patch.object(markets, '_fit_poisson_market_supremacy', side_effect=RuntimeError('solver failed')):
            result = markets.compute_euro_asian_deviation(
                dict(home=.4, draw=.2, away=.4), 0.0,
                total_goals=3.0, asian_home_probability=.5, markets_observed=True)
        self.assertEqual(result['abs_deviation'], 0.0)
        self.assertTrue(result['fit_failed'])
        self.assertEqual(result['error_type'], 'RuntimeError')


class CliDeviationFormattingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load only the pure formatter; importing the CLI loads network/storage adapters.
        path = Path(__file__).resolve().parents[4] / 'src/football/cli.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        formatter = next(node for node in tree.body
                         if isinstance(node, ast.FunctionDef)
                         and node.name == '_format_euro_asian_deviation')
        namespace = {'math': math}
        exec(compile(ast.Module(body=[formatter], type_ignores=[]), str(path), 'exec'), namespace)
        cls.format_deviation = staticmethod(namespace[formatter.name])

    def test_fair_price_uses_the_two_strength_endpoints_not_legacy_lines(self):
        output = self.format_deviation(dict(
            method='poisson_fair_price', euro_supremacy=2.0, asian_supremacy=1.9,
            deviation=.1, implied_handicap=.8, actual_handicap=1.5))
        self.assertIn('欧赔净胜球均值+2.00', output)
        self.assertIn('亚盘净胜球均值+1.90', output)
        self.assertIn('差异+0.10球', output)
        self.assertNotIn('+0.80', output)

    def test_missing_or_failed_inputs_have_readable_output(self):
        self.assertIn('校验失败', self.format_deviation(dict(fit_failed=True, deviation=None)))
        for values in ({}, dict(deviation=None), dict(
                implied_handicap=float('nan'), actual_handicap=0.0, deviation=0.0)):
            self.assertIn('数据不足', self.format_deviation(values))

    def test_legacy_message_remains_compatible(self):
        output = self.format_deviation(dict(implied_handicap=.8, actual_handicap=.75, deviation=.05))
        self.assertEqual(output, '欧赔亚盘偏差: 欧赔隐含让球+0.80 vs 实际盘口+0.75，偏差+0.05')
