"""Joint evidence for the six-number ticket and its linked seven-number pool.

This module never activates a strategy or writes predictions. Existing rolling
backtests already return both plays; the guard compares those same-period
results against the strategy actually used by the live predictor.
"""
from copy import deepcopy
import hashlib
import json
import math

from .config import BACKTEST_MIN_OOS_PERIODS, BACKTEST_FINAL_TEST_PERIODS, KL8_DEFAULT_HISTORY
from .stats import hypergeom_p_ge

MAIN_PLAYS = ('select_6', 'fu_shi_7')


def is_main_play(play_type):
    return play_type in MAIN_PLAYS


def play_family(play_type):
    return 'select_6' if is_main_play(play_type) else play_type


def unsupported_main_options(strategy):
    """Do not validate one ticket and then activate different live repeat rules."""
    unsupported = [key for key in ('final_max_last_numbers', 'final_min_last_numbers')
                   if key in strategy and (key == 'final_max_last_numbers'
                                           or strategy[key] not in (None, 0))]
    for key in ('feature_weights', 'model_weights'):
        weights = strategy.get(key)
        if not isinstance(weights, dict) or any(not _number(value, 0, float('inf')) for value in weights.values()):
            unsupported.append(key)
    if not any(key in unsupported for key in ('feature_weights', 'model_weights')):
        fw, mw = strategy['feature_weights'], strategy['model_weights']
        if not ((mw.get('rank', 0) > 0 and any(value > 0 for value in fw.values()))
                or mw.get('bayesian', 0) > 0 or mw.get('markov', 0) > 0):
            unsupported.append('no_usable_model')
    return unsupported


def strategy_backtest_options(strategy):
    defaults = {
        'window_size': KL8_DEFAULT_HISTORY, 'repeat_direction': 'neutral',
        'repeat_avoid_score': .10, 'repeat_non_avoid_score': .85,
        'repeat_follow_score': .90, 'repeat_non_follow_score': .50,
        'pool_diversify': True, 'pool_max_last_numbers': None,
        'frequency_mode': 'mean_reversion', 'final_selection_mode': 'balanced',
    }
    return {key: strategy.get(key, value) for key, value in defaults.items()}


def _number(value, lower, upper):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and lower <= value <= upper)


def compare_main_play_results(candidate, incumbent, *, minimum):
    """Fail closed when either play regresses or lacks comparable evidence."""
    result = {'passed': False, 'minimum': minimum, 'plays': {}, 'reasons': [],
              'candidate': {}, 'incumbent': {}}
    counts = []
    for play, pick_n in (('select_6', 6), ('fu_shi_7', 7)):
        thresholds = (1, *range(3, pick_n + 1))
        blocks = []
        for label, source in (('candidate', candidate), ('incumbent', incumbent)):
            block = source.get(play, {}) if isinstance(source, dict) else {}
            mean_key = 'mean_hits' if pick_n == 6 else 'pool_mean_hits'
            count = block.get('n_tests')
            probs = block.get('probabilities') or {}
            mean = block.get(mean_key)
            valid = (isinstance(count, int) and not isinstance(count, bool) and count >= minimum
                     and _number(mean, 0, pick_n) and isinstance(probs, dict)
                     and all(_number(probs.get(f'>={k}'), 0, 1) for k in thresholds))
            if valid:
                valid = all(probs[f'>={left}'] >= probs[f'>={right}']
                            for left, right in zip(thresholds, thresholds[1:]))
            if not valid:
                result['reasons'].append(f'{play}:{label}:missing_or_invalid_evidence')
                continue
            kept = {'n_tests': count, mean_key: mean,
                    'probabilities': {f'>={k}': probs[f'>={k}'] for k in thresholds}}
            result[label][play] = kept
            counts.append(count)
            blocks.append({'mean_hits': mean, 'zero_rate': 1 - probs['>=1'],
                           **{f'>={k}': probs[f'>={k}'] for k in range(3, pick_n + 1)}})
            if play == 'fu_shi_7':
                for tier in (3, 4, 5):
                    # A seven-number pool is 21 five-number tickets. High
                    # pool-hit tails affect how many of those tickets win.
                    combo_rate = 0.0
                    for hits in range(tier, 8):
                        mass = probs[f'>={hits}'] - (probs.get(f'>={hits + 1}', 0) if hits < 7 else 0)
                        winners = sum(math.comb(hits, k) * math.comb(7 - hits, 5 - k)
                                      for k in range(tier, 6) if k <= hits and 0 <= 5 - k <= 7 - hits)
                        combo_rate += mass * winners / 21
                    blocks[-1][f'combo>={tier}'] = combo_rate
        if len(blocks) != 2:
            continue
        proposed, current = blocks
        checks = {key: proposed[key] + 1e-12 >= current[key]
                  for key in proposed if key != 'zero_rate'}
        checks['zero_rate'] = proposed['zero_rate'] <= current['zero_rate'] + 1e-12
        # Each play must still clear its own random baseline, even when the
        # incumbent itself is an unvalidated reference strategy.
        random_checks = {'mean_hits': proposed['mean_hits'] > pick_n / 4}
        random_checks.update({f'>={k}': proposed[f'>={k}'] + 1e-12 >= hypergeom_p_ge(pick_n, k)
                              for k in (3, 4, 5)})
        passed = all(checks.values()) and all(random_checks.values())
        result['plays'][play] = {'passed': passed, 'not_worse': checks,
                                 'random_baseline': random_checks,
                                 'candidate': proposed, 'incumbent': current}
        if not passed:
            result['reasons'].append(f'{play}:regression_or_random_baseline_failed')
    if len(counts) != 4 or len(set(counts)) != 1:
        result['reasons'].append('unequal_or_missing_test_periods')
    result['passed'] = not result['reasons'] and len(result['plays']) == 2
    return result


def has_main_play_evidence(report):
    """Read-only check for new joint reports; old single-play flags do not qualify."""
    if not isinstance(report, dict):
        return False
    evidence = report.get('main_play_validation')
    if (not isinstance(evidence, dict) or evidence.get('schema') != 1
            or evidence.get('target') != 'select_6'
            or evidence.get('history_unchanged') is not True
            or not evidence.get('history_fingerprint')
            or any(not isinstance(evidence.get(key), str) or not evidence[key]
                   for key in ('candidate_fingerprint', 'incumbent_fingerprint'))):
        return False
    adjusted_p = evidence.get('adjusted_p')
    probabilities = evidence.get('permutation_p_values') or {}
    if (not _number(adjusted_p, 0, .05) or adjusted_p >= .05
            or not isinstance(probabilities, dict)
            or any(not _number(probabilities.get(play), 0, adjusted_p + 1e-12) for play in MAIN_PLAYS)
            or not isinstance(evidence.get('positive_sub_windows'), int)
            or evidence['positive_sub_windows'] < 3):
        return False
    for stage, minimum in (('validation', BACKTEST_MIN_OOS_PERIODS),
                           ('final_test', BACKTEST_FINAL_TEST_PERIODS)):
        stored = evidence.get(stage)
        if not isinstance(stored, dict):
            return False
        recalculated = compare_main_play_results(stored.get('candidate'), stored.get('incumbent'), minimum=minimum)
        if not recalculated['passed']:
            return False
    return True


def joint_candidate_rank(comparison):
    """Prefer improvements shared by both main plays; mean is only a tie-break."""
    plays = comparison.get('plays', {})
    if not comparison.get('passed') or any(play not in plays for play in MAIN_PLAYS):
        return (float('-inf'),) * 4
    gains = lambda keys: [plays[play]['candidate'][key] - plays[play]['incumbent'][key]
                          for play in MAIN_PLAYS for key in keys]
    high = gains(('>=4', '>=5'))
    return min(high), sum(high), min(gains(('>=3',))), sum(gains(('mean_hits',)))


class MainPlayGate:
    """Freeze the current strategy, and reuse its replay once per stage/range."""
    def __init__(self, backtest):
        from .strategies import resolve_play_strategy
        self.backtest = backtest
        self.incumbent = deepcopy(resolve_play_strategy('select_6'))
        self.cache = {}
        self.error = None
        self.history_fingerprint = self._history_fingerprint()
        if not self.incumbent:
            self.error = 'current_main_play_strategy_unavailable'
        elif unsupported_main_options(self.incumbent):
            self.error = 'current_main_play_configuration_not_supported_in_validation'
        elif self.history_fingerprint is None:
            self.error = 'invalid_main_play_history'

    def _history_fingerprint(self):
        try:
            rows = self.backtest.analyzer.history_data
            content = sorted((str(row.get('issue', '')), sorted(row.get('numbers', []))) for row in rows)
            return hashlib.sha256(json.dumps(content, separators=(',', ':')).encode()).hexdigest()
        except (AttributeError, TypeError, ValueError):
            return None

    def compare(self, strategy, result, bounds, *, minimum):
        if self.error or unsupported_main_options(strategy):
            return {'passed': False, 'reasons': [self.error or 'candidate_main_play_configuration_not_supported_in_validation']}
        if self._history_fingerprint() != self.history_fingerprint:
            return {'passed': False, 'reasons': ['history_changed_during_main_play_validation']}
        bounds = tuple(bounds)
        if bounds not in self.cache:
            self.cache[bounds] = self.backtest._rolling_backtest_parametric(
                self.incumbent['feature_weights'], self.incumbent['model_weights'],
                start_idx=bounds[0], end_idx=bounds[1], min_train=50,
                **strategy_backtest_options(self.incumbent),
            )
        if self._history_fingerprint() != self.history_fingerprint:
            return {'passed': False, 'reasons': ['history_changed_during_main_play_validation']}
        compared = compare_main_play_results(result, self.cache[bounds], minimum=minimum)
        # Missing predictions cannot be hidden by comparing a shorter subset.
        expected = bounds[1] - bounds[0]
        for source in ('candidate', 'incumbent'):
            if any(block['n_tests'] != expected for block in compared[source].values()):
                compared['passed'] = False
                compared['reasons'].append('incomplete_stage_coverage')
        compared['range'] = list(bounds)
        return compared

    def evidence(self, strategy, validation, final_test, *, adjusted_p, permutation_p_values, positive_sub_windows):
        from .records import _strategy_fingerprint
        return {'schema': 1, 'target': 'select_6', 'plays': list(MAIN_PLAYS),
                'history_fingerprint': self.history_fingerprint,
                'history_unchanged': self._history_fingerprint() == self.history_fingerprint,
                'candidate_fingerprint': _strategy_fingerprint(strategy),
                'incumbent_fingerprint': _strategy_fingerprint(self.incumbent),
                'adjusted_p': adjusted_p, 'permutation_p_values': permutation_p_values,
                'positive_sub_windows': positive_sub_windows,
                'validation': validation, 'final_test': final_test}
