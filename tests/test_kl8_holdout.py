"""The final period is consumed before inspection, including failed experiments."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.kl8.holdout import reserve_final_holdout
from src.kl8 import config, records, snapshots, validation
from src.kl8.backtest import KL8RollingBacktest


def history(n=800):
    return [{'issue': str(2026001 + i)} for i in reversed(range(n))]


def reserve(trials, n=800, *, persist=lambda: True, strategy='one', version='v1'):
    return reserve_final_holdout('select_6', {'strategy_id': strategy}, history(n),
                                 (600, n), trials, persist, version=version)


def test_final_period_cannot_be_reused_by_renaming_or_upgrading_after_restart():
    trials = []
    saved = []
    def persist():
        saved.extend(deepcopy(trials))
        return True
    first = reserve(trials, persist=persist)
    assert first['range'] == (600, 800)
    assert saved[0]['first_issue'] == '2026601'
    assert saved[0]['last_issue'] == '2026800'
    # A new process reloads the persisted record; outcome never releases it.
    for name, version in [('one', 'v1'), ('renamed', 'v2')]:
        assert reserve(deepcopy(saved), strategy=name, version=version)['available'] is False
    assert reserve(saved, 999)['fresh_draws'] == 199
    assert reserve(saved, 1000)['range'] == (800, 1000)


@pytest.mark.parametrize('saved', [False, None])
def test_unpersisted_reservation_cannot_open_final(saved):
    trials = []
    result = reserve(trials, persist=lambda: saved)
    assert result['reason'] == 'holdout_reservation_not_persisted'
    assert trials == []


def test_failed_storage_and_unknown_legacy_exposure_fail_closed():
    def fail():
        raise OSError('disk full')
    trials = []
    assert reserve(trials, persist=fail)['available'] is False
    assert trials == []
    trials = [{'play_type': 'select_6', 'raw_p_value': .01}]
    assert reserve(trials)['reason'] == 'legacy_final_exposure_unknown'
    assert trials[-1]['legacy_exposure_boundary'] is True
    assert reserve(trials)['available'] is False
    assert reserve(trials, 1000)['range'] == (800, 1000)


def test_invalid_history_or_exposure_never_opens_test():
    invalid = history()
    invalid[0] = invalid[1]
    assert not reserve_final_holdout('select_6', {}, invalid, (600, 800), [],
                                     lambda: True, version='v1')['available']
    invalid[0] = {}
    assert not reserve_final_holdout('select_6', {}, invalid, (600, 800), [],
                                     lambda: True, version='v1')['available']
    assert reserve([{'play_type': 'select_6', 'tournament_round': 'holdout_exposure'}])[
        'reason'] == 'invalid_prior_holdout_boundary'


def test_same_second_attempts_survive_persistence_and_restart(tmp_path, monkeypatch):
    path = tmp_path / 'trials.json'
    common = {'strategy_id': 'same', 'play_type': 'select_6', 'tested_at': '2026-09-09T10:00:00'}
    rows = [{**common, 'trial_id': 'first', 'raw_p_value': .01},
            {**common, 'trial_id': 'second', 'raw_p_value': .5}]
    monkeypatch.setattr(records, 'KL8_STRATEGY_TRIAL_FILE', path)
    monkeypatch.setattr(config, 'STRATEGY_TRIAL_RESULTS', rows + [rows[0]])
    assert records._persist_trial_results() is True
    assert json.loads(path.read_text(encoding='utf-8')) == rows
    assert records._load_trial_results() == rows


@pytest.mark.parametrize('final_lift,probabilities,expected', [
    (-.1, {'>=3': .3, '>=4': .1, '>=5': .05}, False),
    (.1, {'>=3': .3, '>=4': .1, '>=5': .05}, True),
    (.1, {'>=3': .3, '>=4': float('nan'), '>=5': .05}, False),
    (.1, {}, False),
])
def test_standalone_activation_requires_final_success_and_cannot_retry(final_lift, probabilities, expected):
    metrics = {'lift': .1, 'mean_hits': 2., 'n_tests': 300,
               'probabilities': {'>=3': .3, '>=4': .1, '>=5': .05},
               'profit_roi': -.3, 'random_profit_roi': -.5}
    final_calls = []
    def rolling(*args, **kwargs):
        result = deepcopy(metrics)
        if kwargs['start_idx'] == 600:
            final_calls.append(kwargs)
            result.update(lift=final_lift, probabilities=probabilities)
        return {'select_6': result}
    trials = []
    with patch.object(validation, 'get_kl8_analyzer', return_value=SimpleNamespace(history_data=history())), \
         patch.object(config, 'STRATEGY_TRIAL_RESULTS', trials), \
         patch.object(records, '_persist_trial_results', return_value=True), \
         patch.object(KL8RollingBacktest, '_rolling_backtest_parametric', side_effect=rolling), \
         patch.object(KL8RollingBacktest, '_permutation_test', return_value={'p_value': .001}), \
         patch.object(snapshots, 'activate_verified_strategy') as activate:
        report = validation.validate_and_activate_strategy('select_6', {'frequency': 1}, {'rank': 1},
                                                          100, auto_activate=True)
        assert report['all_conditions_passed'] is expected
        assert activate.call_count == int(expected)
        report2 = validation.validate_and_activate_strategy('select_6', {'frequency': .5}, {'rank': 1},
                                                           100, auto_activate=True)
        assert report2['all_conditions_passed'] is False
        assert len(final_calls) == 1
