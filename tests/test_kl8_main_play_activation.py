"""No single-play improvement may silently replace the two linked main plays."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.kl8 import config, records, snapshots, strategies, validation
from src.kl8.backtest import KL8RollingBacktest
from src.kl8.holdout import reserve_final_holdout
from src.kl8.main_play_validation import (
    MainPlayGate, compare_main_play_results, has_main_play_evidence,
)


def history(n=800):
    return [{'issue': str(2026001 + i)} for i in reversed(range(n))]


def metrics(n=300, *, improvement=0):
    return {play: {'n_tests': n, 'mean_hits': 2.1 + improvement,
                   'pool_mean_hits': 2.3 + improvement, 'pool_expected_random': 1.75,
                   'lift': .4, 'probabilities': {'>=1': .9 + improvement, '>=3': .35 + improvement,
                                                 '>=4': .15 + improvement, '>=5': .04 + improvement,
                                                 '>=6': .005 + improvement, '>=7': .001 + improvement},
                   'profit_roi': -.2, 'random_profit_roi': -.5}
            for play in ('select_6', 'fu_shi_7')}


def candidate():
    return {'strategy_id': 'candidate', 'feature_weights': {'frequency': 1.0},
            'model_weights': {'rank': 1.0}, 'window_size': 100}


def evidence_report(strategy=None):
    strategy = strategy or candidate()
    gate = MainPlayGate(SimpleNamespace(analyzer=SimpleNamespace(history_data=history())))
    val = compare_main_play_results(metrics(improvement=.01), metrics(), minimum=300)
    final = compare_main_play_results(metrics(200, improvement=.01), metrics(200), minimum=200)
    return {'main_play_validation': gate.evidence(
        strategy, val, final, adjusted_p=.02,
        permutation_p_values={'select_6': .01, 'fu_shi_7': .02}, positive_sub_windows=4)}


@pytest.mark.parametrize('play', ['select_6', 'fu_shi_7'])
@pytest.mark.parametrize('metric', ['mean', 'zero', '>=3', '>=4', '>=5', '>=6'])
def test_improvement_in_other_play_cannot_hide_regression(play, metric):
    proposed = metrics(improvement=.01)
    if metric == 'mean':
        proposed[play]['mean_hits' if play == 'select_6' else 'pool_mean_hits'] -= .02
    elif metric == '>=6':
        proposed[play]['probabilities']['>=6'] = .004
        proposed[play]['probabilities']['>=7'] = .001
    else:
        proposed[play]['probabilities']['>=1' if metric == 'zero' else metric] -= .02
    report = compare_main_play_results(proposed, metrics(), minimum=300)
    assert report['passed'] is False
    assert report['plays'][play]['passed'] is False


@pytest.mark.parametrize('defect', ['missing_play', 'missing_zero', 'few_periods', 'unequal_periods', 'nan', 'random'])
def test_missing_or_inadequate_joint_evidence_fails_closed(defect):
    proposed = metrics(improvement=.01)
    if defect == 'missing_play':
        del proposed['fu_shi_7']
    elif defect == 'missing_zero':
        del proposed['fu_shi_7']['probabilities']['>=1']
    elif defect == 'few_periods':
        proposed['select_6']['n_tests'] = 20
    elif defect == 'unequal_periods':
        proposed['fu_shi_7']['n_tests'] = 301
    elif defect == 'nan':
        proposed['select_6']['probabilities']['>=5'] = float('nan')
    else:
        proposed['fu_shi_7']['pool_mean_hits'] = 1.7
    assert not compare_main_play_results(proposed, metrics(), minimum=300)['passed']


def test_alias_cannot_open_the_same_final_period_again():
    trials = []
    result = reserve_final_holdout('fu_shi_7', candidate(), history(), (600, 800), trials,
                                  lambda: True, version='test')
    assert result['available']
    assert trials[-1]['play_type'] == 'select_6'
    repeated = reserve_final_holdout('select_6', candidate(), history(), (600, 800), trials,
                                    lambda: True, version='renamed')
    assert repeated['reason'] == 'insufficient_fresh_final_draws'


def test_legacy_compound_exposure_also_blocks_select6_final():
    trials = [{'play_type': 'fu_shi_7', 'tournament_round': 'holdout_exposure', 'last_issue': '2026800'}]
    assert not reserve_final_holdout('select_6', candidate(), history(), (600, 800), trials,
                                     lambda: True, version='new')['available']


def test_compound_activation_updates_the_live_shared_strategy_only():
    with patch.object(config, 'ACTIVE_STRATEGIES', {}), \
         patch.object(snapshots, '_persist_active_strategies') as persist, \
         patch.object(snapshots, 'clear_cache') as clear:
        report = evidence_report()
        assert has_main_play_evidence(report)
        assert snapshots.activate_verified_strategy('fu_shi_7', candidate(), report) is True
        assert set(config.ACTIVE_STRATEGIES) == {'select_6'}
        assert config.ACTIVE_STRATEGIES['select_6']['feature_weights'] == {'frequency': 1.0}
        assert strategies.resolve_play_strategy('fu_shi_7')['is_validated'] is True
        assert strategies.resolve_play_strategy('fu_shi_7')['strategy_id'] == config.ACTIVE_STRATEGIES['select_6']['strategy_id']
        persist.assert_called_once()
        clear.assert_called_once()


@pytest.mark.parametrize('defect', ['legacy', 'missing_companion', 'stale_baseline', 'changed_candidate', 'unsupported', 'weak_companion_p', 'forged_pass'])
def test_central_activation_rejects_incomplete_changed_or_unvalidated_evidence(defect):
    with patch.object(config, 'ACTIVE_STRATEGIES', {}), \
         patch.object(snapshots, '_persist_active_strategies') as persist, \
         patch.object(snapshots, 'clear_cache') as clear:
        proposed = candidate()
        report = evidence_report(proposed)
        if defect == 'legacy':
            report = {'all_conditions_passed': True, 'play_type': 'select_6'}
        elif defect == 'missing_companion':
            del report['main_play_validation']['final_test']['candidate']['fu_shi_7']
        elif defect == 'stale_baseline':
            config.ACTIVE_STRATEGIES['select_6'] = {**candidate(), 'status': 'validated'}
        elif defect == 'changed_candidate':
            proposed['frequency_mode'] = 'hot'
        elif defect == 'unsupported':
            proposed['final_min_last_numbers'] = 1
        elif defect == 'weak_companion_p':
            report['main_play_validation']['permutation_p_values']['fu_shi_7'] = .1
        else:
            report['main_play_validation']['final_test']['candidate']['select_6']['probabilities']['>=4'] = 0
        before = deepcopy(config.ACTIVE_STRATEGIES)
        assert snapshots.activate_verified_strategy('select_6', proposed, report) is False
        assert config.ACTIVE_STRATEGIES == before
        persist.assert_not_called()
        clear.assert_not_called()


@pytest.mark.parametrize('entrypoint', ['standalone', 'tournament'])
@pytest.mark.parametrize('defect', ['none', 'validation_regression', 'final_regression', 'companion_p', 'missing_companion'])
def test_both_entrypoints_require_both_main_plays(entrypoint, defect):
    trials = []
    seen = []
    def rolling(feature_weights, model_weights, **kwargs):
        proposed = feature_weights == {'frequency': 1.0}
        result = metrics(kwargs['end_idx'] - kwargs['start_idx'], improvement=.01 if proposed else 0)
        seen.append((proposed, kwargs['start_idx'], kwargs['end_idx']))
        if proposed and ((defect == 'validation_regression' and kwargs['start_idx'] == 300)
                         or (defect == 'final_regression' and kwargs['start_idx'] == 600)):
            result['fu_shi_7']['probabilities']['>=4'] = .14
        if proposed and defect == 'missing_companion':
            del result['fu_shi_7']
        return result
    def permutation(*args, **kwargs):
        return {'p_value': .2 if defect == 'companion_p' and kwargs['play_type'] == 'fu_shi_7' else .001}
    analyzer = SimpleNamespace(history_data=history())
    with patch.object(config, 'ACTIVE_STRATEGIES', {'select_6': {}, 'fu_shi_7': {}}), \
         patch.object(config, 'STRATEGY_TRIAL_RESULTS', trials), \
         patch.object(validation, 'get_kl8_analyzer', return_value=analyzer), \
         patch.object(records, '_persist_trial_results', return_value=True), \
         patch.object(KL8RollingBacktest, '_rolling_backtest_parametric', side_effect=rolling), \
         patch.object(KL8RollingBacktest, '_permutation_test', side_effect=permutation) as perm, \
         patch.object(snapshots, '_persist_active_strategies') as save_active, \
         patch.object(snapshots, 'clear_cache'):
        if entrypoint == 'standalone':
            report = validation.validate_and_activate_strategy('select_6', {'frequency': 1.0}, {'rank': 1.0},
                                                               100, auto_activate=True)
        else:
            report = KL8RollingBacktest(analyzer).run_candidate_tournament_per_play_type('fu_shi_7', {'candidate': candidate()})
        assert bool(report.get('activated')) is (defect == 'none')
        assert save_active.call_count == int(defect == 'none')
        if defect != 'missing_companion' or entrypoint == 'standalone':
            assert {call.kwargs['play_type'] for call in perm.call_args_list} == {'select_6', 'fu_shi_7'}
        if defect in ('validation_regression', 'companion_p', 'missing_companion'):
            assert not any(start == 600 for _, start, _ in seen)
        if defect == 'none':
            assert has_main_play_evidence(report)
            assert report['activation_target'] == 'select_6'


def test_auto_activate_false_keeps_live_state_even_when_pair_passes():
    with patch.object(config, 'ACTIVE_STRATEGIES', {'select_6': {}, 'fu_shi_7': {}}), \
         patch.object(config, 'STRATEGY_TRIAL_RESULTS', []), \
         patch.object(validation, 'get_kl8_analyzer', return_value=SimpleNamespace(history_data=history())), \
         patch.object(records, '_persist_trial_results', return_value=True), \
         patch.object(KL8RollingBacktest, '_rolling_backtest_parametric',
                      side_effect=lambda *args, **kw: metrics(kw['end_idx'] - kw['start_idx'])), \
         patch.object(KL8RollingBacktest, '_permutation_test', return_value={'p_value': .001}), \
         patch.object(snapshots, 'activate_verified_strategy') as activate:
        report = validation.validate_and_activate_strategy('select_6', {'frequency': 1}, {'rank': 1}, 100)
        assert report['all_conditions_passed']
        assert has_main_play_evidence(report)
        assert report['activated'] is False
        activate.assert_not_called()


def test_unsupported_current_rules_are_rejected_without_running_a_different_backtest():
    active = {**candidate(), 'status': 'validated', 'final_max_last_numbers': 2}
    with patch.object(config, 'ACTIVE_STRATEGIES', {'select_6': active}), \
         patch.object(KL8RollingBacktest, '_rolling_backtest_parametric') as rolling:
        report = KL8RollingBacktest(SimpleNamespace(history_data=history())).run_candidate_tournament_per_play_type(
            'select_6', {'candidate': candidate()})
        assert report['activated'] is False
        assert 'not_supported' in report['error']
        rolling.assert_not_called()


@pytest.mark.parametrize('cap', [None, 0, 2])
def test_explicit_final_cap_is_rejected_even_when_none(cap):
    proposed = {**candidate(), 'final_max_last_numbers': cap}
    with patch.object(config, 'ACTIVE_STRATEGIES', {}), \
         patch.object(KL8RollingBacktest, '_rolling_backtest_parametric') as rolling:
        report = KL8RollingBacktest(SimpleNamespace(history_data=history())).run_candidate_tournament_per_play_type(
            'select_6', {'candidate': proposed})
        assert report['all_failed']
        rolling.assert_not_called()


def test_high_pool_hits_and_compound_winning_tickets_cannot_be_sacrificed():
    def from_hits(six, seven):
        return {play: {'n_tests': 2, ('mean_hits' if play == 'select_6' else 'pool_mean_hits'): sum(hits) / 2,
                       'probabilities': {f'>={k}': sum(hit >= k for hit in hits) / 2 for k in range(1, pick + 1)}}
                for play, hits, pick in [('select_6', six, 6), ('fu_shi_7', seven, 7)]}
    report = compare_main_play_results(from_hits([5, 3], [5, 3]), from_hits([6, 0], [7, 0]), minimum=2)
    assert not report['passed']
    assert not report['plays']['select_6']['not_worse']['>=6']
    compound = report['plays']['fu_shi_7']
    assert not compound['not_worse']['>=7']
    assert not compound['not_worse']['combo>=5']
    assert compound['candidate']['combo>=5'] == pytest.approx(1 / 42)
    assert compound['incumbent']['combo>=5'] == .5


@pytest.mark.parametrize('when', ['before', 'during'])
def test_same_length_history_change_invalidates_comparison(when):
    analyzer = SimpleNamespace(history_data=history())
    backtest = KL8RollingBacktest(analyzer)
    gate = MainPlayGate(backtest)
    def rolling(*args, **kwargs):
        analyzer.history_data[0]['numbers'] = [1, 2, 3]
        return metrics()
    if when == 'before':
        analyzer.history_data[0]['issue'] = '2026900'
    with patch.object(backtest, '_rolling_backtest_parametric', side_effect=rolling):
        report = gate.compare(candidate(), metrics(), (300, 600), minimum=300)
    assert not report['passed']
    assert report['reasons'] == ['history_changed_during_main_play_validation']


@pytest.mark.parametrize('key', ['feature_weights', 'model_weights'])
@pytest.mark.parametrize('value', [None, [], 'missing'])
def test_incomplete_candidate_configuration_cannot_be_validated_or_activated(key, value):
    proposed = candidate()
    if value == 'missing':
        del proposed[key]
    else:
        proposed[key] = value
    with patch.object(config, 'ACTIVE_STRATEGIES', {}), \
         patch.object(KL8RollingBacktest, '_rolling_backtest_parametric') as rolling, \
         patch.object(snapshots, '_persist_active_strategies') as persist:
        report = KL8RollingBacktest(SimpleNamespace(history_data=history())).run_candidate_tournament_per_play_type(
            'select_6', {'candidate': proposed})
        assert report['all_failed']
        rolling.assert_not_called()
        assert snapshots.activate_verified_strategy('select_6', proposed, {}) is False
        persist.assert_not_called()
