"""Actual KL8 activation decisions must use the evidence for the same trial/play."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.domain.numeric.repository import create_all
from src.domain.numeric.trial_store import TrialStore
from src.foundation.store import Database, make_engine
from src.kl8 import backtest as backtest_module, records, snapshots, trial_sync, validation
from src.kl8.analyzer import KL8Analyzer
from src.kl8.backtest import KL8RollingBacktest


def tournament(history_trials, *, play='select_6', raw_p=.01):
    backtest = KL8RollingBacktest(SimpleNamespace(history_data=[
        {'issue': str(2026800 - i)} for i in range(800)
    ]))
    metrics = {'lift': .1, 'mean_hits': 2.0, 'pool_mean_hits': 2.0,
               'pool_expected_random': 1.75, 'n_tests': 300,
               'probabilities': {'>=1': .9, '>=3': .3, '>=4': .15, '>=5': .08, '>=6': .005, '>=7': .001},
               'theoretical_probs': {'>=3': .2, '>=4': .1, '>=5': .05},
               'profit_roi': -.3, 'random_profit_roi': -.5}
    store = _memory_store(history_trials)
    trial_sync.set_store(store)
    try:
        with patch.object(snapshots, 'activate_verified_strategy', return_value=True) as activation, \
             patch.object(backtest, '_rolling_backtest_parametric', side_effect=lambda *a, **kw: {
                 key: {**metrics, 'n_tests': kw['end_idx'] - kw['start_idx']}
                 for key in {'select_6', 'fu_shi_7', play}}) as rolling, \
             patch.object(backtest, '_permutation_test', return_value={'p_value': raw_p}) as permutation:
            report = backtest.run_candidate_tournament_per_play_type(play, {
                'current': {'strategy_id': 'current', 'feature_weights': {'frequency': 1.0},
                            'model_weights': {'rank': 1.0}, 'window_size': 100},
            }, n_permutations=10)
    finally:
        trial_sync.reset_store()
    return report, store.load(), activation, rolling, permutation


def _memory_store(history_trials):
    db = Database(make_engine('sqlite+pysqlite:///:memory:'))
    create_all(db)
    store = TrialStore(db, game='kl8')
    store.append_many(deepcopy(history_trials))
    return store


def _round(trials, name):
    """按轮次取记录。

    不再按下标取：记录从库里查出来是按 tested_at 排的，而 holdout 预留那条
    的 tested_at 是 issues_sha256，排到哪个位置并不确定。
    """
    return [t for t in trials if (t.get('tournament_round') or '') == name]


def test_earlier_trials_change_the_real_gate_and_block_opening_the_final_slice():
    historical = [{'strategy_id': f'past-{i}', 'play_type': 'select_6',
                   'raw_p_value': .9, 'tournament_round': 'standalone_validation'}
                  for i in range(9)]
    report, trials, activation, rolling, _ = tournament(historical)
    assert report['all_failed'] is True
    activation.assert_not_called()
    assert all(call.kwargs['start_idx'] != 600 for call in rolling.call_args_list)
    assert report['val_results']['current']['raw_p_value'] == .01
    assert report['val_results']['current']['fdr_adjusted_p'] == pytest.approx(.1)
    assert _round(trials, 'per_play_validation')[-1]['fdr_adjusted_p'] == pytest.approx(.1)
    assert report['fdr_audit']['family_size'] == 10


def test_other_plays_do_not_contaminate_the_family_and_new_attempt_uses_its_own_q():
    report, trials, activation, _, permutation = tournament([
        {'strategy_id': 'current', 'play_type': 'select_6', 'raw_p_value': .9, 'evidence_schema': 2},
        *[{'strategy_id': 'irrelevant', 'play_type': 'select_5', 'raw_p_value': .9}
          for _ in range(20)],
    ])
    assert report['activated'] is True
    activation.assert_called_once()
    activation_report = activation.call_args.args[2]
    assert activation_report['adjusted_p'] == pytest.approx(.02)
    assert activation_report['fdr_audit']['family_size'] == 2
    assert _round(trials, 'per_play_validation')[-1]['fdr_adjusted_p'] == pytest.approx(.02)
    assert len(_round(trials, 'holdout_exposure')) == 1
    # 历史条的 fdr_adjusted_p 不再被整族回写（它是纯派生值），
    # 但它必须仍留在族里参与校正——上面的 family_size==2 已经证明了这点。
    assert _round(trials, '')[0]['raw_p_value'] == .9
    assert {call.kwargs['play_type'] for call in permutation.call_args_list} == {'select_6', 'fu_shi_7'}
    assert report['fdr_audit']['controls_repeated_final_test_access'] is True


@pytest.mark.parametrize('invalid_p', [None, float('nan'), float('inf'), -.1, 1.1, False])
def test_malformed_p_values_never_pass_and_historical_failures_remain_in_family(invalid_p):
    report, _, activation, _, _ = tournament([], raw_p=invalid_p)
    assert report['all_failed'] is True
    activation.assert_not_called()
    # 九条必须各自可区分：四元键是库里的主键，键相同的记录只会留下一条，
    # 族就从 10 缩成 2。旧的内存列表靠 trial_id 区分同秒重复，库不靠它。
    report, trials, activation, _, _ = tournament([
        {'strategy_id': f'past-{i}', 'play_type': 'select_6', 'raw_p_value': invalid_p}
        for i in range(9)
    ])
    assert report['all_failed'] is True
    assert _round(trials, 'per_play_validation')[-1]['fdr_adjusted_p'] == pytest.approx(.1)
    activation.assert_not_called()


def test_compound_permutation_measures_actual_linked_seven_number_pool():
    history = [{'issue': str(2026000 + i), 'numbers': list(range(1, 21))}
               for i in range(60)]
    backtest = KL8RollingBacktest(SimpleNamespace(history_data=history))

    def statistics(analyzer, window_size=None):
        analyzer.statistics = {'last_numbers': set(range(61, 81))}

    primary_helper = backtest_module._predict_select6_primary
    compound_helper = backtest_module._predict_fushi7_from_select6
    options = {'window_size': 50, 'pool_diversify': False,
               'final_selection_mode': 'concentrated'}
    with patch.object(backtest, '_rolling_backtest_parametric', return_value={'fu_shi_7': {}}), \
         patch.object(KL8Analyzer, 'update_statistics', statistics), \
         patch.object(KL8Analyzer, 'build_pool_by_strategy', return_value={
             'candidates': [(n, 81 - n) for n in range(1, 81)]}), \
         patch.object(KL8Analyzer, 'multi_model_voting', return_value={
             'candidates': [(n, 81 - n) for n in range(41, 81)]}) as ordinary_vote, \
         patch.object(backtest_module, '_predict_select6_primary', wraps=primary_helper) as primary, \
         patch.object(backtest_module, '_predict_fushi7_from_select6', wraps=compound_helper) as compound:
        report = backtest._permutation_test({'frequency': 1}, {'rank': 1}, 50, 60,
                                            play_type='fu_shi_7', n_permutations=3, **options)
        assert primary.call_count == compound.call_count == 10
        ordinary_vote.assert_not_called()
        assert report['play_type'] == 'fu_shi_7'
        assert report['pick_n'] == 7  # default pick_n=5 cannot silently change the tested pool
        assert report['real_mean_hits'] == 7
        assert report['real_lift'] == 3.0

        ordinary = backtest._permutation_test({'frequency': 1}, {'rank': 1}, 50, 60,
                                              pick_n=7, n_permutations=3, **options)
        assert ordinary['play_type'] == 'select_7'
        assert ordinary['real_mean_hits'] == 0


def test_both_validation_entrypoints_declare_the_compound_play():
    _, _, _, _, permutation = tournament([], play='fu_shi_7')
    assert {call.kwargs['play_type'] for call in permutation.call_args_list} == {'select_6', 'fu_shi_7'}
    with patch.object(validation, 'get_kl8_analyzer', return_value=SimpleNamespace(history_data=[{}] * 800)), \
         patch.object(KL8RollingBacktest, '_rolling_backtest_parametric', return_value={
             'fu_shi_7': {'pool_mean_hits': 2, 'pool_expected_random': 1.75}}), \
         patch.object(KL8RollingBacktest, '_permutation_test', return_value={
             'error': 'stop before persistence'}) as standalone_permutation, \
         patch.object(trial_sync, 'record_trial', side_effect=AssertionError('no storage')):
        report = validation.validate_and_activate_strategy(
            'fu_shi_7', {'frequency': 1}, {'rank': 1}, 100,
        )
    assert 'error' in report
    assert standalone_permutation.call_args.kwargs['play_type'] == 'fu_shi_7'


def test_unknown_permutation_play_is_rejected_before_any_prediction():
    backtest = KL8RollingBacktest(SimpleNamespace(history_data=[]))
    with patch.object(backtest, '_rolling_backtest_parametric') as rolling:
        report = backtest._permutation_test({}, {}, 0, 20, play_type='not_a_play')
    assert 'error' in report
    rolling.assert_not_called()
