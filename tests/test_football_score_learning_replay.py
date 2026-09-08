import csv
from copy import deepcopy
from unittest.mock import patch

import pytest

from scripts.diagnose.replay_football_score_learning import load_csv_rows, replay
from src.common import kv_store
from src.football.market_db import MarketScoreDB


def matrix(_row):
    assert 'actual_score' not in _row
    return {(0, 0): .20, (0, 1): .10, (1, 0): .45, (1, 1): .25}


def row(identity, date, score='1-0'):
    return {'match_id': identity, 'date': date, 'league': 'E0', 'season': '2526',
            'asian': .5, 'total_line': 2.5, 'features': {}, 'actual_score': score}


def unchanged(base, prior, **kwargs):
    return dict(base), {'applied': False}


def test_replay_predicts_the_entire_day_before_ingesting_any_result():
    seen = []

    def inspect_prior(base, prior, **kwargs):
        seen.append((prior['sample_count'], dict(prior['probabilities'])))
        assert kwargs == {'max_weight': .15, 'quality_factor': 1.0}
        return dict(base), {'applied': False}

    records = [row('b', '2025-07-02', '0-1'), row('a', '2025-07-02'), row('c', '2025-07-03')]
    with patch.object(kv_store, 'load', side_effect=AssertionError('must not read DB')), \
         patch.object(kv_store, 'save', side_effect=AssertionError('must not write DB')), \
         patch.object(MarketScoreDB, 'save', side_effect=AssertionError('must not save DB')):
        report = replay(records, matrix_builder=matrix, prior_transform=inspect_prior,
                        confidence_intervals=False)
    assert seen[:2] == [(0, {}), (0, {})]
    assert seen[2] == (2, {'1-0': .5, '0-1': .5})
    assert [item['history_n'] for item in report['observations']] == [0, 0, 2]
    assert [item['history_last_date'] for item in report['observations']] == [None, None, '2025-07-02']
    assert report['second_season']['n'] == 3


def test_future_results_and_input_order_cannot_change_earlier_forecasts():
    records = [row('a', '2025-06-30'), row('b', '2025-07-01'), row('c', '2025-07-02')]
    first = replay(records, matrix_builder=matrix, prior_transform=unchanged, confidence_intervals=False)
    changed = deepcopy(records)
    changed[-1]['actual_score'] = '0-0'
    later = replay(list(reversed(changed)), matrix_builder=matrix, prior_transform=unchanged,
                   confidence_intervals=False)
    assert first['observations'][:2] == later['observations'][:2]
    assert first['warmup_n'] == 1
    assert first['second_season']['n'] == 2
    assert first['release_qualified'] is False
    assert first['policy']['parameters_tuned'] is False


def test_metrics_use_full_support_and_penalize_outside_support_results():
    report = replay([row('a', '2025-07-01', '8-0')], matrix_builder=matrix,
                    prior_transform=unchanged, confidence_intervals=False)
    metrics = report['second_season']['baseline']
    assert metrics['n'] == 1
    assert metrics['rank_n'] == 0
    assert metrics['mean_actual_rank'] is None
    assert metrics['logloss'] > 30
    assert metrics['brier'] == pytest.approx(1 + .2**2 + .1**2 + .45**2 + .25**2)
    assert all(metrics[f'top{k}_accuracy'] == 0 for k in (1, 3, 5, 10))
    assert report['second_season']['paired']['logloss_improvement']['estimate'] == 0


def test_replay_rejects_duplicate_ids_and_partial_score_distributions():
    with pytest.raises(ValueError, match='duplicate match_id'):
        replay([row('a', '2025-07-01'), row('a', '2025-07-02')],
               matrix_builder=matrix, prior_transform=unchanged)
    with pytest.raises(ValueError, match='incomplete score matrix'):
        replay([row('a', '2025-07-01')], matrix_builder=lambda _: {(1, 0): .7},
               prior_transform=unchanged)


def test_replay_runs_the_production_prior_helper_after_history_threshold():
    records = [row(f'warmup-{index:02}', '2025-06-30', ('0-0', '0-1', '1-1')[index % 3])
               for index in range(30)]
    records.append(row('test', '2025-07-01'))
    report = replay(records, matrix_builder=matrix, confidence_intervals=False)
    summary = report['second_season']
    assert summary['history_prior_applied_n'] == 1
    assert summary['learned']['n'] == 1
    assert summary['baseline']['logloss'] != summary['learned']['logloss']
    assert report['observations'][-1]['history_n'] == 30
    assert all(not item['prior_applied'] for item in report['observations'][:-1])


def test_observed_tail_expands_both_declared_supports_without_changing_baseline_mass():
    records = [row(f'warmup-{index:02}', '2025-06-30', ('0-0', '1-1', '8-0')[index % 3])
               for index in range(30)]
    records.append(row('test', '2025-07-01', '8-0'))
    report = replay(records, matrix_builder=matrix, confidence_intervals=False)
    scores = report['observations'][-1]['_score_evaluations']
    assert scores['baseline']['score_distribution_valid'] is True
    assert scores['learned']['score_distribution_valid'] is True
    assert scores['baseline']['actual_score_prob'] == 0
    assert scores['learned']['actual_score_prob'] == pytest.approx(.15 * .1 / 3)
    assert scores['baseline']['actual_score_rank'] is not None


def test_csv_uses_actual_2_5_line_and_nonclosing_odds_and_reports_exclusions(tmp_path):
    fields = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
              'AvgH', 'AvgD', 'AvgA', 'AHh', 'Avg>2.5', 'Avg<2.5', 'AvgCH']
    good = dict(zip(fields, ['01/08/2025', 'Home', 'Away', '2', '1',
                             '1.9', '3.2', '4.1', '-0.5', '1.6', '2.5', '99']))
    bad = {**good, 'HomeTeam': 'Bad', 'AHh': 'nan'}
    with (tmp_path / 'E0_2526.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([good, good, bad])
    rows, audit = load_csv_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]['asian'] == .5
    assert rows[0]['total_line'] == 2.5
    assert rows[0]['features']['AvgH'] == 1.9
    assert 'AvgCH' not in rows[0]['features']
    assert audit['exclusions']['duplicate_match'] == 1
    assert audit['exclusions']['invalid_or_missing_match_data'] == 1
    assert len(audit['files'][0]['sha256']) == 64
