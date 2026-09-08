import csv
import json
import math
from unittest.mock import patch

import pytest

from scripts.diagnose.replay_football_high_totals import (
    load_csv_rows, main, price_projection, replay, summarize,
)
from src.domain.sports.football.markets import fair_over_probability


def row(identity='one', goals=4, date='2025-08-01'):
    return {'match_id': identity, 'date': date, 'league': 'E0', 'season': '2526',
            'total_line': 2.5, 'actual_goals': goals,
            'prices': {'non_closing': [2.0, 2.0], 'closing': [1.4, 3.5]}}


def test_tail_probabilities_are_unconditional_and_match_independent_poisson_sum():
    mean = 3.6
    over = fair_over_probability(mean, 2.5)
    result = price_projection([1 / over, 1 / (1 - over)])
    assert result['implied_mean'] == pytest.approx(mean)
    for k in (4, 5, 6):
        expected = 1 - sum(math.exp(-mean) * mean ** i / math.factorial(i) for i in range(k))
        assert result['tail_probabilities'][str(k)] == pytest.approx(expected)


def test_proper_scores_penalize_false_high_totals_and_count_inclusive_threshold():
    projections = {name: {'tail_probabilities': {'4': p, '5': p, '6': p}}
                   for name, p in [('non_closing', .2), ('closing', .8)]}
    records = [{'date': '2025-08-01', 'actual_goals': goals, 'projections': projections}
               for goals in (2, 4)]
    result = summarize(records)
    four, five = result['tails']['4'], result['tails']['5']
    assert four['actual_events'] == 1
    assert four['actual_rate'] == .5
    assert four['closing']['brier'] == pytest.approx((.8**2 + .2**2) / 2)
    assert four['paired_closing_improvement']['brier'] == pytest.approx(0)
    assert five['actual_events'] == 0
    assert five['closing']['logloss'] == pytest.approx(-math.log(.2))
    assert five['paired_closing_improvement']['brier'] < 0


def test_result_mutation_and_input_order_cannot_change_any_price_projection():
    records = [row(), row('two', 0, '2025-08-02')]
    before = replay(records)
    records[0]['actual_goals'] = 9
    after = replay(list(reversed(records)))
    for first, second in zip(before['observations'], after['observations']):
        assert first['match_id'] == second['match_id']
        assert first['projections'] == second['projections']
    assert before['release_qualified'] is False
    assert before['policy']['parameters_tuned'] is False
    assert before['policy']['observed_total_line'] == 2.5


def test_high_environment_selects_by_price_only_and_keeps_same_pair_cohort():
    records = [row(goals=0), {**row('other', goals=8),
                            'prices': {'non_closing': [2, 2], 'closing': [2, 2]}}]
    report = replay(records)['all']
    high = report['high_environment_groups']['closing']
    assert high['n'] == 1
    assert high['coverage'] == .5
    assert high['tails']['4']['actual_rate'] == 0
    assert report['all']['tails']['4']['actual_rate'] == .5
    assert report['high_environment_groups']['non_closing']['n'] == 0


def test_invalid_duplicate_and_cross_line_inputs_are_not_evaluated_as_fixed_lines():
    with pytest.raises(ValueError, match='duplicate match_id'):
        replay([row(), row()])
    with pytest.raises(ValueError, match='fixed 2.5'):
        replay([{**row(), 'total_line': 3.0}])
    for price in (0, 1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='invalid odds'):
            price_projection([price, 2])
    assert summarize([])['tails']['4']['closing']['brier'] is None


def write_csv(tmp_path):
    good = {'Date': '01/08/2025', 'HomeTeam': 'Home', 'AwayTeam': 'Away',
            'FTHG': '3', 'FTAG': '2', 'Avg>2.5': '2', 'Avg<2.5': '2',
            'AvgC>2.5': '1.4', 'AvgC<2.5': '3.5', 'AHh': '-.5', 'AHCh': '-1.5'}
    with (tmp_path / 'E0_2526.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(good))
        writer.writeheader()
        writer.writerows([good, good, {**good, 'HomeTeam': 'NoClose', 'AvgC>2.5': ''},
                          {**good, 'HomeTeam': 'BadPrice', 'AvgC>2.5': 'nan'},
                          {**good, 'HomeTeam': 'BadResult', 'FTHG': '-1'}])


def test_csv_keeps_actual_contract_and_filters_to_complete_identical_pairs(tmp_path):
    write_csv(tmp_path)
    records, audit = load_csv_rows(tmp_path)
    assert len(records) == 1
    assert records[0]['total_line'] == 2.5
    assert records[0]['prices'] == {'non_closing': [2, 2], 'closing': [1.4, 3.5]}
    assert records[0]['actual_goals'] == 5
    assert audit['exclusions']['duplicate_match'] == 1
    assert audit['exclusions']['invalid_or_missing_match_data'] == 3
    assert len(audit['files'][0]['sha256']) == 64


def test_cli_writes_only_explicit_report_without_network_or_business_store(tmp_path):
    from src.common import kv_store

    write_csv(tmp_path)
    output = tmp_path / 'report.json'
    with patch('socket.socket.connect', side_effect=AssertionError('no network')), \
         patch.object(kv_store, 'load', side_effect=AssertionError('no store read')), \
         patch.object(kv_store, 'save', side_effect=AssertionError('no store write')):
        assert main(['--data-dir', str(tmp_path), '--output', str(output)]) == 0
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['input_audit']['valid_rows'] == 1
    assert report['by_season']['2526']['all']['n'] == 1
    assert report['by_league']['E0']['all']['n'] == 1
    assert report['by_season_and_league']['2526']['E0']['all']['n'] == 1
