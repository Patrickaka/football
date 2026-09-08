"""Observed-score arithmetic and lookup contracts, with all storage mocked."""

from copy import deepcopy
from unittest.mock import patch

import pytest

from src.football.market_db import MarketScoreDB, get_market_score_prob, parse_match_row


def make_db(probabilities=None, sample_counts=None, fresh_counts=None):
    with patch('src.football.market_db.kv_store.load', return_value={
        'probabilities': deepcopy(probabilities or {}),
        'sample_counts': deepcopy(sample_counts or {}),
        'fresh_counts': deepcopy(fresh_counts if fresh_counts is not None else {}),
    }):
        return MarketScoreDB()


def test_online_result_has_one_sample_weight_after_probability_normalization():
    db = make_db({'0.00_2.50': {'1-0': 0.6, '0-0': 0.4}}, {'0.00_2.50': 100})
    assert db.add_match_result(0, 2.5, '2-1') is True
    assert db.get_sample_count(0, 2.5) == 101
    assert db.get_prob(0, 2.5) == pytest.approx({'1-0': 60 / 101, '0-0': 40 / 101, '2-1': 1 / 101})
    assert db.add_match_result(0, 2.5, '1-0') is True
    assert db.get_prob(0, 2.5)['1-0'] == pytest.approx(61 / 102)


def test_batch_online_and_normalized_roundtrips_preserve_same_observations():
    scores = ['1-0'] * 61 + ['0-0'] * 29 + ['2-1'] * 10
    batch, online = make_db(), make_db()
    batch.add_records([{'asian': 0, 'ou': 2.5, 'score': score} for score in scores])
    batch._normalize_all()
    for score in scores:
        online.add_match_result(0, 2.5, score)
        online._normalize_all()
    assert batch.sample_counts == online.sample_counts == {'0.00_2.50': 100}
    assert online.get_prob(0, 2.5) == pytest.approx({'1-0': 0.61, '0-0': 0.29, '2-1': 0.10})
    assert batch.db == online.db
    with patch('src.football.market_db.kv_store.save') as save:
        online.save()
    saved = save.call_args.args[1]
    restored = make_db(saved['probabilities'], saved['sample_counts'])
    restored.add_match_result(0, 2.5, '2-1')
    assert restored.get_prob(0, 2.5)['2-1'] == pytest.approx(11 / 101)


@pytest.mark.parametrize('sample_counts', [{'0.00_2.50': 100}, {}])
def test_legacy_raw_counts_are_recovered_without_treating_them_as_probabilities(sample_counts):
    db = make_db({'0.00_2.50': {'1-0': 60, '0-0': 40}}, sample_counts)
    assert db.get_prob(0, 2.5) == {'1-0': 0.6, '0-0': 0.4}
    db.add_record(0, 2.5, '2-1')
    assert db.get_sample_count(0, 2.5) == 101
    assert db.get_prob(0, 2.5)['2-1'] == pytest.approx(1 / 101)


@pytest.mark.parametrize('bucket,sample_count', [
    ({'1-0': 0.5, '2-1': 0.5}, 101),  # old (p + 1) / 2 corruption
    ({'1-0': 0.6, '0-0': 0.4}, None),
    ({'1-0': 1.0}, None),  # one count or normalized certainty: ambiguous
    ({'1-0': 60, '0-0': 40}, 101),
    ({'1-0': 0.6, '0-0': 0.4}, 0),
    ({'1-0': 0.6, '0-0': 0.4}, True),
    ({'1-0': float('nan')}, 10),
    ({'1-0': float('inf')}, 10),
    ({'1-0': -1, '0-0': 2}, 10),
    ({'bad-score': 1}, 1),
])
def test_ambiguous_buckets_are_preserved_while_real_new_results_restart_independently(bucket, sample_count):
    db = make_db({'0.00_2.50': bucket}, {} if sample_count is None else {'0.00_2.50': sample_count})
    original = repr(db.db), repr(db.sample_counts)
    assert db.get_prob(0, 2.5) is None
    assert db.get_sample_count(0, 2.5) == 0
    assert db.get_prob_with_nearest(0, 2.5)['reason'] == 'invalid_bucket_counts'
    assert db.add_match_result(0, 2.5, '2-0') is True
    db._normalize_all()
    assert (repr(db.db), repr(db.sample_counts)) == original
    assert db.fresh_counts == {'0.00_2.50': {'2-0': 1}}
    result = db.get_prob_with_nearest(0, 2.5)
    assert result['probabilities'] == {'2-0': 1.0}
    assert result['sample_count'] == 1  # old sample_count contributes nothing
    assert result['count_source'] == 'post_repair_observations'
    assert result['legacy_bucket_excluded'] is True


@pytest.mark.parametrize('asian,ou,score', [
    (None, 2.5, '1-0'), (float('nan'), 2.5, '1-0'), (0, float('inf'), '1-0'),
    (0, None, '1-0'), (0, 2.5, '01-0'), (0, 2.5, '-1-0'), (0, 2.5, ''),
])
def test_invalid_new_observations_do_not_create_buckets(asian, ou, score):
    db = make_db()
    assert db.add_match_result(asian, ou, score) is False
    assert db.db == db.sample_counts == {}


def test_nearest_lookup_probability_and_count_always_describe_the_same_bucket():
    db = make_db({'0.25_2.50': {'1-0': 0.75, '0-0': 0.25}}, {'0.25_2.50': 120})
    result = db.get_prob_with_nearest(0, 2.75)
    assert result['matched_key'] == '0.25_2.50'
    assert result['distance'] == 0.5
    assert result['exact_match'] is False
    assert result['sample_count'] == 120
    assert result['probabilities'] == {'1-0': 0.75, '0-0': 0.25}
    assert db.get_sample_count(0, 2.75) == 0  # exact-count API keeps its meaning
    assert db.get_top_scores(0, 2.75, 1) == [('1-0', 0.75)]
    with patch('src.football.market_db.MarketScoreDB', return_value=db):
        public = get_market_score_prob(0, 2.75)
    assert public['sample_count'] == 120
    assert public['matched_key'] == result['matched_key']
    assert public['probabilities'] == result['probabilities']


def test_bad_exact_bucket_is_excluded_before_selecting_a_valid_neighbour():
    db = make_db({'0.00_2.50': {'1-0': 0.5, '0-0': 0.5},
                  '0.25_2.50': {'1-0': 0.6, '0-0': 0.4}},
                 {'0.00_2.50': 101, '0.25_2.50': 50})
    result = db.get_prob_with_nearest(0, 2.5)
    assert result['matched_key'] == '0.25_2.50'
    assert result['excluded_exact_bucket'] == '0.00_2.50'
    assert result['sample_count'] == 50


def test_merging_two_normalized_databases_keeps_both_sample_counts():
    first = make_db({'0.00_2.50': {'1-0': 0.6, '0-0': 0.4}}, {'0.00_2.50': 100})
    second = make_db({'0.00_2.50': {'1-0': 0.2, '2-1': 0.8}}, {'0.00_2.50': 50})
    first.merge_with(second)
    assert first.get_sample_count(0, 2.5) == 150
    assert first.get_prob(0, 2.5) == pytest.approx({'1-0': 70 / 150, '0-0': 40 / 150, '2-1': 40 / 150})


def test_football_data_build_buckets_by_the_observed_2_5_contract():
    common = {'Date': '01/01/2025', 'HomeTeam': 'Home', 'AwayTeam': 'Away',
              'FTHG': '2', 'FTAG': '1', 'AHh': '0'}
    rows = [{**common, 'Avg>2.5': '1.20', 'Avg<2.5': '5.00'},
            {**common, 'FTHG': '0', 'FTAG': '0', 'Avg>2.5': '5.00', 'Avg<2.5': '1.20'},
            {**common}]  # unavailable O/U quotes must not fabricate a market
    assert parse_match_row(rows[0])['total_line'] == 2.5
    assert parse_match_row(rows[2])['total_line'] is None
    db = make_db()
    with patch('src.common.match_store.iter_csv_rows', return_value=iter(rows)):
        db.build_from_matches()
    assert set(db.db) == {'0.00_2.50'}
    assert db.get_sample_count(0, 2.5) == 2
    assert db.get_prob(0, 2.5) == {'2-1': 0.5, '0-0': 0.5}


def test_restarted_sample_survives_reload_and_reaches_the_real_30_sample_gate():
    legacy = {'0.00_2.50': {'1-0': 0.5, '0-0': 0.5}}
    legacy_n = {'0.00_2.50': 101}
    db = make_db(legacy, legacy_n)
    for score in ['2-1'] * 14 + ['1-0'] * 10 + ['0-0'] * 5:
        assert db.add_match_result(0, 2.5, score) is True
    result = db.get_prob_with_nearest(0, 2.5)
    assert result['sample_count'] == 29
    assert result['probabilities']['2-1'] == pytest.approx(14 / 29)
    assert db.db == legacy and db.sample_counts == legacy_n
    with patch('src.football.market_db.kv_store.save') as save:
        db.save()
    saved = save.call_args.args[1]
    assert saved['fresh_counts'] == {'0.00_2.50': {'2-1': 14, '1-0': 10, '0-0': 5}}
    reloaded = make_db(saved['probabilities'], saved['sample_counts'], saved['fresh_counts'])
    assert reloaded.add_match_result(0, 2.5, '2-1') is True
    with patch('src.football.market_db.MarketScoreDB', return_value=reloaded):
        result = get_market_score_prob(0, 2.5)
    assert result['sample_count'] == 30
    assert result['probabilities'] == pytest.approx({'2-1': 0.5, '1-0': 1 / 3, '0-0': 1 / 6})
    assert result['legacy_bucket_excluded'] is True
    assert result['count_source'] == 'post_repair_observations'
    assert reloaded.db == legacy and reloaded.sample_counts == legacy_n


@pytest.mark.parametrize('fresh', [
    {'0.00_2.50': {'1-0': 0.5}}, {'0.00_2.50': {'1-0': 1.0}},
    {'0.00_2.50': {'1-0': True}}, {'0.00_2.50': {'1-0': -1}},
    {'0.00_2.50': {'1-0': 0}}, {'0.00_2.50': {'1-0': float('nan')}},
    {'0.00_2.50': {'bad-score': 1}}, {'0.00_2.50': {}},
    {'0.00_2.50': []}, [],
])
def test_invalid_fresh_count_format_is_preserved_and_cannot_overwrite_or_reuse_legacy(fresh):
    db = make_db({'0.00_2.50': {'1-0': 0.6, '0-0': 0.4}}, {'0.00_2.50': 100}, fresh)
    original = repr(db.db), repr(db.sample_counts), repr(db.fresh_counts)
    assert db.add_match_result(0, 2.5, '2-1') is False
    assert db.get_prob(0, 2.5) is None
    assert db.get_sample_count(0, 2.5) == 0
    db._normalize_all()
    with patch('src.football.market_db.kv_store.save') as save:
        db.save()
    assert repr(save.call_args.args[1]['fresh_counts']) == original[2]
    assert (repr(db.db), repr(db.sample_counts), repr(db.fresh_counts)) == original


def test_fresh_only_bucket_is_found_by_nearest_lookup_and_merged_without_losing_prior_fresh_counts():
    db = make_db(fresh_counts={'0.25_2.50': {'1-0': 20, '0-0': 10}})
    neighbour = db.get_prob_with_nearest(0, 2.5)
    assert neighbour['sample_count'] == 30
    assert neighbour['matched_key'] == '0.25_2.50'
    assert neighbour['count_source'] == 'post_repair_observations'
    other = make_db({'0.25_2.50': {'1-0': 1.0}}, {'0.25_2.50': 10})
    db.merge_with(other)
    assert db.fresh_counts == {'0.25_2.50': {'1-0': 30, '0-0': 10}}
    assert db.db == db.sample_counts == {}
    db.clear()
    assert db.fresh_counts == {}


def test_count_includes_verified_and_fresh_observations_but_excludes_damaged_legacy_counts():
    db = make_db(
        {'0.00_2.50': {'1-0': 0.6, '0-0': 0.4},
         '0.25_2.50': {'1-0': 0.5, '0-0': 0.5},
         '0.50_2.50': {'1-0': 0.5, '0-0': 0.5}},
        {'0.00_2.50': 100, '0.25_2.50': 101, '0.50_2.50': 103},
        {'0.50_2.50': {'2-1': 20, '1-0': 10}, '0.75_2.50': {'0-0': 5}},
    )
    assert db.count() == 135  # 100 valid + 30 restarted + 5 fresh-only
    assert db.add_match_result(0.25, 2.5, '2-1') is True
    assert db.count() == 136
    assert db.sample_counts['0.25_2.50'] == 101
    assert db.sample_counts['0.50_2.50'] == 103
