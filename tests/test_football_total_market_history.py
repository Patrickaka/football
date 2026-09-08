"""Local price history must be attributable and strictly available before prediction."""
from copy import deepcopy

import pytest

from src.domain.sports.football.markets import analyze_total, implied_total_goals
from src.football.total_market_history import restore_total_market_history


AS_OF = '2026-09-08T18:00:00+08:00'
MATCH = {'id': 'sporttery-100', 'hkjc_id': 'HK100', 'time': '2026-09-08T20:00:00+08:00'}


def total_quote(line=3.5, over=1.8, under=2.0):
    quote = {'line': line, 'over_odds': over, 'under_odds': under}
    return {
        **analyze_total({'open': quote, 'close': quote, 'odds_format': 'decimal'}),
        'source': 'hkjc', 'source_matched': True, 'source_event_id': 'HK100',
        'updated_at': '2026-09-08T17:30:00+08:00', 'history_available': False,
    }


def snapshot(line=2.5, captured='2026-09-08T12:00:00+08:00', **quote_args):
    total = total_quote(line, **quote_args)
    total['updated_at'] = '2026-09-08T10:00:00+08:00'
    return {'captured_at': captured, 'is_prematch': True, 'odds': {'total': total}}


def record(*rows):
    return {'match_id': MATCH['id'], 'market_timeline': list(rows)}


def restore(total=None, history=None, match=None, as_of=AS_OF):
    return restore_total_market_history(
        total if total is not None else total_quote(),
        history if history is not None else record(snapshot()),
        match=match if match is not None else MATCH, as_of=as_of,
    )


def test_restore_earliest_real_quote_without_changing_current_quote_or_inputs():
    current = total_quote()
    history = record(snapshot(3.0, '2026-09-08T14:00:00+08:00'), snapshot(2.5),
                     snapshot(2.75, '2026-09-08T05:00:00Z'))
    originals = deepcopy((current, history))
    output, trace = restore(current, history)
    assert output['open_line'] == 2.5
    assert output['line_change'] == 1
    assert output['open_implied_total'] == pytest.approx(implied_total_goals(2.5, 2 / 3.8))
    assert output['implied_change'] == pytest.approx(current['implied_total'] - output['open_implied_total'])
    assert output['trend_direction'] == 'up'
    assert output['prob_change'] == pytest.approx({'over': 0, 'under': 0})
    for key in ('close_line', 'close_prob', 'close_water', 'implied_total', 'updated_at', 'source', 'source_event_id'):
        assert output[key] == current[key]
    assert output['history_available'] is True
    assert output['history_source'] == trace['history_source'] == 'local_observations'
    assert output['history_open_captured_at'] == '2026-09-08T12:00:00+08:00'
    assert output['history_open_updated_at'] == '2026-09-08T10:00:00+08:00'
    assert '本地首次观测' in output['line_trend']
    assert trace['restored'] is True
    assert (current, history) == originals
    output['open_prob']['over'] = 0
    output['close_prob']['over'] = 0
    assert (current, history) == originals


def test_same_line_water_movement_changes_implied_total_without_faking_line_change():
    output, _ = restore(total_quote(3.0, 1.65, 2.3), record(snapshot(3.0, over=2.1, under=1.75)))
    assert output['line_change'] == 0
    assert output['implied_change'] > 0.3
    assert output['prob_change']['over'] > 0.1


def test_downward_total_movement_and_changed_water_do_not_become_big_score_signal():
    output, _ = restore(total_quote(2.5, 2.1, 1.75), record(snapshot(3.5, over=1.65, under=2.3)))
    assert output['line_change'] == -1
    assert output['implied_change'] < -1
    assert output['trend_direction'] == 'down'


def test_existing_supplier_history_is_never_replaced_even_by_an_earlier_local_quote():
    current = total_quote()
    current.update(history_available=True, history_source='provider', open_line=2.75)
    output, trace = restore(current)
    assert output == current and output is not current
    assert trace['restored'] is False and trace['reason'] == 'history_already_available'


@pytest.mark.parametrize('mutation', [
    lambda row: row.update(is_prematch=False),
    lambda row: row.update(is_prematch=1),
    lambda row: row.pop('is_prematch'),
    lambda row: row.update(captured_at='2026-09-08T12:00:00'),
    lambda row: row.update(captured_at='not a date'),
    lambda row: row.update(captured_at=AS_OF),
    lambda row: row.update(captured_at='2026-09-08T20:01:00+08:00'),
    lambda row: row.update(match_id='another-match'),
    lambda row: row.update(source_event_id='HK-other'),
    lambda row: row['odds']['total'].update(source='model_proxy'),
    lambda row: row['odds']['total'].update(source_matched=False),
    lambda row: row['odds']['total'].update(source_matched=1),
    lambda row: row['odds']['total'].pop('source_event_id'),
    lambda row: row['odds']['total'].update(source_event_id='HK-other'),
    lambda row: row['odds']['total'].update(updated_at='2026-09-08T11:00:00'),
    lambda row: row['odds']['total'].update(updated_at='2026-09-08T12:01:00+08:00'),
    lambda row: row['odds']['total'].update(updated_at=AS_OF),
    lambda row: row['odds']['total'].update(updated_at='2026-09-08T20:00:00+08:00'),
    lambda row: row['odds']['total'].update(source_updated_at='2026-09-08T11:00:00'),
    lambda row: row.update(source_updated_at='2026-09-08T20:00:00+08:00'),
    lambda row: row['odds']['total'].update(close_line=float('nan')),
    lambda row: row['odds']['total'].update(close_line=True),
    lambda row: row['odds']['total'].update(close_line=-1),
    lambda row: row['odds']['total'].update(close_prob={'over': 0.7, 'under': 0.4}),
    lambda row: row['odds']['total'].update(close_prob={'over': 0.7, 'under': 0.3}),
    lambda row: row['odds']['total'].update(close_water={'over': 0.8, 'under': 2}),
    lambda row: row['odds']['total'].update(close_water={'over': float('inf'), 'under': 2}),
    lambda row: row['odds']['total'].update(odds_format='hong_kong'),
    lambda row: row['odds']['total'].pop('close_prob'),
    lambda row: row.update(odds=[]),
])
def test_reject_unattributed_invalid_or_leaking_historical_quotes(mutation):
    current, old = total_quote(), snapshot()
    mutation(old)
    output, trace = restore(current, record(old))
    assert output == current
    assert trace['restored'] is False
    assert trace['reason'] == 'no_eligible_local_observation'


@pytest.mark.parametrize('updates,match_updates,reason', [
    ({'source': 'model_proxy'}, {}, 'current_source_unverified'),
    ({'source_matched': False}, {}, 'current_source_unverified'),
    ({'source_event_id': 'different'}, {}, 'current_source_event_mismatch'),
    ({}, {'hkjc_id': None}, 'current_source_event_mismatch'),
    ({}, {'id': 'different'}, 'record_match_mismatch'),
    ({}, {'time': '2026-09-08T20:00:00'}, 'aware_prediction_and_kickoff_required'),
    ({'close_line': float('inf')}, {}, 'invalid_current_quote'),
    ({'history_available': None}, {}, 'history_availability_unknown'),
])
def test_current_context_must_be_verified(updates, match_updates, reason):
    current, match = total_quote(), dict(MATCH)
    current.update(updates)
    match.update(match_updates)
    output, trace = restore(current, match=match)
    assert output == current and trace['reason'] == reason


@pytest.mark.parametrize('as_of,reason', [
    ('2026-09-08T18:00:00', 'aware_prediction_and_kickoff_required'),
    ('2026-09-08T20:00:00+08:00', 'prediction_not_prematch'),
    ('2026-09-08T21:00:00+08:00', 'prediction_not_prematch'),
])
def test_prediction_time_must_be_aware_and_strictly_prematch(as_of, reason):
    current = total_quote()
    output, trace = restore(current, as_of=as_of)
    assert output == current and trace['reason'] == reason


def test_snapshot_without_source_update_keeps_absence_instead_of_inventing_timestamp():
    old = snapshot()
    old['odds']['total'].pop('updated_at')
    output, trace = restore(history=record(old))
    assert output['history_open_updated_at'] is None
    assert trace['open_updated_at'] is None and trace['restored'] is True


def test_source_update_equal_to_observation_is_allowed():
    old = snapshot()
    old['odds']['total']['updated_at'] = old['captured_at']
    output, trace = restore(history=record(old))
    assert trace['restored'] is True
    assert output['history_open_updated_at'] == old['captured_at']


def test_invalid_earliest_row_does_not_hide_a_later_valid_observation():
    invalid, valid = snapshot(2.0), snapshot(2.75, '2026-09-08T14:00:00+08:00')
    invalid['odds']['total']['source_event_id'] = 'other'
    output, trace = restore(history=record(invalid, valid))
    assert output['open_line'] == 2.75 and trace['restored'] is True


def test_more_unchanged_observations_do_not_change_audit_or_invent_source_updates():
    first, trace = restore(history=record(snapshot()))
    more, more_trace = restore(history=record(snapshot(), snapshot(captured='2026-09-08T15:00:00+08:00')))
    assert more == first
    assert more_trace == trace


def test_equal_earlier_price_is_true_observed_history_but_zero_movement():
    old = snapshot(3.5)
    old['match_id'], old['source_event_id'] = MATCH['id'], MATCH['hkjc_id']
    output, trace = restore(history=record(old))
    assert trace['restored'] is True
    assert output['line_change'] == output['implied_change'] == 0


def test_current_adjusted_prediction_target_is_not_overwritten_by_quote_history():
    current = total_quote()
    current['implied_total'] = 3.0
    output, _ = restore(current)
    assert output['implied_total'] == 3.0
    assert output['implied_change'] == pytest.approx(
        implied_total_goals(current['close_line'], current['close_prob']['over'])
        - output['open_implied_total'])
