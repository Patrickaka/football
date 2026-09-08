"""Restore a total-goals market's first *observed* quote, never invent an opener.

HKJC supplies a current quote, so its initial open=close pair is not price
history. Only an earlier, attributable prematch snapshot can fill that gap.
This module is pure: callers own storage, clocks and match-time normalization.
"""
from copy import deepcopy
from datetime import datetime
import math

from ..domain.sports.football.markets import analyze_total, implied_total_goals


def _timestamp(value):
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
        return parsed if parsed.tzinfo is not None and parsed.utcoffset() is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _identity(value):
    if value is None or isinstance(value, bool):
        return None
    return str(value).strip() or None


def _number(value):
    if isinstance(value, bool):
        raise ValueError('boolean market value')
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('nonfinite market value')
    return number


def _quote(total):
    """Validate HKJC's decimal quote and its stored two-way fair probabilities."""
    if total.get('odds_format', 'decimal') != 'decimal':
        raise ValueError('unexpected HKJC odds format')
    line = _number(total['close_line'])
    probability = {side: _number(total['close_prob'][side]) for side in ('over', 'under')}
    water = {side: _number(total['close_water'][side]) for side in ('over', 'under')}
    if not 0 < line <= 20 or not all(0 < value < 1 for value in probability.values()):
        raise ValueError('invalid total market quote')
    if not math.isclose(sum(probability.values()), 1.0, abs_tol=1e-6):
        raise ValueError('total probabilities are not normalized')
    if not all(value > 1 for value in water.values()):
        raise ValueError('invalid decimal prices')
    expected_over = water['under'] / (water['over'] + water['under'])
    if not math.isclose(probability['over'], expected_over, abs_tol=1e-6):
        raise ValueError('stored probabilities and prices disagree')
    return line, probability, water


def restore_total_market_history(total, record, *, match, as_of):
    """Return a detached total market and an audit trace of history recovery.

    Match identity is bound by the containing record, and the HKJC event ID is
    additionally required on each historical total quote. All observation and
    supplied source-update times must be timezone aware and strictly precede
    both this prediction and kickoff. Legacy naive timestamps are not repaired.
    An existing provider history is left untouched.
    """
    restored = deepcopy(total)
    trace = {'restored': False, 'history_source': None, 'reason': None}

    def unchanged(reason):
        trace['reason'] = reason
        return restored, trace

    if not isinstance(total, dict) or not isinstance(match, dict):
        return unchanged('invalid_input')
    if total.get('history_available') is not False:
        return unchanged('history_already_available' if total.get('history_available') is True
                         else 'history_availability_unknown')
    if total.get('source') != 'hkjc' or total.get('source_matched') is not True:
        return unchanged('current_source_unverified')
    event_id = _identity(match.get('hkjc_id'))
    if not event_id or _identity(total.get('source_event_id')) != event_id:
        return unchanged('current_source_event_mismatch')
    match_id = _identity(match.get('match_id') or match.get('id'))
    if (not match_id or not isinstance(record, dict)
            or _identity(record.get('match_id')) != match_id):
        return unchanged('record_match_mismatch')
    cutoff = _timestamp(as_of)
    kickoff = _timestamp(match.get('kickoff') or match.get('match_time') or match.get('time'))
    if cutoff is None or kickoff is None:
        return unchanged('aware_prediction_and_kickoff_required')
    if cutoff >= kickoff:
        return unchanged('prediction_not_prematch')
    try:
        close_line, close_probability, close_water = _quote(total)
    except (KeyError, TypeError, ValueError, OverflowError):
        return unchanged('invalid_current_quote')

    candidates = []
    rows = record.get('market_timeline')
    if not isinstance(rows, (list, tuple)):
        return unchanged('no_eligible_local_observation')
    for row in rows:
        if not isinstance(row, dict) or row.get('is_prematch') is not True:
            continue
        if 'match_id' in row and _identity(row['match_id']) != match_id:
            continue
        captured = _timestamp(row.get('captured_at'))
        if captured is None or captured >= cutoff or captured >= kickoff:
            continue
        odds = row.get('odds')
        prior = odds.get('total') if isinstance(odds, dict) else None
        if (not isinstance(prior, dict) or prior.get('source') != 'hkjc'
                or prior.get('source_matched') is not True
                or _identity(prior.get('source_event_id')) != event_id):
            continue
        if 'source_event_id' in row and _identity(row['source_event_id']) != event_id:
            continue
        source_times = [value for value in (
            prior.get('updated_at'), prior.get('source_updated_at'), row.get('source_updated_at'),
        ) if value is not None]
        parsed_times = [_timestamp(value) for value in source_times]
        if any(value is None or value > captured or value >= cutoff or value >= kickoff
               for value in parsed_times):
            continue
        try:
            quote = _quote(prior)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        candidates.append((captured, row, prior, quote))
    if not candidates:
        return unchanged('no_eligible_local_observation')

    captured, row, prior, (open_line, open_probability, open_water) = min(
        candidates, key=lambda candidate: candidate[0])
    # Reuse existing movement calculations. Only open/history-derived fields
    # are copied back; current quote, closing target and source stay intact.
    movement = analyze_total({
        'odds_format': 'decimal',
        'open': {'line': open_line, 'over_odds': open_water['over'], 'under_odds': open_water['under']},
        'close': {'line': close_line, 'over_odds': close_water['over'], 'under_odds': close_water['under']},
    })
    for key in ('line_change', 'trend_direction', 'trend_strength', 'signal_strength', 'lambda_adjust'):
        restored[key] = movement[key]
    open_mean = implied_total_goals(open_line, open_probability['over'])
    restored.update({
        'open_line': open_line,
        'open_prob': deepcopy(prior['close_prob']),
        'open_water': deepcopy(prior['close_water']),
        'open_implied_total': open_mean,
        'implied_change': implied_total_goals(close_line, close_probability['over']) - open_mean,
        'prob_change': {side: close_probability[side] - open_probability[side] for side in ('over', 'under')},
        'line_trend': f"本地首次观测：{movement['line_trend']}",
        'history_available': True,
        'history_source': 'local_observations',
        'history_open_captured_at': row['captured_at'],
        'history_open_updated_at': prior.get('updated_at'),
    })
    trace.update({
        'restored': True,
        'reason': 'earliest_eligible_local_observation',
        'history_source': 'local_observations',
        'source_event_id': event_id,
        'open_captured_at': row['captured_at'],
        'open_updated_at': prior.get('updated_at'),
    })
    return restored, trace
