"""Evaluate frozen prediction events at a fixed decision horizon; never fit a model."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import math
import random

from .validation import multiclass_metrics


def _timestamp(value):
    try:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return result.astimezone(timezone.utc) if result.tzinfo is not None else None
    except (TypeError, ValueError):
        return None


def _probabilities(value):
    if not isinstance(value, dict) or set(value) != set('HDA'):
        return None
    try:
        values = {k: float(value[k]) for k in 'HDA'}
    except (ValueError, TypeError):
        return None
    if any(not math.isfinite(v) or v < 0 or v > 1 for v in values.values()):
        return None
    total = sum(values.values())
    return {k: v/total for k, v in values.items()} if abs(total - 1) <= 1e-6 else None


def paired_date_comparison(rows, *, draws=2000, seed=20260908):
    """Paired bootstrap by kickoff day, preserving within-day dependence."""
    groups = defaultdict(list)
    for row in rows:
        p, q, actual = row['probabilities'], row['market_probabilities'], row['actual']
        pick = lambda v: max(('H', 'D', 'A'), key=v.get)
        groups[row['date']].append((
            float(pick(p) == actual) - float(pick(q) == actual),
            math.log(max(p[actual], 1e-15)) - math.log(max(q[actual], 1e-15)),
            sum((q[k] - (k == actual))**2 - (p[k] - (k == actual))**2 for k in 'HDA'),
        ))
    daily = [(len(items), [sum(item[i] for item in items) for i in range(3)])
             for _, items in sorted(groups.items())]
    n = len(rows)
    result = {'n': n, 'independent_days': len(daily), 'method': 'paired_kickoff_day_bootstrap',
              'bootstrap_draws': draws, 'seed': seed}
    intervals = [[], [], []]
    rng = random.Random(seed)
    if daily:
        for _ in range(draws):
            sampled = [rng.choice(daily) for _ in daily]
            count = sum(day[0] for day in sampled)
            for i in range(3):
                intervals[i].append(sum(day[1][i] for day in sampled) / count)
    for i, key in enumerate(('accuracy_difference', 'logloss_improvement', 'brier_improvement')):
        values = sorted(intervals[i])
        result[key] = {
            'estimate': sum(day[1][i] for day in daily)/n if n else None,
            'ci95': [values[int(.025*(draws-1))], values[int(.975*(draws-1))]] if values else None,
        }
    return result


def build_release_report(payload, *, now=None):
    """A manifest plus immutable events is required; legacy aggregate exports cannot qualify.

    Each event contains explicit version/cutoff, model and market probabilities,
    prediction/kickoff/market/settlement timestamps and a frozen marker. Unknown
    provenance is excluded with a reason instead of inferred from a result label.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError('now must include a timezone')
    if payload.get('schema_version') != 'football-frozen-prediction-events-v1':
        raise ValueError('a football-frozen-prediction-events-v1 manifest is required')
    training = _timestamp(payload.get('training_cutoff_at'))
    version, logic = payload.get('model_version'), payload.get('prediction_logic_version')
    horizon = payload.get('prediction_horizon_seconds', 3600)
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 0:
        raise ValueError('prediction_horizon_seconds must be a nonnegative integer')
    samples = payload.get('samples') or []
    ids = Counter(str(row.get('match_id') or '') for row in samples)
    rejected = Counter()
    rows = []
    for row in samples:
        reason = None
        times = {key: _timestamp(row.get(key)) for key in
                 ('predicted_at', 'kickoff_at', 'market_captured_at', 'settled_at', 'training_cutoff_at')}
        p, q = _probabilities(row.get('probabilities')), _probabilities(row.get('market_probabilities'))
        if not row.get('match_id') or ids[str(row['match_id'])] > 1:
            reason = 'missing_or_duplicate_match_id'
        elif not version or not logic or row.get('model_version') != version or row.get('prediction_logic_version') != logic:
            reason = 'model_version_mismatch'
        elif row.get('frozen') is not True:
            reason = 'prediction_not_frozen'
        elif training is None or any(value is None for value in times.values()):
            reason = 'missing_timezone_aware_timestamp'
        elif times['training_cutoff_at'] != training or training >= times['predicted_at']:
            reason = 'training_cutoff_mismatch_or_leakage'
        elif not times['predicted_at'] <= times['kickoff_at'] < times['settled_at'] <= now:
            reason = 'invalid_prediction_or_settlement_time'
        elif not horizon <= (times['kickoff_at']-times['predicted_at']).total_seconds() <= horizon+900:
            reason = 'outside_fixed_decision_horizon'
        elif not 0 <= (times['predicted_at']-times['market_captured_at']).total_seconds() <= 300:
            reason = 'market_not_available_at_prediction_time'
        elif not p or not q or row.get('actual') not in ('H', 'D', 'A'):
            reason = 'invalid_probabilities_or_result'
        if reason:
            rejected[reason] += 1
            continue
        rows.append({**row, 'probabilities': p, 'market_probabilities': q,
                     'date': times['kickoff_at'].date().isoformat(), '_times': times})
    rows.sort(key=lambda row: (row['_times']['kickoff_at'], str(row['match_id'])))
    report = {
        'schema_version': 'football-model-acceptance-v1',
        'evaluation_scope': payload.get('evaluation_scope'),
        'model_version': version, 'prediction_logic_version': logic,
        'generated_at': now.isoformat(), 'training_cutoff_at': training.isoformat() if training else None,
        'evaluation_start_at': min(row['_times']['predicted_at'] for row in rows).isoformat() if rows else None,
        'evaluation_end_at': max(row['_times']['kickoff_at'] for row in rows).isoformat() if rows else None,
        'data_cutoff_at': max(row['_times']['settled_at'] for row in rows).isoformat() if rows else None,
        'prediction_horizon_seconds': horizon, 'out_of_sample_n': len(rows),
        'model_metrics': multiclass_metrics(rows),
        'market_baseline_metrics': multiclass_metrics(rows, 'market_probabilities'),
        'paired_comparison': paired_date_comparison(rows),
        'audit': {'immutable_prematch': bool(rows), 'same_snapshot_market': bool(rows),
                  'unique_matches': bool(rows) and not any(count > 1 for count in ids.values()),
                  'input_count': len(samples), 'accepted_count': len(rows),
                  'rejected_count': sum(rejected.values()), 'rejection_reasons': dict(rejected)},
    }
    return report
