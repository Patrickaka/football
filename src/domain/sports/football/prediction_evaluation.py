"""Read-only evaluation of explicitly frozen, verifiable prematch observations.

Legacy prediction documents remain useful for descriptive statistics, but are never
promoted into out-of-sample observations by this module.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import random
import re

from .release_evaluation import _probabilities as _release_probabilities, _timestamp, paired_date_comparison
from .validation import multiclass_metrics


EVENT_SCHEMA = 'football-prediction-event-v1'
SELECTION_POLICY = 'first_before_kickoff-v1'
VARIANTS = ('statistical', 'market_adjusted', 'agent_adjusted', 'market_only', 'production')
_SCORE = re.compile(r'^(0|[1-9]\d*)-(0|[1-9]\d*)$')


def _probabilities(value):
    if isinstance(value, dict) and any(isinstance(p, bool) for p in value.values()):
        return None
    return _release_probabilities(value)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def event_hash(event):
    return canonical_hash({key: value for key, value in event.items() if key not in ('hash', 'event_id')})


def score_matrix(value):
    """Accept a complete rectangular finite-support distribution, never a Top-K list."""
    if not isinstance(value, dict) or not value or len(value) > 10000:
        return None
    result, pairs = {}, set()
    for score, probability in value.items():
        match = _SCORE.fullmatch(str(score))
        if not match or isinstance(probability, bool):
            return None
        try:
            probability = float(probability)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            return None
        h, a = map(int, match.groups())
        if max(h, a) > 99:
            return None
        pairs.add((h, a))
        result[score] = probability
    max_h, max_a = max(h for h, _ in pairs), max(a for _, a in pairs)
    if len(pairs) != (max_h + 1) * (max_a + 1) or abs(sum(result.values()) - 1) > 1e-6:
        return None
    return result


def matrix_outcomes(matrix):
    result = dict.fromkeys('HDA', 0.0)
    for score, p in matrix.items():
        h, a = map(int, score.split('-'))
        result['H' if h > a else 'A' if a > h else 'D'] += p
    return result


def score_metrics(value, actual_score):
    """Rank/Top-K use available scores; proper scoring rules require full support.

    An omitted score in a partial list has unknown probability. A score outside a
    declared complete finite support has probability zero and is penalized as such.
    """
    empty = {'actual_score_rank': None, 'actual_score_prob': None,
             'score_brier': None, 'score_logloss': None, 'score_distribution_valid': False,
             **{f'hit_top{k}': None for k in (1, 3, 5, 10, 20, 30)}}
    if not _SCORE.fullmatch(str(actual_score)) or not isinstance(value, dict) or not value:
        return empty
    try:
        scores = {key: float(p) for key, p in value.items()
                  if _SCORE.fullmatch(str(key)) and not isinstance(p, bool)}
    except (TypeError, ValueError):
        return empty
    if len(scores) != len(value) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores.values()):
        return empty
    ordered = sorted(scores, key=lambda score: (-scores[score], tuple(map(int, score.split('-')))))
    rank = ordered.index(actual_score) + 1 if actual_score in ordered else None
    matrix = score_matrix(scores)
    probability = scores.get(actual_score, 0.0 if matrix is not None else None)
    return {**empty, 'actual_score_rank': rank, 'actual_score_prob': probability,
            'score_distribution_valid': matrix is not None,
            'score_brier': (sum(p*p for p in matrix.values()) + 1 - 2*probability) if matrix is not None else None,
            'score_logloss': -math.log(max(probability, 1e-15)) if matrix is not None else None,
            **{f'hit_top{k}': (actual_score in ordered[:k] if matrix is not None or len(ordered) >= k else None)
               for k in (1, 3, 5, 10, 20, 30)}}


def validate_event(event):
    """Return a rejection reason; timestamps must prove actual recording before kickoff."""
    if not isinstance(event, dict) or event.get('schema_version') != EVENT_SCHEMA:
        return 'missing_event_schema'
    if event.get('frozen') is not True or event.get('selection_policy') != SELECTION_POLICY:
        return 'not_registered_frozen_policy'
    try:
        expected = event_hash(event)
    except (TypeError, ValueError):
        return 'invalid_event_encoding'
    if event.get('hash') != expected or event.get('event_id') != expected:
        return 'event_hash_mismatch'
    as_of, recorded, kickoff = (_timestamp(event.get(key)) for key in ('as_of', 'recorded_at', 'kickoff_at'))
    if any(t is None for t in (as_of, recorded, kickoff)):
        return 'missing_timezone_aware_timestamp'
    if not as_of <= recorded < kickoff:
        return 'not_recorded_before_kickoff'
    if any(not isinstance(event.get(key), str) or not event[key].strip()
           for key in ('model_version', 'prediction_logic_version', 'match_id')):
        return 'missing_version_or_match_id'
    if event.get('execution_trace') is not None and not isinstance(event['execution_trace'], dict):
        return 'invalid_execution_trace'
    training = event.get('training_cutoff_at')
    if training is not None and (_timestamp(training) is None or _timestamp(training) >= as_of):
        return 'invalid_training_cutoff'
    variants = event.get('variants')
    if not isinstance(variants, dict) or not variants:
        return 'missing_variants'
    for name, variant in variants.items():
        if not isinstance(variant, dict) or not isinstance(variant.get('applied'), bool):
            return f'invalid_variant_execution:{name}'
        if variant.get('status') not in ('applied', 'fallback', 'unavailable'):
            return f'invalid_variant_status:{name}'
        if variant['applied'] != (variant['status'] == 'applied'):
            return f'inconsistent_variant_execution:{name}'
        p = variant.get('probabilities')
        if p is not None and _probabilities(p) is None:
            return f'invalid_variant_probabilities:{name}'
        if p is None and variant['status'] != 'unavailable':
            return f'missing_variant_probabilities:{name}'
        matrix = variant.get('score_probabilities')
        if p is not None and name != 'market_only' and matrix is None:
            return f'missing_complete_score_matrix:{name}'
        if matrix is not None:
            valid = score_matrix(matrix)
            if valid is None:
                return f'incomplete_score_matrix:{name}'
            probabilities = _probabilities(p)
            outcomes = matrix_outcomes(valid)
            if probabilities is None or any(abs(outcomes[key] - probabilities[key]) > 1e-6 for key in 'HDA'):
                return f'inconsistent_score_and_1x2:{name}'
        captured = variant.get('captured_at')
        if captured is not None and (_timestamp(captured) is None or _timestamp(captured) > as_of):
            return f'invalid_variant_capture_time:{name}'
    return None


def selected_event(record):
    """First stored event is the registered observation; never silently pick a later winner."""
    events = record.get('prediction_events') or []
    if not isinstance(events, list) or not events:
        return None, 'no_frozen_event'
    first = events[0]
    error = validate_event(first)
    if error:
        return None, error
    if record.get('selected_prediction_event_id') != first['event_id']:
        return None, 'selected_event_policy_mismatch'
    if str(first['match_id']) != str(record.get('match_id')):
        return None, 'event_match_id_mismatch'
    return first, None


def _settled_rows(records, as_of=None):
    cutoff = _timestamp(as_of) if as_of is not None else datetime.now(timezone.utc)
    if cutoff is None:
        raise ValueError('as_of must include a timezone')
    rows, excluded, seen = [], Counter(), set()
    for record in records:
        event, reason = selected_event(record)
        if reason:
            excluded[reason] += 1
            continue
        if not record.get('settled') or record.get('actual_result') not in ('H', 'D', 'A') or not _SCORE.fullmatch(str(record.get('actual_score'))):
            excluded['missing_settlement'] += 1
            continue
        settled = _timestamp(record.get('settled_at'))
        if settled is None or not _timestamp(event['kickoff_at']) < settled <= cutoff or _timestamp(event['recorded_at']) > cutoff:
            excluded['invalid_or_future_settlement_time'] += 1
            continue
        h, a = map(int, record['actual_score'].split('-'))
        if record['actual_result'] != ('H' if h > a else 'A' if h < a else 'D'):
            excluded['score_result_mismatch'] += 1
            continue
        quality = record.get('result_quality') or {}
        if record.get('exclude_from_stats') or record.get('skip_training_ingest') or (isinstance(quality, dict) and quality.get('grade') in ('reject', 'rejected', 'low')):
            excluded['excluded_result_quality'] += 1
            continue
        identity = str(event['match_id'])
        if identity in seen:
            excluded['duplicate_match_id'] += 1
            continue
        seen.add(identity)
        rows.append({'event': event, 'record': record, 'actual': record['actual_result'],
                     'date': _timestamp(event['kickoff_at']).date().isoformat()})
    return rows, dict(excluded)


def _metrics(rows, key='probabilities'):
    return multiclass_metrics(rows, key) if rows else {'n': 0, 'accuracy': None, 'logloss': None, 'brier': None}


def _score_summary(rows):
    metrics = [row['_score_evaluation'] for row in rows]
    valid = [m for m in metrics if m['score_distribution_valid']]
    result = {'n': len(valid), 'brier': None, 'logloss': None, 'mean_actual_rank': None,
              'rank_n': 0, **{f'top{k}_accuracy': None for k in (1, 3, 5, 10)}}
    if valid:
        result.update(brier=sum(m['score_brier'] for m in valid)/len(valid),
                      logloss=sum(m['score_logloss'] for m in valid)/len(valid))
        for k in (1, 3, 5, 10):
            result[f'top{k}_accuracy'] = sum(m[f'hit_top{k}'] for m in valid)/len(valid)
        ranks = [m['actual_score_rank'] for m in valid if m['actual_score_rank'] is not None]
        result.update(rank_n=len(ranks), mean_actual_rank=sum(ranks)/len(ranks) if ranks else None)
    return result


def _paired_scores(rows, candidate, baseline, draws=2000):
    """Score proper-score improvements, on the same events and resampled by day."""
    groups = defaultdict(list)
    for row in rows:
        p = row['_score_evaluations'][candidate]
        q = row['_score_evaluations'][baseline]
        if p['score_distribution_valid'] and q['score_distribution_valid']:
            groups[row['date']].append((q['score_logloss'] - p['score_logloss'], q['score_brier'] - p['score_brier']))
    daily = [(len(values), tuple(sum(v[i] for v in values) for i in range(2))) for _, values in sorted(groups.items())]
    n, rng = sum(day[0] for day in daily), random.Random(20260908)
    samples = [[], []]
    if daily:
        for _ in range(draws):
            picked = [rng.choice(daily) for _ in daily]
            count = sum(day[0] for day in picked)
            for i in range(2):
                samples[i].append(sum(day[1][i] for day in picked)/count)
    result = {'n': n, 'independent_days': len(daily), 'method': 'paired_kickoff_day_bootstrap',
              'bootstrap_draws': draws, 'seed': 20260908}
    for i, key in enumerate(('logloss_improvement', 'brier_improvement')):
        values = sorted(samples[i])
        result[key] = {'estimate': sum(day[1][i] for day in daily)/n if n else None,
                       'ci95': [values[int(.025*(draws-1))], values[int(.975*(draws-1))]] if values and len(daily) > 1 else None}
    return result


def _comparison(rows, candidate, baseline, *, include_confidence_intervals=True):
    paired = []
    for row in rows:
        variants = row['event']['variants']
        p = _probabilities((variants.get(candidate) or {}).get('probabilities'))
        q = _probabilities((variants.get(baseline) or {}).get('probabilities'))
        if p and q:
            paired.append({'probabilities': p, 'market_probabilities': q, 'actual': row['actual'], 'date': row['date']})
    draws = 2000 if include_confidence_intervals else 0
    report = paired_date_comparison(paired, draws=draws)
    if report['independent_days'] < 2:
        for key in ('accuracy_difference', 'logloss_improvement', 'brier_improvement'):
            report[key]['ci95'] = None
    report['score'] = _paired_scores(rows, candidate, baseline, draws=draws)
    if not include_confidence_intervals:
        report['method'] = report['score']['method'] = 'paired_kickoff_day_point_estimates'
    return report


def evaluate_frozen_events(records, *, model_version=None, as_of=None, include_confidence_intervals=True):
    rows, excluded = _settled_rows(records, as_of)
    if model_version is not None:
        excluded['model_version_mismatch'] = sum(row['event']['model_version'] != model_version for row in rows)
        rows = [row for row in rows if row['event']['model_version'] == model_version]
    versions = sorted({row['event']['model_version'] for row in rows})
    # Reporting separate versions is allowed; comparative means must never mix them.
    if model_version is None and len(versions) > 1:
        return {'schema_version': 'football-variant-evaluation-v1', 'selection_policy': SELECTION_POLICY,
                'status': 'version_filter_required', 'n': 0, 'model_versions': versions,
                'by_model_version': {v: evaluate_frozen_events(records, model_version=v, as_of=as_of,
                                                              include_confidence_intervals=include_confidence_intervals) for v in versions},
                'exclusions': excluded, 'release_qualified': False}
    report = {'schema_version': 'football-variant-evaluation-v1', 'selection_policy': SELECTION_POLICY,
              'status': 'evaluated' if rows else 'not_evaluable', 'n': len(rows),
              'model_version': model_version or (versions[0] if versions else None), 'exclusions': excluded,
              'trained_provenance_n': sum(_timestamp(row['event'].get('training_cutoff_at')) is not None for row in rows),
              'release_qualified': False, 'comparison_date_basis': 'kickoff_utc_date',
              'confidence_intervals_requested': include_confidence_intervals,
              'release_note': 'Descriptive first-before-kickoff cohort; a separately preregistered release acceptance gate is required.',
              'variants': {}, 'comparisons': {}}
    for row in rows:
        row['_score_evaluations'] = {
            name: score_metrics((row['event']['variants'].get(name) or {}).get('score_probabilities'), row['record']['actual_score'])
            for name in VARIANTS}
    for name in VARIANTS:
        available, executions = [], Counter()
        for row in rows:
            variant = row['event']['variants'].get(name) or {}
            executions[variant.get('status', 'unavailable')] += 1
            p = _probabilities(variant.get('probabilities'))
            if p:
                available.append({'probabilities': p, 'actual': row['actual'],
                                  '_score_evaluation': row['_score_evaluations'][name]})
        report['variants'][name] = {'one_x_two': _metrics(available), 'score': _score_summary(available),
                                     'execution_counts': dict(executions), 'missing_n': len(rows)-len(available)}
    for candidate, baseline in (('statistical', 'market_only'), ('market_adjusted', 'statistical'),
                                ('market_adjusted', 'market_only'), ('agent_adjusted', 'market_adjusted'),
                                ('production', 'market_only')):
        report['comparisons'][f'{candidate}_vs_{baseline}'] = _comparison(
            rows, candidate, baseline, include_confidence_intervals=include_confidence_intervals)
    agent_rows = [row for row in rows if (row['event']['variants'].get('agent_adjusted') or {}).get('applied') is True]
    report['agent_research_uplift_applied_only'] = _comparison(
        agent_rows, 'agent_adjusted', 'market_adjusted', include_confidence_intervals=include_confidence_intervals)
    report['agent_actual_applied_n'] = len(agent_rows)
    report['agent_fallback_n'] = sum((row['event']['variants'].get('agent_adjusted') or {}).get('status') == 'fallback' for row in rows)
    return report


def evaluate_frozen_ml(records, *, min_samples=45, model_version=None, as_of=None, include_confidence_intervals=True):
    rows, excluded = _settled_rows(records, as_of)
    candidates = []
    for row in rows:
        trace = (row['event'].get('execution_trace') or {}).get('ml_candidate') or {}
        if not isinstance(trace, dict):
            trace = {}
        p, q = _probabilities(trace.get('probabilities')), _probabilities(trace.get('base_probabilities'))
        training = _timestamp(trace.get('training_cutoff_at'))
        if not p or not q or not isinstance(trace.get('model_version'), str) or not trace['model_version'].strip():
            excluded['missing_valid_ml_pair'] = excluded.get('missing_valid_ml_pair', 0) + 1
            continue
        if training is None or training >= _timestamp(row['event']['as_of']):
            excluded['ml_training_cutoff_unknown_or_leaking'] = excluded.get('ml_training_cutoff_unknown_or_leaking', 0) + 1
            continue
        quality = row['record'].get('result_quality') or {}
        audit = trace.get('feature_audit') or {}
        if (row['record'].get('exclude_from_calibration')
                or not isinstance(quality, dict)
                or not (quality.get('grade') in ('high', 'medium') or quality.get('usable_for_calibration') is True)):
            excluded['ml_result_quality_not_verified'] = excluded.get('ml_result_quality_not_verified', 0) + 1
            continue
        if not isinstance(audit, dict) or audit.get('complete') is not True:
            excluded['ml_features_not_verified'] = excluded.get('ml_features_not_verified', 0) + 1
            continue
        if model_version is not None and trace['model_version'] != model_version:
            excluded['ml_model_version_mismatch'] = excluded.get('ml_model_version_mismatch', 0) + 1
            continue
        candidates.append({**row, 'trace': trace, 'probabilities': p, 'market_probabilities': q})
    versions = sorted({row['trace']['model_version'] for row in candidates})
    if model_version is None and len(versions) > 1:
        excluded['ml_version_filter_required'] = len(candidates)
        candidates = []

    def summarize(items):
        base, ml = _metrics(items, 'market_probabilities'), _metrics(items)
        result = {'sample_count': len(items), 'qualified': len(items) >= min_samples,
                  'base_1x2_logloss': base['logloss'], 'base_1x2_brier': base['brier'], 'base_1x2_hit_rate': base['accuracy'],
                  'ml_1x2_logloss': ml['logloss'], 'ml_1x2_brier': ml['brier'], 'ml_1x2_hit_rate': ml['accuracy']}
        for weight in (5, 10):
            blended = [{**row, 'probabilities': {k: (1-weight/100)*row['market_probabilities'][k] + weight/100*row['probabilities'][k] for k in 'HDA'}} for row in items]
            metrics = _metrics(blended)
            result[f'fused_{weight}pct_logloss'], result[f'fused_{weight}pct_brier'] = metrics['logloss'], metrics['brier']
        paired = paired_date_comparison(items, draws=2000 if include_confidence_intervals else 0)
        if not include_confidence_intervals:
            paired['method'] = 'paired_kickoff_day_point_estimates'
        if paired['independent_days'] < 2:
            for key in ('accuracy_difference', 'logloss_improvement', 'brier_improvement'):
                paired[key]['ci95'] = None
        result['paired_comparison'] = paired
        return result

    dimensions = {name: defaultdict(list) for name in ('by_league', 'by_handicap_type', 'by_total_line', 'by_result')}
    for row in candidates:
        record = row['record']
        asian = record.get('asian')
        handicap = 'unknown' if not isinstance(asian, (int, float)) else 'home_favored' if asian < 0 else 'away_favored' if asian > 0 else 'level'
        for dimension, key in (('by_league', record.get('league') or 'unknown'), ('by_handicap_type', handicap),
                               ('by_total_line', str(record.get('total_line')) if record.get('total_line') is not None else 'unknown'),
                               ('by_result', row['actual'])):
            dimensions[dimension][key].append(row)
    return {'overall': summarize(candidates), **{name: {key: summarize(values) for key, values in groups.items()} for name, groups in dimensions.items()},
            'min_samples_required': min_samples, 'model_version': model_version or (versions[0] if len(versions) == 1 else None),
            'available_model_versions': versions, 'selection_policy': SELECTION_POLICY, 'exclusions': excluded,
            'evaluation_source': 'immutable_prematch_ml_pairs', 'legacy_evaluations_included': False,
            'confidence_intervals_requested': include_confidence_intervals}
