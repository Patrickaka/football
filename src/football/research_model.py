"""Trainable, local statistical residuals over audited intelligence facts.

No LLM-supplied probability, impact score or coefficient is accepted. A JSON
artifact only becomes applicable after chronological, paired holdout validation.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math

MODEL_SCHEMA = 'football-intelligence-residual-v1'
FEATURE_NAMES = tuple(f'{side}_{name}' for side in ('home', 'away')
                      for name in ('rest_days', 'rest_known', 'unavailable_players', 'injuries_known', 'lineup_confirmed')) + (
                          'temperature', 'rain', 'wind', 'weather_known')


def intelligence_features(intelligence, *, as_of):
    from .research import timestamp
    cutoff = timestamp(as_of)
    values = dict.fromkeys(FEATURE_NAMES, 0.0)
    empty = {'features': values, 'known_facts': 0, 'feature_version': MODEL_SCHEMA}
    if not isinstance(intelligence, dict):
        return empty
    captured = timestamp(intelligence.get('captured_at'))
    snapshot_cutoff = timestamp(intelligence.get('as_of'))
    kickoff = timestamp(intelligence.get('kickoff') or intelligence.get('kickoff_at'))
    if (not cutoff or not captured or not snapshot_cutoff or not kickoff
            or not snapshot_cutoff <= captured <= cutoff < kickoff):
        return {**empty, 'reason': 'invalid_intelligence_snapshot_time'}
    # A JSON flag is not evidence. Recheck provenance, freshness, data shape and
    # source conflicts identically for live inference and historical training.
    from ..domain.sports.football.match_context import build_match_context
    try:
        intelligence = build_match_context({**intelligence, 'kickoff': kickoff.isoformat()},
                                           intelligence.get('evidence') or [],
                                           as_of=cutoff, captured_at=cutoff)
    except (KeyError, TypeError, ValueError):
        return {**empty, 'reason': 'invalid_intelligence_snapshot'}
    seen = set()
    known = 0
    for fact in intelligence.get('evidence') or []:
        if fact.get('verified') is not True or fact.get('confirmation_status') != 'confirmed':
            continue
        published, collected = timestamp(fact.get('published_at')), timestamp(fact.get('collected_at'))
        if not cutoff or not published or not collected or max(published, collected) > cutoff:
            continue
        if not str(fact.get('source_url', '')).startswith(('https://', 'http://')):
            continue
        data, category, team = fact.get('data') or {}, fact.get('category'), fact.get('team')
        if not isinstance(data, dict):
            continue
        try:
            if category == 'schedule' and team in ('home', 'away'):
                previous = timestamp(data.get('previous_kickoff'))
                kickoff = timestamp(intelligence.get('kickoff') or intelligence.get('kickoff_at'))
                if not previous or not kickoff or previous >= cutoff or previous >= kickoff:
                    continue
                values[f'{team}_rest_days'] = min(30, (kickoff-previous).total_seconds()/86400)/7
                values[f'{team}_rest_known'] = 1.0
            elif category == 'injuries' and team in ('home', 'away'):
                status, player = data.get('status'), str(data.get('player') or '').strip().casefold()
                if status not in ('injured', 'suspended', 'unavailable', 'available', 'none_reported'):
                    continue
                key = (team, category, player)
                if key in seen or (status != 'none_reported' and not player):
                    continue
                seen.add(key)
                values[f'{team}_injuries_known'] = 1.0
                if status in ('injured', 'suspended', 'unavailable'):
                    values[f'{team}_unavailable_players'] += .1
            elif category == 'lineup' and team in ('home', 'away'):
                players = data.get('players')
                if not isinstance(players, list) or len(set(map(str, players))) != 11:
                    continue
                values[f'{team}_lineup_confirmed'] = 1.0
            elif category == 'weather':
                # Missing individual weather fields are not measurements of zero.
                keys = ('temperature_celsius', 'precipitation_mm', 'wind_kmh')
                numbers = [float(data[k]) for k in keys]
                forecast = timestamp(data.get('forecast_for'))
                kickoff = timestamp(intelligence.get('kickoff') or intelligence.get('kickoff_at'))
                if not forecast or not kickoff or abs((forecast-kickoff).total_seconds()) > 10800:
                    continue
                if not all(math.isfinite(x) for x in numbers):
                    continue
                values.update(temperature=max(-50, min(60, numbers[0]))/30,
                              rain=max(0, min(100, numbers[1]))/10,
                              wind=max(0, min(200, numbers[2]))/30, weather_known=1.0)
            else:
                continue
        except (ValueError, TypeError, KeyError):
            continue
        known += 1
    return {'features': values, 'known_facts': known, 'feature_version': MODEL_SCHEMA}


def _softmax(values):
    peak = max(values)
    terms = [math.exp(max(-700, v-peak)) for v in values]
    return [v/sum(terms) for v in terms]


def _residual_probabilities(base, features, coefficients):
    x = [features[name] for name in FEATURE_NAMES]
    logits = [math.log(max(base[k], 1e-15)) + sum(w*v for w, v in zip(coefficients[i], x))
              for i, k in enumerate('HDA')]
    return dict(zip('HDA', _softmax(logits)))


def _fingerprint(artifact):
    fields = {k: artifact.get(k) for k in ('schema_version', 'baseline_version', 'feature_names', 'coefficients',
                                          'weight', 'training_cutoff_at', 'validation')}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def artifact_eligibility(artifact, *, as_of):
    from .research import RESEARCH_VERSION, timestamp
    if not isinstance(artifact, dict) or artifact.get('schema_version') != MODEL_SCHEMA:
        return False, 'no_compatible_trained_intelligence_model'
    if artifact.get('baseline_version') != RESEARCH_VERSION:
        return False, 'research_baseline_version_mismatch'
    cutoff, now = timestamp(artifact.get('training_cutoff_at')), timestamp(as_of)
    if not cutoff or not now or cutoff >= now:
        return False, 'unknown_or_future_model_cutoff'
    if artifact.get('feature_names') != list(FEATURE_NAMES):
        return False, 'feature_contract_mismatch'
    try:
        coefs = artifact['coefficients']
        if len(coefs) != 3 or any(len(row) != len(FEATURE_NAMES) or
                                  not all(math.isfinite(v) for v in row) for row in coefs):
            return False, 'invalid_coefficients'
        if not 0 < artifact['weight'] <= .1 or artifact.get('artifact_id') != _fingerprint(artifact):
            return False, 'invalid_weight_or_artifact_hash'
    except (KeyError, TypeError, ValueError):
        return False, 'invalid_artifact'
    validation = artifact.get('validation') or {}
    if not isinstance(validation, dict):
        return False, 'invalid_validation_report'
    paired = validation.get('paired_comparison') or {}
    if not isinstance(paired, dict):
        return False, 'invalid_validation_report'
    def lower(key):
        metric = paired.get(key)
        interval = metric.get('ci95') if isinstance(metric, dict) else None
        if (not isinstance(interval, list) or len(interval) != 2 or
                any(not isinstance(v, (float, int)) or isinstance(v, bool) or not math.isfinite(v) for v in interval)):
            return float('-inf')
        return interval[0] if interval[0] <= interval[1] else float('-inf')
    for report, key in ((validation, 'train_n'), (validation, 'selection_n'), (validation, 'test_n'),
                        (paired, 'independent_days')):
        if not isinstance(report.get(key), int) or isinstance(report[key], bool) or report[key] < 0:
            return False, 'invalid_validation_sample_count'
    # Preserve the project's complete-model minimum: 1,000 independent held-out
    # matches and 30 kickoff days. A candidate can be trained before it qualifies.
    eligible = (validation.get('chronological') is True and validation.get('train_n', 0) >= 200
                and validation.get('selection_n', 0) >= 100 and validation.get('test_n', 0) >= 1000
                and paired.get('independent_days', 0) >= 30 and lower('logloss_improvement') > 0
                and lower('brier_improvement') >= 0 and lower('accuracy_difference') >= 0)
    return eligible, 'validated_intelligence_residual' if eligible else 'independent_holdout_validation_pending'


def apply_intelligence_residual(candidates, intelligence, artifact, *, as_of):
    from .research import outcome_probabilities, project_outcome_probabilities
    payload = intelligence_features(intelligence, as_of=as_of)
    trace = {'applied': False, 'weight': 0.0, **payload}
    eligible, reason = artifact_eligibility(artifact, as_of=as_of)
    trace.update(eligible=eligible, reason=reason)
    if not eligible or not payload['known_facts']:
        if eligible:
            trace['reason'] = 'no_verified_intelligence_features'
        return candidates, trace
    base = outcome_probabilities(candidates)
    target = _residual_probabilities(base, payload['features'], artifact['coefficients'])
    weight = artifact['weight']
    mixed = {k: (1-weight)*base[k] + weight*target[k] for k in 'HDA'}
    result = project_outcome_probabilities(candidates, mixed)
    trace.update(applied=True, weight=weight, model_version=artifact['artifact_id'],
                 training_cutoff_at=artifact['training_cutoff_at'])
    return result, trace


def train_residual(rows, *, iterations=300):
    """Fit from immutable event/settlement pairs; select weight on validation only.

    Input rows: match_id, kickoff_at, as_of, settled_at, frozen, intelligence,
    base_probabilities (B), actual. Caller obtains them from the event ledger.
    The test block is read only after all parameters and weights are frozen.
    """
    from .research import RESEARCH_VERSION, timestamp, valid_probabilities
    from ..domain.sports.football.release_evaluation import paired_date_comparison
    from ..domain.sports.football.validation import multiclass_metrics
    counts = Counter(str(row.get('match_id', '')) for row in rows)
    cohorts = {(r.get('production_model_version'), r.get('prediction_logic_version'), r.get('baseline_version'))
               for r in rows}
    if len(cohorts) > 1:
        return {'schema_version': MODEL_SCHEMA, 'status': 'version_filter_required', 'eligible': False}
    accepted = []
    excluded = Counter()
    for row in rows:
        quality = row.get('result_quality')
        quality = quality if isinstance(quality, dict) else {}
        if (row.get('exclude_from_calibration') or quality.get('usable_for_calibration') is False
                or not (quality.get('usable_for_calibration') is True or quality.get('grade') in ('high', 'medium'))):
            excluded['result_not_qualified_for_training'] += 1
            continue
        if (row.get('baseline_version') != RESEARCH_VERSION or not row.get('production_model_version')
                or not row.get('prediction_logic_version')):
            excluded['unknown_baseline_or_model_version'] += 1
            continue
        kickoff, as_of, settled = map(timestamp, (row.get('kickoff_at'), row.get('as_of'), row.get('settled_at')))
        if (not row.get('match_id') or counts[str(row['match_id'])] != 1 or row.get('frozen') is not True
                or not kickoff or not as_of or not settled or not as_of < kickoff < settled
                or settled > datetime.now(timezone.utc) or row.get('actual') not in ('H', 'D', 'A')
                or not valid_probabilities(row.get('base_probabilities'))):
            excluded['invalid_or_unfrozen_row'] += 1
            continue
        payload = intelligence_features(row.get('intelligence') or {}, as_of=as_of)
        if not payload['known_facts']:
            excluded['no_verified_features'] += 1
            continue
        accepted.append({**row, 'base_probabilities': {k: float(row['base_probabilities'][k]) for k in 'HDA'},
                         'date': kickoff.date().isoformat(), '_kickoff': kickoff,
                         '_settled': settled, '_as_of': as_of, '_features': payload['features']})
    accepted.sort(key=lambda row: (row['_kickoff'], str(row['match_id'])))
    days = sorted({row['date'] for row in accepted})
    if len(days) < 5 or len(accepted) < 60:
        return {'schema_version': MODEL_SCHEMA, 'status': 'insufficient_samples',
                'accepted_n': len(accepted), 'excluded': dict(excluded), 'eligible': False}
    train_end, selection_end = days[max(0, int(len(days)*.6)-1)], days[max(1, int(len(days)*.8)-1)]
    train = [r for r in accepted if r['date'] <= train_end]
    selection = [r for r in accepted if train_end < r['date'] <= selection_end]
    test = [r for r in accepted if r['date'] > selection_end]
    # Purge matches whose outcomes were not known when the next fold predicted.
    selection_as_of = min(r['_as_of'] for r in selection)
    test_as_of = min(r['_as_of'] for r in test)
    train = [r for r in train if r['_settled'] < selection_as_of]
    selection = [r for r in selection if r['_settled'] < test_as_of]
    if not train or not selection or not test:
        return {'schema_version': MODEL_SCHEMA, 'status': 'insufficient_chronological_samples', 'eligible': False}
    coefficients = [[0.0]*len(FEATURE_NAMES) for _ in 'HDA']
    for step in range(iterations):
        gradients = [[0.0]*len(FEATURE_NAMES) for _ in 'HDA']
        for row in train:
            prediction = _residual_probabilities(row['base_probabilities'], row['_features'], coefficients)
            for i, key in enumerate('HDA'):
                error = prediction[key] - float(row['actual'] == key)
                for j, name in enumerate(FEATURE_NAMES):
                    gradients[i][j] += error*row['_features'][name]
        rate = .1 / math.sqrt(1 + step/50)
        for i in range(3):
            for j in range(len(FEATURE_NAMES)):
                coefficients[i][j] -= rate*(gradients[i][j]/len(train) + .1*coefficients[i][j])
    def predictions(records, weight):
        result = []
        for row in records:
            base = row['base_probabilities']
            residual = _residual_probabilities(base, row['_features'], coefficients)
            result.append({'actual': row['actual'], 'date': row['date'],
                           'probabilities': {k: (1-weight)*base[k]+weight*residual[k] for k in 'HDA'},
                           'market_probabilities': base})
        return result
    trials = [(weight, multiclass_metrics(predictions(selection, weight))) for weight in (0.0, .025, .05, .1)]
    baseline = trials[0][1]
    qualifying = [(w, m) for w, m in trials if m['brier'] <= baseline['brier']
                  and m['logloss'] <= baseline['logloss'] and m['accuracy'] >= baseline['accuracy']]
    weight = min(qualifying, key=lambda pair: (pair[1]['logloss'], pair[0]))[0]
    heldout = predictions(test, weight)
    artifact = {'schema_version': MODEL_SCHEMA, 'baseline_version': RESEARCH_VERSION, 'status': 'trained_candidate',
                'feature_names': list(FEATURE_NAMES), 'coefficients': coefficients, 'weight': weight,
                'training_cutoff_at': datetime.now(timezone.utc).isoformat(),
                'validation': {'chronological': True, 'train_n': len(train), 'selection_n': len(selection),
                               'production_model_version': accepted[0]['production_model_version'],
                               'prediction_logic_version': accepted[0]['prediction_logic_version'],
                               'data_cutoff_at': max(r['_settled'] for r in accepted).isoformat(),
                               'test_n': len(test), 'paired_comparison': paired_date_comparison(heldout),
                               'baseline_metrics': multiclass_metrics(heldout, 'market_probabilities'),
                               'candidate_metrics': multiclass_metrics(heldout),
                               'selection_trials': trials, 'excluded': dict(excluded)}}
    artifact['artifact_id'] = _fingerprint(artifact)
    artifact['eligible'], artifact['reason'] = artifact_eligibility(artifact, as_of=datetime.now(timezone.utc))
    return artifact
