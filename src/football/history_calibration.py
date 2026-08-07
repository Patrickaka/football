#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Conservative production-history calibration for football score distributions."""

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


MIN_HISTORY_SAMPLES = 60
HISTORY_HALF_LIFE = 180.0
MAX_GOAL_BETA = 0.18
PROFILE_PATH = Path(__file__).resolve().parents[2] / 'data' / 'football_history_calibration.json'

_PROFILE_CACHE = {'key': None, 'profile': None}


def _score_tuple(value):
    try:
        home, away = map(int, str(value).split('-'))
    except (TypeError, ValueError):
        return None
    if home < 0 or away < 0 or home > 15 or away > 15:
        return None
    return home, away


def _outcome(score: Tuple[int, int]) -> str:
    return 'H' if score[0] > score[1] else 'A' if score[0] < score[1] else 'D'


def _normalized_scores(record: Dict) -> Dict[Tuple[int, int], float]:
    values = {}
    for score_text, probability in (record.get('predicted_scores') or {}).items():
        score = _score_tuple(score_text)
        try:
            probability = float(probability)
        except (TypeError, ValueError):
            continue
        if score is not None and probability > 0:
            values[score] = probability
    total = sum(values.values())
    return {score: value / total for score, value in values.items()} if total > 0 else {}


def _quality_weight(record: Dict) -> float:
    try:
        from .sample_quality import assess_record_quality

        quality = assess_record_quality(record)
        weight = max(0.0, min(1.0, float(quality.get('calibration_weight', 0.0))))
        if weight > 0:
            return weight

        # Older production snapshots contain the immutable prediction, final
        # score and a successful sync marker, but pre-date the odds/result
        # quality metadata.  Rejecting all of them leaves runtime calibration
        # permanently disabled (even with hundreds of settled predictions).
        # They are sufficient for the score-shape calibration performed here;
        # use a deliberately small weight and never admit failed/ignored rows.
        legacy_score_sample = (
            record.get('settled')
            and record.get('sync_status') == 'synced'
            and _score_tuple(record.get('actual_score')) is not None
            and bool(_normalized_scores(record))
            and not record.get('exclude_from_calibration')
        )
        if legacy_score_sample and set(quality.get('reasons') or []).issubset({
            'missing_predicted_1x2',
            'missing_result_quality',
            'missing_asian_line',
            'missing_total_line',
            'missing_odds_snapshot',
        }):
            return 0.35
        return 0.0
    except Exception:
        return 0.0


def _mean_after_beta(rows, beta: float, outcome_weights=None) -> float:
    outcome_weights = outcome_weights or {'H': 1.0, 'D': 1.0, 'A': 1.0}
    weighted_mean = 0.0
    weight_sum = 0.0
    for _, distribution, sample_weight in rows:
        adjusted = {
            score: probability * math.exp(beta * sum(score)) * outcome_weights.get(_outcome(score), 1.0)
            for score, probability in distribution.items()
        }
        total = sum(adjusted.values())
        if total <= 0:
            continue
        expected = sum(sum(score) * value for score, value in adjusted.items()) / total
        weighted_mean += expected * sample_weight
        weight_sum += sample_weight
    return weighted_mean / weight_sum if weight_sum > 0 else 0.0


def estimate_history_calibration(records: Iterable[Dict], min_samples: int = MIN_HISTORY_SAMPLES) -> Dict:
    """Estimate a guarded profile from settled predictions without changing history."""
    prepared = []
    for record in records or []:
        actual = _score_tuple(record.get('actual_score'))
        distribution = _normalized_scores(record)
        if actual is None or not distribution:
            continue
        prepared.append((record, actual, distribution, _quality_weight(record)))

    prepared.sort(key=lambda item: str(item[0].get('created_at') or item[0].get('match_time') or ''))
    if len(prepared) < min_samples:
        return {
            'applied': False,
            'reason': 'insufficient_history',
            'sample_count': len(prepared),
            'min_samples': min_samples,
        }

    rows = []
    actual_goals_sum = 0.0
    predicted_goals_sum = 0.0
    actual_outcomes = {'H': 0.0, 'D': 0.0, 'A': 0.0}
    predicted_outcomes = {'H': 0.0, 'D': 0.0, 'A': 0.0}
    weight_sum = 0.0
    total_rows = len(prepared)
    for index, (_, actual, distribution, quality_weight) in enumerate(prepared):
        if quality_weight <= 0:
            continue
        age = total_rows - index - 1
        recency_weight = 0.5 ** (age / HISTORY_HALF_LIFE)
        sample_weight = quality_weight * recency_weight
        rows.append((actual, distribution, sample_weight))
        actual_goals_sum += sum(actual) * sample_weight
        predicted_goals_sum += sum(sum(score) * prob for score, prob in distribution.items()) * sample_weight
        actual_outcomes[_outcome(actual)] += sample_weight
        for score, probability in distribution.items():
            predicted_outcomes[_outcome(score)] += probability * sample_weight
        weight_sum += sample_weight

    if weight_sum < min_samples * 0.35:
        return {
            'applied': False,
            'reason': 'insufficient_effective_weight',
            'sample_count': len(prepared),
            'effective_weight': round(weight_sum, 3),
        }

    actual_goal_mean = actual_goals_sum / weight_sum
    predicted_goal_mean = predicted_goals_sum / weight_sum
    # Effective weights are deliberately below one for legacy snapshots.  A
    # second linear shrink here made hundreds of settled matches behave like a
    # tiny sample and left a persistent goal-mean error almost untouched.  Use
    # square-root shrinkage: it remains conservative for small samples, while
    # allowing a broad, consistently biased history to make a visible repair.
    reliability = min(1.0, math.sqrt(weight_sum / 180.0))

    raw_outcome_weights = {}
    outcome_weights = {}
    for key in ('H', 'D', 'A'):
        actual_rate = actual_outcomes[key] / weight_sum
        predicted_rate = predicted_outcomes[key] / weight_sum
        raw_ratio = actual_rate / predicted_rate if predicted_rate > 1e-9 else 1.0
        raw_outcome_weights[key] = raw_ratio
        # Outcome calibration is intentionally weaker than the goal-tail repair.
        adjusted_ratio = 1.0 + reliability * 0.50 * (raw_ratio - 1.0)
        outcome_weights[key] = max(0.90, min(1.10, adjusted_ratio))

    target_beta = min(
        (step / 100.0 for step in range(-20, 41)),
        key=lambda beta: abs(_mean_after_beta(rows, beta, outcome_weights) - actual_goal_mean),
    )
    goal_beta = max(-MAX_GOAL_BETA, min(MAX_GOAL_BETA, target_beta * reliability))

    return {
        'applied': True,
        'source': 'settled_prediction_history',
        'sample_count': len(prepared),
        'effective_weight': round(weight_sum, 3),
        'reliability': round(reliability, 4),
        'actual_goal_mean': round(actual_goal_mean, 4),
        'predicted_goal_mean': round(predicted_goal_mean, 4),
        'goal_gap': round(actual_goal_mean - predicted_goal_mean, 4),
        'raw_goal_beta': round(target_beta, 4),
        'goal_beta': round(goal_beta, 4),
        'outcome_weights': {key: round(value, 4) for key, value in outcome_weights.items()},
        'raw_outcome_weights': {key: round(value, 4) for key, value in raw_outcome_weights.items()},
        'guards': {
            'min_samples': min_samples,
            'half_life': HISTORY_HALF_LIFE,
            'max_goal_beta': MAX_GOAL_BETA,
            'reliability_curve': 'sqrt_effective_weight',
        },
    }


def apply_history_calibration(candidates: List, profile: Dict) -> Tuple[List, Dict]:
    """Adjust score shape while preserving the current match's 1X2 marginals."""
    if not profile or not profile.get('applied') or not candidates:
        return candidates, {'applied': False, 'reason': (profile or {}).get('reason', 'missing_profile')}
    try:
        beta = float(profile.get('goal_beta', 0.0))
        outcome_weights = profile.get('outcome_weights') or {'H': 1.0, 'D': 1.0, 'A': 1.0}
        adjusted = []
        expected_before = 0.0
        total_before = 0.0
        original_outcome_mass = {'H': 0.0, 'D': 0.0, 'A': 0.0}
        for score, probability in candidates:
            score = int(score[0]), int(score[1])
            probability = float(probability)
            factor = math.exp(beta * sum(score)) * float(outcome_weights.get(_outcome(score), 1.0))
            adjusted.append((score, probability * factor))
            expected_before += sum(score) * probability
            total_before += probability
            original_outcome_mass[_outcome(score)] += probability

        adjusted_outcome_mass = {'H': 0.0, 'D': 0.0, 'A': 0.0}
        for score, probability in adjusted:
            adjusted_outcome_mass[_outcome(score)] += probability
        adjusted = [
            (
                score,
                probability * original_outcome_mass[_outcome(score)]
                / adjusted_outcome_mass[_outcome(score)],
            )
            for score, probability in adjusted
            if adjusted_outcome_mass[_outcome(score)] > 0
        ]
        total = sum(probability for _, probability in adjusted)
        if total <= 0:
            return candidates, {'applied': False, 'reason': 'zero_adjusted_mass'}
        adjusted = [(score, probability / total) for score, probability in adjusted]
        adjusted.sort(key=lambda item: -item[1])
        expected_after = sum(sum(score) * probability for score, probability in adjusted)
        return adjusted, {
            'applied': True,
            'sample_count': profile.get('sample_count'),
            'effective_weight': profile.get('effective_weight'),
            'goal_beta': beta,
            'outcome_weights': outcome_weights,
            'preserved_1x2': True,
            'expected_goals_before': round(expected_before / total_before, 4) if total_before > 0 else None,
            'expected_goals_after': round(expected_after, 4),
            'source': profile.get('source'),
        }
    except Exception as exc:
        return candidates, {'applied': False, 'reason': str(exc)}


def _load_fallback_profile() -> Dict:
    try:
        with PROFILE_PATH.open('r', encoding='utf-8') as handle:
            profile = json.load(handle)
        return profile if isinstance(profile, dict) else {}
    except Exception:
        return {}


def get_runtime_history_profile() -> Dict:
    """Read the current production history, caching until its latest record changes."""
    try:
        from .result_sync import get_history

        records = get_history().records
        latest = max((str(item.get('updated_at') or item.get('settled_at') or '') for item in records), default='')
        cache_key = (len(records), latest)
        if _PROFILE_CACHE['key'] == cache_key and _PROFILE_CACHE['profile'] is not None:
            return _PROFILE_CACHE['profile']
        profile = estimate_history_calibration(records)
        if not profile.get('applied'):
            profile = _load_fallback_profile() or profile
        _PROFILE_CACHE.update({'key': cache_key, 'profile': profile})
        return profile
    except Exception:
        return _load_fallback_profile()
