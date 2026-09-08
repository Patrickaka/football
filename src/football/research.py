"""Deterministic research variants and guarded production fusion.

The statistical branch never receives odds. Research output is supplementary:
the existing production model remains the fallback until a candidate qualifies.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math
import os

from ..domain.sports.football.lambdas import team_poisson_lambdas
from ..domain.sports.football.market_anchoring import anchor_candidates_to_market
from ..domain.sports.football.scoring_model import build_score_matrix

RESEARCH_VERSION = 'football-research-v1'
LOCAL_TIMEZONE = timezone(timedelta(hours=8))


def timestamp(value):
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else None
    try:
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except (TypeError, ValueError):
        return None


def kickoff_timestamp(match, *, now):
    """Resolve the existing Chinese schedule format without inventing an hour."""
    value = match.get('kickoff_at') or match.get('kickoff') or match.get('time')
    aware = timestamp(value)
    if aware:
        return aware
    if not isinstance(value, str) or not value.strip():
        return None
    for fmt in ('%Y-%m-%d %H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.strptime(value.strip(), fmt).replace(tzinfo=LOCAL_TIMEZONE).astimezone(timezone.utc)
        except ValueError:
            pass
    try:
        local_now = now.astimezone(LOCAL_TIMEZONE)
        choices = [datetime.strptime(f'{year}-{value.strip()}', '%Y-%m-%d %H:%M').replace(tzinfo=LOCAL_TIMEZONE)
                   for year in (local_now.year - 1, local_now.year, local_now.year + 1)]
        return min(choices, key=lambda item: abs((item-local_now).total_seconds())).astimezone(timezone.utc)
    except ValueError:
        return None


def outcome_probabilities(candidates):
    result = dict.fromkeys('HDA', 0.0)
    for (h, a), p in candidates:
        result['H' if h > a else 'A' if h < a else 'D'] += float(p)
    return result


def valid_probabilities(value):
    if not isinstance(value, dict) or set(value) != set('HDA'):
        return False
    try:
        return (all(not isinstance(p, bool) and math.isfinite(float(p)) and 0 <= float(p) <= 1
                    for p in value.values()) and abs(sum(map(float, value.values()))-1) < 1e-6)
    except (TypeError, ValueError):
        return False


def project_outcome_probabilities(candidates, target):
    """Preserve within-outcome score shape while replacing the H/D/A marginals."""
    if not valid_probabilities(target):
        raise ValueError('invalid target distribution')
    base = outcome_probabilities(candidates)
    if not valid_probabilities(base) or any(base[k] <= 0 and target[k] > 0 for k in 'HDA'):
        raise ValueError('score distribution cannot support target outcomes')
    result = []
    for (h, a), probability in candidates:
        if not math.isfinite(probability) or probability < 0:
            raise ValueError('invalid score distribution')
        key = 'H' if h > a else 'A' if h < a else 'D'
        result.append(((h, a), probability * target[key] / base[key] if base[key] else 0.0))
    return sorted(result, key=lambda item: (-item[1], item[0]))


def apply_qualified_ml(candidates, ml_response, *, as_of, stats=None, metadata=None, enabled=True):
    """Use the existing qualification rules and report what was actually applied."""
    base = outcome_probabilities(candidates)
    trace = {'loaded': False, 'eligible': False, 'executed': False, 'applied': False,
             'weight': 0.0, 'reason': 'model_unavailable', 'base_probabilities': base,
             'probabilities': None, 'model_version': None, 'training_cutoff_at': None}
    response = ml_response or {}
    if not response.get('available'):
        trace['reason'] = response.get('reason') or trace['reason']
        return candidates, trace
    metadata = metadata or {}
    target = {k: response.get(k) for k in 'HDA'}
    trace.update(loaded=True, executed=True, probabilities=target,
                 model_version=response.get('model_version'),
                 training_cutoff_at=metadata.get('training_cutoff_at'),
                 feature_audit=response.get('feature_audit'))
    cutoff = timestamp(metadata.get('training_cutoff_at'))
    prediction_time = timestamp(as_of)
    if not cutoff or not prediction_time or cutoff >= prediction_time:
        trace['reason'] = 'unknown_or_future_training_cutoff'
        return candidates, trace
    if not valid_probabilities(target):
        trace['reason'] = 'invalid_ml_probabilities'
        return candidates, trace
    from ..domain.sports.football.settlement import check_ml_fusion_eligibility
    eligibility = check_ml_fusion_eligibility(stats or {}, metadata.get('test_count', 0))
    trace.update(eligible=eligibility['eligible'], eligibility=eligibility)
    if not eligibility['eligible']:
        trace['reason'] = 'paired_shadow_validation_pending'
        return candidates, trace
    if not enabled:
        trace['reason'] = 'fusion_disabled'
        return candidates, trace
    # The gate validates a 5% blend. More samples alone do not validate 10%.
    weight = 0.05
    mixed = {k: (1-weight)*base[k] + weight*float(target[k]) for k in 'HDA'}
    result = project_outcome_probabilities(candidates, mixed)
    trace.update(applied=True, weight=weight, reason='qualified_5pct_blend',
                 final_probabilities=outcome_probabilities(result))
    return result, trace


def variant(candidates, *, source, captured_at, status='applied', applied=True, **extra):
    return {'probabilities': outcome_probabilities(candidates),
            'score_probabilities': {f'{h}-{a}': p for (h, a), p in candidates},
            'source': source, 'captured_at': captured_at, 'status': status,
            'applied': applied, 'fallback_reason': None, **extra}


def statistical_candidates(team, league_profile):
    """A real team-only baseline, using the existing Poisson implementation."""
    if not team or any(not isinstance(team.get(key), (int, float)) or
                       not math.isfinite(team[key]) or team[key] <= 0
                       for key in ('attack_home', 'defense_home', 'attack_away', 'defense_away')):
        return None
    if any((team.get(f'{side}_recent') or {}).get('games', 0) < 5 for side in ('home', 'away')):
        return None
    # No market-implied total enters A. Both mean and home advantage are the
    # existing league prior, recorded with this version for later comparison.
    profile = league_profile or {}
    total = 2 * float(profile.get('avg_goal', 1.35))
    if not math.isfinite(total) or total <= 0:
        return None
    lh, la = team_poisson_lambdas(team, total, profile)
    matrix = build_score_matrix(lh, la, max_goals=7, rho=0, distribution='poisson')
    candidates = [(score, float(p)) for score, p in matrix.items()]
    mass = sum(p for _, p in candidates)
    return sorted([(score, p/mass) for score, p in candidates], key=lambda item: (-item[1], item[0]))


def build_research_variants(*, team, league_profile, euro, asian, total, as_of,
                            intelligence=None, residual_artifact=None):
    """A/B/C share inputs and an identity calibration protocol; M is 1X2 only."""
    captured_at = timestamp(as_of).isoformat()
    unavailable = {'probabilities': None, 'score_probabilities': None, 'applied': False,
                   'status': 'unavailable', 'fallback_reason': 'insufficient_team_history',
                   'source': RESEARCH_VERSION, 'captured_at': captured_at}
    variants = {k: deepcopy(unavailable) for k in ('statistical', 'market_adjusted', 'agent_adjusted')}
    odds = (euro.get('raw_odds') or {}).get('close') or {}
    try:
        inverse = {k: 1/float(odds[name]) for k, name in zip('HDA', ('home', 'draw', 'away'))}
        mass = sum(inverse.values())
        market = {k: p/mass for k, p in inverse.items()}
        if any(float(odds[name]) <= 1 for name in ('home', 'draw', 'away')) or not valid_probabilities(market):
            raise ValueError('invalid market odds')
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        market = None
    variants['market_only'] = {**deepcopy(unavailable), 'probabilities': market,
                              'applied': market is not None, 'status': 'applied' if market else 'unavailable',
                              'fallback_reason': None if market else 'missing_observed_1x2_market',
                              'source': 'same_snapshot_no_vig_1x2'}
    # A proxy from RQSPF is not a directly observed ordinary 1X2 market.
    if euro.get('ordinary_market_observed') is False:
        market = None
        variants['market_only'].update(probabilities=None, applied=False, status='unavailable',
                                       fallback_reason='ordinary_1x2_not_observed')
    a = statistical_candidates(team, league_profile)
    if a is None:
        return variants
    protocol = {'baseline_version': RESEARCH_VERSION,
                'calibration': {'applied': False, 'method': 'identity', 'version': RESEARCH_VERSION}}
    variants['statistical'] = variant(a, source='team_poisson_no_market_v1', captured_at=captured_at,
                                      parameters={'league_profile': deepcopy(league_profile)}, **protocol)
    if market is None:
        for name in ('market_adjusted', 'agent_adjusted'):
            variants[name]['fallback_reason'] = 'missing_observed_1x2_market'
        return variants
    # Unobserved proxy Asian/totals must not become extra market evidence.
    observed_asian = {} if asian.get('source') == 'model_proxy' else asian
    observed_total = {} if total.get('source') == 'model_proxy' else total
    b, anchor_trace = anchor_candidates_to_market(a, observed_total, euro, observed_asian, {})
    variants['market_adjusted'] = variant(b, source='statistical_plus_observed_market_v1',
                                         captured_at=captured_at, execution_trace=anchor_trace, **protocol)
    from .research_model import apply_intelligence_residual
    try:
        c, residual_trace = apply_intelligence_residual(b, intelligence or {}, residual_artifact,
                                                      as_of=as_of)
    except Exception as exc:
        c, residual_trace = b, {'applied': False, 'weight': 0.0,
                               'reason': 'residual_error:' + type(exc).__name__}
    variants['agent_adjusted'] = variant(c, source='market_plus_trained_intelligence_v1',
                                        captured_at=captured_at, applied=residual_trace['applied'],
                                        status='applied' if residual_trace['applied'] else 'fallback',
                                        fallback_reason=None if residual_trace['applied'] else residual_trace['reason'],
                                        execution_trace=residual_trace, **protocol)
    return variants


def load_residual_artifact():
    """An explicitly configured local JSON artifact; never load executable pickles."""
    import json
    path = os.getenv('FOOTBALL_INTELLIGENCE_MODEL', '').strip()
    if not path:
        return None
    try:
        with open(path, encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None
