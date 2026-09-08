# -*- coding: utf-8 -*-
"""临场情报的新鲜度与融合。

**时钟由调用方注入**（判据 16）：`grade_live_context` 判「这条情报够不够新」
要跟当前时间比，不注入的话黄金隔天就红。
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _parse_timestamp(value) -> Optional[datetime]:
    if not value or value == 'UNAVAILABLE':
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None

def _audit_context_entry(entry, now: datetime, max_age_hours: float) -> Dict:
    if not isinstance(entry, dict) or not entry:
        return {'verified': False, 'status': 'missing', 'age_hours': None,
                'source': None, 'source_at': None}
    source = entry.get('source')
    source_valid = isinstance(source, str) and source.strip().lower() not in {
        '', 'unavailable', 'unknown', 'default', 'model_proxy', 'none', 'null',
    }
    timestamp = _parse_timestamp(entry.get('ts'))
    # Offset-free local timestamps are ambiguous. Keep the legacy parser's UTC
    # assumption for its other callers, but do not call such a source verified.
    try:
        explicit_timezone = datetime.fromisoformat(str(entry.get('ts')).replace('Z', '+00:00')).tzinfo is not None
    except (TypeError, ValueError):
        explicit_timezone = False
    age = (now - timestamp).total_seconds() / 3600.0 if timestamp else None
    if not source_valid or not timestamp or not explicit_timezone:
        status = 'unverified'
    elif age < 0:
        status = 'future'
    elif age > max_age_hours:
        status = 'stale'
    else:
        status = 'verified'
    return {
        'verified': status == 'verified', 'status': status,
        'source': source if source_valid else None,
        'source_at': timestamp.isoformat() if timestamp and explicit_timezone else None,
        'age_hours': round(age, 2) if age is not None else None,
    }


def assess_live_context(
    context: Dict,
    now: Optional[datetime] = None,
    require_confirmed_lineup: bool = True,
    max_age_hours: float = 24.0,
) -> Dict:
    """Require sourced injuries and confirmed lineups for the default live gate.

    `age_hours` describes the oldest supplied observation, so a fresher entry
    cannot hide a stale one. Callers can opt out of requiring a lineup when
    scoring early research snapshots; that does not waive source verification.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    context = context or {}
    checks = {}
    score = 1.0

    injuries = context.get('injuries') or []
    injury_audit = [_audit_context_entry(entry, now, max_age_hours) for entry in injuries] if isinstance(injuries, list) else []
    injuries_verified = bool(injury_audit) and all(item['verified'] for item in injury_audit)
    checks['injuries'] = 'available' if injuries_verified else ('unverified' if injuries else 'missing')
    if not injuries_verified:
        score -= 0.10

    lineup = context.get('lineup') or {}
    lineup_audit = _audit_context_entry(lineup, now, max_age_hours)
    lineup_confirmed = isinstance(lineup, dict) and lineup.get('confirmed') is True
    lineup_audit['confirmed'] = lineup_confirmed
    lineup_verified = lineup_audit['verified'] and lineup_confirmed
    checks['lineup'] = 'available' if lineup_verified else ('unverified' if lineup else 'missing')
    if not lineup_verified:
        score -= 0.15

    possession = context.get('possession')
    checks['performance_context'] = 'available' if possession else 'missing'
    if not possession:
        score -= 0.05

    if context.get('injury_conflict'):
        checks['source_conflict'] = 'failed'
        score -= 0.20
    else:
        checks['source_conflict'] = 'passed'

    audits = list(injury_audit)
    if lineup:
        audits.append(lineup_audit)
    for key in ('schedule_density', 'form'):
        if context.get(key):
            audits.append(_audit_context_entry(context[key], now, max_age_hours))
    timestamps = [_parse_timestamp(item['source_at']) for item in audits if item.get('source_at')]
    freshest = max(timestamps) if timestamps else None
    oldest = min(timestamps) if timestamps else None
    age_hours = (now - oldest).total_seconds() / 3600.0 if oldest else None
    # A current lineup must not conceal a stale or undated injury observation.
    statuses = {item['status'] for item in audits}
    checks['freshness'] = ('future' if 'future' in statuses else
                           'stale' if 'stale' in statuses else
                           'passed' if audits and all(item['verified'] for item in audits) else 'unknown')
    if checks['freshness'] != 'passed':
        score -= 0.10

    blockers = []
    if not injuries_verified:
        blockers.append('injuries_missing_or_unverified')
    if require_confirmed_lineup and not lineup_verified:
        blockers.append('confirmed_lineup_missing' if not lineup else 'confirmed_lineup_unverified')
    if audits and checks['freshness'] != 'passed':
        blockers.append('context_sources_not_fresh_and_verified')
    if context.get('injury_conflict'):
        blockers.append('injury_source_conflict')
    return {
        'quality_score': round(max(0.0, score), 3),
        'confidence_multiplier': round(max(0.50, score), 3),
        'official_bet_allowed': not blockers,
        'blockers': blockers,
        'checks': checks,
        'freshest_source_at': freshest.isoformat() if freshest else None,
        'oldest_source_at': oldest.isoformat() if oldest else None,
        'age_hours': round(age_hours, 2) if age_hours is not None else None,
        'source_audit': {'injuries': injury_audit, 'lineup': lineup_audit},
    }


def _number(value, default=0.0):
    try:
        number = float(value)
        return number if math.isfinite(number) and not isinstance(value, bool) else default
    except (TypeError, ValueError):
        return default

def apply_contextual_fusion(
    candidates: List[Tuple[Tuple[int, int], float]],
    context: Dict,
    now: Optional[datetime] = None,
    max_age_hours: float = 24.0,
) -> Tuple[List[Tuple[Tuple[int, int], float]], Dict]:
    """Apply small sourced H2H/motivation corrections.

    Expected structured context keys:
      h2h: games, home_wins, draws, away_wins, avg_goals, quality_score, source, ts
      motivation: home, away (-1..1), quality_score, source, ts
    Free-form text is deliberately ignored.
    """
    context = context or {}
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    h2h = context.get('h2h') if isinstance(context.get('h2h'), dict) else {}
    motivation = context.get('motivation') if isinstance(context.get('motivation'), dict) else {}
    outcome_weights = {'H': 1.0, 'D': 1.0, 'A': 1.0}
    goal_beta = 0.0
    evidence = []

    games = int(_number(h2h.get('games'), 0))
    h2h_quality = max(0.0, min(1.0, _number(h2h.get('quality_score'), 0.0)))
    h2h_audit = _audit_context_entry(h2h, now, max_age_hours)
    motivation_audit = _audit_context_entry(motivation, now, max_age_hours)
    source_audit = {'h2h': h2h_audit, 'motivation': motivation_audit}
    outcome_counts = [_number(h2h.get(key), -1) for key in ('home_wins', 'draws', 'away_wins')]
    counts_valid = (_number(h2h.get('games'), -1) == games
                    and all(count >= 0 and count.is_integer() for count in outcome_counts)
                    and sum(outcome_counts) == games)
    if games >= 5 and counts_valid and h2h_quality >= 0.6 and h2h_audit['verified']:
        n = max(games, 1)
        empirical = {
            'H': _number(h2h.get('home_wins')) / n,
            'D': _number(h2h.get('draws')) / n,
            'A': _number(h2h.get('away_wins')) / n,
        }
        strength = min(0.08, 0.02 + games * 0.003) * h2h_quality
        for key, rate in empirical.items():
            outcome_weights[key] *= 1.0 + strength * (rate - 1 / 3) * 3
        avg_goals = _number(h2h.get('avg_goals'), 0.0)
        if 0.5 <= avg_goals <= 6.0:
            goal_beta += max(-0.05, min(0.05, (avg_goals - 2.6) * 0.035)) * h2h_quality
        evidence.append({'type': 'h2h', 'games': games, 'quality_score': h2h_quality})

    motivation_quality = max(0.0, min(1.0, _number(motivation.get('quality_score'), 0.0)))
    source = motivation.get('source')
    if motivation_quality >= 0.7 and motivation_audit['verified']:
        home = max(-1.0, min(1.0, _number(motivation.get('home'))))
        away = max(-1.0, min(1.0, _number(motivation.get('away'))))
        delta = (home - away) * 0.06 * motivation_quality
        outcome_weights['H'] *= 1.0 + delta
        outcome_weights['A'] *= 1.0 - delta
        # High motivation on both sides often increases tempo; one-sided
        # motivation primarily changes direction rather than total goals.
        goal_beta += max(0.0, min(home, away)) * 0.025 * motivation_quality
        evidence.append({'type': 'motivation', 'source': source, 'quality_score': motivation_quality})

    if not evidence or not candidates:
        return candidates, {'applied': False, 'reason': 'no_qualified_structured_context',
                            'source_audit': source_audit}

    def outcome(score):
        return 'H' if score[0] > score[1] else 'A' if score[0] < score[1] else 'D'

    adjusted = []
    for score, probability in candidates:
        factor = outcome_weights[outcome(score)] * math.exp(goal_beta * sum(score))
        adjusted.append((score, max(0.0, float(probability)) * factor))
    total = sum(probability for _, probability in adjusted)
    if total <= 0:
        return candidates, {'applied': False, 'reason': 'zero_probability'}
    adjusted = [(score, probability / total) for score, probability in adjusted]
    adjusted.sort(key=lambda item: -item[1])
    return adjusted, {
        'applied': True,
        'outcome_weights': {key: round(value, 4) for key, value in outcome_weights.items()},
        'goal_beta': round(goal_beta, 4),
        'evidence': evidence,
        'source_audit': source_audit,
        'guards': {'max_direction_adjustment': 0.12, 'max_goal_beta': 0.075},
    }
