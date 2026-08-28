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

def assess_live_context(
    context: Dict,
    now: Optional[datetime] = None,
    require_confirmed_lineup: bool = False,
    max_age_hours: float = 24.0,
) -> Dict:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    context = context or {}
    checks = {}
    score = 1.0

    injuries = context.get('injuries') or []
    checks['injuries'] = 'available' if injuries else 'missing'
    if not injuries:
        score -= 0.10

    lineup = context.get('lineup') or {}
    checks['lineup'] = 'available' if lineup else 'missing'
    if not lineup:
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

    timestamps = []
    for entry in injuries:
        timestamp = _parse_timestamp(entry.get('ts'))
        if timestamp:
            timestamps.append(timestamp)
    for value in (lineup, context.get('schedule_density'), context.get('form')):
        if isinstance(value, dict):
            timestamp = _parse_timestamp(value.get('ts'))
            if timestamp:
                timestamps.append(timestamp)
    freshest = max(timestamps) if timestamps else None
    age_hours = (now - freshest).total_seconds() / 3600.0 if freshest else None
    checks['freshness'] = (
        'passed' if age_hours is not None and age_hours <= max_age_hours
        else ('stale' if age_hours is not None else 'unknown')
    )
    if checks['freshness'] != 'passed':
        score -= 0.10

    blockers = []
    if require_confirmed_lineup and not lineup:
        blockers.append('confirmed_lineup_missing')
    if context.get('injury_conflict'):
        blockers.append('injury_source_conflict')
    return {
        'quality_score': round(max(0.0, score), 3),
        'confidence_multiplier': round(max(0.50, score), 3),
        'official_bet_allowed': not blockers,
        'blockers': blockers,
        'checks': checks,
        'freshest_source_at': freshest.isoformat() if freshest else None,
        'age_hours': round(age_hours, 2) if age_hours is not None else None,
    }


def _number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def apply_contextual_fusion(
    candidates: List[Tuple[Tuple[int, int], float]],
    context: Dict,
) -> Tuple[List[Tuple[Tuple[int, int], float]], Dict]:
    """Apply small sourced H2H/motivation corrections.

    Expected structured context keys:
      h2h: games, home_wins, draws, away_wins, avg_goals, quality_score
      motivation: home, away (-1..1), quality_score, source
    Free-form text is deliberately ignored.
    """
    context = context or {}
    h2h = context.get('h2h') or {}
    motivation = context.get('motivation') or {}
    outcome_weights = {'H': 1.0, 'D': 1.0, 'A': 1.0}
    goal_beta = 0.0
    evidence = []

    games = int(_number(h2h.get('games'), 0))
    h2h_quality = max(0.0, min(1.0, _number(h2h.get('quality_score'), 0.0)))
    if games >= 5 and h2h_quality >= 0.6:
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
    if source and source != 'UNAVAILABLE' and motivation_quality >= 0.7:
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
        return candidates, {'applied': False, 'reason': 'no_qualified_structured_context'}

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
        'guards': {'max_direction_adjustment': 0.12, 'max_goal_beta': 0.075},
    }
