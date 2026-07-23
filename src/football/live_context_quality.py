"""Quality gate for volatile pre-match context.

Missing context does not invent a prediction adjustment. The gate produces an
auditable confidence multiplier and can block official bets when lineups were
required but are unavailable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional


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

