"""Time-bounded football facts. No network, model inference or prediction weights."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import TypedDict
from urllib.parse import urlsplit

from .context import assess_live_context

SCHEMA_VERSION = 'football-match-context-v1'
CATEGORIES = ('injuries', 'lineup', 'news', 'weather', 'schedule', 'h2h')
CONFIRMATION_STATES = ('confirmed', 'reported', 'disputed', 'unknown')
_MAX_AGE_HOURS = {'injuries': 24, 'lineup': 6, 'news': 72, 'weather': 12}


class EvidenceFact(TypedDict, total=False):
    evidence_id: str
    category: str
    team: str
    data: dict
    source_url: str
    published_at: str
    collected_at: str
    confirmation_status: str
    verified: bool
    reasons: list


class MatchContext(TypedDict, total=False):
    schema_version: str
    match_id: str
    kickoff: str
    as_of: str
    captured_at: str
    status: str
    evidence: list[EvidenceFact]
    missing_categories: list[str]
    live_context: dict
_FIELDS = {
    'injuries': {'player', 'status'},
    'lineup': {'players'},
    'news': {'headline'},
    'weather': {'temperature_celsius', 'precipitation_mm', 'wind_kmh', 'forecast_for'},
    'schedule': {'previous_kickoff', 'rest_days'},
    'h2h': {'games', 'home_wins', 'draws', 'away_wins', 'avg_goals', 'most_recent_match_at'},
}


def timestamp(value):
    """Unlike legacy parsers, never guess a timezone for pre-match evidence."""
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def require_timestamp(value, name='as_of'):
    parsed = timestamp(value)
    if parsed is None:
        raise ValueError(f'{name} must be an ISO timestamp with explicit timezone')
    return parsed


def source_url_valid(value):
    try:
        parsed = urlsplit(value)
        return (parsed.scheme in ('http', 'https') and bool(parsed.hostname)
                and not parsed.username and not parsed.password)
    except (TypeError, ValueError):
        return False


def _numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_data(category, data, cutoff, kickoff):
    if not isinstance(data, dict) or not data or set(data) - _FIELDS[category]:
        return None, 'unsupported_fact_fields'
    data = deepcopy(data)
    if category == 'injuries':
        if (not isinstance(data.get('player'), str) or not data['player'].strip()
                or data.get('status') not in ('injured', 'suspended', 'unavailable', 'doubtful', 'available', 'none_reported')):
            return None, 'invalid_injury'
    elif category == 'lineup':
        players = data.get('players')
        if (not isinstance(players, list) or len(players) != 11
                or any(not isinstance(p, str) or not p.strip() for p in players)
                or len(set(players)) != 11):
            return None, 'lineup_requires_eleven_named_players'
    elif category == 'news':
        if not isinstance(data.get('headline'), str) or not data['headline'].strip():
            return None, 'empty_news'
        data['headline'] = data['headline'][:500]
    elif category == 'schedule':
        previous = timestamp(data.get('previous_kickoff'))
        if previous is None or previous >= cutoff or previous >= kickoff:
            return None, 'schedule_not_before_cutoff'
        # A quoted "rest" number cannot override observable fixture timestamps.
        data = {'previous_kickoff': previous.isoformat(),
                'rest_days': round((kickoff - previous).total_seconds() / 86400, 3)}
    elif category == 'weather':
        forecast = timestamp(data.get('forecast_for'))
        if forecast is None or abs((forecast - kickoff).total_seconds()) > 3600:
            return None, 'weather_not_for_match'
        bounds = {'temperature_celsius': (-80, 65), 'precipitation_mm': (0, 1000), 'wind_kmh': (0, 500)}
        for key, (low, high) in bounds.items():
            if key in data and (not _numeric(data[key]) or not low <= data[key] <= high):
                return None, 'invalid_weather_value'
        if not any(key in data for key in bounds):
            return None, 'missing_weather_values'
    elif category == 'h2h':
        recent = timestamp(data.get('most_recent_match_at'))
        counts = [data.get(k) for k in ('games', 'home_wins', 'draws', 'away_wins')]
        if (recent is None or recent >= cutoff or recent >= kickoff
                or any(not _numeric(n) or int(n) != n or n < 0 for n in counts)
                or counts[0] < 1 or sum(counts[1:]) != counts[0]
                or not _numeric(data.get('avg_goals')) or not 0 <= data['avg_goals'] <= 30):
            return None, 'invalid_or_future_h2h'
    return data, None


def audit_fact(raw, *, as_of, kickoff):
    """Validate a claim without granting its publisher or text any authority."""
    cutoff, kickoff = require_timestamp(as_of), require_timestamp(kickoff, 'kickoff')
    if not isinstance(raw, dict):
        return {'verified': False, 'reasons': ['invalid_fact']}
    category = raw.get('category')
    reasons = []
    published, collected = timestamp(raw.get('published_at')), timestamp(raw.get('collected_at'))
    if category not in CATEGORIES:
        reasons.append('unsupported_category')
    if raw.get('team') not in ('home', 'away', 'match'):
        reasons.append('unresolved_team')
    if category in ('injuries', 'lineup', 'schedule') and raw.get('team') not in ('home', 'away'):
        reasons.append('team_required')
    if not source_url_valid(raw.get('source_url')):
        reasons.append('source_url_missing_or_invalid')
    if published is None:
        reasons.append('publication_time_unknown')
    elif published > cutoff or published >= kickoff:
        reasons.append('published_after_cutoff')
    elif category in _MAX_AGE_HOURS and (cutoff - published).total_seconds() > _MAX_AGE_HOURS[category] * 3600:
        reasons.append('stale_fact')
    if collected is None:
        reasons.append('collection_time_unknown')
    elif collected > cutoff or collected >= kickoff:
        reasons.append('collected_after_cutoff')
    if published and collected and published > collected:
        reasons.append('publication_after_collection')
    confirmation = raw.get('confirmation_status', 'unknown')
    if confirmation not in CONFIRMATION_STATES:
        confirmation = 'unknown'
    if confirmation != 'confirmed':
        reasons.append('confirmation_' + confirmation)
    if confirmation == 'disputed' and 'source_conflict' in (raw.get('reasons') or []):
        reasons.append('source_conflict')
    data, invalid = (_validate_data(category, raw.get('data'), cutoff, kickoff)
                     if category in CATEGORIES else (None, None))
    if invalid:
        reasons.append(invalid)
    result = {
        'category': category, 'team': raw.get('team'), 'data': data,
        'source_url': raw.get('source_url'), 'source': str(raw.get('source') or '')[:100],
        'published_at': published.isoformat() if published else None,
        'collected_at': collected.isoformat() if collected else None,
        'confirmation_status': confirmation,
        'extraction_method': str(raw.get('extraction_method') or 'structured_source')[:80],
        'quote': str(raw.get('quote') or '')[:1000],
        'verified': not reasons, 'reasons': sorted(set(reasons)),
    }
    identity = {k: result[k] for k in ('category', 'team', 'data', 'source_url', 'published_at')}
    result['evidence_id'] = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
    return result


def _conflicts(facts):
    groups = {}
    for fact in facts:
        if not fact['verified']:
            continue
        data = fact['data']
        subject = data.get('player', '') if fact['category'] == 'injuries' else ''
        groups.setdefault((fact['category'], fact['team'], subject), []).append(fact)
    conflicts = []
    for key, group in groups.items():
        if key[0] not in ('injuries', 'lineup'):
            continue
        if len({json.dumps(f['data'], sort_keys=True) for f in group}) > 1:
            conflicts.append(':'.join(key))
            for fact in group:
                fact.update(verified=False, confirmation_status='disputed')
                fact['reasons'].append('source_conflict')
    # "No injuries" and a named absence are contradictory even across subjects.
    for team in ('home', 'away'):
        group = [f for f in facts if f['verified'] and f['category'] == 'injuries' and f['team'] == team]
        if (any(f['data']['status'] == 'none_reported' for f in group)
                and any(f['data']['status'] in ('injured', 'suspended', 'unavailable') for f in group)):
            conflicts.append(f'injuries:{team}:availability')
            for fact in group:
                fact.update(verified=False, confirmation_status='disputed')
                fact['reasons'].append('source_conflict')
    return conflicts


def facts_to_live_context(facts, *, as_of, conflicts=()):
    """Project only validated facts onto the existing live-context contract."""
    valid = [fact for fact in facts if fact.get('verified')]
    live = {'injuries': [], 'lineup': {}, 'schedule_density': {}, 'form': {},
            'possession': None, 'news': [], 'weather': {}, 'h2h': {}}

    def provenance(fact):
        return {'source': fact['source_url'], 'source_url': fact['source_url'],
                'ts': fact['published_at'], 'published_at': fact['published_at'],
                'collected_at': fact['collected_at'], 'evidence_id': fact['evidence_id']}

    for fact in valid:
        category, team, data = fact['category'], fact['team'], fact['data']
        if category == 'injuries':
            live['injuries'].append(dict(data, team=team, **provenance(fact)))
        elif category == 'news':
            live['news'].append(dict(data, team=team, **provenance(fact)))
        elif category in ('weather', 'h2h'):
            live[category] = dict(data, **provenance(fact))
            # No LLM confidence/quality scalar may activate legacy fusion.
            if category == 'h2h':
                live[category]['evidence_only'] = True
    for category, key in (('lineup', 'lineup'), ('schedule', 'schedule_density')):
        entries = [f for f in valid if f['category'] == category]
        by_team = {team: next((f for f in reversed(entries) if f['team'] == team), None) for team in ('home', 'away')}
        if all(by_team.values()):
            oldest = min(by_team.values(), key=lambda f: f['published_at'])
            live[key] = dict(provenance(oldest), **{team: f['data'] for team, f in by_team.items()})
            live[key]['evidence'] = [provenance(f) for f in by_team.values()]
            if category == 'lineup':
                live[key]['confirmed'] = True
    if any(c.startswith('injuries:') for c in conflicts):
        live['injury_conflict'] = 'Sources disagree; conflicting facts are excluded.'
    live['quality'] = assess_live_context(live, now=require_timestamp(as_of))
    # The legacy gate permits a single team's injury report; this adapter also
    # requires both teams so absence of an away report cannot appear complete.
    if {f['team'] for f in valid if f['category'] == 'injuries'} != {'home', 'away'}:
        live['quality']['official_bet_allowed'] = False
        live['quality']['blockers'].append('injury_coverage_incomplete')
    return live


def build_match_context(match, facts, *, as_of, captured_at, tool_log=(), errors=()):
    cutoff = require_timestamp(as_of)
    kickoff = require_timestamp(match.get('kickoff') or match.get('match_time') or match.get('time'), 'kickoff')
    captured = require_timestamp(captured_at, 'captured_at')
    if cutoff > captured:
        raise ValueError('as_of cannot exceed actual snapshot capture time')
    audited = [audit_fact(f, as_of=cutoff, kickoff=kickoff) for f in facts]
    unique = {f.get('evidence_id', str(i)): f for i, f in enumerate(audited)}
    evidence = list(unique.values())
    conflicts = _conflicts(evidence)
    available = {f['category'] for f in evidence if f['verified']}
    for category in ('injuries', 'lineup', 'schedule'):
        if {f['team'] for f in evidence if f['verified'] and f['category'] == category} != {'home', 'away'}:
            available.discard(category)
    missing = [c for c in CATEGORIES if c not in available]
    return {
        'schema_version': SCHEMA_VERSION, 'match_id': str(match['match_id']),
        'home': match.get('home', ''), 'away': match.get('away', ''),
        'kickoff': kickoff.isoformat(), 'as_of': cutoff.isoformat(), 'captured_at': captured.isoformat(),
        'status': 'complete' if not missing else 'partial' if available else 'unavailable',
        'evidence': evidence, 'missing_categories': missing, 'conflicts': conflicts,
        'tool_log': list(tool_log), 'errors': list(errors),
        'live_context': facts_to_live_context(evidence, as_of=cutoff, conflicts=conflicts),
    }
