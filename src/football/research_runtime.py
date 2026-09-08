"""Optional adapters around the existing prediction pipeline, with no UI changes."""
from __future__ import annotations

from datetime import datetime, timezone
import os

from .research import kickoff_timestamp, timestamp


def team_strength_from_history(records, match, *, as_of):
    """Use actual, timestamped local results when an official fixture lacks a 500 ID.

    Exact team names only. No fabricated xG, Elo, or assumed healthy lineup.
    """
    from .ml_features import _dated_history, histories_from_records

    cutoff = timestamp(as_of)
    if cutoff is None:
        return None
    kickoff = kickoff_timestamp(match, now=cutoff)
    if not kickoff:
        return None
    teams = [match.get('home'), match.get('away')]
    if not all(teams) or teams[0] == teams[1]:
        return None
    # Share ML's actual-result observation time, exclusion flags, score/result
    # agreement, and selected frozen-event kickoff. In particular, historical
    # month/day strings must never be assigned a year from today's fixture.
    observed = histories_from_records(records, teams[0], teams[1], match.get('league'), as_of=cutoff)
    history = {team: [] for team in teams}
    for side, name in zip(('home', 'away'), teams):
        seen = set()
        for row in _dated_history(observed[side + '_history'], min(cutoff, kickoff)):
            played = row['date']
            # Also collapse the same fixture stored under two source IDs.
            event_key = (played.isoformat(), row.get('opponent'), row.get('venue'))
            if event_key in seen:
                continue
            seen.add(event_key)
            gf, ga = int(row['goals_for']), int(row['goals_against'])
            history[name].append({'played_at': played, 'gf': gf, 'ga': ga,
                                  'points': 3 if gf > ga else 1 if gf == ga else 0,
                                  'venue': row.get('venue')})
    if any(len(items) < 5 for items in history.values()):
        return None
    result = {'data_provenance': {'team_form': {
        'kind': 'historical_results', 'source': 'local_settled_prediction_records',
        'collected_at': cutoff.isoformat(), 'timestamp_basis': 'result_available_before_as_of'}}}
    for side, name in zip(('home', 'away'), teams):
        recent = sorted(history[name], key=lambda row: row['played_at'], reverse=True)[:10]
        result[f'{side}_recent'] = {'games': len(recent), 'gf': sum(r['gf'] for r in recent),
                                   'ga': sum(r['ga'] for r in recent),
                                   'form_pts': sum(r['points'] for r in recent)}
        result[f'attack_{side}'] = max(.05, sum(r['gf'] for r in recent)/len(recent))
        result[f'defense_{side}'] = max(.05, sum(r['ga'] for r in recent)/len(recent))
    return result


def completed_intelligence(match, *, as_of):
    """Read completed research only; schedule missing work without waiting for it."""
    try:
        from .intelligence import get_cached_context, submit_research
        kickoff = kickoff_timestamp(match, now=as_of)
        if not kickoff or kickoff <= as_of:
            return None
        research_match = {**match, 'kickoff_at': kickoff.isoformat(), 'kickoff': kickoff.isoformat()}
        cached = get_cached_context(str(match['match_id']), as_of=as_of, kickoff=kickoff.isoformat())
        if os.getenv('FOOTBALL_INTELLIGENCE_ENABLED', '1').lower() not in ('0', 'false', 'off'):
            submit_research(research_match)
        return cached
    except Exception:
        return None


def ml_runtime_fusion(candidates, response, *, as_of):
    from .research import apply_qualified_ml
    stats = {}
    if (response or {}).get('available'):
        try:
            from .result_sync import get_history
            stats = get_history().get_ml_evaluation_stats(
                model_version=response.get('model_version'), as_of=as_of,
                include_confidence_intervals=False)
        except Exception:
            pass
    try:
        return apply_qualified_ml(candidates, response, as_of=as_of, stats=stats,
                                  metadata={key: (response or {}).get(key) for key in ('test_count', 'training_cutoff_at')},
                                  enabled=os.getenv('FOOTBALL_ML_FUSION_ENABLED', '1').lower() not in ('0', 'false', 'off'))
    except Exception as exc:
        return apply_qualified_ml(candidates, {'available': False,
                                  'reason': 'fusion_error:' + type(exc).__name__}, as_of=as_of)
