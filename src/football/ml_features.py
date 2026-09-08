"""Shared, explicit v2 features for training and live football predictions.

No imputation is performed here. Missing observations remain missing and make
the ML path unavailable; the statistical/market baseline can still run.
"""

from datetime import datetime, timezone
import math
from numbers import Real

from .ml_feature_schema import FEATURE_VERSION, get_feature_names

FEATURE_BUILDER_VERSION = 'v2-strict-2'
MIN_TEAM_HISTORY = 10
MIN_VENUE_HISTORY = 5


def audit_prediction_features(features):
    names = get_feature_names()
    payload = features if isinstance(features, dict) else {}
    missing = sorted(set(names) - payload.keys())
    unknown = sorted(payload.keys() - set(names), key=str)
    invalid = {}
    for name in set(names) & payload.keys():
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            invalid[name] = 'finite_number_required'
        elif name.startswith('has_') or name == 'is_home_favorite':
            if value not in (0, 1):
                invalid[name] = 'binary_flag_required'
        elif name.endswith('_count'):
            if value < 0 or int(value) != value:
                invalid[name] = 'nonnegative_integer_required'
            elif name in ('home_matches_count', 'away_matches_count') and value < MIN_TEAM_HISTORY:
                invalid[name] = 'at_least_10_observed_matches_required'
        elif 'prob' in name or '_rate_' in name:
            if not 0 <= value <= 1:
                invalid[name] = 'probability_out_of_range'
        elif name not in ('elo_diff', 'asian_handicap') and value < 0:
            invalid[name] = 'negative_measurement'
    for group in (('euro_home_prob', 'euro_draw_prob', 'euro_away_prob'),
                  ('asian_home_prob', 'asian_away_prob'), ('over_prob', 'under_prob')):
        if all(name in payload and name not in invalid for name in group):
            if abs(sum(payload[name] for name in group) - 1.0) > 1e-5:
                for name in group:
                    invalid[name] = 'probabilities_must_sum_to_one'
    if all(name in payload and name not in invalid for name in ('elo_home', 'elo_away', 'elo_diff')):
        if abs(payload['elo_home'] - payload['elo_away'] - payload['elo_diff']) > 1e-5:
            invalid['elo_diff'] = 'inconsistent_elo_difference'
    return {'feature_version': FEATURE_VERSION, 'expected_count': len(names),
            'supplied_count': len(payload), 'missing': missing, 'unknown': unknown,
            'invalid': invalid, 'complete': not missing and not unknown and not invalid}


def feature_vector(features, feature_names=None):
    """The same ordered, checked vector is used by train and predict."""
    names = get_feature_names()
    if feature_names is not None and list(feature_names) != names:
        raise ValueError('incompatible_feature_order')
    audit = audit_prediction_features(features)
    if not audit['complete']:
        raise ValueError(f'invalid_feature_payload: {audit}')
    return [float(features[name]) for name in names]


def _time(value, *, allow_date=False):
    try:
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            # Day-only historical fixtures have an explicit, conservative date
            # convention. Live observations never infer a missing timezone.
            if allow_date and len(str(value)) == 10:
                return parsed.replace(tzinfo=timezone.utc)
            return None
        return parsed
    except (ValueError, TypeError):
        return None


def _number(value):
    return (float(value) if isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(float(value)) else None)


def _dated_history(rows, cutoff):
    if not cutoff:
        return []
    output = []
    seen = set()
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        played = _time(row.get('date') or row.get('match_time'), allow_date=True)
        gf, ga = _number(row.get('goals_for')), _number(row.get('goals_against'))
        observed = _time(row.get('observed_at'))
        if (not played or played >= cutoff or not observed or not played < observed <= cutoff
                or gf is None or ga is None or min(gf, ga) < 0 or gf != int(gf) or ga != int(ga)):
            continue
        identity = row.get('match_id') or (played.isoformat(), row.get('opponent'), row.get('venue'), gf, ga)
        if identity in seen:
            continue
        seen.add(identity)
        output.append({**row, 'date': played, 'goals_for': gf, 'goals_against': ga})
    return sorted(output, key=lambda row: row['date'])


def histories_from_records(records, home, away, league, *, as_of):
    """Use settled observations, preserving their result-observation time."""
    result = {'home_history': [], 'away_history': [], 'league_history': []}
    cutoff = _time(as_of)
    for record in records or []:
        if (not isinstance(record, dict) or not record.get('settled')
                or record.get('exclude_from_calibration') or record.get('exclude_from_stats')
                or record.get('skip_training_ingest')):
            continue
        quality = record.get('result_quality') or {}
        if (not isinstance(quality, dict) or quality.get('usable_for_calibration') is False
                or not (quality.get('usable_for_calibration') is True or quality.get('grade') in ('high', 'medium'))):
            continue
        try:
            hg, ag = map(int, str(record.get('actual_score')).split('-'))
        except (TypeError, ValueError):
            continue
        if min(hg, ag) < 0:
            continue
        actual = 'H' if hg > ag else 'A' if hg < ag else 'D'
        if record.get('actual_result') not in (None, actual):
            continue
        played = record.get('match_time')
        if record.get('prediction_events') or record.get('selected_prediction_event_id'):
            from ..domain.sports.football.prediction_evaluation import selected_event
            event, reason = selected_event(record)
            if reason:
                continue
            played = event['kickoff_at']
        observed = record.get('result_observed_at') or record.get('settled_at') or record.get('result_updated_at')
        observed_time = _time(observed)
        played_time = _time(played, allow_date=True)
        if (not cutoff or not observed_time or not played_time
                or not played_time < observed_time <= cutoff):
            continue
        base = {'match_id': record.get('match_id'), 'date': played,
                'observed_at': observed}
        for side, team_name in (('home', home), ('away', away)):
            if team_name and team_name == record.get('home'):
                result[side + '_history'].append({**base, 'goals_for': hg, 'goals_against': ag,
                                                 'venue': 'home', 'opponent': record.get('away')})
            elif team_name and team_name == record.get('away'):
                result[side + '_history'].append({**base, 'goals_for': ag, 'goals_against': hg,
                                                 'venue': 'away', 'opponent': record.get('home')})
        if league and league == record.get('league'):
            result['league_history'].append({**base, 'goals_for': hg, 'goals_against': ag})
    return result


def build_prediction_features(*, euro=None, asian=None, total=None, team=None,
                              league_profile=None, match_time=None, feature_snapshot=None,
                              history_records=(), home=None, away=None, league=None, as_of=None):
    """Build observable v2 features; aggregates are never relabeled as 5/10 games.

    Training rows can supply ``feature_snapshot`` (their canonical features).
    Live callers supply market analyses and date-stamped ``home_history`` /
    ``away_history`` in team, and ``history`` in league_profile. The cutoff is
    exclusive, and current/future matches are excluded.
    """
    features = dict(feature_snapshot) if isinstance(feature_snapshot, dict) else {}
    team, league_profile = team or {}, league_profile or {}
    target, observation_cutoff = _time(match_time), _time(as_of)
    cutoff = min(target, observation_cutoff) if target and observation_cutoff else None
    if history_records:
        observed = histories_from_records(history_records, home, away, league, as_of=as_of)
        team = {**team, **{key: value for key, value in observed.items() if value}}
    if feature_snapshot is None:
        for name in ('elo_home', 'elo_away'):
            if _number(team.get(name)) is not None:
                features[name] = float(team[name])
        if 'elo_home' in features and 'elo_away' in features:
            features['elo_diff'] = features['elo_home'] - features['elo_away']

        for market, flag, source_keys, target_keys in (
            (euro, 'has_euro_odds', ('home', 'draw', 'away'), ('euro_home_prob', 'euro_draw_prob', 'euro_away_prob')),
            (asian, 'has_asian_odds', ('home', 'away'), ('asian_home_prob', 'asian_away_prob')),
            (total, 'has_total_odds', ('over', 'under'), ('over_prob', 'under_prob')),
        ):
            market = market or {}
            source = market.get('close' if flag == 'has_euro_odds' else 'close_prob') or {}
            if flag == 'has_asian_odds':
                source = {'home': source.get('home', source.get('home_give', source.get('home_recv'))),
                          'away': source.get('away', source.get('away_recv', source.get('away_give')))}
            values = [_number(source.get(key)) for key in source_keys]
            available = (market.get('source') != 'model_proxy'
                         and market.get('source_matched') is not False
                         and all(value is not None and 0 <= value <= 1 for value in values)
                         and sum(values) > 0)
            features[flag] = int(available)
            if available:
                features.update({key: value / sum(values) for key, value in zip(target_keys, values)})
        if features.get('has_euro_odds'):
            features['is_home_favorite'] = int(features['euro_home_prob'] >= features['euro_away_prob'])
        if features.get('has_asian_odds') and _number((asian or {}).get('handicap')) is not None:
            # Analysis uses positive = home gives; training's home-view line is negative.
            features['asian_handicap'] = -float(asian['handicap'])
        if features.get('has_total_odds') and _number((total or {}).get('close_line')) is not None:
            features['total_line'] = float(total['close_line'])

        for side, venue in (('home', 'home'), ('away', 'away')):
            history = _dated_history(team.get(side + '_history'), cutoff)
            if not history:
                continue
            features[side + '_matches_count'] = len(history)
            for window in (5, 10):
                recent = history[-window:]
                n = len(recent)
                if n < window:
                    continue
                gf = sum(row['goals_for'] for row in recent) / n
                ga = sum(row['goals_against'] for row in recent) / n
                wins = sum(row['goals_for'] > row['goals_against'] for row in recent) / n
                draws = sum(row['goals_for'] == row['goals_against'] for row in recent) / n
                features[f'{side}_win_rate_{window}'] = wins
                if window == 5:
                    features.update({f'{side}_attack_5': gf, f'{side}_defense_5': ga,
                                     f'{side}_form_points_5': 3 * wins + draws,
                                     f'{side}_draw_rate_5': draws})
                else:
                    features.update({f'{side}_goals_for_10': gf, f'{side}_goals_against_10': ga})
            venue_rows = [row for row in history if row.get('venue') == venue][-5:]
            if len(venue_rows) >= MIN_VENUE_HISTORY:
                prefix = 'home_h' if side == 'home' else 'away_a'
                features[prefix + '_goals_for_5'] = sum(row['goals_for'] for row in venue_rows) / len(venue_rows)
                features[prefix + '_goals_against_5'] = sum(row['goals_against'] for row in venue_rows) / len(venue_rows)
        league_rows = _dated_history(league_profile.get('history') or team.get('league_history'), cutoff)[-100:]
        if league_rows:
            features['league_avg_goals_100'] = sum(row['goals_for'] + row['goals_against'] for row in league_rows) / len(league_rows)
            features['league_draw_rate_100'] = sum(row['goals_for'] == row['goals_against'] for row in league_rows) / len(league_rows)
    audit = audit_prediction_features(features)
    return {'features': features, 'audit': audit, 'available': audit['complete'], 'feature_version': FEATURE_VERSION}
