#!/usr/bin/env python
"""Read-only, date-ordered ablation of the static market score learning chain.

This is not a replay of the complete production predictor or a release gate.
CSV prices lack original collection timestamps. No database, trained artifact,
remote source, or prediction history is read or written. Only an explicitly
requested report file is written.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.sports.football.lambdas import euro_implied_supremacy
from src.domain.sports.football.markets import implied_total_goals, remove_vig
from src.domain.sports.football.prediction_evaluation import (
    _paired_scores, _score_summary, score_metrics,
)
from src.domain.sports.football.scoring import fit_lambdas_from_markets
from src.domain.sports.football.scoring_model import build_score_matrix, calibrate_to_euro

DIVISIONS = ('E0', 'SP1', 'D1', 'I1', 'F1')
SEASONS = ('2425', '2526')
TEST_START = '2025-07-01'
STATIC_CAP = 0.15
FEATURE_COLUMNS = ('AvgH', 'AvgD', 'AvgA', 'AHh', 'Avg>2.5', 'Avg<2.5')


def load_csv_rows(data_dir):
    """Use only dated, complete, distinct matches and non-closing price columns."""
    rows, exclusions, files, identities = [], Counter(), [], set()
    for division in DIVISIONS:
        for season in SEASONS:
            path = Path(data_dir) / f'{division}_{season}.csv'
            if not path.exists():
                exclusions['missing_csv'] += 1
                continue
            files.append({'name': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            with path.open(encoding='utf-8-sig', newline='') as handle:
                for source in csv.DictReader(handle):
                    try:
                        date = datetime.strptime(source['Date'], '%d/%m/%Y').date().isoformat()
                        home, away = source['HomeTeam'].strip(), source['AwayTeam'].strip()
                        if not home or not away:
                            raise ValueError('missing team')
                        home_goals, away_goals = int(source['FTHG']), int(source['FTAG'])
                        if min(home_goals, away_goals) < 0:
                            raise ValueError('invalid result')
                        features = {key: float(source[key]) for key in FEATURE_COLUMNS}
                        if any(not math.isfinite(value) for value in features.values()):
                            raise ValueError('nonfinite feature')
                        if any(value <= 1 for key, value in features.items() if key != 'AHh'):
                            raise ValueError('invalid price')
                    except (KeyError, TypeError, ValueError):
                        exclusions['invalid_or_missing_match_data'] += 1
                        continue
                    identity = (division, date, home, away)
                    if identity in identities:
                        exclusions['duplicate_match'] += 1
                        continue
                    identities.add(identity)
                    rows.append({'match_id': '_'.join(identity), 'date': date,
                                 'league': division, 'season': season, 'features': features,
                                 # Football-Data AHh has the opposite sign to the app.
                                 'asian': -features['AHh'], 'total_line': 2.5,
                                 'actual_score': f'{home_goals}-{away_goals}'})
    rows.sort(key=lambda row: (row['date'], row['match_id']))
    return rows, {'files': files, 'exclusions': dict(exclusions), 'valid_rows': len(rows)}


def baseline_matrix(row):
    """Existing Poisson/DC market fitting, with one fixed generic league profile.

Only the feature dictionary reaches model fitting. No later team history,
league calibration, residual model, or in-memory result is an input.
"""
    features = row['features']
    ph, pd, pa = remove_vig(*(features[key] for key in ('AvgH', 'AvgD', 'AvgA')))
    p_over, _ = remove_vig(features['Avg>2.5'], features['Avg<2.5'])
    total = implied_total_goals(2.5, p_over)
    supremacy = euro_implied_supremacy(ph, pd, pa, total)
    lh, la, _, rho = fit_lambdas_from_markets(
        supremacy, 2.5, p_over, ph, pd, pa, handicap=row['asian'],
        league_profile={'avg_goal': 1.42, 'home_boost': 1.06,
                        'low_score': .92, 'draw_mult': 1.0},
    )
    matrix = build_score_matrix(lh, la, rho=rho, distribution='poisson')
    return calibrate_to_euro(matrix, ph, pd, pa)


def summarize(rows, *, confidence_intervals=False):
    result = {'n': len(rows), 'independent_days': len({row['date'] for row in rows}),
              'history_prior_applied_n': sum(row['prior_applied'] for row in rows)}
    for name in ('baseline', 'learned'):
        result[name] = _score_summary([
            {'_score_evaluation': row['_score_evaluations'][name]} for row in rows
        ])
    result['paired'] = _paired_scores(rows, 'learned', 'baseline',
                                      draws=2000 if confidence_intervals else 0)
    result['paired']['topk_accuracy_difference'] = {
        f'top{k}': (result['learned'][f'top{k}_accuracy'] - result['baseline'][f'top{k}_accuracy'])
        if rows else None for k in (1, 3, 5, 10)
    }
    if not confidence_intervals:
        result['paired']['method'] = 'paired_kickoff_day_point_estimates'
    return result


def replay(rows, *, matrix_builder=baseline_matrix, prior_transform=None,
           confidence_intervals=True):
    """Freeze all predictions for a date before adding any results from that date."""
    from src.football.market_db import MarketScoreDB
    if prior_transform is None:
        from src.football.scoring import apply_static_score_prior
        prior_transform = apply_static_score_prior

    # Constructor state stays compatible with the production class; loading is
    # intentionally suppressed instead of temporarily emptying the real store.
    with patch.object(MarketScoreDB, '_load', return_value=None):
        database = MarketScoreDB()
    ordered = sorted(rows, key=lambda row: (row['date'], row['match_id']))
    if len({row['match_id'] for row in ordered}) != len(ordered):
        raise ValueError('duplicate match_id in replay input')
    observations, history_count, history_last_date = [], 0, None
    for date, group in groupby(ordered, key=lambda row: row['date']):
        matches = list(group)
        for row in matches:
            if history_last_date is not None and history_last_date >= date:
                raise AssertionError('history must be strictly earlier than the prediction date')
            base = matrix_builder({key: value for key, value in row.items() if key != 'actual_score'})
            market = database.get_prob_with_nearest(row['asian'], row['total_line'])
            # Newer DB queries return the matched bucket count. The fallback is
            # solely for compatibility with an exact-match query on old code.
            if 'sample_count' not in market:
                market['sample_count'] = (database.get_sample_count(row['asian'], row['total_line'])
                                          if market.get('exact_match') else 0)
            learned, trace = prior_transform(base, market, max_weight=STATIC_CAP, quality_factor=1.0)
            distributions = {}
            for name, matrix in (('baseline', base), ('learned', learned)):
                value = {f'{h}-{a}': float(probability) for (h, a), probability in matrix.items()}
                if not score_metrics(value, row['actual_score'])['score_distribution_valid']:
                    raise ValueError(f'{name} returned an incomplete score matrix for {row["match_id"]}')
                distributions[name] = value
            # A previously observed 8-0 can expand the learned support beyond
            # the model's 0..7 range. Give the baseline the identical declared
            # rectangle, with zero extra mass, so rank and scoring agree.
            support = set(distributions['baseline']) | set(distributions['learned'])
            metrics = {name: score_metrics({score: value.get(score, 0.0) for score in support},
                                          row['actual_score'])
                       for name, value in distributions.items()}
            observations.append({'match_id': row['match_id'], 'date': date, 'league': row['league'],
                                 'history_n': history_count, 'history_last_date': history_last_date,
                                 'prior_applied': bool(trace.get('applied')),
                                 '_score_evaluations': metrics})
        # Even an early kickoff on the same date cannot train a later kickoff.
        for row in matches:
            database.add_match_result(row['asian'], row['total_line'], row['actual_score'])
        history_count += len(matches)
        history_last_date = date

    test = [row for row in observations if row['date'] >= TEST_START]
    dates = sorted({row['date'] for row in test})
    middle = dates[len(dates) // 2] if dates else None
    return {
        'schema_version': 'football-static-score-prior-replay-v1',
        'evaluation_scope': 'isolated_static_score_prior_date_ordered_csv_ablation',
        'release_qualified': False,
        'policy': {'static_cap': STATIC_CAP, 'quality_factor': 1.0, 'test_start': TEST_START,
                   'parameters_tuned': False, 'same_date_results_excluded': True,
                   'feature_columns': list(FEATURE_COLUMNS), 'csv_total_market_line': 2.5,
                   'baseline': 'existing_poisson_dc_market_fit_generic_profile'},
        'limitations': [
            'Retrospective ordered ablation, not a frozen production forecast or untouched release test.',
            'CSV non-closing price columns have no original collection timestamps.',
            'Team form, injuries, lineups, news, trained calibration and other production stages are excluded.',
            'Only five leagues and the 2.5 total-goals price market are covered.',
            'Both arms use the same finite score support, expanded only by observed historical scores; outside results receive zero probability.',
            'Second-season predictions learn from strictly earlier dates, including earlier test dates; no parameter search.',
        ],
        'warmup_n': len(observations) - len(test),
        'all_walk_forward': summarize(observations),
        'second_season': summarize(test, confidence_intervals=confidence_intervals),
        'second_season_periods': {
            'first_half': summarize([row for row in test if row['date'] < middle]),
            'second_half': summarize([row for row in test if row['date'] >= middle]),
            'split_date': middle,
        } if middle else {},
        'second_season_by_league': {league: summarize([row for row in test if row['league'] == league])
                                    for league in sorted({row['league'] for row in test})},
        'observations': observations,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, help='optional JSON report; no business records are changed')
    args = parser.parse_args(argv)
    rows, audit = load_csv_rows(args.data_dir)
    if not rows:
        parser.error('no complete local CSV matches')
    report = replay(rows)
    report['input_audit'] = audit
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
        print(str(args.output.resolve()))
        print(json.dumps(report['second_season'], ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
