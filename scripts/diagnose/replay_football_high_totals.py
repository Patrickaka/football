#!/usr/bin/env python
"""Read-only comparison of fixed-2.5 O/U prices and high-total probabilities.

The CSV's non-closing prices are not verified opening snapshots. Closing
prices arrive later and cannot be used to backfill an earlier prediction.
This descriptive comparison never trains, tunes, or modifies business data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.sports.football.markets import (
    fair_over_probability, implied_total_goals, remove_vig,
)

DIVISIONS = ('E0', 'SP1', 'D1', 'I1', 'F1')
SEASONS = ('2425', '2526')
TAIL_THRESHOLDS = (4, 5, 6)
HIGH_ENVIRONMENT_MEAN = 3.5
PRICE_COLUMNS = {
    'non_closing': ('Avg>2.5', 'Avg<2.5'),
    'closing': ('AvgC>2.5', 'AvgC<2.5'),
}


def load_csv_rows(data_dir):
    """Keep one complete paired observation per league/date/home/away."""
    rows, identities, files, exclusions = [], set(), [], Counter()
    for league in DIVISIONS:
        for season in SEASONS:
            path = Path(data_dir) / f'{league}_{season}.csv'
            if not path.exists():
                exclusions['missing_csv'] += 1
                continue
            files.append({'name': path.name,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            with path.open(encoding='utf-8-sig', newline='') as handle:
                for source in csv.DictReader(handle):
                    try:
                        date = datetime.strptime(source['Date'], '%d/%m/%Y').date().isoformat()
                        home, away = source['HomeTeam'].strip(), source['AwayTeam'].strip()
                        if not home or not away or home == away:
                            raise ValueError('missing or identical teams')
                        goals = [int(source[key]) for key in ('FTHG', 'FTAG')]
                        if min(goals) < 0:
                            raise ValueError('negative goals')
                        prices = {name: [float(source[key]) for key in columns]
                                  for name, columns in PRICE_COLUMNS.items()}
                        if any(not math.isfinite(p) or p <= 1
                               for values in prices.values() for p in values):
                            raise ValueError('invalid odds')
                    except (KeyError, TypeError, ValueError, AttributeError):
                        exclusions['invalid_or_missing_match_data'] += 1
                        continue
                    identity = (league, date, home, away)
                    if identity in identities:
                        exclusions['duplicate_match'] += 1
                        continue
                    identities.add(identity)
                    rows.append({'match_id': '|'.join(identity), 'date': date,
                                 'league': league, 'season': season,
                                 'total_line': 2.5, 'prices': prices,
                                 'actual_goals': sum(goals)})
    rows.sort(key=lambda row: (row['date'], row['match_id']))
    return rows, {'files': files, 'exclusions': dict(exclusions), 'valid_rows': len(rows)}


def price_projection(prices):
    """Only the price pair reaches the model; outcomes never reach inversion."""
    over, under = map(float, prices)
    if any(not math.isfinite(p) or p <= 1 for p in (over, under)):
        raise ValueError('invalid odds')
    fair_over = remove_vig(over, under)[0]
    mean = implied_total_goals(2.5, fair_over)
    # k - .5 is a half-goal line: no push/half-win conditioning is involved.
    tails = {str(k): fair_over_probability(mean, k - .5) for k in TAIL_THRESHOLDS}
    return {'fair_over_2_5': fair_over, 'implied_mean': mean, 'tail_probabilities': tails}


def summarize(rows):
    n = len(rows)
    report = {'n': n, 'independent_dates': len({row['date'] for row in rows}),
              'mean_actual_goals': sum(row['actual_goals'] for row in rows) / n if n else None,
              'tails': {}}
    for k in TAIL_THRESHOLDS:
        events = sum(row['actual_goals'] >= k for row in rows)
        tail = {'actual_events': events, 'actual_rate': events / n if n else None}
        for name in PRICE_COLUMNS:
            pairs = [(row['projections'][name]['tail_probabilities'][str(k)],
                      int(row['actual_goals'] >= k)) for row in rows]
            tail[name] = {
                'mean_probability': sum(p for p, _ in pairs) / n if n else None,
                'brier': sum((p - y) ** 2 for p, y in pairs) / n if n else None,
                'logloss': sum(-(y * math.log(max(p, 1e-12))
                                 + (1 - y) * math.log(max(1 - p, 1e-12)))
                               for p, y in pairs) / n if n else None,
            }
        tail['paired_closing_improvement'] = {
            metric: tail['non_closing'][metric] - tail['closing'][metric] if n else None
            for metric in ('brier', 'logloss')
        }
        report['tails'][str(k)] = tail
    return report


def cohort_report(rows):
    result = {'all': summarize(rows), 'high_environment_groups': {}}
    for name in PRICE_COLUMNS:
        selected = [row for row in rows
                    if row['projections'][name]['implied_mean'] >= HIGH_ENVIRONMENT_MEAN]
        result['high_environment_groups'][name] = {
            'selected_by': name + '_implied_mean',
            'threshold': HIGH_ENVIRONMENT_MEAN,
            'coverage': len(selected) / len(rows) if rows else None,
            **summarize(selected),
        }
    return result


def replay(rows):
    if len({row['match_id'] for row in rows}) != len(rows):
        raise ValueError('duplicate match_id in replay input')
    observations = []
    for row in sorted(rows, key=lambda value: (value['date'], value['match_id'])):
        if row.get('total_line') != 2.5:
            raise ValueError('only observed fixed 2.5-goal contracts are supported')
        goals = row['actual_goals']
        if isinstance(goals, bool) or not isinstance(goals, int) or goals < 0:
            raise ValueError('invalid actual goals')
        observations.append({key: row[key] for key in ('match_id', 'date', 'league', 'season', 'actual_goals')}
                            | {'projections': {name: price_projection(row['prices'][name])
                                               for name in PRICE_COLUMNS}})
    return {
        'schema_version': 'football-high-totals-csv-replay-v1',
        'evaluation_scope': 'descriptive_fixed_2_5_market_tail_probability_comparison',
        'release_qualified': False,
        'policy': {'parameters_tuned': False, 'tail_thresholds': list(TAIL_THRESHOLDS),
                   'high_environment_mean': HIGH_ENVIRONMENT_MEAN,
                   'price_columns': PRICE_COLUMNS, 'observed_total_line': 2.5,
                   'probability_model': 'existing_poisson_fair_price_inversion',
                   'paired_improvement_sign': 'positive means closing has lower loss'},
        'limitations': [
            'Non-closing CSV prices are not verified opening quotes and have no collection timestamps.',
            'Closing prices arrive later; this is not evidence of uplift at an identical prediction time.',
            'Both quoted markets are fixed at 2.5 goals; actual O/U line changes are unavailable.',
            'No rise-in-line strategy or cross-line movement benefit can be validated by these CSV columns.',
            'High-environment groups are descriptive; a closing-selected group is unavailable at the earlier snapshot.',
            'Tail probabilities use the existing Poisson assumption; they are not probabilities of any single exact score.',
            'No calibrated production matrix, team data, trained correction, parameter search or release gate is included.',
            'Only five leagues and two already-observed seasons are covered; point estimates have no significance claim.',
        ],
        'all': cohort_report(observations),
        'by_season': {season: cohort_report([row for row in observations if row['season'] == season])
                      for season in sorted({row['season'] for row in observations})},
        'by_league': {league: cohort_report([row for row in observations if row['league'] == league])
                      for league in sorted({row['league'] for row in observations})},
        'by_season_and_league': {
            season: {league: cohort_report([row for row in observations
                                            if row['season'] == season and row['league'] == league])
                     for league in sorted({row['league'] for row in observations if row['season'] == season})}
            for season in sorted({row['season'] for row in observations})
        },
        'observations': observations,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, help='optional JSON report; no business data are modified')
    args = parser.parse_args(argv)
    rows, audit = load_csv_rows(args.data_dir)
    if not rows:
        parser.error('no complete paired local CSV matches')
    report = replay(rows)
    report['input_audit'] = audit
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
        print(str(args.output.resolve()))
        print(json.dumps(report['by_season'], ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
