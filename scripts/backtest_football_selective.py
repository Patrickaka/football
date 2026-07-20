#!/usr/bin/env python
"""Tune the football 1X2 official-pick gate on pre-match market probabilities."""

import csv
import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
LEAGUES = ('E0', 'SP1', 'D1', 'I1', 'F1')


def _number(row, key):
    try:
        return float(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def load_rows():
    rows = []
    for league in LEAGUES:
        for season in ('2425', '2526'):
            path = os.path.join(DATA, f'{league}_{season}.csv')
            if not os.path.exists(path):
                continue
            with open(path, encoding='utf-8-sig') as handle:
                for row in csv.DictReader(handle):
                    odds = [_number(row, key) for key in ('AvgH', 'AvgD', 'AvgA')]
                    if not all(value > 1 for value in odds):
                        continue
                    inv = [1 / value for value in odds]
                    total = sum(inv)
                    probs = [value / total for value in inv]
                    result = row.get('FTR')
                    if result in ('H', 'D', 'A'):
                        rows.append((probs, result))
    return rows


def evaluate(rows, min_probability, min_margin):
    hits = total = 0
    labels = ('H', 'D', 'A')
    for probs, actual in rows:
        ranked = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)
        if ranked[0][1] < min_probability or ranked[0][1] - ranked[1][1] < min_margin:
            continue
        total += 1
        hits += labels[ranked[0][0]] == actual
    return {
        'min_probability': min_probability,
        'min_margin': min_margin,
        'total': total,
        'accuracy': hits / total if total else 0.0,
        'coverage': total / len(rows) if rows else 0.0,
    }


def main():
    rows = load_rows()
    candidates = [
        evaluate(rows, probability, margin)
        for probability in (0.48, 0.50, 0.52, 0.54, 0.56)
        for margin in (0.10, 0.12, 0.15, 0.18, 0.20)
    ]
    # Require useful coverage, then maximize accuracy; coverage breaks close ties.
    eligible = [item for item in candidates if item['coverage'] >= 0.30]
    eligible.sort(key=lambda item: (item['accuracy'], item['coverage']), reverse=True)
    print(f'样本: {len(rows)}')
    for item in eligible[:10]:
        print(
            f"p>={item['min_probability']:.2f} margin>={item['min_margin']:.2f} "
            f"accuracy={item['accuracy']:.2%} coverage={item['coverage']:.2%} n={item['total']}"
        )


if __name__ == '__main__':
    main()
