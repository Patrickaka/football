#!/usr/bin/env python
"""Chronological CatBoost backtest with price settlement and CLV.

No model sees the matches it predicts. Strategy thresholds are subsequently
selected from earlier out-of-sample predictions only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.football.ml_feature_schema import audit_feature_payload, get_feature_names
from src.football.professional_validation import walk_forward_evaluate


DATA = os.path.join(ROOT, 'data')
DIVISIONS = ('E0', 'SP1', 'D1', 'I1', 'F1')


def _float(row, name):
    try:
        return float(row.get(name) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def load_prices():
    prices = {}
    for division in DIVISIONS:
        for season in ('2425', '2526'):
            path = os.path.join(DATA, f'{division}_{season}.csv')
            if not os.path.exists(path):
                continue
            with open(path, encoding='utf-8-sig') as handle:
                for row in csv.DictReader(handle):
                    from src.football.ml_dataset_builder import parse_date
                    match_id = f"{division}_{parse_date(row.get('Date', ''))}_{row.get('HomeTeam')}_{row.get('AwayTeam')}"
                    offered = {label: _float(row, key) for label, key in zip(('H', 'D', 'A'), ('AvgH', 'AvgD', 'AvgA'))}
                    closing = {label: _float(row, key) for label, key in zip(('H', 'D', 'A'), ('AvgCH', 'AvgCD', 'AvgCA'))}
                    if all(value > 1 for value in offered.values()):
                        prices[match_id] = {'odds': offered, 'closing_odds': closing}
    return prices


def load_samples():
    path = os.path.join(DATA, 'ml_training_data.jsonl')
    with open(path, encoding='utf-8') as handle:
        samples = [json.loads(line) for line in handle if line.strip()]
    samples.sort(key=lambda sample: (sample['match_date'], sample['match_id']))
    audit = audit_feature_payload(samples[0]['features'])
    if not audit['complete']:
        raise RuntimeError(f'feature contract failed: {audit}')
    return samples


def generate_oos_predictions(samples, prices, warmup, fold_size):
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError('CatBoost is required for this backtest') from exc

    names = get_feature_names()
    labels = {'H': 0, 'D': 1, 'A': 2}
    output = []
    for start in range(warmup, len(samples), fold_size):
        train = samples[:start]
        test = samples[start:start + fold_size]
        x_train = np.asarray([[row['features'][name] for name in names] for row in train], dtype=float)
        y_train = np.asarray([labels[row['target']['result']] for row in train])
        x_test = np.asarray([[row['features'][name] for name in names] for row in test], dtype=float)
        model = CatBoostClassifier(
            iterations=250,
            depth=6,
            learning_rate=.05,
            loss_function='MultiClass',
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)
        for row, probs in zip(test, probabilities):
            price = prices.get(row['match_id'])
            if not price:
                continue
            output.append({
                'match_id': row['match_id'],
                'date': row['match_date'],
                'league': row['league'],
                'actual': row['target']['result'],
                'probabilities': {'H': float(probs[0]), 'D': float(probs[1]), 'A': float(probs[2])},
                **price,
            })
        print(f'fold train={len(train)} test={len(test)} oos_total={len(output)}')
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--warmup', type=int, default=1200)
    parser.add_argument('--fold-size', type=int, default=250)
    parser.add_argument('--threshold-train', type=int, default=500)
    parser.add_argument('--out', default=os.path.join(ROOT, 'reports', 'professional_football_backtest.json'))
    args = parser.parse_args()

    samples = load_samples()
    prices = load_prices()
    oos = generate_oos_predictions(samples, prices, args.warmup, args.fold_size)
    if len(oos) <= args.threshold_train:
        raise RuntimeError('not enough out-of-sample predictions for threshold validation')
    report = walk_forward_evaluate(
        oos,
        initial_train=args.threshold_train,
        test_size=args.fold_size,
        min_training_bets=30,
    )
    report['audit'] = {
        'model_split': 'expanding chronological window',
        'threshold_split': 'earlier out-of-sample predictions only',
        'offered_price_columns': ['AvgH', 'AvgD', 'AvgA'],
        'closing_price_columns': ['AvgCH', 'AvgCD', 'AvgCA'],
        'raw_samples': len(samples),
        'model_oos_samples': len(oos),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'model': report['model_metrics'], 'market': report['market_baseline_metrics'],
                      'strategy': report['strategy'], 'out': args.out}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
