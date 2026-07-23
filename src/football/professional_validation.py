"""Leakage-resistant evaluation utilities for football betting models.

The module intentionally separates probability quality from betting returns.
Records must be in chronological order and thresholds are selected using only
the training side of each walk-forward fold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


LABELS = ('H', 'D', 'A')


def normalize_probabilities(values: Dict[str, float]) -> Dict[str, float]:
    clean = {label: max(0.0, float(values.get(label, 0.0) or 0.0)) for label in LABELS}
    total = sum(clean.values())
    if total <= 0:
        return {label: 1.0 / 3.0 for label in LABELS}
    return {label: clean[label] / total for label in LABELS}


def probabilities_from_odds(odds: Dict[str, float]) -> Dict[str, float]:
    inverse = {
        label: (1.0 / float(odds[label]) if float(odds.get(label, 0.0) or 0.0) > 1 else 0.0)
        for label in LABELS
    }
    return normalize_probabilities(inverse)


def multiclass_metrics(records: Sequence[Dict], probability_key: str = 'probabilities') -> Dict:
    if not records:
        return {'n': 0, 'accuracy': 0.0, 'logloss': 0.0, 'brier': 0.0}
    hits = 0
    logloss = 0.0
    brier = 0.0
    for record in records:
        probs = normalize_probabilities(record[probability_key])
        actual = record['actual']
        hits += max(LABELS, key=lambda label: probs[label]) == actual
        logloss -= math.log(max(1e-15, probs[actual]))
        brier += sum((probs[label] - (1.0 if label == actual else 0.0)) ** 2 for label in LABELS)
    n = len(records)
    return {'n': n, 'accuracy': hits / n, 'logloss': logloss / n, 'brier': brier / n}


def _max_drawdown(equity: Iterable[float]) -> float:
    peak = 0.0
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        worst = max(worst, peak - value)
    return worst


def evaluate_strategy(
    records: Sequence[Dict],
    min_probability: float = 0.0,
    min_edge: float = 0.0,
    stake: float = 1.0,
) -> Dict:
    """Settle flat-stake 1X2 bets and report returns and closing-line value."""
    profit = 0.0
    equity = []
    bets = wins = 0
    clv_values: List[float] = []
    for record in records:
        probs = normalize_probabilities(record['probabilities'])
        market = probabilities_from_odds(record['odds'])
        pick = max(LABELS, key=lambda label: probs[label] - market[label])
        edge = probs[pick] - market[pick]
        offered = float(record['odds'].get(pick, 0.0) or 0.0)
        if offered <= 1.0 or probs[pick] < min_probability or edge < min_edge:
            continue
        bets += 1
        won = record['actual'] == pick
        wins += won
        profit += stake * (offered - 1.0) if won else -stake
        equity.append(profit)
        close = float((record.get('closing_odds') or {}).get(pick, 0.0) or 0.0)
        if close > 1.0:
            # Positive when the taken price beats the later closing price.
            clv_values.append(offered / close - 1.0)
    turnover = bets * stake
    return {
        'bets': bets,
        'wins': wins,
        'hit_rate': wins / bets if bets else 0.0,
        'coverage': bets / len(records) if records else 0.0,
        'profit': profit,
        'roi': profit / turnover if turnover else 0.0,
        'max_drawdown_units': _max_drawdown(equity),
        'mean_clv': sum(clv_values) / len(clv_values) if clv_values else None,
        'clv_samples': len(clv_values),
    }


def select_threshold(
    training_records: Sequence[Dict],
    probability_grid: Sequence[float] = (0.0, 0.45, 0.50, 0.55),
    edge_grid: Sequence[float] = (0.0, 0.02, 0.04, 0.06),
    min_bets: int = 30,
) -> Dict:
    """Choose a threshold on past data only, penalising tiny samples."""
    candidates = []
    for probability in probability_grid:
        for edge in edge_grid:
            result = evaluate_strategy(training_records, probability, edge)
            if result['bets'] >= min_bets:
                score = result['roi'] - 0.002 * result['max_drawdown_units']
                candidates.append((score, probability, edge, result))
    if not candidates:
        return {'min_probability': 1.0, 'min_edge': 1.0, 'training': evaluate_strategy([])}
    _, probability, edge, result = max(candidates, key=lambda item: item[0])
    return {'min_probability': probability, 'min_edge': edge, 'training': result}


def walk_forward_evaluate(
    records: Sequence[Dict],
    initial_train: int,
    test_size: int,
    min_training_bets: int = 30,
) -> Dict:
    """Expanding-window evaluation with thresholds frozen before each test fold."""
    ordered = sorted(records, key=lambda record: (record.get('date', ''), record.get('match_id', '')))
    folds = []
    out_of_sample: List[Dict] = []
    start = initial_train
    while start < len(ordered):
        test = ordered[start:start + test_size]
        if not test:
            break
        selected = select_threshold(ordered[:start], min_bets=min_training_bets)
        settled = evaluate_strategy(
            test,
            selected['min_probability'],
            selected['min_edge'],
        )
        folds.append({
            'train_n': start,
            'test_n': len(test),
            'test_start': test[0].get('date'),
            'test_end': test[-1].get('date'),
            'min_probability': selected['min_probability'],
            'min_edge': selected['min_edge'],
            'training_roi': selected['training']['roi'],
            'test': settled,
        })
        for record in test:
            copied = dict(record)
            copied['_fold_min_probability'] = selected['min_probability']
            copied['_fold_min_edge'] = selected['min_edge']
            out_of_sample.append(copied)
        start += test_size

    # Settle each OOS fold using its frozen threshold, then aggregate cashflows.
    aggregate_parts = [
        evaluate_strategy(
            [record],
            record['_fold_min_probability'],
            record['_fold_min_edge'],
        )
        for record in out_of_sample
    ]
    bets = sum(part['bets'] for part in aggregate_parts)
    wins = sum(part['wins'] for part in aggregate_parts)
    profit = sum(part['profit'] for part in aggregate_parts)
    clv = [part['mean_clv'] for part in aggregate_parts if part['mean_clv'] is not None]
    running_profit = 0.0
    equity = []
    for part in aggregate_parts:
        if part['bets']:
            running_profit += part['profit']
            equity.append(running_profit)
    strategy = {
        'bets': bets,
        'wins': wins,
        'hit_rate': wins / bets if bets else 0.0,
        'coverage': bets / len(out_of_sample) if out_of_sample else 0.0,
        'profit': profit,
        'roi': profit / bets if bets else 0.0,
        'max_drawdown_units': _max_drawdown(equity),
        'mean_clv': sum(clv) / len(clv) if clv else None,
        'clv_samples': len(clv),
    }
    market_records = [
        {**record, 'market_probabilities': probabilities_from_odds(record['odds'])}
        for record in out_of_sample
    ]
    return {
        'method': 'expanding-window-walk-forward',
        'out_of_sample_n': len(out_of_sample),
        'folds': folds,
        'model_metrics': multiclass_metrics(out_of_sample),
        'market_baseline_metrics': multiclass_metrics(market_records, 'market_probabilities'),
        'strategy': strategy,
    }
