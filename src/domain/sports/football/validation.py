# -*- coding: utf-8 -*-
"""策略验证：多分类指标、阈值选择、回撤、前进式评估。

**没有存储**：收记录、出指标。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


LABELS = ('H', 'D', 'A')

RQSPF_LABELS = ('让胜', '让平', '让负')

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

def _normalize_labeled(values: Dict[str, float], labels: Sequence[str]) -> Dict[str, float]:
    clean = {label: max(0.0, float((values or {}).get(label, 0.0) or 0.0)) for label in labels}
    total = sum(clean.values())
    return {label: clean[label] / total for label in labels} if total else {}

def _rqspf_actual(record: Dict) -> str | None:
    if record.get('actual_rqspf') in RQSPF_LABELS:
        return record['actual_rqspf']
    try:
        home, away = map(int, str(record.get('actual_score') or '').split('-'))
        margin = home + int(record.get('lottery_handicap')) - away
    except (TypeError, ValueError):
        return None
    return '让胜' if margin > 0 else ('让平' if margin == 0 else '让负')

def _rqspf_odds(record: Dict) -> Dict[str, float]:
    direct = record.get('rqspf_odds') or {}
    if direct:
        return direct
    snapshots = [record.get('odds_snapshot') or {}]
    timeline = record.get('market_timeline') or []
    if timeline:
        snapshots.insert(0, (timeline[-1].get('odds') or {}))
    for snapshot in snapshots:
        lottery = snapshot.get('lottery') or {}
        odds = lottery.get('rqspf_odds') or {}
        if odds:
            return odds
    return {}

def evaluate_rqspf_records(
    records: Sequence[Dict],
    min_probability: float = 0.0,
    min_edge: float = 0.0,
) -> Dict:
    """Independently settle official integer-handicap 3-way predictions."""
    rows = []
    for record in records:
        actual = _rqspf_actual(record)
        model = _normalize_labeled(record.get('predicted_rqspf') or {}, RQSPF_LABELS)
        odds = _rqspf_odds(record)
        inverse_odds = {}
        for label in RQSPF_LABELS:
            try:
                price = float(odds.get(label, 0.0) or 0.0)
            except (TypeError, ValueError):
                price = 0.0
            inverse_odds[label] = 1.0 / price if price > 1.0 else 0.0
        market = _normalize_labeled(inverse_odds, RQSPF_LABELS)
        if actual not in RQSPF_LABELS or not model or not market:
            continue
        rows.append((record, actual, model, market, odds))

    hits = market_hits = bets = wins = 0
    logloss = brier = market_logloss = market_brier = profit = 0.0
    for _, actual, model, market, odds in rows:
        model_pick = max(RQSPF_LABELS, key=model.get)
        market_pick = max(RQSPF_LABELS, key=market.get)
        hits += model_pick == actual
        market_hits += market_pick == actual
        logloss -= math.log(max(1e-15, model[actual]))
        market_logloss -= math.log(max(1e-15, market[actual]))
        brier += sum((model[label] - (label == actual)) ** 2 for label in RQSPF_LABELS)
        market_brier += sum((market[label] - (label == actual)) ** 2 for label in RQSPF_LABELS)
        pick = max(RQSPF_LABELS, key=lambda label: model[label] - market[label])
        edge = model[pick] - market[pick]
        offered = float(odds.get(pick, 0.0) or 0.0)
        if edge <= 1e-12 or edge < min_edge or model[pick] < min_probability or offered <= 1.0:
            continue
        bets += 1
        won = pick == actual
        wins += won
        profit += offered - 1.0 if won else -1.0

    n = len(rows)
    return {
        'market': 'rqspf',
        'independent_integer_handicap_validation': True,
        'n': n,
        'accuracy': hits / n if n else None,
        'logloss': logloss / n if n else None,
        'brier': brier / n if n else None,
        'market_baseline': {
            'accuracy': market_hits / n if n else None,
            'logloss': market_logloss / n if n else None,
            'brier': market_brier / n if n else None,
        },
        'strategy': {
            'bets': bets,
            'wins': wins,
            'profit': profit,
            'roi': profit / bets if bets else None,
        },
        'production_ready': (
            n >= 500
            and logloss < market_logloss
            and bets >= 100
            and profit > 0
        ),
    }

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

def blend_record_with_market(record: Dict, model_weight: float) -> Dict:
    """Blend a model only as a residual around the observable market prior."""
    weight = max(0.0, min(1.0, float(model_weight)))
    model = normalize_probabilities(record.get('probabilities') or {})
    market = probabilities_from_odds(record.get('odds') or {})
    blended = normalize_probabilities({
        label: (1.0 - weight) * market[label] + weight * model[label]
        for label in LABELS
    })
    return {
        **record,
        'raw_model_probabilities': model,
        'market_probabilities': market,
        'probabilities': blended,
        'market_residual_weight': weight,
    }

def select_market_residual_weight(
    training_records: Sequence[Dict],
    weight_grid: Sequence[float] = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50),
    min_logloss_improvement: float = 0.001,
) -> Dict:
    """Select model weight on past OOS predictions; otherwise return market-only."""
    if not training_records:
        return {'weight': 0.0, 'reason': 'empty_training_records'}
    market_rows = [blend_record_with_market(record, 0.0) for record in training_records]
    market_metrics = multiclass_metrics(market_rows)
    candidates = []
    for weight in weight_grid:
        rows = [blend_record_with_market(record, weight) for record in training_records]
        metrics = multiclass_metrics(rows)
        candidates.append({'weight': float(weight), 'metrics': metrics})
    best = min(
        candidates,
        key=lambda item: (item['metrics']['logloss'], item['metrics']['brier']),
    )
    improves_logloss = (
        best['metrics']['logloss']
        <= market_metrics['logloss'] - min_logloss_improvement
    )
    protects_brier = best['metrics']['brier'] <= market_metrics['brier']
    if best['weight'] <= 0 or not improves_logloss or not protects_brier:
        return {
            'weight': 0.0,
            'reason': 'model_residual_not_proven',
            'market_metrics': market_metrics,
            'best_candidate': best,
            'candidates': candidates,
        }
    return {
        'weight': best['weight'],
        'reason': 'validated_market_residual',
        'market_metrics': market_metrics,
        'best_candidate': best,
        'candidates': candidates,
    }

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
        if offered <= 1.0 or edge <= 1e-12 or probs[pick] < min_probability or edge < min_edge:
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
    raw_out_of_sample: List[Dict] = []
    start = initial_train
    while start < len(ordered):
        raw_test = ordered[start:start + test_size]
        if not raw_test:
            break
        residual = select_market_residual_weight(ordered[:start])
        residual_weight = residual['weight']
        training = [
            blend_record_with_market(record, residual_weight)
            for record in ordered[:start]
        ]
        test = [blend_record_with_market(record, residual_weight) for record in raw_test]
        selected = select_threshold(training, min_bets=min_training_bets)
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
            'market_residual_weight': residual_weight,
            'market_residual_reason': residual['reason'],
            'training_roi': selected['training']['roi'],
            'test': settled,
        })
        for record in test:
            copied = dict(record)
            copied['_fold_min_probability'] = selected['min_probability']
            copied['_fold_min_edge'] = selected['min_edge']
            out_of_sample.append(copied)
        raw_out_of_sample.extend(raw_test)
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
        'raw_model_metrics': multiclass_metrics(raw_out_of_sample),
        'model_metrics': multiclass_metrics(out_of_sample),
        'market_baseline_metrics': multiclass_metrics(market_records, 'market_probabilities'),
        'strategy': strategy,
    }
