"""Fixed development comparisons for one select-6 and one seven-number select-5 pool.

Reuse the live ranking and paired accuracy audit. No activation, search API,
network fetch, extra tickets, or prediction/settlement writes. Previously
inspected historical draws are explicitly not a fresh final holdout.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose.replay_kl8_primary_accuracy import load_history, analyzer_for
from src.kl8.backtest import _predict_select6_primary, _predict_fushi7_from_select6
from src.kl8.main_play_accuracy import evaluate_main_play_pair
from src.kl8.strategies import resolve_play_strategy
from src.kl8.config import KL8_PREDICTOR_VERSION


def candidate_slate(incumbent):
    """Predeclared ablations, not a grid search fitted to recent losing draws."""
    slate = {'current': deepcopy(incumbent)}
    for name, dropped in (
        ('without_sparse_transition_pair', ('next_transition', 'pair_cooccurrence')),
        ('without_position_residuals', ('position_residual', 'position_residual_cross', 'road_residual')),
    ):
        strategy = deepcopy(incumbent)
        strategy['feature_weights'] = {**strategy.get('feature_weights', {}), **dict.fromkeys(dropped, 0.)}
        strategy['strategy_id'] = name
        slate[name] = strategy
    slate['window_150'] = {**deepcopy(incumbent), 'strategy_id': 'window_150', 'window_size': 150}
    return slate


def predict_pair(history, strategy):
    analyzer = analyzer_for(history)
    six, ranking = _predict_select6_primary(analyzer, strategy)
    seven = _predict_fushi7_from_select6(analyzer, strategy, six, ranking)
    return {'select_6': six, 'fu_shi_7': seven}


def compare(history, strategies, *, warmup=100, predictor=predict_pair):
    if 'current' not in strategies:
        raise ValueError('The current strategy is required as a paired control')
    issues = [row['issue'] for row in history]
    if issues != sorted(set(issues)):
        raise ValueError('History must have unique, ascending issues')
    if warmup < 100 or len(history) - warmup < 100:
        raise ValueError('Need at least 100 warmup draws and 100 development targets')
    observations = {name: [] for name in strategies}
    frozen = deepcopy(strategies)
    for index in range(warmup, len(history)):
        target = history[index]
        for name, strategy in frozen.items():
            tickets = predictor(deepcopy(history[:index]), deepcopy(strategy))
            observations[name].append({
                'issue': target['issue'], 'based_on_issue': history[index - 1]['issue'],
                'actual_numbers': list(target['numbers']), 'tickets': tickets,
            })
    split = len(observations['current']) // 2
    early = {name: evaluate_main_play_pair(observations['current'][:split], rows[:split])
             for name, rows in observations.items()}
    winner = max(early, key=lambda name: early[name]['ranking_key'])
    later = evaluate_main_play_pair(observations['current'][split:], observations[winner][split:])
    complete = {name: evaluate_main_play_pair(observations['current'], rows)
                for name, rows in observations.items()}
    return {
        'version': KL8_PREDICTOR_VERSION, 'strategies': frozen,
        'strategy_sha256': hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'objective': 'both_primary_plays_non_worse_mean_zero_and_3_4_5_plus',
        'ticket_sizes': {'select_6': 6, 'fu_shi_7': 7},
        'development_earlier': early,
        'locked_earlier_winner': winner,
        'development_later_check': later,
        'development_complete': complete,
        'observations': observations,
        'promotion_allowed': False,
        'data_role': 'previously_inspected_development_data',
        'decision': 'retain_current_strategy_pending_fresh_joint_validation',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/kl8_main_play_comparison.json')
    args = parser.parse_args()
    history, sources = load_history(args.input)
    incumbent = resolve_play_strategy('select_6', allow_reference=True)
    if not incumbent:
        raise ValueError('No current select-6 strategy to compare')
    report = {**compare(history, candidate_slate(incumbent)), 'data': sources}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'data': sources, 'locked_earlier_winner': report['locked_earlier_winner'],
        'promotion_allowed': False,
        'results': {name: {'candidate': row['candidate'], 'gate': row['gate']}
                    for name, row in report['development_complete'].items()},
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
