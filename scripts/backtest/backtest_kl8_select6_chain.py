"""Walk-forward audit for the select-6 primary/exclusion chain.

The primary recommendation is round 0.  Every later round removes every
number already recommended and recalculates the next six numbers.  Candidate
strategies are ranked on the older validation slice; the newest slice is only
used once for the locked final report.
"""

import argparse
import json
import logging
import os
import math
import sys
from collections import defaultdict
from copy import deepcopy


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
logging.disable(logging.CRITICAL)

from src.kl8 import (  # noqa: E402
    KL8Analyzer,
    VALIDATION_CANDIDATES,
    _adaptive_repeat_cap,
    _select_final_candidate_pool,
    resolve_play_strategy,
)


SELECT_SIZE = 6


def _strategy_slate():
    current = deepcopy(resolve_play_strategy('select_6', allow_reference=True))
    slate = {
        'current_reference': current,
        'previous_reference_v3': {
            'strategy_id': 'select_6_ref_transition_repeat_v3',
            'feature_weights': {
                'frequency': 0.18,
                'gap': 0.14,
                'trend': 0.12,
                'next_transition': 0.22,
                'pair_cooccurrence': 0.04,
                'position_residual': 0.11,
                'position_residual_cross': 0.08,
                'road_residual': 0.08,
                'repeat': 0.03,
                'odd_even': 0.0,
                'big_small': 0.0,
            },
            'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
            'window_size': 100,
            'repeat_direction': 'follow',
            'pool_max_last_numbers': 3,
            'pool_diversify': False,
            'final_selection_mode': 'concentrated',
        },
        'previous_chain_v4': {
            'strategy_id': 'select_6_ref_transition_chain_v4',
            'feature_weights': {
                'frequency': 0.20,
                'gap': 0.10,
                'trend': 0.10,
                'next_transition': 0.35,
                'pair_cooccurrence': 0.05,
                'position_residual': 0.10,
                'position_residual_cross': 0.0,
                'road_residual': 0.05,
                'repeat': 0.05,
                'odd_even': 0.0,
                'big_small': 0.0,
            },
            'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
            'window_size': 150,
            'repeat_direction': 'follow',
            'pool_max_last_numbers': 4,
            'pool_diversify': False,
            'final_selection_mode': 'shape_balanced',
        },
        # Pre-registered compromise between the current 100-draw reference and
        # transition_repeat_150_cap4.  Its weights are fixed before the oldest
        # untouched 600-draw audit is opened.
        'select6_chain_consensus_150': {
            'strategy_id': 'candidate_select6_chain_consensus_150',
            'feature_weights': {
                'frequency': 0.19,
                'gap': 0.12,
                'trend': 0.11,
                'next_transition': 0.285,
                'pair_cooccurrence': 0.045,
                'position_residual': 0.105,
                'position_residual_cross': 0.04,
                'road_residual': 0.065,
                'repeat': 0.04,
                'odd_even': 0.0,
                'big_small': 0.0,
            },
            'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
            'window_size': 150,
            'repeat_direction': 'follow',
            'pool_max_last_numbers': 4,
            'pool_diversify': False,
            'final_selection_mode': 'shape_balanced',
        },
    }
    for name in (
        'select6_balanced_100',
        'select6_repeat_follow_75',
        'select6_hot_balanced_150',
        'transition_repeat_75_cap2',
        'transition_repeat_100_cap3',
        'random_shape_50',
    ):
        strategy = deepcopy(VALIDATION_CANDIDATES[name])
        # The live select-6 chain is a ranking/exclusion chain.  Keeping the
        # full ranking intact avoids changing the candidate order merely
        # because a later round requested a larger pool.
        strategy['pool_diversify'] = False
        slate[name] = strategy
    return slate


def _one_chain(analyzer, strategy, rounds):
    pool = analyzer.build_pool_by_strategy(strategy, pool_size=80)
    candidates = pool.get('candidates', [])
    if len(candidates) < SELECT_SIZE:
        return []

    cap = strategy.get('pool_max_last_numbers')
    if cap is None:
        cap = _adaptive_repeat_cap(analyzer.history_data, SELECT_SIZE)
    cap = max(0, min(SELECT_SIZE, int(cap)))
    selection_mode = strategy.get('final_selection_mode', 'concentrated')
    last_numbers = analyzer.statistics.get('last_numbers', set())

    selected, _ = _select_final_candidate_pool(
        candidates[:20],
        SELECT_SIZE,
        last_numbers,
        max_last_numbers=cap,
        selection_mode=selection_mode,
    )
    groups = [[num for num, _ in selected]]
    excluded = set(groups[0])

    for _ in range(1, rounds):
        remaining = [(num, score) for num, score in candidates if num not in excluded]
        if len(remaining) < SELECT_SIZE:
            break
        selected, _ = _select_final_candidate_pool(
            remaining,
            SELECT_SIZE,
            last_numbers,
            max_last_numbers=cap,
            selection_mode=selection_mode,
        )
        group = [num for num, _ in selected]
        groups.append(group)
        excluded.update(group)
    return groups


def _summarize(rows, rounds):
    if not rows:
        return {'n_tests': 0}
    n_tests = len(rows)
    round_metrics = []
    for round_index in range(rounds):
        hits = [row[round_index] for row in rows if len(row) > round_index]
        round_metrics.append({
            'round': round_index,
            'mean_hits': round(sum(hits) / len(hits), 4) if hits else None,
            'hit_3_rate': round(sum(hit >= 3 for hit in hits) / len(hits), 4) if hits else None,
            'hit_4_rate': round(sum(hit >= 4 for hit in hits) / len(hits), 4) if hits else None,
        })

    early = [row[:rounds] for row in rows]
    first_ge3 = [next((i for i, hit in enumerate(row) if hit >= 3), rounds) for row in early]
    first_ge4 = [next((i for i, hit in enumerate(row) if hit >= 4), rounds) for row in early]
    weights = [1.0 / (index + 1) for index in range(rounds)]
    discounted_hits = [
        sum(hit * weights[index] for index, hit in enumerate(row))
        for row in early
    ]
    discounted_followup_hits = [
        sum(hit / index for index, hit in enumerate(row[1:], 1))
        for row in early
    ]
    return {
        'n_tests': n_tests,
        'rounds': round_metrics,
        'primary_mean_hits': round_metrics[0]['mean_hits'],
        'primary_hit_3_rate': round_metrics[0]['hit_3_rate'],
        'primary_hit_4_rate': round_metrics[0]['hit_4_rate'],
        'early_best_mean_hits': round(sum(max(row) for row in early) / n_tests, 4),
        'early_any_hit_3_rate': round(sum(max(row) >= 3 for row in early) / n_tests, 4),
        'early_any_hit_4_rate': round(sum(max(row) >= 4 for row in early) / n_tests, 4),
        'mean_first_hit_3_round': round(sum(first_ge3) / n_tests, 4),
        'mean_first_hit_4_round': round(sum(first_ge4) / n_tests, 4),
        'discounted_hits': round(sum(discounted_hits) / n_tests, 4),
        'discounted_followup_hits': round(
            sum(discounted_followup_hits) / n_tests, 4,
        ),
    }


def _score(metrics):
    """Rank primarily by the formal select-6 ticket, not near-full coverage."""
    return round(
        metrics.get('primary_mean_hits', 0) * 0.65
        + metrics.get('primary_hit_3_rate', 0) * 0.20
        + metrics.get('discounted_followup_hits', 0) * 0.10
        + metrics.get('early_any_hit_3_rate', 0) * 0.05,
        6,
    )


def _row_score(row):
    followups = row[1:]
    discounted_followups = sum(
        hit / (index + 1) for index, hit in enumerate(followups)
    )
    return (
        row[0] * 0.65
        + (row[0] >= 3) * 0.20
        + discounted_followups * 0.10
        + (max(row) >= 3) * 0.05
    )


def _paired_summary(values):
    if not values:
        return {'mean': 0.0, 'ci_95': [0.0, 0.0]}
    mean = sum(values) / len(values)
    variance = (
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        if len(values) > 1 else 0.0
    )
    margin = 1.96 * math.sqrt(variance / len(values))
    return {
        'mean': round(mean, 6),
        'ci_95': [round(mean - margin, 6), round(mean + margin, 6)],
    }


def _run_slice(raw, target_indices, strategies, rounds):
    hits_by_strategy = defaultdict(list)
    for position, target_index in enumerate(target_indices, 1):
        target = set(raw[target_index]['numbers'])
        history = raw[target_index + 1:]
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = history
        analyzer.using_simulated_data = False
        analyzer.history_file = ''
        analyzer._data_mtime = 0
        analyzer.statistics = {}
        analyzer.update_statistics()

        for name, strategy in strategies.items():
            groups = _one_chain(analyzer, strategy, rounds)
            if len(groups) == rounds:
                hits_by_strategy[name].append([
                    len(set(group) & target) for group in groups
                ])

        if position % 50 == 0:
            print(f'  completed {position}/{len(target_indices)} targets', flush=True)
    summaries = {}
    current_rows = hits_by_strategy.get('current_reference', [])
    for name in strategies:
        metrics = _summarize(hits_by_strategy[name], rounds)
        metrics['score'] = _score(metrics)
        if name != 'current_reference' and len(hits_by_strategy[name]) == len(current_rows):
            metrics['comparison_to_current'] = {
                'primary_hits': _paired_summary([
                    candidate[0] - current[0]
                    for candidate, current in zip(hits_by_strategy[name], current_rows)
                ]),
                'chain_score': _paired_summary([
                    _row_score(candidate) - _row_score(current)
                    for candidate, current in zip(hits_by_strategy[name], current_rows)
                ]),
            }
        summaries[name] = metrics
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', default='data/kl8_history.json')
    parser.add_argument('--validation-periods', type=int, default=100)
    parser.add_argument('--final-periods', type=int, default=100)
    parser.add_argument('--offset', type=int, default=0, help='skip this many newest draws')
    parser.add_argument('--rounds', type=int, default=5, help='includes primary round 0')
    parser.add_argument('--strategies', default='', help='comma-separated strategy names')
    parser.add_argument('--output', default='reports/kl8_select6_chain_backtest.json')
    args = parser.parse_args()

    raw_doc = json.load(open(args.history, encoding='utf-8'))
    raw = raw_doc.get('results', raw_doc) if isinstance(raw_doc, dict) else raw_doc
    raw = sorted(raw, key=lambda row: row['issue'], reverse=True)
    required = args.offset + args.validation_periods + args.final_periods + 50
    if len(raw) < required:
        raise SystemExit(f'need at least {required} periods, got {len(raw)}')

    strategies = _strategy_slate()
    if args.strategies:
        requested = [name.strip() for name in args.strategies.split(',') if name.strip()]
        missing = [name for name in requested if name not in strategies]
        if missing:
            raise SystemExit(f'unknown strategies: {missing}')
        strategies = {name: strategies[name] for name in requested}
    final_indices = list(range(args.offset, args.offset + args.final_periods))
    validation_indices = list(range(
        args.offset + args.final_periods,
        args.offset + args.final_periods + args.validation_periods,
    ))

    print('validation slice', flush=True)
    validation = _run_slice(raw, validation_indices, strategies, args.rounds)
    ranking = sorted(strategies, key=lambda name: validation[name]['score'], reverse=True)
    winner = ranking[0]

    # Lock the validation winner before looking at the newest final slice.  We
    # still report every pre-registered control so regressions stay visible.
    print(f'locked validation winner: {winner}', flush=True)
    print('final slice', flush=True)
    final = _run_slice(raw, final_indices, strategies, args.rounds)

    report = {
        'history_periods': len(raw),
        'latest_issue': raw[0]['issue'],
        'offset': args.offset,
        'rounds_including_primary': args.rounds,
        'validation_periods': args.validation_periods,
        'final_periods': args.final_periods,
        'strategies': {
            name: strategy.get('strategy_id', name)
            for name, strategy in strategies.items()
        },
        'validation_ranking': ranking,
        'locked_winner': winner,
        'validation': validation,
        'final': final,
        'note': (
            'Round 0 is the primary select-6 recommendation. Later rounds use '
            'cumulative exclusion. The final slice was not used to choose the winner.'
        ),
    }
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
