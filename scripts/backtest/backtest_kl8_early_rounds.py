"""Audit round 1/2 of both live exclusion chains without writing predictions.

Candidates are fixed before validation; the latest holdout cannot select weights.
The objective counts >=4 and >=5 in individual rounds, never union coverage.
"""
import argparse
from copy import deepcopy
import json
import gzip
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.backtest.backtest_kl8_select6_chain import _paired_summary
from src.kl8 import (KL8Analyzer, VALIDATION_CANDIDATES, resolve_play_strategy, _adaptive_repeat_cap,
                     _select_final_candidate_pool, _enforce_minimum_repeats)
from src.kl8.analyzer import _fushi7_from_select6


def slate(expanded=False):
    baseline = resolve_play_strategy('select_6', allow_reference=True)
    result = {'current': baseline}
    for mode in ('shape_balanced', 'repeat_follow', 'low_repeat'):
        candidate = deepcopy(baseline)
        candidate['exclusion_selection_mode'] = mode
        result[mode] = candidate
    if expanded:
        for name in ('select6_balanced_100', 'select6_repeat_follow_75',
                     'select6_hot_balanced_150', 'transition_repeat_150_cap4'):
            candidate = deepcopy(VALIDATION_CANDIDATES[name])
            candidate['pool_diversify'] = False
            result[name] = candidate
    return result


def first_round_slate():
    """Fixed hypotheses; primary select-6 ranking is identical in every arm."""
    baseline = resolve_play_strategy('select_6', allow_reference=True)
    result = {'current': baseline}
    for window in (75, 150):
        candidate = deepcopy(baseline)
        candidate['first_exclusion_strategy'] = {
            'strategy_id': f'candidate_first_exclusion_window_{window}',
            'window_size': window,
        }
        result[f'first_window_{window}'] = candidate
    for name in ('select6_repeat_follow_75', 'select6_hot_balanced_150'):
        candidate = deepcopy(baseline)
        candidate['first_exclusion_strategy'] = deepcopy(VALIDATION_CANDIDATES[name])
        result[f'first_{name}'] = candidate
    return result


def simplified_first_round_slate():
    """Two fixed ablations of the early ranking, not a weight search.

    Keep primary select-6 unchanged. Test whether sparse conditional counts
    hurt round 1, and compare a simple hot-frequency ranking as a control.
    Both exclusion rounds and both plays still have to pass the holdout.
    """
    baseline = resolve_play_strategy('select_6', allow_reference=True)
    result = {'current': baseline}
    weights = deepcopy(baseline['feature_weights'])
    weights.update(next_transition=0.0, pair_cooccurrence=0.0)
    overrides = {
        'first_without_sparse_pairs': {'feature_weights': weights},
        'first_hot_frequency_100': {
            'feature_weights': {'frequency': 1.0},
            'frequency_mode': 'hot', 'window_size': 100,
            'repeat_direction': 'neutral',
        },
    }
    for name, override in overrides.items():
        candidate = deepcopy(baseline)
        candidate['first_exclusion_strategy'] = {
            **override, 'strategy_id': f'candidate_{name}',
        }
        result[name] = candidate
    return result


def live_groups(analyzer, strategy):
    """Use production recalculation and linked fushi supplementation; no writes."""
    with patch('src.kl8.strategies.resolve_play_strategy', return_value=strategy), \
         patch.object(analyzer, '_save_exclude_recalculation', return_value={}):
        ranking = analyzer.build_pool_by_strategy(strategy, pool_size=80)['candidates'][:80]
        adaptive = _adaptive_repeat_cap(analyzer.history_data, 6)
        cap = strategy.get('final_max_last_numbers',
                           min(strategy.get('pool_max_last_numbers', adaptive) or adaptive, adaptive))
        primary_pool, _ = _select_final_candidate_pool(
            ranking[:20], 6, analyzer.statistics.get('last_numbers', set()),
            max_last_numbers=cap, selection_mode=strategy.get('final_selection_mode', 'balanced'),
        )
        if strategy.get('final_min_last_numbers', 0) > 0:
            primary_pool = _enforce_minimum_repeats(
                primary_pool, ranking, analyzer.statistics.get('last_numbers', set()),
                strategy['final_min_last_numbers'],
            )
        primary = sorted(n for n, _ in primary_pool)
        fushi, _ = _fushi7_from_select6(primary, ranking)
        groups = {'select_6': [primary], 'fu_shi_7': [fushi]}
        excluded6, excluded7 = set(primary), set(fushi)
        for _ in range(2):
            single, _ = analyzer._calculate_select_recalculation('select_6', sorted(excluded6))
            compound = analyzer.recalculate_play_excluding(
                'fu_shi_7', sorted(excluded7), record_context={'select6_round': single},
            )
            groups['select_6'].append(single['numbers'])
            groups['fu_shi_7'].append(compound['core_numbers'])
            excluded6.update(single['numbers'])
            excluded7.update(compound['core_numbers'])
        return groups


def objective(row, first_round=False):
    # Both plays, round 1 and 2 equally weighted. Round 0 stays a guardrail.
    hits = [hit for play in row.values() for hit in play[1:2 if first_round else 3]]
    return sum((hit >= 4) + (hit >= 5) for hit in hits) / len(hits)


def summarize(rows, first_round=False):
    result = {'n_tests': len(rows),
              'objective': sum(objective(row, first_round) for row in rows) / len(rows)}
    for play in ('select_6', 'fu_shi_7'):
        result[play] = [
            {'round': r, 'mean_hits': sum(row[play][r] for row in rows) / len(rows),
             **{f'hit_{k}_rate': sum(row[play][r] >= k for row in rows) / len(rows)
                for k in (4, 5)}} for r in range(3)
        ]
    return result


def round_non_regression(candidate, baseline, round_number):
    """A gain in one play must not hide a loss in the other play."""
    return all(
        sum(metric(row[play][round_number]) for row in candidate) >=
        sum(metric(row[play][round_number]) for row in baseline)
        for play in ('select_6', 'fu_shi_7')
        for metric in (lambda hit: hit, lambda hit: hit >= 4, lambda hit: hit >= 5)
    )


def promotion_checks(winner, candidate, baseline, comparison):
    """Never let an aggregate gain conceal a losing play or exclusion round."""
    primary_guard = all(
        sum(row[play][0] for row in candidate) >= sum(row[play][0] for row in baseline)
        for play in ('select_6', 'fu_shi_7')
    )
    first_guard = round_non_regression(candidate, baseline, 1)
    second_guard = round_non_regression(candidate, baseline, 2)
    return {
        'primary_mean_non_regression': primary_guard,
        'first_round_non_regression': first_guard,
        'second_round_non_regression': second_guard,
        'promotion_supported': winner != 'current' and comparison['ci_95'][0] > 0
                               and primary_guard and first_guard and second_guard,
    }


def run_slice(raw, indices, strategies):
    rows = {name: [] for name in strategies}
    for position, index in enumerate(indices, 1):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = raw[index + 1:]
        analyzer.using_simulated_data = False
        analyzer.history_file = ''
        analyzer._data_mtime = 0
        analyzer.statistics = {}
        analyzer.update_statistics()
        target = set(raw[index]['numbers'])
        for name, strategy in strategies.items():
            groups = live_groups(analyzer, strategy)
            rows[name].append({play: [len(set(g) & target) for g in chain]
                               for play, chain in groups.items()})
        if position % 25 == 0:
            print(f'completed {position}/{len(indices)}', flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', default='data/kl8_history.json')
    parser.add_argument('--periods', type=int, default=100, help='periods per chronological slice')
    parser.add_argument('--offset', type=int, default=0, help='skip newest draws to audit a separate historical interval')
    parser.add_argument('--expanded', action='store_true', help='also compare fixed alternative feature rankings')
    parser.add_argument('--first-round', action='store_true', help='freeze select-6 primary and optimize round 1 only')
    parser.add_argument('--simplified-first-round', action='store_true',
                        help='test two fixed simpler round-1 rankings; score both exclusion rounds')
    parser.add_argument('--output', default='reports/kl8_early_rounds_audit.json')
    args = parser.parse_args()
    if sum((args.first_round, args.expanded, args.simplified_first_round)) > 1:
        parser.error('choose only one of --first-round, --expanded, --simplified-first-round')
    opener = gzip.open if args.history.endswith('.gz') else open
    with opener(args.history, 'rt', encoding='utf-8') as handle:
        doc = json.load(handle)
    raw = sorted(doc['results'] if isinstance(doc, dict) else doc,
                 key=lambda row: row['issue'], reverse=True)
    if args.offset < 0 or args.periods < 2 or len(raw) < args.offset + 2 * args.periods + 150:
        parser.error('need positive slices and at least 150 older training draws')
    if len({row['issue'] for row in raw}) != len(raw):
        parser.error('duplicate issues')
    for row in raw:
        numbers = row['numbers']
        if len(numbers) != 20 or len(set(numbers)) != 20 or any(
            type(n) is not int or not 1 <= n <= 80 for n in numbers
        ):
            parser.error(f'invalid draw: {row["issue"]}')
    strategies = (simplified_first_round_slate() if args.simplified_first_round else
                  first_round_slate() if args.first_round else slate(expanded=args.expanded))
    validation = run_slice(raw, range(args.offset + args.periods, args.offset + args.periods * 2), strategies)
    winner = max(strategies, key=lambda name: summarize(validation[name], args.first_round)['objective'])
    print(f'locked winner: {winner}', flush=True)
    final = run_slice(raw, range(args.offset, args.offset + args.periods),
                      {n: strategies[n] for n in dict.fromkeys(['current', winner])})
    comparison = _paired_summary([objective(a, args.first_round) - objective(b, args.first_round)
                                  for a, b in zip(final[winner], final['current'])])
    report = {
        'history_source': args.history,
        'history_sha256': hashlib.sha256(Path(args.history).read_bytes()).hexdigest(),
        'objective_rounds': [1] if args.first_round else [1, 2],
        'offset': args.offset,
        'holdout_issues': [raw[args.offset + args.periods - 1]['issue'], raw[args.offset]['issue']],
        'validation_issues': [raw[args.offset + 2 * args.periods - 1]['issue'], raw[args.offset + args.periods]['issue']],
        'latest_issue': raw[0]['issue'], 'periods_per_slice': args.periods,
        'locked_winner': winner, 'strategies': strategies,
        'validation': {n: summarize(r, args.first_round) for n, r in validation.items()},
        'holdout': {n: summarize(r, args.first_round) for n, r in final.items()},
        'paired_objective_difference': comparison,
        **promotion_checks(winner, final[winner], final['current'], comparison),
        'note': 'Round 0 is primary; objective_rounds lists the separately scored exclusion rounds. '
                'Fushi hits are 7-number pool hits (best 5-number ticket capped at 5). '
                'No target or future draw enters training. No automatic promotion.',
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('locked_winner', 'holdout', 'paired_objective_difference', 'promotion_supported')}, indent=2))


if __name__ == '__main__':
    main()
