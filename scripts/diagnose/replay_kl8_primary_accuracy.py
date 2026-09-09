"""Audit primary KL8 tickets, not the union/best of a full exclusion chain.

Uses explicit historical files and pure production selection helpers. No
prediction snapshots, strategy activation, database writes or parameter search.
The hot-frequency arm is a fixed descriptive control, never auto-promoted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kl8 import config
from src.kl8.analyzer import KL8Analyzer, _fushi7_from_select6
from src.kl8.backtest import _predict_select6_primary, _predict_fushi7_from_select6
from src.kl8.strategies import resolve_play_strategy
from src.kl8.stats import hypergeom_p_ge, hypergeom_pmf
from src.kl8.candidates import _select_final_candidate_pool, _adaptive_repeat_cap


def load_history(paths):
    records, conflicts, excluded, sources = {}, set(), Counter(), []
    for path in map(Path, paths):
        raw = path.read_bytes()
        sources.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
        data = json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)
        rows = data.get('results', []) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ValueError(f'Expected a list of draws: {path}')
        for row in rows:
            if not isinstance(row, dict):
                excluded['invalid_record'] += 1
                continue
            issue, numbers = str(row.get('issue', '')), row.get('numbers')
            if (len(issue) != 7 or not issue.isdigit() or not isinstance(numbers, list)
                    or len(numbers) != 20 or any(type(n) is not int or not 1 <= n <= 80 for n in numbers)
                    or len(set(numbers)) != 20):
                excluded['invalid_record'] += 1
                continue
            cleaned = {'issue': issue, 'date': row.get('date'), 'numbers': sorted(numbers)}
            if issue in records:
                if records[issue]['numbers'] != cleaned['numbers']:
                    conflicts.add(issue)
                else:
                    excluded['duplicate_issue'] += 1
                continue
            records[issue] = cleaned
    if conflicts:
        raise ValueError(f'Conflicting draw results for issues: {sorted(conflicts)}')
    return sorted(records.values(), key=lambda row: row['issue']), {
        'sources': sources, 'exclusions': dict(excluded), 'unique_draws': len(records)}


def reference_strategies():
    # The audit must not silently benchmark a workstation's active strategy.
    with patch.object(config, 'ACTIVE_STRATEGIES', {}):
        return {f'select_{n}': deepcopy(resolve_play_strategy(f'select_{n}', allow_reference=True))
                for n in (5, 6, 10)}


def analyzer_for(history):
    analyzer = KL8Analyzer.__new__(KL8Analyzer)
    analyzer.history_data = list(reversed(history))
    analyzer.history_file, analyzer._data_mtime = '', 0
    analyzer.using_simulated_data = False
    analyzer.update_statistics()
    return analyzer


def primary(analyzer, strategy, pick):
    if pick == 6:
        return _predict_select6_primary(analyzer, strategy)
    ranking = analyzer.build_pool_by_strategy(strategy, pool_size=80 if pick == 5 else 20)['candidates']
    adaptive = _adaptive_repeat_cap(analyzer.history_data, pick)
    cap = strategy.get('final_max_last_numbers', min(strategy.get('pool_max_last_numbers') or adaptive, adaptive))
    selected, _ = _select_final_candidate_pool(
        ranking[:20], pick, analyzer.statistics['last_numbers'],
        max_last_numbers=cap, selection_mode=strategy.get('final_selection_mode', 'balanced'))
    return sorted(number for number, _ in selected), ranking


def evaluate_rows(history, warmup=100, predictor=None):
    strategies = reference_strategies()
    hot = {**deepcopy(strategies['select_6']), 'strategy_id': 'audit_hot_frequency_100',
           'feature_weights': {'frequency': 1.0}, 'frequency_mode': 'hot',
           'repeat_direction': 'neutral', 'window_size': 100}
    observations = []
    for index in range(max(100, warmup), len(history)):
        target = history[index]
        past = history[:index]
        if predictor is not None:
            tickets = predictor(deepcopy(past))
        else:
            analyzer = analyzer_for(past)
            six, ranking = primary(analyzer, strategies['select_6'], 6)
            old_first, _ = analyzer._calculate_select_recalculation('select_6', six, strategy=strategies['select_6'])
            legacy_seven, _ = _fushi7_from_select6(six, ranking, old_first['numbers'])
            corrected_seven = _predict_fushi7_from_select6(analyzer, strategies['select_6'], six, ranking)
            tickets = {
                'select_5': primary(analyzer, strategies['select_5'], 5)[0],
                'select_6': six, 'select_10': primary(analyzer, strategies['select_10'], 10)[0],
                'fu_shi_7_legacy_reserved': legacy_seven,
                'fu_shi_7_primary_ranked': corrected_seven,
                'select_6_hot_control': primary(analyzer, hot, 6)[0],
            }
        actual = set(target['numbers'])
        observations.append({'issue': target['issue'], 'based_on_issue': past[-1]['issue'],
                             'tickets': tickets,
                             'hits': {name: len(set(numbers) & actual) for name, numbers in tickets.items()}})
    return observations, strategies


def summarize(rows):
    if not rows:
        return {'n': 0, 'plays': {}}
    plays = {}
    for name in rows[0]['tickets']:
        tickets = [row['tickets'][name] for row in rows]
        pick = len(tickets[0])
        if not pick or any(len(t) != pick or len(set(t)) != pick for t in tickets):
            raise ValueError('Invalid or changing ticket size')
        hits = [row['hits'][name] for row in rows]
        frequency = Counter(number for ticket in tickets for number in ticket)
        plays[name] = {
            'pick_size': pick, 'n': len(hits), 'mean_hits': sum(hits) / len(hits),
            'fair_mean_hits': pick * .25, 'zero_rate': hits.count(0) / len(hits),
            'fair_zero_rate': hypergeom_pmf(pick, 0),
            'hit_distribution': dict(sorted(Counter(hits).items())),
            'at_least': {str(k): {'observed': sum(h >= k for h in hits) / len(hits),
                                 'fair_baseline': hypergeom_p_ge(pick, k)}
                         for k in (3, 4, 5) if k <= pick},
            'most_selected_numbers': frequency.most_common(10),
            'mean_consecutive_ticket_overlap': (
                sum(len(set(a) & set(b)) for a, b in zip(tickets, tickets[1:])) / (len(tickets) - 1)
                if len(tickets) > 1 else None),
        }
    return {'n': len(rows), 'first_issue': rows[0]['issue'], 'last_issue': rows[-1]['issue'], 'plays': plays}


def paired_seventh_number(rows):
    differences = [row['hits']['fu_shi_7_primary_ranked'] - row['hits']['fu_shi_7_legacy_reserved'] for row in rows]
    if not differences:
        return {'n': 0}
    n, mean = len(differences), sum(differences) / len(differences)
    se = math.sqrt(sum((d - mean) ** 2 for d in differences) / (n - 1) / n) if n > 1 else None
    return {'n': n, 'mean_hit_difference': mean,
            'descriptive_normal_ci95': [mean - 1.96 * se, mean + 1.96 * se] if se is not None else None,
            'new_better_issues': sum(d > 0 for d in differences),
            'old_better_issues': sum(d < 0 for d in differences)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/kl8_primary_accuracy_replay.json')
    args = parser.parse_args()
    paths = args.input or [ROOT / 'tests/fixtures/numeric/kl8_history.json.gz']
    history, audit = load_history(paths)
    rows, strategies = evaluate_rows(history)
    midpoint = len(rows) // 2
    report = {
        'version': config.KL8_PREDICTOR_VERSION, 'data': audit, 'strategies': strategies,
        'release_qualified': False, 'parameter_search': False,
        'limitations': [
            'Historical results do not prove what source information was available before each draw.',
            'No candidate is selected or activated using this report; these periods may have been inspected before.',
            'Primary tickets only: late-round best hits or near-full-universe coverage are not prediction uplift.',
            'Fair independent draws give each number probability 20/80 regardless of hot/cold labels.',
        ],
        'all': summarize(rows), 'older_half': summarize(rows[:midpoint]),
        'newer_half': summarize(rows[midpoint:]), 'latest_30': summarize(rows[-30:]),
        'paired_primary_seventh': paired_seventh_number(rows), 'observations': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'all': report['all'],
                      'paired_primary_seventh': report['paired_primary_seventh']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
