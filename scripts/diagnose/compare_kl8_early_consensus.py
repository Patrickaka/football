"""Frozen multi-window hypotheses; historical development evidence only.

Choose on the older slice using low-hit frequency, then check the newer slice.
Recent complaint draws are reported separately and cannot choose the winner.
No activation, snapshots or settlement writes.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnose.replay_kl8_primary_accuracy import load_history
from scripts.backtest.backtest_kl8_early_rounds import run_slice, summarize, round_non_regression
from scripts.backtest.backtest_kl8_select6_chain import _paired_summary
from src.kl8.strategies import resolve_play_strategy


def candidates():
    base = resolve_play_strategy('select_6', allow_reference=True)
    result = {'current': deepcopy(base)}
    for windows in ((50, 100, 200), (100, 150, 250)):
        name = 'consensus_' + '_'.join(map(str, windows))
        result[name] = deepcopy(base)
        result[name]['early_exclusion_strategy'] = {
            'strategy_id': name, 'exclusion_windows': list(windows),
        }
    return result


def low_hit_rate(row):
    return sum(hit <= 2 for play in ('select_6', 'fu_shi_7')
               for hit in row[play][1:3]) / 4


def summary(rows):
    result = summarize(rows)
    result['low_hit_rate'] = sum(map(low_hit_rate, rows)) / len(rows)
    for play in ('select_6', 'fu_shi_7'):
        for r in range(3):
            result[play][r]['hit_0_to_2_rate'] = sum(row[play][r] <= 2 for row in rows) / len(rows)
        low_runs = longest = current = 0
        for row in rows:
            both_low = all(hit <= 2 for hit in row[play][1:3])
            low_runs += both_low
            current = current + 1 if both_low else 0
            longest = max(longest, current)
        result[play + '_both_early_rounds_low'] = {
            'rate': low_runs / len(rows), 'longest_streak': longest,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', required=True)
    parser.add_argument('--periods', type=int, default=200)
    parser.add_argument('--recent-from', default='2026247')
    parser.add_argument('--output', default='reports/kl8_early_consensus_20260920.json')
    args = parser.parse_args()
    history, sources = load_history(args.input)
    raw = list(reversed(history))
    older = [i for i, row in enumerate(raw) if row['issue'] < args.recent_from]
    recent = [i for i, row in enumerate(raw) if row['issue'] >= args.recent_from]
    n = args.periods
    if n < 2 or len(older) < 2 * n + 250:
        parser.error('need two slices and 250 older training draws')
    slate = candidates()
    earlier = run_slice(raw, older[n:2*n], slate)
    winner = min(slate, key=lambda name: summary(earlier[name])['low_hit_rate'])
    print('locked winner: ' + winner, flush=True)
    locked = {name: slate[name] for name in dict.fromkeys(('current', winner))}
    later = run_slice(raw, older[:n], locked)
    candidate, base = later[winner], later['current']
    comparison = _paired_summary([low_hit_rate(b) - low_hit_rate(a) for a, b in zip(candidate, base)])
    checks = {
        'primary_unchanged': all(a[p][0] == b[p][0] for a, b in zip(candidate, base) for p in a),
        'first_round_non_regression': round_non_regression(candidate, base, 1),
        'second_round_non_regression': round_non_regression(candidate, base, 2),
        'low_hits_non_regression_each_round': all(
            sum(row[p][r] <= 2 for row in candidate) <= sum(row[p][r] <= 2 for row in base)
            for p in ('select_6', 'fu_shi_7') for r in (1, 2)),
        'positive_low_hit_reduction_ci': comparison['ci_95'][0] > 0,
    }
    report = {
        'data': sources, 'data_role': 'historical_development_not_fresh_holdout',
        'strategies': slate, 'locked_winner': winner,
        'earlier_issues': [raw[i]['issue'] for i in older[n:2*n]],
        'later_issues': [raw[i]['issue'] for i in older[:n]],
        'earlier': {name: summary(rows) for name, rows in earlier.items()},
        'later': {name: summary(rows) for name, rows in later.items()},
        'paired_low_hit_reduction': comparison, 'checks': checks,
        'development_checks_passed': winner != 'current' and all(checks.values()),
        'promotion_allowed': False,
        'recent_issues': [raw[i]['issue'] for i in recent],
        'recent_rows': run_slice(raw, recent, locked),
        'observations': {'earlier': earlier, 'later': later},
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('locked_winner', 'later', 'checks', 'recent_rows')}, indent=2))


if __name__ == '__main__':
    main()
