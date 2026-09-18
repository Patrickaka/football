"""Frozen early-round window/neutrality hypotheses, with recent draws separate.

Previously inspected history is development data, not a fresh holdout.
This script neither activates candidates nor rewrites predictions.
"""
import json
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnose.replay_kl8_primary_accuracy import load_history
from scripts.backtest.backtest_kl8_early_rounds import run_slice, summarize, objective, promotion_checks
from scripts.backtest.backtest_kl8_select6_chain import _paired_summary
from src.kl8.strategies import resolve_play_strategy


def candidates():
    base = resolve_play_strategy('select_6', allow_reference=True)
    result = {'current': deepcopy(base)}
    neutral = deepcopy(base['feature_weights'])
    neutral.update(gap=0.0, repeat=0.0)
    for name, options in (
        ('window_150', {'window_size': 150}),
        ('window_250', {'window_size': 250}),
        ('neutral_gap_repeat', {'feature_weights': neutral, 'repeat_direction': 'neutral'}),
    ):
        result[name] = deepcopy(base)
        result[name]['early_exclusion_strategy'] = {
            'strategy_id': 'development_early_' + name, **options,
        }
    return result


def main():
    history, sources = load_history([
        ROOT / 'reports/kl8_research_history.json', ROOT / 'data/kl8_history.json',
        ROOT / 'reports/kl8_recent_20260918.json',
    ])
    raw = list(reversed(history))
    recent = [i for i, r in enumerate(raw) if r['issue'] >= '2026247']
    older = [i for i, r in enumerate(raw) if r['issue'] <= '2026246']
    if len(older) < 650:
        raise ValueError('Need 400 targets and 250 earlier training draws')
    slate = candidates()
    early = run_slice(raw, older[200:400], slate)
    winner = max(slate, key=lambda n: summarize(early[n])['objective'])
    locked = {n: slate[n] for n in dict.fromkeys(('current', winner))}
    print('locked winner: ' + winner, flush=True)
    later = run_slice(raw, older[:200], locked)
    differences = [objective(a) - objective(b) for a, b in zip(later[winner], later['current'])]
    comparison = _paired_summary(differences)
    # Recent disappointing draws do not select the candidate.
    recent_rows = run_slice(raw, recent, locked)
    report = {
        'data': sources, 'data_role': 'historical_development_not_fresh_holdout',
        'strategies': slate, 'locked_winner': winner,
        'earlier': {n: summarize(v) for n, v in early.items()},
        'later': {n: summarize(v) for n, v in later.items()},
        'recent_issues': [raw[i]['issue'] for i in recent], 'recent_rows': recent_rows,
        'paired': comparison,
        'checks': promotion_checks(winner, later[winner], later['current'], comparison),
        'promotion_allowed': False,
        'observations': {'earlier': early, 'later': later},
    }
    out = ROOT / 'reports/kl8_early_windows_20260918.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('locked_winner', 'later', 'checks', 'recent_rows')}, indent=2))


if __name__ == '__main__':
    main()
