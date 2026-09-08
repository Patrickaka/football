"""Replay only the changed totals stage; no retraining or network access."""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.football.ml import predict_goal_counts_from_candidates
from src.football import backtest, result_sync


def main():
    records = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))['records']
    rows = []
    for r in records:
        gc = r.get('goal_count') or {}
        if not r.get('settled') or not r.get('actual_score') or not gc.get('matrix') or not gc.get('distribution_dict'):
            continue
        candidates = [((h, a), p) for h, row in enumerate(gc['matrix']) for a, p in enumerate(row)]
        new = predict_goal_counts_from_candidates(
            candidates, max_goals=len(gc['matrix'])-1, use_history=False)['distribution_dict']
        old = {int(k): float(p) for k, p in gc['distribution_dict'].items()}
        actual = sum(map(int, r['actual_score'].split('-')))
        rows.append((r.get('created_at', ''), old, new, actual))
    rows.sort(key=lambda row: row[0])

    def metrics(subset, index):
        n = len(subset)
        hits = sum(max(row[index], key=row[index].get) == row[3] for row in subset)
        return {'n': n, 'hits': hits, 'accuracy': hits/n if n else None,
                'logloss': sum(-math.log(max(row[index].get(row[3], 0), 1e-12)) for row in subset)/n if n else None}

    result = {'scope': 'Historical final matrices, replacing only the subsequent totals stage. Retrospective ablation, not a fresh holdout or a complete model replay.',
              'before': metrics(rows, 1), 'after': metrics(rows, 2), 'chronological_slices': {}}
    for label, subset in [('first_60pct', rows[:int(len(rows)*.6)]),
                          ('next_20pct', rows[int(len(rows)*.6):int(len(rows)*.8)]),
                          ('last_20pct', rows[int(len(rows)*.8):])]:
        result['chronological_slices'][label] = {'before': metrics(subset, 1), 'after': metrics(subset, 2)}
    with patch.object(result_sync, 'get_history', return_value=SimpleNamespace(records=records)):
        report = backtest.rolling_backtest_from_history(limit=180, windows=(30,60,90))
    result['diagnostics'] = {key: report.get(key) for key in ('available_samples','sample_quality','error')}
    Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
