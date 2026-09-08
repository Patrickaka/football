#!/usr/bin/env python
"""Evaluate frozen football variants or train a local intelligence candidate.

Evaluation never rewrites predictions. Training writes a JSON candidate only;
it does not promote it, change the running model, or select on the test block.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_records(path=None):
    if path:
        data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        records = data.get('records') if isinstance(data, dict) else data
        if not isinstance(records, list):
            raise ValueError('input must be prediction records or an export with records')
        from src.common.football_storage import decode_record
        return [decode_record(record) for record in records]
    from src.football.result_sync import get_history
    return get_history().records


def training_rows(records, *, model_version=None):
    from src.domain.sports.football.prediction_evaluation import _settled_rows
    rows, excluded = _settled_rows(records)
    result = []
    for row in rows:
        event = row['event']
        record = row['record']
        if model_version and event['model_version'] != model_version:
            excluded['model_version_mismatch'] = excluded.get('model_version_mismatch', 0) + 1
            continue
        base = (event['variants'].get('market_adjusted') or {}).get('probabilities')
        result.append({'match_id': event['match_id'], 'kickoff_at': event['kickoff_at'],
                       'as_of': event['as_of'], 'settled_at': row['record']['settled_at'],
                       'frozen': True, 'intelligence': (event.get('context') or {}).get('intelligence') or {},
                       'base_probabilities': base, 'actual': row['actual']})
        result[-1].update(production_model_version=event['model_version'],
                          prediction_logic_version=event['prediction_logic_version'],
                          baseline_version=(event['variants'].get('market_adjusted') or {}).get('baseline_version'),
                          result_quality=record.get('result_quality'),
                          exclude_from_calibration=record.get('exclude_from_calibration', False))
    return result, excluded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('evaluate', 'train-intelligence', 'status'))
    parser.add_argument('--input', help='local export JSON; omit to read the configured store')
    parser.add_argument('--output', help='write report/candidate JSON')
    parser.add_argument('--model-version', help='evaluate only one production version')
    args = parser.parse_args(argv)
    if args.command == 'status':
        from src.football.intelligence import build_agent_from_environment
        from src.football.research import load_residual_artifact
        from src.football.research_model import artifact_eligibility
        agent = build_agent_from_environment()
        eligible, reason = artifact_eligibility(load_residual_artifact(), as_of=datetime.now(timezone.utc))
        report = {'sources': [source.name for source in agent.sources],
                  'ollama_configured': agent.extractor is not None,
                  'intelligence_model_eligible': eligible, 'reason': reason,
                  'note': 'Configuration status only; no requests or model training are performed.'}
    else:
        records = load_records(args.input)
        if args.command == 'evaluate':
            from src.domain.sports.football.prediction_evaluation import evaluate_frozen_events
            report = evaluate_frozen_events(records, model_version=args.model_version)
        else:
            if not args.output:
                parser.error('train-intelligence requires --output for a reviewable candidate artifact')
            from src.football.research_model import train_residual
            rows, excluded = training_rows(records, model_version=args.model_version)
            report = train_residual(rows)
            report['ledger_exclusions'] = excluded
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + '\n', encoding='utf-8')
        print(str(path.resolve()))
    else:
        print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
