#!/usr/bin/env python
"""Collect factual match context in a background/operations process, not an API request."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.football.intelligence import build_agent_from_environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', action='store_true', help='Report configuration without network/model calls')
    parser.add_argument('--match-file', help='JSON object with match_id, home, away, timezone-aware kickoff')
    parser.add_argument('--findings', help='Existing fetch_live_context findings, with explicit publication/collection metadata')
    parser.add_argument('--as-of', help='Strict historical cutoff; disables new network collection')
    parser.add_argument('--cached', action='store_true', help='Read completed cache only')
    parser.add_argument('--force', action='store_true', help='Research even if an unexpired snapshot exists')
    parser.add_argument('--out', help='Optional output MatchContext JSON file')
    parser.add_argument('--budget-seconds', type=float, default=12)
    args = parser.parse_args(argv)
    agent = build_agent_from_environment()
    agent.budget_seconds = max(.1, min(args.budget_seconds, 60))
    if args.status:
        print(json.dumps(agent.get_status(), ensure_ascii=False, indent=2))
        return 0
    if not args.match_file:
        parser.error('--match-file is required unless --status is used')
    match = json.loads(Path(args.match_file).read_text(encoding='utf-8'))
    findings = json.loads(Path(args.findings).read_text(encoding='utf-8')) if args.findings else []
    if args.cached:
        result = agent.get_cached_context(match['match_id'],
                                          as_of=args.as_of or datetime.now(timezone.utc),
                                          kickoff=match.get('kickoff') or match.get('time') or match.get('match_time'))
    else:
        result = agent.research_match_context(match, as_of=args.as_of, findings=findings, force=args.force)
    if result is None:
        print(json.dumps({'status': 'cache_missing'}, ensure_ascii=False))
        return 2
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({
        'match_id': result['match_id'], 'status': result['status'],
        'captured_at': result['captured_at'], 'facts': len(result['evidence']),
        'verified_facts': sum(row.get('verified') is True for row in result['evidence']),
        'missing_categories': result['missing_categories'], 'errors': result['errors'],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
