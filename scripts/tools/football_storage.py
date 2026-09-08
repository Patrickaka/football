#!/usr/bin/env python
"""Preview or run football cache retention and lossless history archiving."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='apply cache cleanup and lossless compression')
    parser.add_argument('--dry-run', action='store_true', help='preview only (the default)')
    parser.add_argument('--output', help='optional maintenance report JSON')
    args = parser.parse_args(argv)
    if args.apply and args.dry_run:
        parser.error('--apply and --dry-run cannot be combined')
    from src.football.storage_maintenance import run_football_storage_maintenance
    report = run_football_storage_maintenance(dry_run=not args.apply)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + '\n', encoding='utf-8')
    print(rendered)
    return int(any(value.get('errors') for value in report.values() if isinstance(value, dict)))


if __name__ == '__main__':
    raise SystemExit(main())
