"""Build and assess a release report from frozen events; no deployment or fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.domain.sports.football.acceptance import assess_model_acceptance
from src.domain.sports.football.release_evaluation import build_release_report
from src.domain.sports.football.settlement import PRODUCTION_MODEL_VERSION
from src.football.config import FOOTBALL_PREDICTION_LOGIC_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Frozen-event manifest JSON or an existing acceptance report')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    payload = json.loads(raw)
    if payload.get('schema_version') == 'football-model-acceptance-v1':
        report = payload
    elif 'samples' in payload:
        report = build_release_report(payload)
    else:
        parser.error('Input must be a frozen-event manifest (samples) or an acceptance report; legacy exports cannot certify a release')
    report['source_sha256'] = hashlib.sha256(raw).hexdigest()
    report['acceptance'] = assess_model_acceptance(
        report, model_version=PRODUCTION_MODEL_VERSION,
        prediction_logic_version=FOOTBALL_PREDICTION_LOGIC_VERSION)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report['acceptance'], ensure_ascii=False, indent=2))
    return 0 if report['acceptance']['prediction_ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
