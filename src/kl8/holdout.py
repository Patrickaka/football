"""Persist final-test exposure before reading results; never recycle a failed test."""
from bisect import bisect_right
import hashlib
import json
import threading
from .main_play_validation import play_family

_lock = threading.RLock()
EXPOSURE_KIND = 'holdout_exposure'


def reserve_final_holdout(play_type, strategy, history, bounds, trials, persist, *, version, minimum=200):
    rows = sorted(history, key=lambda row: str(row.get('issue', '')))
    issues = [str(row.get('issue', '')) for row in rows]
    if (len(set(issues)) != len(issues) or any(len(i) != 7 or not i.isdigit() for i in issues)
            or minimum < 1 or bounds[0] < 0 or bounds[1] > len(rows) or bounds[0] >= bounds[1]):
        return {'available': False, 'reason': 'invalid_holdout_issue_boundary'}
    with _lock:
        family = play_family(play_type)
        prior = [t for t in trials if isinstance(t, dict) and play_family(t.get('play_type')) == family]
        exposures = [t for t in prior if t.get('tournament_round') == EXPOSURE_KIND]
        if any(len(str(t.get('last_issue', ''))) != 7 or not str(t.get('last_issue', '')).isdigit()
               for t in exposures):
            return {'available': False, 'reason': 'invalid_prior_holdout_boundary'}
        legacy = [t for t in prior if t.get('tournament_round') != EXPOSURE_KIND and t.get('evidence_schema') != 2]
        # Old trials have no exposure provenance. Mark the existing history
        # once as already inspected, then wait for new draws rather than guess.
        baseline_only = bool(legacy and not exposures)
        last_seen = max((str(t.get('last_issue', '')) for t in exposures), default='')
        start = max(bounds[0], bisect_right(issues, last_seen))
        if not baseline_only and bounds[1] - start < minimum:
            return {'available': False, 'reason': 'insufficient_fresh_final_draws',
                    'fresh_draws': max(0, bounds[1] - start), 'required': minimum,
                    'last_exposed_issue': last_seen or None}
        end = bounds[1] if baseline_only else start + minimum
        selected = issues[bounds[0]:end] if baseline_only else issues[start:end]
        digest = hashlib.sha256(json.dumps(strategy, sort_keys=True, ensure_ascii=False,
                                           separators=(',', ':')).encode()).hexdigest()
        reservation = {
            'play_type': family, 'requested_play_type': play_type,
            'strategy_id': strategy.get('strategy_id', ''),
            'strategy_fingerprint': digest, 'version': version, 'evidence_schema': 2,
            'tournament_round': EXPOSURE_KIND, 'first_issue': selected[0], 'last_issue': selected[-1],
            'issue_count': len(selected), 'issues_sha256': hashlib.sha256('|'.join(selected).encode()).hexdigest(),
            'legacy_exposure_boundary': baseline_only,
        }
        reservation['tested_at'] = reservation['issues_sha256']
        trials.append(reservation)
        try:
            saved = persist() is True
        except Exception:
            saved = False
        if not saved:
            trials.remove(reservation)
            return {'available': False, 'reason': 'holdout_reservation_not_persisted'}
        if baseline_only:
            return {'available': False, 'reason': 'legacy_final_exposure_unknown',
                    'last_exposed_issue': selected[-1], 'required': minimum}
        return {'available': True, 'reason': 'reserved_fresh_final_draws',
                'range': (start, end), 'reservation': reservation}
