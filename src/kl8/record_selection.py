"""Pure selection/audit of the first formal prediction for an independent draw.

An actual aware draw timestamp takes precedence. If the source only supplies
the draw's calendar date, use the published 21:30 Asia/Shanghai schedule:
https://www.szlottery.org/fcw/fcxw/csdt/content/post_1646091.html
The cutoff is labelled as scheduled, never as an observed draw timestamp.
Never infer a draw date from its issue number or a settlement's creation time.
"""
from datetime import datetime, time, timezone, timedelta
import hashlib
import json
import re

LOCAL = timezone(timedelta(hours=8))
POLICY = 'first-formal-predraw-v1'


def _aware(value):
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        return parsed.astimezone(timezone.utc) if parsed.utcoffset() is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _prediction_time(snapshot):
    explicit = _aware(snapshot.get('predicted_at'))
    raw_ns = snapshot.get('predicted_at_ns')
    if raw_ns not in (None, '', 0):
        try:
            if isinstance(raw_ns, bool) or not re.fullmatch(r'[0-9]+', str(raw_ns)):
                raise ValueError('invalid epoch')
            ns = int(raw_ns)
            instant = datetime.fromtimestamp(ns / 1_000_000_000, timezone.utc)
            if instant.year < 2020 or instant.year > 2200:
                raise ValueError('invalid epoch range')
            if explicit and abs((instant - explicit).total_seconds()) >= 1:
                return None, 'conflicting_prediction_timestamps'
            return instant, 'epoch_ns'
        except (TypeError, ValueError, OverflowError, OSError):
            return None, 'invalid_prediction_epoch'
    if explicit:
        return explicit, 'aware_timestamp'
    return None, 'unverified_legacy_timestamp'


def _draw_cutoff(draw):
    if not isinstance(draw, dict):
        return None, 'draw_date_unavailable'
    for key in ('draw_at', 'draw_timestamp', 'draw_time'):
        if draw.get(key):
            actual = _aware(draw[key])
            return actual, 'actual_draw_timestamp' if actual else 'invalid_actual_draw_timestamp'
    value = draw.get('date') or draw.get('draw_date')
    try:
        date = datetime.strptime(str(value)[:10], '%Y-%m-%d').date()
        return datetime.combine(date, time(21, 30), LOCAL).astimezone(timezone.utc), 'scheduled_2130_asia_shanghai'
    except (ValueError, TypeError):
        return None, 'draw_date_unavailable'


def audit_prediction(snapshot, draw=None):
    predicted, source = _prediction_time(snapshot)
    cutoff, cutoff_source = _draw_cutoff(draw)
    result = {'eligible': False, 'selection_policy': POLICY,
              'prediction_time_source': source, 'draw_cutoff_source': cutoff_source,
              'prediction_at': predicted.isoformat() if predicted else None,
              'draw_cutoff_at': cutoff.isoformat() if cutoff else None}
    issue, based = str(snapshot.get('target_issue') or ''), str(snapshot.get('based_on_issue') or '')
    if snapshot.get('is_experiment') is True:
        reason = 'experimental_prediction'
    elif snapshot.get('is_experiment') is not False:
        reason = 'formal_status_unverified'
    elif not snapshot.get('version') or not snapshot.get('snapshot_id'):
        reason = 'missing_prediction_identity'
    elif not re.fullmatch(r'[0-9]{7}', issue) or not re.fullmatch(r'[0-9]{7}', based) or based >= issue:
        reason = 'invalid_prediction_issue_boundary'
    elif isinstance(draw, dict) and str(draw.get('issue') or issue) != issue:
        reason = 'draw_issue_mismatch'
    elif predicted is None:
        reason = source
    elif cutoff is None:
        reason = cutoff_source
    elif predicted >= cutoff:
        reason = 'prediction_at_or_after_draw'
    else:
        reason = 'verified_formal_before_draw'
        result['eligible'] = True
    result['reason'] = reason
    return result


def select_canonical_snapshots(snapshots, draws=None):
    """Keep one earliest formal record per issue without rewriting any record.

    Proven predraw records win. Unverifiable old formal records remain visible
    with eligible=False. A newer version/configuration or experiment cannot
    replace the historical first forecast solely by being newer.
    """
    groups = {}
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        issue = str(snapshot.get('target_issue') or '')
        if not issue:
            continue
        audit = audit_prediction(snapshot, (draws or {}).get(issue))
        groups.setdefault(issue, []).append({**snapshot, 'prediction_audit': audit})

    def order(snapshot):
        audit = snapshot['prediction_audit']
        rank = 0 if audit['eligible'] else (2 if snapshot.get('is_experiment') is True else 1)
        # Naive legacy text is used only for display ordering, never as proof.
        instant = audit['prediction_at'] or str(snapshot.get('predicted_at') or '9999')
        try:
            ns = int(snapshot.get('predicted_at_ns') or 0)
        except (TypeError, ValueError):
            ns = 0
        return rank, instant, ns, str(snapshot.get('snapshot_id') or snapshot.get('file') or '')

    return [min(groups[issue], key=order) for issue in sorted(groups, reverse=True)]


def audit_settlement(snapshot, settlement, draw, *, snapshot_sha256):
    audit = audit_prediction(snapshot, draw)
    if not audit['eligible']:
        return audit
    reason = None
    if (not isinstance(settlement, dict)
            or settlement.get('snapshot_id') != snapshot.get('snapshot_id')
            or str(settlement.get('actual_issue')) != str(snapshot.get('target_issue'))):
        reason = 'settlement_identity_mismatch'
    elif not snapshot_sha256 or settlement.get('snapshot_sha256') != snapshot_sha256:
        reason = 'settlement_snapshot_hash_mismatch'
    else:
        actual = settlement.get('actual_numbers')
        expected = (draw or {}).get('numbers')
        def valid_numbers(values):
            return (isinstance(values, list) and len(values) == 20
                    and all(isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 80 for n in values)
                    and len(set(values)) == 20)
        if not valid_numbers(actual) or not valid_numbers(expected) or set(actual) != set(expected):
            reason = 'settlement_draw_numbers_mismatch'
        else:
            # The immutable snapshot, not a mutable stored hit counter, is the evidence.
            from .config import SELECT_TYPES, FUSHI_CONFIG
            plays = [(f'select_{n}', f'select_{n}', n, 'prize_settlement', 'hits') for n in SELECT_TYPES]
            plays += [(play, play, cfg['pool_size'], 'fushi_settlement', 'pool_hits')
                      for play, cfg in FUSHI_CONFIG.items()]
            for play, field, size, section, counter in plays:
                row = (settlement.get(section) or {}).get(play) or {}
                if not row.get('placed'):
                    continue
                numbers = snapshot.get(field) or []
                if (not isinstance(numbers, list) or len(numbers) != size
                        or any(not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 80 for n in numbers)
                        or len(set(numbers)) != size
                        or row.get(counter) != len(set(numbers) & set(actual))):
                    reason = 'settlement_hit_count_mismatch'
                    break
    return {**audit, 'eligible': reason is None, 'reason': reason or audit['reason']}


def strategy_cohort(snapshot, play_type):
    """Group by the saved version and actual saved strategy, never today's config."""
    strategy_id = (snapshot.get('play_strategies') or {}).get(play_type, '')
    payload = {'version': snapshot.get('version'), 'play_type': play_type,
               'strategy_id': strategy_id,
               'resolved_strategy': (snapshot.get('resolved_strategies') or {}).get(play_type),
               'configuration': snapshot.get('strategy_config_fingerprint')}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                   separators=(',', ':')).encode()).hexdigest()[:20]
    return {'key': key, 'version': payload['version'], 'strategy_id': strategy_id, 'play_type': play_type}
