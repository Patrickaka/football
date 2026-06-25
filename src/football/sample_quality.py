#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Historical sample quality helpers for football prediction calibration/backtests.
"""

from typing import Dict, Iterable, List, Tuple


GRADE_RANK = {
    'reject': 0,
    'low': 1,
    'medium': 2,
    'high': 3,
}


FRIENDLY_KEYWORDS = (
    '友谊',
    'friendly',
    '热身',
    'club friendly',
)


def _has_any(record: Dict, keys: Iterable[str]) -> bool:
    return any(record.get(key) not in (None, '', [], {}) for key in keys)


def _is_friendly(record: Dict) -> bool:
    league = str(record.get('league') or '').lower()
    match_type = str(record.get('match_type') or record.get('type') or '').lower()
    text = f"{league} {match_type}"
    return any(keyword.lower() in text for keyword in FRIENDLY_KEYWORDS)


def assess_record_quality(record: Dict) -> Dict:
    """Return a conservative quality assessment for one historical record."""
    reasons: List[str] = []
    score = 1.0
    source_weight = 0.70

    if not record.get('settled') and not record.get('actual_score'):
        reasons.append('unsettled')
        score -= 0.55

    if not record.get('actual_score'):
        reasons.append('missing_actual_score')
        score -= 0.45

    if not record.get('predicted_scores'):
        reasons.append('missing_predicted_scores')
        score -= 0.35

    if not record.get('predicted_1x2'):
        reasons.append('missing_predicted_1x2')
        score -= 0.15

    if record.get('sync_status') in {'failed', 'ignored'}:
        reasons.append(f"sync_{record.get('sync_status')}")
        score -= 0.25

    result_quality = record.get('result_quality') or {}
    result_grade = result_quality.get('grade')
    result_quality_usable = True
    result_source = result_quality.get('source') or record.get('result_source') or record.get('sync_source')
    if result_source == 'live_fid':
        source_weight = 1.0
    elif result_source == 'live_team':
        source_weight = 0.85
    elif result_source == 'shuju':
        source_weight = 0.60
    elif result_source:
        source_weight = 0.70
    if result_grade in {'reject', 'low'}:
        reasons.append(f"result_quality_{result_grade}")
        score -= 0.35 if result_grade == 'low' else 0.65
        result_quality_usable = False
    elif not result_quality and record.get('settled'):
        reasons.append('missing_result_quality')
        score -= 0.12

    if _is_friendly(record):
        reasons.append('friendly_match')
        score -= 0.25

    if _has_any(record, ('red_card', 'red_cards', 'home_red_cards', 'away_red_cards')):
        reasons.append('red_card_flag')
        score -= 0.25

    if record.get('asian') is None:
        reasons.append('missing_asian_line')
        score -= 0.12

    if record.get('total_line') is None:
        reasons.append('missing_total_line')
        score -= 0.12

    odds_snapshot = record.get('odds_snapshot') or record.get('odds_data') or {}
    if not odds_snapshot:
        reasons.append('missing_odds_snapshot')
        score -= 0.08

    score = max(0.0, min(1.0, score))
    if score >= 0.80:
        grade = 'high'
    elif score >= 0.55:
        grade = 'medium'
    elif score >= 0.30:
        grade = 'low'
    else:
        grade = 'reject'

    usable_for_calibration = GRADE_RANK[grade] >= GRADE_RANK['medium'] and result_quality_usable
    calibration_weight = 0.0
    if usable_for_calibration:
        grade_weight = {'high': 1.0, 'medium': 0.75}.get(grade, 0.0)
        calibration_weight = max(0.0, min(1.0, grade_weight * source_weight))

    return {
        'score': round(score, 3),
        'grade': grade,
        'reasons': reasons,
        'usable_for_calibration': usable_for_calibration,
        'calibration_weight': round(calibration_weight, 3),
        'source_weight': round(source_weight, 3),
    }


def filter_quality_records(records: List[Dict],
                           min_grade: str = 'medium',
                           exclude_friendly: bool = True) -> Tuple[List[Dict], Dict]:
    """Filter historical records and return kept records plus a compact report."""
    min_rank = GRADE_RANK.get(min_grade, GRADE_RANK['medium'])
    kept: List[Dict] = []
    rejected: List[Dict] = []
    reason_counts: Dict[str, int] = {}
    grade_counts: Dict[str, int] = {'high': 0, 'medium': 0, 'low': 0, 'reject': 0}

    for record in records:
        quality = assess_record_quality(record)
        grade_counts[quality['grade']] += 1
        for reason in quality['reasons']:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        reject_friendly = exclude_friendly and 'friendly_match' in quality['reasons']
        if GRADE_RANK[quality['grade']] >= min_rank and not reject_friendly:
            enriched = record.copy()
            enriched['sample_quality'] = quality
            kept.append(enriched)
        else:
            rejected.append({
                'match_id': record.get('match_id'),
                'league': record.get('league'),
                'home': record.get('home'),
                'away': record.get('away'),
                'quality': quality,
            })

    return kept, {
        'input_count': len(records),
        'kept_count': len(kept),
        'rejected_count': len(rejected),
        'min_grade': min_grade,
        'exclude_friendly': exclude_friendly,
        'grade_counts': grade_counts,
        'reason_counts': dict(sorted(reason_counts.items())),
        'rejected_examples': rejected[:10],
    }
