"""Shared decision rules for local football and basketball match analysis.

The sport-specific modules remain responsible for producing probabilities.  This
module only turns those probabilities into a conservative, consistent decision;
it never invents an additional prediction signal.
"""

from math import isfinite


LOCAL_ANALYST_VERSION = "local-analyst-v1"


def normalize_probabilities(probabilities):
    """Return finite, non-negative probabilities normalized to one."""
    clean = {}
    for key, value in (probabilities or {}).items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        clean[key] = number if isfinite(number) and number > 0 else 0.0
    total = sum(clean.values())
    if total <= 0:
        count = len(clean)
        return {key: (1.0 / count if count else 0.0) for key in clean}
    return {key: value / total for key, value in clean.items()}


def build_decision(probabilities, confidence=None, upset_alert=False, min_single=0.56,
                   min_margin=0.10):
    """Build the common single/double/pass decision used by all match modules."""
    probs = normalize_probabilities(probabilities)
    ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return {
            "action": "观望", "primary": None, "secondary": None,
            "reason": "缺少有效概率", "playable": False,
        }
    primary, primary_prob = ranked[0]
    secondary, secondary_prob = ranked[1] if len(ranked) > 1 else (None, 0.0)
    margin = primary_prob - secondary_prob

    confidence_low = confidence in ("low", "低", None)
    if upset_alert or primary_prob < 0.50 or margin < min_margin:
        action = "双选" if secondary is not None else "观望"
        reason = "结果分歧较大，覆盖次选更稳妥" if secondary is not None else "信号不足"
    elif confidence_low or primary_prob < min_single:
        action = "观望"
        reason = "优势或数据可信度不足，不建议强行单选"
    else:
        action = "单选"
        reason = "概率优势和数据可信度同时达到阈值"

    return {
        "action": action,
        "primary": primary,
        "secondary": secondary if action == "双选" else None,
        "primary_prob": primary_prob,
        "margin": margin,
        "reason": reason,
        "playable": action in ("单选", "双选"),
    }


def pick_high_score_scenario(candidates, min_goals=4, min_mass=0.20):
    """Return the likeliest high-score cell when its aggregate tail is meaningful.

    A 4+ goal event is spread across many exact scores, so none of those cells may
    enter a raw Top3 even when the combined high-score probability is substantial.
    """
    eligible = []
    mass = 0.0
    for score, probability in candidates or []:
        try:
            home, away = int(score[0]), int(score[1])
            probability = float(probability)
        except (TypeError, ValueError, IndexError):
            continue
        if home + away >= min_goals:
            mass += probability
            eligible.append(((home, away), probability))
    if mass < min_mass or not eligible:
        return None
    score, probability = max(eligible, key=lambda item: item[1])
    return {'score': score, 'probability': probability, 'tail_probability': mass}


def build_score_strategy(candidates, confidence=None, upset_alert=False):
    """Decide whether an exact score is usable or only a score range is honest."""
    ranked = sorted(candidates or [], key=lambda item: float(item[1]), reverse=True)
    if not ranked:
        return {'action': '观望', 'playable': False, 'reason': '缺少比分分布'}
    top1_prob = float(ranked[0][1])
    top3_mass = sum(float(item[1]) for item in ranked[:3])
    confidence_low = confidence in (None, 'low', '低')
    playable = (
        not upset_alert and not confidence_low and
        top1_prob >= 0.14 and top3_mass >= 0.34
    )
    if playable:
        action = '谨慎单比分'
        reason = '单比分集中度和前三覆盖率达到历史筛选门槛'
    else:
        action = '比分区间'
        reason = '单比分概率不足，采用前三比分覆盖，避免强行单挑'
    return {
        'action': action,
        'playable': playable,
        'top1': f"{ranked[0][0][0]}-{ranked[0][0][1]}",
        'top1_prob': top1_prob,
        'top3_mass': top3_mass,
        'reason': reason,
    }
