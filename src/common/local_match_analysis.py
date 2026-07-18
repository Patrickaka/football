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
