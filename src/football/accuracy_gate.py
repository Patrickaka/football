"""High-precision selection gates for football 1X2 markets.

The gate deliberately abstains when the available evidence is not strong
enough.  A target accuracy is not presented as achieved until an independent
out-of-sample audit supports it.
"""

from __future__ import annotations

from typing import Any, Mapping


TARGET_ACCURACY = 0.80
SPF_MIN_PROBABILITY = 0.70
RQSPF_MIN_PROBABILITY = 0.80
MIN_TOP2_MARGIN = 0.18
MIN_PREDICTION_RELIABILITY = 0.80
SPF_HISTORICAL_PROXY = {
    "accuracy": 0.8027,
    "sample_count": 375,
    "coverage": 0.1070,
    "basis": "historical_market_implied_probability_gte_70pct",
}


def _top_pick(probabilities: Mapping[str, Any] | None) -> tuple[str | None, float, float]:
    values = []
    for key, value in (probabilities or {}).items():
        try:
            values.append((str(key), float(value)))
        except (TypeError, ValueError):
            continue
    values.sort(key=lambda item: item[1], reverse=True)
    if not values:
        return None, 0.0, 0.0
    second = values[1][1] if len(values) > 1 else 0.0
    return values[0][0], values[0][1], values[0][1] - second


def _market_agrees(market: Mapping[str, Any] | None, prediction: str | None) -> bool:
    market_pick, _, _ = _top_pick(market)
    return bool(prediction and market_pick and prediction == market_pick)


def prediction_reliability(probability: float, information_completeness: float) -> float:
    """Shrink a 3-way top-pick probability toward the 1/3 ignorance prior."""
    probability = max(0.0, min(1.0, float(probability)))
    information_completeness = max(0.0, min(1.0, float(information_completeness)))
    baseline = 1.0 / 3.0
    return baseline + (probability - baseline) * information_completeness


def build_accuracy_gate(
    lottery: Mapping[str, Any],
    *,
    confidence: Mapping[str, Any] | None = None,
    anomaly: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return separate abstention decisions for SPF and RQSPF."""

    confidence = confidence or {}
    anomaly = anomaly or {}
    euro_asian = anomaly.get("euro_asian_deviation") or {}
    conflict = abs(float(euro_asian.get("abs_deviation") or 0.0)) >= 0.50
    low_data_quality = confidence.get("level") == "low"
    try:
        information_completeness = float(confidence.get("score", 1.0))
    except (TypeError, ValueError):
        information_completeness = 1.0

    decisions: dict[str, Any] = {
        "target_accuracy": TARGET_ACCURACY,
        "policy": "selective_prediction_with_abstention",
    }
    configs = (
        ("spf", lottery.get("standard"), SPF_MIN_PROBABILITY, True),
        ("rqspf", lottery.get("handicap"), RQSPF_MIN_PROBABILITY, False),
    )
    for key, market, threshold, historically_supported in configs:
        market = market or {}
        prediction, probability, margin = _top_pick(market.get("probabilities"))
        reliability = prediction_reliability(probability, information_completeness)
        market_probs = market.get("market_probabilities")
        reasons = []
        if not prediction:
            reasons.append("缺少有效概率")
        if probability < threshold:
            reasons.append(f"最高概率低于{threshold:.0%}")
        if margin < MIN_TOP2_MARGIN:
            reasons.append(f"领先第二选项不足{MIN_TOP2_MARGIN:.0%}")
        if reliability < MIN_PREDICTION_RELIABILITY:
            reasons.append(f"预测可信度低于{MIN_PREDICTION_RELIABILITY:.0%}")
        if not market_probs:
            reasons.append("缺少官方赔率去水概率")
        elif not _market_agrees(market_probs, prediction):
            reasons.append("模型与官方赔率方向不一致")
        if low_data_quality:
            reasons.append("基础数据置信度低")
        if conflict:
            reasons.append("欧赔与亚盘明显冲突")

        selected = not reasons
        decisions[key] = {
            "selected": selected,
            "decision": prediction if selected else "观望",
            "candidate": prediction,
            "probability": probability,
            "information_completeness": information_completeness,
            "prediction_reliability": reliability,
            "margin": margin,
            "minimum_probability": threshold,
            "minimum_reliability": MIN_PREDICTION_RELIABILITY,
            "reasons": reasons,
            "validation_status": (
                "historical_proxy_supported" if historically_supported
                else "pending_independent_rqspf_validation"
            ),
            "validation": SPF_HISTORICAL_PROXY if historically_supported else None,
        }
    return decisions
