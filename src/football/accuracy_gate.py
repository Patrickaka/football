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
MIN_PREDICTION_RELIABILITY = 0.60
SPF_HISTORICAL_PROXY = {
    "minimum_accuracy": 0.7988,
    "sample_count": 390,
    "seasons": {
        "2024/25": {"accuracy": 0.8230, "sample_count": 226, "coverage": 0.1290},
        "2025/26": {"accuracy": 0.7988, "sample_count": 164, "coverage": 0.0936},
    },
    "basis": "dual_season_closing_market_probability_gte_70pct_margin_gte_10pct",
}

SPF_LEAGUE_POLICIES = {
    "SP1": {
        "aliases": ("SP1", "西甲", "西班牙甲级联赛", "La Liga"),
        "minimum_probability": 0.65,
        "validation_status": "dual_season_market_proxy_supported",
        "validation": {
            "minimum_accuracy": 0.8356,
            "sample_count": 130,
            "seasons": {
                "2024/25": {"accuracy": 0.8356, "sample_count": 73, "coverage": 0.1921},
                "2025/26": {"accuracy": 0.8596, "sample_count": 57, "coverage": 0.1500},
            },
            "basis": (
                "SP1_threshold_selected_on_2024_25_then_frozen_for_2025_26_holdout; "
                "closing_market_probability_gte_65pct_margin_gte_10pct"
            ),
        },
    },
    "D1": {
        "aliases": ("D1", "德甲", "德国甲级联赛", "Bundesliga"),
        "minimum_probability": 0.67,
        "validation_status": "dual_season_market_proxy_supported",
        "validation": {
            "minimum_accuracy": 0.8113,
            "sample_count": 91,
            "seasons": {
                "2024/25": {"accuracy": 0.8113, "sample_count": 53, "coverage": 0.1732},
                "2025/26": {"accuracy": 0.8158, "sample_count": 38, "coverage": 0.1242},
            },
            "basis": (
                "D1_threshold_selected_on_2024_25_then_frozen_for_2025_26_holdout; "
                "closing_market_probability_gte_67pct_margin_gte_10pct"
            ),
        },
    },
}
MIN_MARKET_MARGIN = 0.10


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


def _spf_policy(league: Any) -> dict[str, Any]:
    text = str(league or "").strip().lower()
    for code, policy in SPF_LEAGUE_POLICIES.items():
        if text and any(text == str(alias).strip().lower() for alias in policy["aliases"]):
            return {"code": code, **policy}
    return {
        "code": "global",
        "minimum_probability": SPF_MIN_PROBABILITY,
        "validation_status": "chronological_holdout_near_target",
        "validation": SPF_HISTORICAL_PROXY,
    }


def build_accuracy_gate(
    lottery: Mapping[str, Any],
    *,
    confidence: Mapping[str, Any] | None = None,
    anomaly: Mapping[str, Any] | None = None,
    league: Any = None,
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
    spf_policy = _spf_policy(league)
    configs = (
        ("spf", lottery.get("standard"), spf_policy["minimum_probability"], True),
        ("rqspf", lottery.get("handicap"), RQSPF_MIN_PROBABILITY, False),
    )
    for key, market, threshold, historically_supported in configs:
        market = market or {}
        prediction, probability, margin = _top_pick(market.get("probabilities"))
        reliability = prediction_reliability(probability, information_completeness)
        market_probs = market.get("market_probabilities")
        market_pick, market_probability, market_margin = _top_pick(market_probs)
        qualifying_probability = market_probability if key == "spf" else probability
        reasons = []
        if not prediction:
            reasons.append("缺少有效概率")
        if qualifying_probability < threshold:
            label = "官方赔率去水概率" if key == "spf" else "最高概率"
            reasons.append(f"{label}低于{threshold:.0%}")
        if margin < MIN_TOP2_MARGIN:
            reasons.append(f"领先第二选项不足{MIN_TOP2_MARGIN:.0%}")
        if key == "spf" and market_probs and market_margin < MIN_MARKET_MARGIN:
            reasons.append(f"官方赔率领先第二选项不足{MIN_MARKET_MARGIN:.0%}")
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
            "market_probability": market_probability,
            "market_margin": market_margin,
            "information_completeness": information_completeness,
            "prediction_reliability": reliability,
            "margin": margin,
            "minimum_probability": threshold,
            "threshold_scope": spf_policy["code"] if key == "spf" else "rqspf_global",
            "minimum_reliability": MIN_PREDICTION_RELIABILITY,
            "reasons": reasons,
            "validation_status": (
                spf_policy["validation_status"] if historically_supported
                else "pending_independent_rqspf_validation"
            ),
            "validation": spf_policy["validation"] if historically_supported else None,
        }
    return decisions
