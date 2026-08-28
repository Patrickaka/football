# -*- coding: utf-8 -*-
"""准确率闸门：拿历史命中判断某个口径能不能对外展示。"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


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

TOTAL_GOALS_LEAGUE_POLICIES = {
    "SP1": {
        "aliases": ("SP1", "西甲", "西班牙甲级联赛", "La Liga"),
        "minimum_probability": 0.65,
        "training": {"accuracy": 0.7126, "sample_count": 87},
        "holdout": {"accuracy": 0.6757, "sample_count": 74},
    },
    "D1": {
        "aliases": ("D1", "德甲", "德国甲级联赛", "Bundesliga"),
        "minimum_probability": 0.65,
        "training": {"accuracy": 0.7778, "sample_count": 72},
        "holdout": {"accuracy": 0.7927, "sample_count": 82},
    },
    "F1": {
        "aliases": ("F1", "法甲", "法国甲级联赛", "Ligue 1"),
        "minimum_probability": 0.62,
        "training": {"accuracy": 0.7167, "sample_count": 60},
        "holdout": {"accuracy": 0.7037, "sample_count": 54},
    },
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

def _league_policy(league: Any, policies: Mapping[str, Any]) -> tuple[str | None, dict]:
    """按别名找联赛策略——**别名比较去空白、不分大小写**。

    这段逻辑原本在本文件里有**逐字相同的两份**（另一份在
    `_static_spf_policy`），变异验证时同一个锚点匹配到两处才发现。
    同一件事实现两遍就会漂（判据 11），现在只剩这一份。
    """
    text = str(league or "").strip().lower()
    for code, policy in policies.items():
        if text and any(text == str(alias).strip().lower() for alias in policy["aliases"]):
            return code, policy
    return None, {}

def _static_spf_policy(league: Any) -> dict[str, Any] | None:
    code, policy = _league_policy(league, SPF_LEAGUE_POLICIES)
    return {"code": code, **policy} if code else None

def has_static_spf_policy(league: Any) -> bool:
    return _static_spf_policy(league) is not None

def _spf_policy(
    league: Any,
    production_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    static = _static_spf_policy(league)
    if static:
        return static
    production_policy = production_policy or {}
    if production_policy.get("supported"):
        league_label = str(production_policy.get("league") or league or "production")
        return {
            "code": f"production:{league_label}",
            "minimum_probability": float(production_policy["minimum_probability"]),
            "validation_status": "production_chronological_holdout_supported",
            "validation": {
                "training": production_policy.get("training"),
                "holdout": production_policy.get("holdout"),
                "sample_count": production_policy.get("sample_count"),
                "target_accuracy": production_policy.get("target_accuracy"),
                "selection_rule": production_policy.get("selection_rule"),
                "source": "mysql_first_settled_prediction_history",
            },
        }
    return {
        "code": "global",
        "minimum_probability": SPF_MIN_PROBABILITY,
        "validation_status": "chronological_holdout_near_target",
        "validation": SPF_HISTORICAL_PROXY,
    }


def build_total_goals_gate(
    total: Mapping[str, Any] | None,
    *,
    league: Any = None,
    goal_count: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Selective O/U 2.5 gate validated by frozen cross-season league rules."""
    total = total or {}
    code, policy = _league_policy(league, TOTAL_GOALS_LEAGUE_POLICIES)
    try:
        line = float(total.get("close_line"))
    except (TypeError, ValueError):
        line = None
    market_probabilities = total.get("close_prob") or {}
    direction, probability, margin = _top_pick(market_probabilities)
    threshold = float(policy.get("minimum_probability", 1.0))
    reasons = []
    if not code:
        reasons.append("该联赛大小球规则未通过冻结跨赛季验证")
    if line is None or abs(line - 2.5) > 1e-9:
        reasons.append("当前验证仅覆盖2.5球盘口")
    if direction not in {"over", "under"}:
        reasons.append("缺少有效大小球去水概率")
    if probability < threshold:
        reasons.append(f"大小球去水概率低于{threshold:.0%}")

    model_over_under = (goal_count or {}).get("over_under") or {}
    model_direction, model_probability, _ = _top_pick({
        key: model_over_under.get(key) for key in ("over", "under")
    })
    selected = not reasons
    return {
        "selected": selected,
        "decision": direction if selected else "观望",
        "candidate": direction,
        "line": line,
        "probability": probability,
        "margin": margin,
        "minimum_probability": threshold if code else None,
        "threshold_scope": code,
        "model_direction": model_direction,
        "model_probability": model_probability,
        "reasons": reasons,
        "validation_status": (
            "dual_season_market_proxy_supported" if code
            else "pending_league_validation"
        ),
        "validation": ({
            "training_season": "2024/25",
            "holdout_season": "2025/26",
            "training": policy.get("training"),
            "holdout": policy.get("holdout"),
            "selection_rule": "maximize training accuracy with n>=40, then freeze threshold",
            "market": "bookmaker_average_closing_over_under_2_5",
        } if code else None),
    }

def build_accuracy_gate(
    lottery: Mapping[str, Any],
    *,
    confidence: Mapping[str, Any] | None = None,
    anomaly: Mapping[str, Any] | None = None,
    upset: Mapping[str, Any] | None = None,
    league: Any = None,
    production_spf_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return separate abstention decisions for SPF and RQSPF."""

    confidence = confidence or {}
    anomaly = anomaly or {}
    upset = upset or {}
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
    spf_policy = _spf_policy(league, production_spf_policy)
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
        if key == "spf" and upset.get("alert"):
            reasons.append("爆冷信号触发，正路降为防冷观察")

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
    decisions["upset"] = {
        "selected": False,
        "watch": upset.get("alert") is True,
        "decision": upset.get("recommended_cover") if upset.get("alert") else "无防冷信号",
        "candidate": upset.get("recommended_cover"),
        "level": upset.get("level"),
        "risk_score": upset.get("risk_score"),
        "signals": list(upset.get("signals") or []),
        "defensive_selections": list(upset.get("defensive_selections") or []),
        "validation_status": "persisted_prematch_audit_pending",
        "reasons": (
            ["防冷模型正在积累独立生产结算样本"]
            if upset.get("alert") else []
        ),
    }
    return decisions
