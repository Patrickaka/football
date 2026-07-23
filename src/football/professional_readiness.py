"""Auditable per-match evidence coverage and system capability reporting."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def _present(value: Any) -> bool:
    return value not in (None, "", [], {})


def _probability_divergence(
    model: Mapping[str, Any] | None,
    market: Mapping[str, Any] | None,
) -> float | None:
    """Return total-variation distance in [0, 1] for two 3-way markets."""
    if not model or not market:
        return None
    keys = set(model) | set(market)
    try:
        return round(
            min(1.0, 0.5 * sum(abs(float(model.get(k, 0)) - float(market.get(k, 0))) for k in keys)),
            4,
        )
    except (TypeError, ValueError):
        return None


def build_match_evidence_profile(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Score evidence availability without pretending missing data is negative evidence."""
    result = result or {}
    lottery = result.get("lottery") or {}
    standard = lottery.get("standard") or {}
    team = result.get("team") or {}
    model = result.get("model") or {}
    ml = model.get("ml") or {}
    similar = result.get("similar_market") or {}
    live = result.get("live_context") or {}

    checks = [
        ("euro_odds", "欧赔初终盘", 16, _present(result.get("euro"))),
        ("asian_odds", "亚盘初终盘", 16, _present(result.get("asian"))),
        ("total_odds", "大小球初终盘", 10, _present(result.get("total"))),
        ("team_form", "球队近期攻防", 12, _present(team)),
        (
            "expected_goals",
            "xG/xGA",
            10,
            any(_present(team.get(key)) for key in ("home_xg_last5", "away_xg_last5")),
        ),
        (
            "bookmaker_consensus",
            "多公司一致性",
            8,
            _present(result.get("bookmaker_consensus")),
        ),
        (
            "market_movement",
            "盘口时序变化",
            8,
            bool((result.get("market_change") or {}).get("used"))
            or _present(result.get("steam_move")),
        ),
        (
            "historical_analogs",
            "相似盘口历史",
            7,
            int(similar.get("sample_count", 0) or 0) >= 30,
        ),
        (
            "ml_shadow",
            "机器学习影子模型",
            5,
            bool(ml.get("ml_available") or ml.get("is_trained")),
        ),
        (
            "official_lottery_odds",
            "官方胜平负/让球赔率",
            4,
            bool(lottery.get("spf_odds") or lottery.get("rqspf_odds")),
        ),
        (
            "lineup_injuries",
            "确认首发与伤停",
            4,
            bool(live.get("lineup")) and bool(live.get("injuries")),
        ),
    ]
    earned = sum(weight for _, _, weight, available in checks if available)
    total = sum(weight for _, _, weight, _ in checks)
    score = earned / total if total else 0.0
    details = [
        {"key": key, "label": label, "weight": weight, "available": available}
        for key, label, weight, available in checks
    ]
    missing = [item["label"] for item in details if not item["available"]]

    divergence = _probability_divergence(
        standard.get("model_probabilities") or standard.get("probabilities"),
        standard.get("market_probabilities"),
    )
    if divergence is None:
        agreement = "unavailable"
    elif divergence <= 0.06:
        agreement = "strong"
    elif divergence <= 0.12:
        agreement = "moderate"
    else:
        agreement = "conflict"

    if score >= 0.85:
        grade = "A"
    elif score >= 0.70:
        grade = "B"
    elif score >= 0.50:
        grade = "C"
    else:
        grade = "D"
    blockers = []
    if not standard.get("market_probabilities"):
        blockers.append("缺少官方赔率去水概率")
    if agreement == "conflict":
        blockers.append("模型与市场概率分歧较大")
    if not bool(live.get("lineup")):
        blockers.append("未取得确认首发")
    if score < 0.70:
        blockers.append("专业证据覆盖不足70%")

    return {
        "schema_version": "football-match-evidence-v1",
        "coverage_score": round(score, 3),
        "coverage_grade": grade,
        "available_weight": earned,
        "total_weight": total,
        "checks": details,
        "missing": missing,
        "model_market_divergence": divergence,
        "model_market_agreement": agreement,
        "blockers": blockers,
    }


def build_system_gap_assessment(validation: Mapping[str, Any]) -> Dict[str, Any]:
    """Expose professional capabilities and remaining gaps in priority order."""
    validation = validation or {}
    model = validation.get("model_metrics") or {}
    market = validation.get("market_baseline_metrics") or {}
    strategy = validation.get("strategy") or {}
    capabilities = [
        "欧赔、亚盘、大小球联合建模",
        "概率校准与多模型集成",
        "严格时间顺序 Walk-forward 回测",
        "赛后自动回填、滚动诊断与样本质量过滤",
        "ROI、CLV、LogLoss、Brier 与回撤监控",
        "低置信自动观望和生产磁盘保护",
    ]
    gaps = []
    if not model or not market or float(model.get("logloss", 99)) >= float(market.get("logloss", 99)):
        gaps.append({"priority": "P0", "name": "模型尚未跑赢市场概率", "action": "继续积累无泄漏样本，按联赛和时间层重训并校准"})
    if float(strategy.get("roi", 0) or 0) <= 0:
        gaps.append({"priority": "P0", "name": "样本外ROI尚未转正", "action": "保持观望门控，禁止用命中率替代盈利验证"})
    if float(strategy.get("mean_clv", 0) or 0) <= 0:
        gaps.append({"priority": "P0", "name": "平均CLV尚未转正", "action": "持续保存开盘、推荐时点和收盘赔率"})
    gaps.extend([
        {"priority": "P1", "name": "确认首发、伤停和停赛覆盖不足", "action": "接入可靠实时数据源并记录来源与时间戳"},
        {"priority": "P1", "name": "让球胜平负独立历史验证不足", "action": "单独沉淀体彩让球赔率、预测和赛果，禁止沿用胜平负代理指标"},
        {"priority": "P1", "name": "跨公司赔率时序深度不足", "action": "保存多公司T-24h/T-6h/T-1h/临场快照并监测异常漂移"},
        {"priority": "P2", "name": "联赛/赛季漂移监控仍需加强", "action": "按联赛、月份、概率桶监控校准误差与覆盖率"},
    ])
    return {"capabilities": capabilities, "gaps": gaps}
