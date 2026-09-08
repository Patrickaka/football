"""Pure, fail-closed acceptance of the current football prediction pipeline.

This verifies the report contract and paired out-of-sample evidence. It does
not infer a betting edge or validate ROI/CLV. The producer remains responsible
for computing the paired intervals from immutable, unique prematch records.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from numbers import Integral, Real
from typing import Any, Mapping


ACCEPTANCE_SCHEMA_VERSION = "football-model-acceptance-v1"
MIN_OUT_OF_SAMPLE_N = 1000
MIN_INDEPENDENT_DAYS = 30
MAX_REPORT_AGE_DAYS = 14
CLOCK_SKEW_SECONDS = 300
ESTIMATE_TOLERANCE = 1e-6


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _count(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value > 0


def _timestamp(value: Any, *, allow_datetime: bool = False) -> datetime | None:
    try:
        if isinstance(value, datetime) and allow_datetime:
            parsed = value
        elif isinstance(value, str) and "T" in value:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            return None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(timezone.utc)
    except (OverflowError, TypeError, ValueError):
        return None


def _valid_metrics(metrics: Mapping[str, Any]) -> bool:
    accuracy, brier, logloss = (metrics.get(key) for key in ("accuracy", "brier", "logloss"))
    return (
        _count(metrics.get("n"))
        and _number(accuracy) and 0 <= accuracy <= 1
        and _number(brier) and 0 <= brier <= 2
        and _number(logloss) and logloss >= 0
    )


def _valid_interval(value: Any, *, bound: float | None = None) -> bool:
    interval = _mapping(value)
    estimate, ci = interval.get("estimate"), interval.get("ci95")
    if not (
        _number(estimate) and isinstance(ci, (list, tuple)) and len(ci) == 2
        and all(_number(item) for item in ci)
    ):
        return False
    lo, hi = ci
    return lo <= estimate <= hi and (
        bound is None or (-bound <= lo and hi <= bound)
    )


def assess_model_acceptance(
    report: Any,
    *,
    model_version: str,
    prediction_logic_version: str,
    now: datetime | str | None = None,
) -> dict[str, Any]:
    """Accept only current, reproducible production-pipeline evidence.

    Accuracy must not regress, with a nonnegative paired 95% lower bound;
    logloss must show a strictly positive paired improvement lower bound;
    Brier must also have a nonnegative paired improvement lower bound.
    A promising point estimate alone never makes ``prediction_ready`` true.
    """
    payload = _mapping(report)
    model = _mapping(payload.get("model_metrics"))
    market = _mapping(payload.get("market_baseline_metrics"))
    paired = _mapping(payload.get("paired_comparison"))
    audit = _mapping(payload.get("audit"))
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    def check(key: str, condition: Any, reason: str) -> None:
        checks[key] = bool(condition)
        if not checks[key]:
            reasons.append(reason)

    check("report_available", bool(payload), "缺少可核验的模型验收报告")
    check("schema_current", payload.get("schema_version") == ACCEPTANCE_SCHEMA_VERSION,
          "验收协议缺失或过期，必须重新生成 football-model-acceptance-v1 报告")
    check("production_pipeline", payload.get("evaluation_scope") == "production_pipeline",
          "验证范围不是完整生产预测流程，离线子模型成绩不能替代线上版本验收")
    check("current_model_version", bool(model_version) and payload.get("model_version") == model_version,
          "验收报告的模型版本与当前运行版本不一致")
    check("current_prediction_logic_version", bool(prediction_logic_version)
          and payload.get("prediction_logic_version") == prediction_logic_version,
          "验收报告的预测逻辑版本与当前运行版本不一致")

    evaluated_at = _timestamp(now, allow_datetime=True) if now is not None else datetime.now(timezone.utc)
    times = {key: _timestamp(payload.get(key)) for key in (
        "generated_at", "training_cutoff_at", "evaluation_start_at",
        "evaluation_end_at", "data_cutoff_at",
    )}
    missing = [key for key, value in times.items() if value is None]
    valid_times = not missing and evaluated_at is not None
    check("timestamps_present", valid_times,
          "时间字段必须是带时区的 ISO 时间：" + ", ".join(missing + (["now"] if evaluated_at is None else [])))
    generated, training, start, end, cutoff = (times[key] for key in (
        "generated_at", "training_cutoff_at", "evaluation_start_at", "evaluation_end_at", "data_cutoff_at",
    ))
    ordered = bool(valid_times and training < start <= end <= cutoff <= generated)
    check("time_order", ordered,
          "时间顺序无效：必须满足训练截止 < 验证开始 ≤ 验证结束 ≤ 数据截止 ≤ 报告生成，禁止训练与验收重叠")
    check("not_future", valid_times and generated <= evaluated_at + timedelta(seconds=CLOCK_SKEW_SECONDS),
          "验收报告时间在未来，超过允许的 5 分钟时钟偏差")
    max_age = timedelta(days=MAX_REPORT_AGE_DAYS)
    check("generated_at_recent", valid_times and evaluated_at - generated <= max_age,
          "验收报告生成时间缺失或已超过 14 天有效期")
    check("data_cutoff_recent", valid_times and evaluated_at - cutoff <= max_age,
          "验收数据截止时间缺失或已超过 14 天有效期，重新生成旧数据报告不能延期")
    check("evaluation_end_recent", valid_times and evaluated_at - end <= max_age,
          "验证期最后一场比赛已超过 14 天，最近补录旧比赛赛果不能更新模型验收有效期")

    n = payload.get("out_of_sample_n")
    check("enough_out_of_sample", _count(n) and n >= MIN_OUT_OF_SAMPLE_N,
          "严格样本外比赛数量不足 1000 场或数量格式无效")
    valid_model, valid_market = _valid_metrics(model), _valid_metrics(market)
    check("valid_model_metrics", valid_model,
          "模型指标不完整或无效，需提供 n、accuracy∈[0,1]、Brier∈[0,2]、非负 LogLoss，且数值必须有限")
    check("valid_market_metrics", valid_market,
          "同快照市场基准指标不完整或无效，需提供合法的 n、accuracy、Brier 和 LogLoss")
    check("matched_sample_counts", _count(n) and valid_model and valid_market
          and model.get("n") == market.get("n") == n,
          "模型、市场与样本外统计的比赛数量不一致，必须在同一批比赛上比较")
    check("paired_sample_count", _count(n) and _count(paired.get("n")) and paired.get("n") == n,
          "缺少配对比较或其样本数与样本外比赛数量不一致")
    days = paired.get("independent_days")
    calendar_days = (end.date() - start.date()).days + 1 if valid_times and start <= end else 0
    check("independent_evaluation_days", _count(days) and days >= MIN_INDEPENDENT_DAYS
          and _count(n) and days <= n and days <= calendar_days,
          "配对比较必须覆盖至少 30 个独立比赛日，且比赛日数量不得超过比赛数或验证期 UTC 日期跨度")

    both_metrics = valid_model and valid_market
    check("accuracy_no_regression", both_metrics and model["accuracy"] >= market["accuracy"],
          "模型胜平负准确率低于同场市场基准或指标缺失")
    check("logloss_no_regression", both_metrics and model["logloss"] <= market["logloss"],
          "模型 LogLoss 高于同场市场基准或指标缺失")
    check("brier_no_regression", both_metrics and model["brier"] <= market["brier"],
          "模型 Brier 高于同场市场基准或指标缺失")

    interval_rules = (
        ("accuracy_difference", "accuracy", 1, 1.0, "准确率差（模型减市场）", False),
        ("logloss_improvement", "logloss", -1, None, "LogLoss 改善（市场减模型）", True),
        ("brier_improvement", "brier", -1, 2.0, "Brier 改善（市场减模型）", False),
    )
    for name, metric, direction, bound, label, strict in interval_rules:
        interval = _mapping(paired.get(name))
        valid = _valid_interval(interval, bound=bound)
        check(f"valid_{name}_interval", valid,
              f"缺少有效的配对 95% 置信区间：{label}，区间须包含估计值且数值有限")
        matches = valid and both_metrics and math.isclose(
            interval["estimate"], direction * (model[metric] - market[metric]),
            rel_tol=ESTIMATE_TOLERANCE, abs_tol=ESTIMATE_TOLERANCE,
        )
        check(f"consistent_{name}_estimate", matches,
              f"{label}的配对点估计与模型、市场指标差不一致")
        lower_passes = valid and (interval["ci95"][0] > 0 if strict else interval["ci95"][0] >= 0)
        check(f"supported_{name}", lower_passes,
              f"{label}的配对 95% 置信区间下界必须{'大于' if strict else '不小于'} 0，不能仅凭点估计判断通过")

    for key, reason in (
        ("immutable_prematch", "未确认使用冻结的赛前预测，存在事后修改或信息泄漏风险"),
        ("same_snapshot_market", "未确认市场基准与模型使用同一赛前快照"),
        ("unique_matches", "未确认按唯一比赛去重，同场多个快照不能重复计为独立比赛"),
    ):
        check(key, audit.get(key) is True, reason)

    return {
        "schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "prediction_ready": all(checks.values()),
        "checks": checks,
        "reasons": reasons,
        "policy": {
            "evaluation_scope": "production_pipeline",
            "model_version": model_version,
            "prediction_logic_version": prediction_logic_version,
            "min_out_of_sample_n": MIN_OUT_OF_SAMPLE_N,
            "min_independent_days": MIN_INDEPENDENT_DAYS,
            "independent_day_timezone": "UTC",
            "max_report_age_days": MAX_REPORT_AGE_DAYS,
            "max_data_age_days": MAX_REPORT_AGE_DAYS,
            "max_evaluation_age_days": MAX_REPORT_AGE_DAYS,
            "clock_skew_seconds": CLOCK_SKEW_SECONDS,
            "estimate_tolerance": ESTIMATE_TOLERANCE,
            "confidence_level": 0.95,
            "accuracy_difference_lower_bound": ">=0",
            "logloss_improvement_lower_bound": ">0",
            "brier_improvement_lower_bound": ">=0",
            "readiness_scope": "prediction_accuracy_only",
        },
    }
