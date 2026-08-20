"""Read-only production validation for league-specific SPF admission.

The threshold is selected on an earlier chronological segment and must hold on
the later segment before it can be used by live recommendations.  No database
state is changed here; MySQL-backed settled prediction records are only read.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Iterable, Mapping

from .professional_monitoring import wilson_interval


TARGET_ACCURACY = 0.80
MIN_TOTAL_ROWS = 100
MIN_TRAIN_SELECTIONS = 30
MIN_HOLDOUT_SELECTIONS = 30
MIN_HOLDOUT_CI_LOW = 0.65
THRESHOLDS = (0.60, 0.62, 0.65, 0.67, 0.70, 0.72, 0.75, 0.78, 0.80)
_CACHE_TTL_SECONDS = 600
_POLICY_CACHE: tuple[float, dict[str, dict[str, Any]]] | None = None


def _normalise_league(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _candidate_label(value: Any) -> str | None:
    return {"胜": "H", "平": "D", "负": "A"}.get(str(value), str(value) if value else None)


def _gate_row(record: Mapping[str, Any]) -> dict[str, Any] | None:
    if not (record.get("settled") or record.get("actual_score")):
        return None
    actual = str(record.get("actual_result") or "")
    if actual not in {"H", "D", "A"}:
        return None
    gate = ((((record.get("professional_snapshot") or {}).get("accuracy_gate") or {}).get("spf")) or {})
    candidate = _candidate_label(gate.get("candidate"))
    try:
        market_probability = float(gate.get("market_probability"))
    except (TypeError, ValueError):
        return None
    if candidate not in {"H", "D", "A"} or not 0 < market_probability <= 1:
        return None
    reasons = [str(reason) for reason in (gate.get("reasons") or [])]
    # Re-evaluate only the probability threshold.  Every other contemporaneous
    # guard (margin, reliability, data quality, model/market conflict) remains
    # binding so historical evaluation matches the live decision path.
    blocking_reasons = [
        reason for reason in reasons
        if not reason.startswith("官方赔率去水概率低于")
    ]
    return {
        "time": str(record.get("match_time") or record.get("created_at") or ""),
        "match_id": str(record.get("match_id") or ""),
        "market_probability": market_probability,
        "hit": candidate == actual,
        "base_eligible": not blocking_reasons,
    }


def _metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row["base_eligible"] and row["market_probability"] >= threshold
    ]
    hits = sum(bool(row["hit"]) for row in selected)
    n = len(selected)
    low, high = wilson_interval(hits, n)
    return {
        "sample_count": n,
        "hits": hits,
        "accuracy": round(hits / n, 4) if n else None,
        "coverage": round(n / len(rows), 4) if rows else 0.0,
        "ci95_low": round(low, 4),
        "ci95_high": round(high, 4),
    }


def validate_league_spf_policy(
    records: Iterable[Mapping[str, Any]],
    league: Any,
) -> dict[str, Any]:
    """Select on the earlier 65% of records and verify on the later 35%."""
    league_key = _normalise_league(league)
    rows = []
    for record in records:
        if _normalise_league(record.get("league")) != league_key:
            continue
        row = _gate_row(record)
        if row:
            rows.append(row)
    rows.sort(key=lambda row: (row["time"], row["match_id"]))
    if len(rows) < MIN_TOTAL_ROWS:
        return {
            "supported": False,
            "league": str(league or ""),
            "sample_count": len(rows),
            "reason": f"生产可审计样本不足{MIN_TOTAL_ROWS}场",
        }

    split = max(1, min(len(rows) - 1, round(len(rows) * 0.65)))
    training, holdout = rows[:split], rows[split:]
    chosen = None
    for threshold in THRESHOLDS:
        metrics = _metrics(training, threshold)
        if (
            metrics["sample_count"] >= MIN_TRAIN_SELECTIONS
            and metrics["accuracy"] is not None
            and metrics["accuracy"] >= TARGET_ACCURACY
        ):
            chosen = (threshold, metrics)
            break
    if chosen is None:
        return {
            "supported": False,
            "league": str(league or ""),
            "sample_count": len(rows),
            "reason": "训练时间段无法找到达标且覆盖足够的阈值",
        }

    threshold, train_metrics = chosen
    holdout_metrics = _metrics(holdout, threshold)
    supported = bool(
        holdout_metrics["sample_count"] >= MIN_HOLDOUT_SELECTIONS
        and holdout_metrics["accuracy"] is not None
        and holdout_metrics["accuracy"] >= TARGET_ACCURACY
        and holdout_metrics["ci95_low"] >= MIN_HOLDOUT_CI_LOW
    )
    return {
        "supported": supported,
        "league": str(league or ""),
        "sample_count": len(rows),
        "minimum_probability": threshold,
        "training": train_metrics,
        "holdout": holdout_metrics,
        "target_accuracy": TARGET_ACCURACY,
        "selection_rule": "choose_threshold_on_earlier_65pct_then_freeze_for_later_35pct",
        "reason": None if supported else "冻结阈值未通过后续时间段验证",
    }


def build_production_league_spf_policies(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped = defaultdict(list)
    for record in records:
        league = str(record.get("league") or "").strip()
        if league:
            grouped[league].append(record)
    return {
        league: validate_league_spf_policy(rows, league)
        for league, rows in grouped.items()
    }


def load_production_league_spf_policy(league: Any) -> dict[str, Any]:
    """Load cached policies from MySQL-first storage and return one league.

    The complete production history is read at most once per TTL window.  Live
    match batches commonly contain several previously unseen leagues, so a
    per-league database scan would multiply the same expensive read.
    """
    global _POLICY_CACHE
    key = _normalise_league(league)
    if not key:
        return {"supported": False, "reason": "缺少联赛标识"}
    now = time.monotonic()
    if _POLICY_CACHE is None or now - _POLICY_CACHE[0] >= _CACHE_TTL_SECONDS:
        try:
            from ..common import repositories

            built = build_production_league_spf_policies(
                repositories.football_prediction_load()
            )
            policies = {
                _normalise_league(name): policy
                for name, policy in built.items()
            }
        except Exception as exc:
            policies = {
                "__load_error__": {
                    "supported": False,
                    "reason": f"生产预测历史读取失败: {exc}",
                }
            }
        _POLICY_CACHE = (now, policies)

    policies = _POLICY_CACHE[1]
    if "__load_error__" in policies:
        return {
            **policies["__load_error__"],
            "league": str(league or ""),
        }
    return policies.get(key, {
        "supported": False,
        "league": str(league or ""),
        "sample_count": 0,
        "reason": f"生产可审计样本不足{MIN_TOTAL_ROWS}场",
    })
