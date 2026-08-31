# -*- coding: utf-8 -*-
"""职业化口径的监控报告：校准、冷门告警、审计基线。

**没有存储**：入口收的都是记录列表。
`build_professional_monitoring` 里那条指向 `production_league_gate`
的延迟 import 现在指向同包的 `league_gate`——`wilson_interval` 已提到
`stats`，环断了，不必再延迟。
"""

from __future__ import annotations

import math
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping, Sequence

from .league_gate import build_production_league_spf_policies
from .validation import evaluate_rqspf_records
from .stats import wilson_interval


BASELINE_VERSION = "football-oos-2026-07-23-v1"

BASELINE_GENERATED_AT = "2026-07-23T14:13:38+08:00"

AUDITED_PROFESSIONAL_BASELINE = {
    "method": "expanding-window-walk-forward",
    "out_of_sample_n": 1804,
    "model_metrics": {
        "n": 1804,
        "accuracy": 0.5238359201773836,
        "logloss": 0.9965126754063698,
        "brier": 0.593481284345111,
    },
    "market_baseline_metrics": {
        "n": 1804,
        "accuracy": 0.5360310421286031,
        "logloss": 0.9773157271504305,
        "brier": 0.5816143229560724,
    },
    "strategy": {
        "bets": 293,
        "wins": 175,
        "hit_rate": 0.5972696245733788,
        "coverage": 0.16241685144124168,
        "profit": -5.620000000000001,
        "roi": -0.019180887372013657,
        "max_drawdown_units": 14.170000000000002,
        "mean_clv": -0.006273251680351913,
        "clv_samples": 293,
    },
    "audit": {
        "model_split": "expanding chronological window",
        "threshold_split": "earlier out-of-sample predictions only",
        "raw_samples": 3504,
        "model_oos_samples": 2304,
    },
}

def bundled_professional_baseline():
    return deepcopy(AUDITED_PROFESSIONAL_BASELINE)


def _normalise(values: Mapping[str, Any] | None, labels: Sequence[str]) -> Dict[str, float]:
    aliases = {
        "H": ("H", "home", "胜"), "D": ("D", "draw", "平"), "A": ("A", "away", "负"),
        "让胜": ("让胜",), "让平": ("让平",), "让负": ("让负",),
    }
    clean = {}
    for label in labels:
        value = next((values.get(key) for key in aliases[label] if values and values.get(key) is not None), 0)
        try:
            clean[label] = max(0.0, float(value))
        except (TypeError, ValueError):
            clean[label] = 0.0
    total = sum(clean.values())
    return {key: value / total for key, value in clean.items()} if total else {}

def _actual_rqspf(record: Mapping[str, Any]) -> str | None:
    if record.get("actual_rqspf") in {"让胜", "让平", "让负"}:
        return record["actual_rqspf"]
    score = str(record.get("actual_score") or "")
    try:
        home, away = map(int, score.split("-"))
        margin = home + int(record.get("lottery_handicap")) - away
    except (TypeError, ValueError):
        return None
    return "让胜" if margin > 0 else ("让平" if margin == 0 else "让负")

def calibration_report(
    records: Iterable[Mapping[str, Any]],
    market: str = "spf",
    bucket_width: float = 0.10,
) -> Dict[str, Any]:
    labels = ("H", "D", "A") if market == "spf" else ("让胜", "让平", "让负")
    rows = []
    for record in records:
        probs = _normalise(
            record.get("predicted_1x2") if market == "spf" else record.get("predicted_rqspf"),
            labels,
        )
        actual = record.get("actual_result") if market == "spf" else _actual_rqspf(record)
        if not probs or actual not in labels:
            continue
        pick = max(labels, key=probs.get)
        rows.append((probs[pick], pick == actual, probs, actual))

    buckets = defaultdict(lambda: {"n": 0, "hits": 0, "probability_sum": 0.0})
    logloss = brier = 0.0
    for probability, hit, probs, actual in rows:
        lower = min(0.9, math.floor(probability / bucket_width) * bucket_width)
        key = f"{int(lower * 100)}-{int(min(1, lower + bucket_width) * 100)}%"
        bucket = buckets[key]
        bucket["n"] += 1
        bucket["hits"] += int(hit)
        bucket["probability_sum"] += probability
        logloss -= math.log(max(1e-15, probs[actual]))
        brier += sum((probs[label] - (1 if label == actual else 0)) ** 2 for label in labels)

    output_buckets = []
    for key, bucket in sorted(buckets.items(), key=lambda item: int(item[0].split("-")[0])):
        low, high = wilson_interval(bucket["hits"], bucket["n"])
        output_buckets.append({
            "bucket": key,
            "n": bucket["n"],
            "mean_predicted": round(bucket["probability_sum"] / bucket["n"], 4),
            "observed_accuracy": round(bucket["hits"] / bucket["n"], 4),
            "ci95_low": round(low, 4),
            "ci95_high": round(high, 4),
        })
    n = len(rows)
    return {
        "market": market,
        "n": n,
        "accuracy": round(sum(hit for _, hit, _, _ in rows) / n, 4) if n else None,
        "logloss": round(logloss / n, 4) if n else None,
        "brier": round(brier / n, 4) if n else None,
        "buckets": output_buckets,
    }

def _window_metrics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    report = calibration_report(records, "spf")
    return {key: report[key] for key in ("n", "accuracy", "logloss", "brier")}

def upset_alert_report(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Measure pre-match upset alerts without reconstructing them after result."""
    result_aliases = {"H": "胜", "D": "平", "A": "负"}
    all_favorites = 0
    all_failures = 0
    alerts = []
    levels = defaultdict(lambda: {"n": 0, "hits": 0})
    for record in records:
        actual = result_aliases.get(str(record.get("actual_result") or ""))
        upset = ((record.get("professional_snapshot") or {}).get("upset") or {})
        favorite = upset.get("favorite")
        if actual not in {"胜", "平", "负"} or favorite not in {"胜", "平", "负"}:
            continue
        all_favorites += 1
        favorite_failed = actual != favorite
        all_failures += int(favorite_failed)
        if not upset.get("alert"):
            continue
        defensive = {
            item.get("result") for item in (upset.get("defensive_selections") or [])
            if isinstance(item, Mapping)
        }
        level = str(upset.get("level") or "unknown")
        alerts.append((favorite_failed, actual in defensive))
        levels[level]["n"] += 1
        levels[level]["hits"] += int(favorite_failed)

    n = len(alerts)
    hits = sum(int(hit) for hit, _ in alerts)
    direction_hits = sum(int(hit) for _, hit in alerts)
    low, high = wilson_interval(hits, n)
    return {
        "n": n,
        "realized_upsets": hits,
        "alert_precision": round(hits / n, 4) if n else None,
        "ci95_low": round(low, 4),
        "ci95_high": round(high, 4),
        "defensive_direction_hit_rate": round(direction_hits / n, 4) if n else None,
        "baseline_n": all_favorites,
        "baseline_favorite_failure_rate": (
            round(all_failures / all_favorites, 4) if all_favorites else None
        ),
        "levels": {
            level: {
                "n": values["n"],
                "realized_upsets": values["hits"],
                "alert_precision": round(values["hits"] / values["n"], 4),
            }
            for level, values in levels.items()
        },
        "production_ready": n >= 100 and low >= 0.50,
        "source": "persisted_prematch_upset_snapshot",
    }

def build_professional_monitoring(
    records: Sequence[Mapping[str, Any]],
    recent_window: int = 50,
    baseline_window: int = 200,
) -> Dict[str, Any]:
    ordered = sorted(
        (record for record in records if record.get("settled") or record.get("actual_score")),
        key=lambda record: (record.get("match_time", ""), record.get("match_id", "")),
    )
    recent = ordered[-recent_window:]
    baseline = ordered[-(recent_window + baseline_window):-recent_window]
    recent_metrics = _window_metrics(recent)
    baseline_metrics = _window_metrics(baseline)
    drift_reasons = []
    if recent_metrics["n"] >= 30 and baseline_metrics["n"] >= 50:
        if recent_metrics["logloss"] is not None and baseline_metrics["logloss"] is not None:
            if recent_metrics["logloss"] > baseline_metrics["logloss"] + 0.08:
                drift_reasons.append("近期LogLoss显著恶化")
        if recent_metrics["brier"] is not None and baseline_metrics["brier"] is not None:
            if recent_metrics["brier"] > baseline_metrics["brier"] + 0.05:
                drift_reasons.append("近期Brier显著恶化")

    def has_timed_snapshot(record):
        layers = record.get("odds_layers") or {}
        return any(layers.get(key) for key in ("T-24h", "T-6h", "T-1h", "T-15min"))

    def has_closing_snapshot(record):
        layers = record.get("odds_layers") or {}
        return bool(
            record.get("closing_odds")
            or record.get("closing_odds_snapshot")
            or layers.get("T-15min")
        )

    def has_verified_closing_snapshot(record):
        return bool(
            record.get("closing_odds_snapshot")
            and record.get("closing_odds_source") == "official_close"
        )

    closing_samples = sum(has_closing_snapshot(record) for record in ordered)
    verified_closing_samples = sum(has_verified_closing_snapshot(record) for record in ordered)
    timed_snapshot_samples = sum(has_timed_snapshot(record) for record in ordered)
    try:
        rqspf_independent = evaluate_rqspf_records(ordered, min_probability=0.65, min_edge=0.03)
    except Exception as exc:
        rqspf_independent = {
            "market": "rqspf", "n": 0, "production_ready": False,
            "reason": "internal_error", "error_type": type(exc).__name__,
        }
    try:
        league_spf_validation = build_production_league_spf_policies(ordered)
    except Exception as exc:
        league_spf_validation = {"error": str(exc)}
    return {
        "schema_version": "football-professional-monitoring-v1",
        "settled_records": len(ordered),
        "spf": calibration_report(ordered, "spf"),
        "rqspf": calibration_report(ordered, "rqspf"),
        "upset": upset_alert_report(ordered),
        "rqspf_independent_validation": rqspf_independent,
        "league_spf_validation": league_spf_validation,
        "drift": {
            "detected": bool(drift_reasons),
            "reasons": drift_reasons,
            "recent": recent_metrics,
            "baseline": baseline_metrics,
        },
        "market_timing": {
            "closing_odds_samples": closing_samples,
            "closing_odds_coverage": round(closing_samples / len(ordered), 4) if ordered else 0.0,
            "verified_closing_odds_samples": verified_closing_samples,
            "verified_closing_odds_coverage": (
                round(verified_closing_samples / len(ordered), 4) if ordered else 0.0
            ),
            "timed_snapshot_samples": timed_snapshot_samples,
            "timed_snapshot_coverage": round(timed_snapshot_samples / len(ordered), 4) if ordered else 0.0,
            "clv_ready": verified_closing_samples >= 100,
        },
    }
