"""Production monitoring for calibrated football probabilities and market timing."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Sequence


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


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = hits / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


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
        from .professional_validation import evaluate_rqspf_records
        rqspf_independent = evaluate_rqspf_records(ordered, min_probability=0.65, min_edge=0.03)
    except Exception as exc:
        rqspf_independent = {
            "market": "rqspf", "n": 0, "production_ready": False, "reason": str(exc),
        }
    return {
        "schema_version": "football-professional-monitoring-v1",
        "settled_records": len(ordered),
        "spf": calibration_report(ordered, "spf"),
        "rqspf": calibration_report(ordered, "rqspf"),
        "rqspf_independent_validation": rqspf_independent,
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
