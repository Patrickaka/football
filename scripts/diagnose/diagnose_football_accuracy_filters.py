#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnose high-precision football/Beidan selection filters chronologically."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
LEAGUES = ("E0", "SP1", "D1", "I1", "F1")
SEASONS = ("2425", "2526")
LABELS = ("H", "D", "A")


def _number(row, key):
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _probabilities(row, keys):
    odds = [_number(row, key) for key in keys]
    if not all(value > 1.0 for value in odds):
        return None
    inverse = [1.0 / value for value in odds]
    total = sum(inverse)
    return dict(zip(LABELS, (value / total for value in inverse)))


def load_rows():
    rows = []
    for league in LEAGUES:
        for season in SEASONS:
            path = DATA / f"{league}_{season}.csv"
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for raw in csv.DictReader(handle):
                    actual = raw.get("FTR")
                    opening = _probabilities(raw, ("AvgH", "AvgD", "AvgA"))
                    closing = _probabilities(raw, ("AvgCH", "AvgCD", "AvgCA")) or opening
                    if actual not in LABELS or not opening or not closing:
                        continue
                    close_ranked = sorted(closing, key=closing.get, reverse=True)
                    open_pick = max(opening, key=opening.get)
                    pick = close_ranked[0]
                    line = _number(raw, "AHCh")
                    asian_agrees = (
                        (pick == "H" and line < -0.01)
                        or (pick == "A" and line > 0.01)
                        or (pick == "D" and abs(line) <= 0.25)
                    )
                    rows.append({
                        "league": league,
                        "season": season,
                        "actual": actual,
                        "pick": pick,
                        "probability": closing[pick],
                        "margin": closing[pick] - closing[close_ranked[1]],
                        "stable_top": pick == open_pick,
                        "probability_move": closing[pick] - opening[pick],
                        "asian_agrees": asian_agrees,
                    })
    return rows


def metrics(rows, selector):
    selected = [row for row in rows if selector(row)]
    hits = sum(row["pick"] == row["actual"] for row in selected)
    return {
        "selected": len(selected),
        "accuracy": hits / len(selected) if selected else 0.0,
        "coverage": len(selected) / len(rows) if rows else 0.0,
    }


def show(name, rows, selector):
    values = []
    for season in SEASONS:
        result = metrics([row for row in rows if row["season"] == season], selector)
        values.append(
            f"{result['accuracy']:6.2%}/{result['coverage']:6.2%} n={result['selected']:3d}"
        )
    print(f"{name:<32} {values[0]} | {values[1]}")


def main():
    rows = load_rows()
    base = lambda row: row["probability"] >= 0.60 and row["margin"] >= 0.10
    cases = (
        ("p>=.60", base),
        ("p>=.60 + stable top", lambda row: base(row) and row["stable_top"]),
        ("p>=.60 + nonnegative move", lambda row: base(row) and row["probability_move"] >= 0.0),
        ("p>=.60 + stable + move>=0", lambda row: base(row) and row["stable_top"] and row["probability_move"] >= 0.0),
        ("p>=.60 + Asian agrees", lambda row: base(row) and row["asian_agrees"]),
        ("p>=.60 + stable + Asian", lambda row: base(row) and row["stable_top"] and row["asian_agrees"]),
        ("p>=.62", lambda row: row["probability"] >= 0.62 and row["margin"] >= 0.10),
        ("p>=.65", lambda row: row["probability"] >= 0.65 and row["margin"] >= 0.10),
        ("p>=.65 + nonnegative move", lambda row: row["probability"] >= 0.65 and row["margin"] >= 0.10 and row["probability_move"] >= 0.0),
        ("p>=.67", lambda row: row["probability"] >= 0.67 and row["margin"] >= 0.10),
        ("p>=.70", lambda row: row["probability"] >= 0.70 and row["margin"] >= 0.10),
    )
    print("filter                           2024/25 accuracy/cov     | 2025/26 accuracy/cov")
    for name, selector in cases:
        show(name, rows, selector)

    print("\nper-league thresholds:")
    for league in LEAGUES:
        sample = [row for row in rows if row["league"] == league]
        for threshold in (0.60, 0.62, 0.65, 0.67, 0.70, 0.72, 0.75):
            show(
                f"{league} p>={threshold:.2f}",
                sample,
                lambda row, value=threshold: row["probability"] >= value and row["margin"] >= 0.10,
            )

    print("\nchronological frozen league policies (select 2024/25 -> verify 2025/26):")
    thresholds = (0.60, 0.62, 0.65, 0.67, 0.70, 0.72, 0.75)
    for league in LEAGUES:
        training = [row for row in rows if row["league"] == league and row["season"] == "2425"]
        holdout = [row for row in rows if row["league"] == league and row["season"] == "2526"]
        selected = None
        for threshold in thresholds:
            selector = lambda row, value=threshold: (
                row["probability"] >= value and row["margin"] >= 0.10
            )
            train_metrics = metrics(training, selector)
            if train_metrics["selected"] >= 40 and train_metrics["accuracy"] >= 0.80:
                selected = (threshold, train_metrics, metrics(holdout, selector))
                break
        if not selected:
            print(f"{league}: abstain (no training threshold reaches target with n>=40)")
            continue
        threshold, train_metrics, test_metrics = selected
        supported = test_metrics["selected"] >= 30 and test_metrics["accuracy"] >= 0.80
        print(
            f"{league}: p>={threshold:.2f} train={train_metrics['accuracy']:.2%} "
            f"n={train_metrics['selected']} holdout={test_metrics['accuracy']:.2%} "
            f"n={test_metrics['selected']} status={'supported' if supported else 'rejected'}"
        )


if __name__ == "__main__":
    main()
