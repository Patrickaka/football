#!/usr/bin/env python
"""Chronological audit of closing over/under 2.5 selective policies."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
LEAGUES = ("E0", "SP1", "D1", "I1", "F1")
SEASONS = ("2425", "2526")
THRESHOLDS = (0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65)


def _number(row, key):
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _probabilities(row, over_key, under_key):
    over_odds = _number(row, over_key)
    under_odds = _number(row, under_key)
    if over_odds <= 1.0 or under_odds <= 1.0:
        return None
    over_inverse, under_inverse = 1.0 / over_odds, 1.0 / under_odds
    total = over_inverse + under_inverse
    return {"over": over_inverse / total, "under": under_inverse / total}


def load_rows():
    rows = []
    for league in LEAGUES:
        for season in SEASONS:
            path = DATA / f"{league}_{season}.csv"
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for raw in csv.DictReader(handle):
                    opening = _probabilities(raw, "Avg>2.5", "Avg<2.5")
                    closing = _probabilities(raw, "AvgC>2.5", "AvgC<2.5") or opening
                    home_goals = _number(raw, "FTHG")
                    away_goals = _number(raw, "FTAG")
                    if not opening or not closing:
                        continue
                    pick = max(closing, key=closing.get)
                    open_pick = max(opening, key=opening.get)
                    rows.append({
                        "league": league,
                        "season": season,
                        "actual": "over" if home_goals + away_goals >= 3 else "under",
                        "pick": pick,
                        "probability": closing[pick],
                        "margin": abs(closing["over"] - closing["under"]),
                        "stable_top": pick == open_pick,
                        "probability_move": closing[pick] - opening[pick],
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


def _selector(threshold, stable_only=False):
    return lambda row: (
        row["probability"] >= threshold
        and (not stable_only or row["stable_top"])
    )


def choose_frozen_policy(training, holdout, min_train, min_test, target=0.60):
    """Choose only on training; holdout is used solely for pass/fail reporting."""
    candidates = []
    for threshold in THRESHOLDS:
        selector = _selector(threshold)
        train = metrics(training, selector)
        if train["selected"] >= min_train and train["accuracy"] >= target:
            candidates.append((train["accuracy"], train["selected"], threshold, train))
    if not candidates:
        return None
    _, _, threshold, train = max(candidates)
    test = metrics(holdout, _selector(threshold))
    return {
        "threshold": threshold,
        "stable_only": False,
        "training": train,
        "holdout": test,
        "supported": test["selected"] >= min_test and test["accuracy"] >= target,
    }


def main():
    rows = load_rows()
    training = [row for row in rows if row["season"] == "2425"]
    holdout = [row for row in rows if row["season"] == "2526"]
    scopes = [("global", training, holdout, 150, 100)]
    scopes.extend((
        league,
        [row for row in training if row["league"] == league],
        [row for row in holdout if row["league"] == league],
        40,
        30,
    ) for league in LEAGUES)

    print("frozen O/U 2.5 policies (select 2024/25 -> verify 2025/26):")
    for name, train_rows, test_rows, min_train, min_test in scopes:
        policy = choose_frozen_policy(train_rows, test_rows, min_train, min_test)
        if not policy:
            print(f"{name}: abstain (no training rule reaches 60% with enough samples)")
            continue
        train, test = policy["training"], policy["holdout"]
        print(
            f"{name}: p>={policy['threshold']:.2f} stable={policy['stable_only']} "
            f"train={train['accuracy']:.2%} n={train['selected']} "
            f"holdout={test['accuracy']:.2%} n={test['selected']} "
            f"status={'supported' if policy['supported'] else 'rejected'}"
        )


if __name__ == "__main__":
    main()
