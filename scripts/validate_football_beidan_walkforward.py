#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Walk-forward validation for football/Beidan market selection policies.

The older diagnostic scripts pool seasons together.  This script keeps
2024/25 and 2025/26 separate so recommendation thresholds are only promoted
when they remain useful in the later season.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LEAGUES = ("E0", "SP1", "D1", "I1", "F1")
SEASONS = ("2425", "2526")
LABELS = ("H", "D", "A")
LEAGUE_GOALS = {"E0": 2.8, "SP1": 2.7, "D1": 3.1, "I1": 2.5, "F1": 2.6}


def _number(row, key):
    try:
        value = float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _market_probabilities(row):
    # Prefer the closing market.  Fall back to the pre-closing average for
    # feeds/seasons that do not expose AvgCH/AvgCD/AvgCA.
    odds = [_number(row, key) for key in ("AvgCH", "AvgCD", "AvgCA")]
    source = "closing"
    if not all(value > 1.0 for value in odds):
        odds = [_number(row, key) for key in ("AvgH", "AvgD", "AvgA")]
        source = "average"
    if not all(value > 1.0 for value in odds):
        return None, source
    inverse = [1.0 / value for value in odds]
    total = sum(inverse)
    return dict(zip(LABELS, (value / total for value in inverse))), source


def load_rows():
    rows = []
    for league in LEAGUES:
        for season in SEASONS:
            path = DATA / f"{league}_{season}.csv"
            if not path.exists():
                continue
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for raw in csv.DictReader(handle):
                    actual = raw.get("FTR")
                    probabilities, source = _market_probabilities(raw)
                    if actual not in LABELS or not probabilities:
                        continue
                    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
                    rows.append({
                        "league": league,
                        "season": season,
                        "actual": actual,
                        "probabilities": probabilities,
                        "prediction": ranked[0][0],
                        "top_probability": ranked[0][1],
                        "margin": ranked[0][1] - ranked[1][1],
                        "source": source,
                        "home_goals": int(_number(raw, "FTHG")),
                        "away_goals": int(_number(raw, "FTAG")),
                        "ou_line": 2.5,
                        "over_odds": _number(raw, "AvgC>2.5") or _number(raw, "Avg>2.5"),
                        "under_odds": _number(raw, "AvgC<2.5") or _number(raw, "Avg<2.5"),
                    })
    return rows


def _poisson_probability(goals, mean):
    return math.exp(-mean) * mean ** goals / math.factorial(goals)


def _implied_total(over_odds, under_odds, line=2.5):
    if over_odds <= 1.0 or under_odds <= 1.0:
        return None
    over_implied = 1.0 / over_odds
    fair_over = over_implied / (over_implied + 1.0 / under_odds)
    fair_decimal = 1.0 / fair_over

    def expected_profit(mean):
        profit = 0.0
        for goals in range(16):
            if goals > line:
                payoff = fair_decimal - 1.0
            elif goals == line:
                payoff = 0.0
            else:
                payoff = -1.0
            profit += _poisson_probability(goals, mean) * payoff
        return profit

    lo, hi = 0.4, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if expected_profit(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _score_distribution(row, anchor_strength=0.0, use_ou=True):
    probs = row["probabilities"]
    target_total = LEAGUE_GOALS[row["league"]]
    if use_ou:
        market_total = _implied_total(row["over_odds"], row["under_odds"], row["ou_line"])
        if market_total:
            target_total = 0.6 * market_total + 0.4 * target_total
    supremacy = probs["H"] - probs["A"]
    home_mean = max(0.01, target_total * (0.5 + 0.45 * supremacy))
    away_mean = max(0.01, target_total * (0.5 - 0.45 * supremacy))
    matrix = {
        (home, away): _poisson_probability(home, home_mean) * _poisson_probability(away, away_mean)
        for home in range(8) for away in range(8)
    }
    total = sum(matrix.values())
    matrix = {score: value / total for score, value in matrix.items()}
    if anchor_strength > 0:
        current = {"H": 0.0, "D": 0.0, "A": 0.0}
        for (home, away), value in matrix.items():
            current["H" if home > away else "D" if home == away else "A"] += value
        adjusted = {}
        for (home, away), value in matrix.items():
            key = "H" if home > away else "D" if home == away else "A"
            factor = (probs[key] / max(current[key], 1e-12)) ** anchor_strength
            adjusted[(home, away)] = value * factor
        total = sum(adjusted.values())
        matrix = {score: value / total for score, value in adjusted.items()}
    return matrix


def evaluate_scores(rows, anchor_strength=0.0, use_ou=True):
    score_top1 = score_top3 = goal_top1 = goal_top2 = 0
    score_loss = goal_loss = 0.0
    for row in rows:
        matrix = _score_distribution(row, anchor_strength=anchor_strength, use_ou=use_ou)
        actual_score = row["home_goals"], row["away_goals"]
        ranked = sorted(matrix, key=matrix.get, reverse=True)
        score_top1 += actual_score == ranked[0]
        score_top3 += actual_score in ranked[:3]
        score_loss -= math.log(max(matrix.get(actual_score, 0.0), 1e-12))
        goals = defaultdict(float)
        for score, probability in matrix.items():
            goals[min(sum(score), 7)] += probability
        actual_goals = min(sum(actual_score), 7)
        goal_ranked = sorted(goals, key=goals.get, reverse=True)
        goal_top1 += actual_goals == goal_ranked[0]
        goal_top2 += actual_goals in goal_ranked[:2]
        goal_loss -= math.log(max(goals.get(actual_goals, 0.0), 1e-12))
    count = len(rows) or 1
    return {
        "score_top1": score_top1 / count,
        "score_top3": score_top3 / count,
        "goal_top1": goal_top1 / count,
        "goal_top2": goal_top2 / count,
        "score_logloss": score_loss / count,
        "goal_logloss": goal_loss / count,
    }


def evaluate(rows, min_probability=0.0, min_margin=0.0):
    selected = [
        row for row in rows
        if row["top_probability"] >= min_probability and row["margin"] >= min_margin
    ]
    hits = sum(row["prediction"] == row["actual"] for row in selected)
    top2_hits = 0
    log_loss = 0.0
    brier = 0.0
    for row in rows:
        probs = row["probabilities"]
        ranked = sorted(probs, key=probs.get, reverse=True)
        top2_hits += row["actual"] in ranked[:2]
        log_loss -= math.log(max(probs[row["actual"]], 1e-12))
        brier += sum((probs[key] - (key == row["actual"])) ** 2 for key in LABELS)
    return {
        "count": len(rows),
        "selected": len(selected),
        "accuracy": hits / len(selected) if selected else 0.0,
        "coverage": len(selected) / len(rows) if rows else 0.0,
        "top2_accuracy": top2_hits / len(rows) if rows else 0.0,
        "log_loss": log_loss / len(rows) if rows else 0.0,
        "brier": brier / len(rows) if rows else 0.0,
    }


def _show(tag, metrics):
    print(
        f"{tag:<22} n={metrics['selected']:4d}/{metrics['count']:4d} "
        f"accuracy={metrics['accuracy']:6.2%} coverage={metrics['coverage']:6.2%} "
        f"top2={metrics['top2_accuracy']:6.2%} logloss={metrics['log_loss']:.4f} "
        f"brier={metrics['brier']:.4f}"
    )


def main():
    rows = load_rows()
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["season"]].append(row)

    print(f"all samples: {len(rows)} (closing odds preferred)\n")
    for season in SEASONS:
        _show(f"season {season} all", evaluate(grouped[season]))

    print("\nselective single-pick walk-forward:")
    print("threshold              2024/25 accuracy/cov     2025/26 accuracy/cov")
    for probability in (0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70):
        train = evaluate(grouped["2425"], probability, 0.10)
        test = evaluate(grouped["2526"], probability, 0.10)
        print(
            f"p>={probability:.2f}, gap>=.10     "
            f"{train['accuracy']:6.2%}/{train['coverage']:6.2%}        "
            f"{test['accuracy']:6.2%}/{test['coverage']:6.2%}"
        )

    print("\nper-league later-season audit (p>=.60, gap>=.10):")
    for league in LEAGUES:
        sample = [row for row in grouped["2526"] if row["league"] == league]
        _show(league, evaluate(sample, 0.60, 0.10))

    print("\nlater-season score-matrix outcome anchoring:")
    for strength in (0.0, 0.5, 0.75, 1.0):
        metrics = evaluate_scores(grouped["2526"], anchor_strength=strength)
        print(
            f"anchor={strength:>4.2f} score T1/T3={metrics['score_top1']:6.2%}/{metrics['score_top3']:6.2%} "
            f"goals T1/T2={metrics['goal_top1']:6.2%}/{metrics['goal_top2']:6.2%} "
            f"LL={metrics['score_logloss']:.4f}/{metrics['goal_logloss']:.4f}"
        )


if __name__ == "__main__":
    main()
