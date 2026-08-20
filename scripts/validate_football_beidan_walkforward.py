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
                        "asian_line": _number(raw, "AHCh"),
                        "asian_home_odds": _number(raw, "AvgCAHH") or _number(raw, "AvgAHH"),
                        "asian_away_odds": _number(raw, "AvgCAHA") or _number(raw, "AvgAHA"),
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


def _fair_decimal_odds(side_odds, other_odds):
    """Remove the two-way overround and return fair decimal odds for one side."""
    if side_odds <= 1.0 or other_odds <= 1.0:
        return None
    side_inverse = 1.0 / side_odds
    fair_probability = side_inverse / (side_inverse + 1.0 / other_odds)
    return 1.0 / fair_probability


def _quarter_line_parts(line):
    """Split Asian quarter lines into their two settlement lines."""
    doubled = round(float(line) * 2.0) / 2.0
    if abs(float(line) - doubled) < 1e-8:
        return (doubled,)
    lower = math.floor(float(line) * 2.0) / 2.0
    return (lower, lower + 0.5)


def _settlement_profit(value, line, fair_decimal, over=True):
    profit = 0.0
    parts = _quarter_line_parts(line)
    for settlement_line in parts:
        distance = value - settlement_line if over else settlement_line - value
        if distance > 1e-8:
            profit += fair_decimal - 1.0
        elif distance < -1e-8:
            profit -= 1.0
    return profit / len(parts)


def _exponential_tilt(matrix, feature, theta):
    adjusted = {
        score: probability * math.exp(max(-20.0, min(20.0, theta * feature(score))))
        for score, probability in matrix.items()
    }
    total = sum(adjusted.values())
    return {score: probability / total for score, probability in adjusted.items()}


def _constrain_fair_market(matrix, feature, strength=1.0):
    """Maximum-entropy tilt until the selected market side has zero fair profit."""
    if strength <= 0:
        return matrix

    feature_values = {score: feature(score) for score in matrix}
    buckets = defaultdict(float)
    for score, probability in matrix.items():
        buckets[round(feature_values[score], 12)] += probability

    def expected_profit(theta):
        weighted = [
            (value, probability * math.exp(max(-20.0, min(20.0, theta * value))))
            for value, probability in buckets.items()
        ]
        total = sum(probability for _, probability in weighted)
        return sum(value * probability for value, probability in weighted) / total

    lo, hi = -12.0, 12.0
    lo_value, hi_value = expected_profit(lo), expected_profit(hi)
    if lo_value > 0 or hi_value < 0:
        return matrix
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if expected_profit(mid) < 0:
            lo = mid
        else:
            hi = mid
    theta = strength * (lo + hi) / 2.0
    adjusted = {
        score: probability * math.exp(
            max(-20.0, min(20.0, theta * feature_values[score]))
        )
        for score, probability in matrix.items()
    }
    total = sum(adjusted.values())
    return {score: probability / total for score, probability in adjusted.items()}


def _anchor_outcomes(matrix, probabilities, strength):
    if strength <= 0:
        return matrix
    current = {"H": 0.0, "D": 0.0, "A": 0.0}
    for (home, away), value in matrix.items():
        current["H" if home > away else "D" if home == away else "A"] += value
    adjusted = {}
    for (home, away), value in matrix.items():
        key = "H" if home > away else "D" if home == away else "A"
        adjusted[(home, away)] = value * (
            probabilities[key] / max(current[key], 1e-12)
        ) ** strength
    total = sum(adjusted.values())
    return {score: value / total for score, value in adjusted.items()}


def _score_distribution(row, anchor_strength=0.0, use_ou=True,
                        strength_split=0.45, ou_blend=0.60,
                        asian_constraint=0.0, ou_constraint=0.0):
    probs = row["probabilities"]
    target_total = LEAGUE_GOALS[row["league"]]
    if use_ou:
        market_total = _implied_total(row["over_odds"], row["under_odds"], row["ou_line"])
        if market_total:
            target_total = ou_blend * market_total + (1.0 - ou_blend) * target_total
    supremacy = probs["H"] - probs["A"]
    home_mean = max(0.01, target_total * (0.5 + strength_split * supremacy))
    away_mean = max(0.01, target_total * (0.5 - strength_split * supremacy))
    matrix = {
        (home, away): _poisson_probability(home, home_mean) * _poisson_probability(away, away_mean)
        for home in range(8) for away in range(8)
    }
    total = sum(matrix.values())
    matrix = {score: value / total for score, value in matrix.items()}
    asian_fair = _fair_decimal_odds(row["asian_home_odds"], row["asian_away_odds"])
    over_fair = _fair_decimal_odds(row["over_odds"], row["under_odds"])
    matrix = _anchor_outcomes(matrix, probs, anchor_strength)
    # Alternate the handicap and total constraints because changing one
    # marginal also moves the other. Damping prevents a noisy market from
    # overwhelming the 1X2/league prior.
    for _ in range(3):
        if asian_fair:
            line = row["asian_line"]
            matrix = _constrain_fair_market(
                matrix,
                lambda score, line=line, odds=asian_fair: _settlement_profit(
                    score[0] - score[1], -line, odds, over=True
                ),
                asian_constraint / 3.0,
            )
        if over_fair:
            line = row["ou_line"]
            matrix = _constrain_fair_market(
                matrix,
                lambda score, line=line, odds=over_fair: _settlement_profit(
                    score[0] + score[1], line, odds, over=True
                ),
                ou_constraint / 3.0,
            )
    return matrix


def evaluate_scores(rows, anchor_strength=0.0, use_ou=True,
                    strength_split=0.45, ou_blend=0.60,
                    asian_constraint=0.0, ou_constraint=0.0):
    score_top1 = score_top3 = goal_top1 = goal_top2 = 0
    outcome_hits = 0
    score_loss = goal_loss = outcome_loss = 0.0
    for row in rows:
        matrix = _score_distribution(
            row,
            anchor_strength=anchor_strength,
            use_ou=use_ou,
            strength_split=strength_split,
            ou_blend=ou_blend,
            asian_constraint=asian_constraint,
            ou_constraint=ou_constraint,
        )
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
        outcomes = {"H": 0.0, "D": 0.0, "A": 0.0}
        for (home, away), probability in matrix.items():
            outcomes["H" if home > away else "D" if home == away else "A"] += probability
        outcome_hits += max(outcomes, key=outcomes.get) == row["actual"]
        outcome_loss -= math.log(max(outcomes[row["actual"]], 1e-12))
    count = len(rows) or 1
    return {
        "score_top1": score_top1 / count,
        "score_top3": score_top3 / count,
        "goal_top1": goal_top1 / count,
        "goal_top2": goal_top2 / count,
        "outcome_accuracy": outcome_hits / count,
        "score_logloss": score_loss / count,
        "goal_logloss": goal_loss / count,
        "outcome_logloss": outcome_loss / count,
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

    print("\nchronological score strength split (anchor=.75):")
    for split in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        train = evaluate_scores(grouped["2425"], anchor_strength=0.75, strength_split=split)
        test = evaluate_scores(grouped["2526"], anchor_strength=0.75, strength_split=split)
        print(
            f"split={split:.2f} "
            f"T1={train['score_top1']:6.2%}/{test['score_top1']:6.2%} "
            f"T3={train['score_top3']:6.2%}/{test['score_top3']:6.2%} "
            f"LL={train['score_logloss']:.4f}/{test['score_logloss']:.4f}"
        )

    print("\nchronological O/U blend (anchor=.75, split=.45):")
    for blend in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        train = evaluate_scores(grouped["2425"], anchor_strength=0.75, ou_blend=blend)
        test = evaluate_scores(grouped["2526"], anchor_strength=0.75, ou_blend=blend)
        print(
            f"ou={blend:.2f} goals T1={train['goal_top1']:6.2%}/{test['goal_top1']:6.2%} "
            f"T2={train['goal_top2']:6.2%}/{test['goal_top2']:6.2%} "
            f"score T1={train['score_top1']:6.2%}/{test['score_top1']:6.2%} "
            f"LL={train['goal_logloss']:.4f}/{test['goal_logloss']:.4f}"
        )

    print("\nmaximum-entropy joint market constraints (current O/U mean prior retained):")
    print("strength      score T1 train/test   score LL train/test   goals LL train/test   1X2 train/test")
    for strength in (0.0, 0.20, 0.35, 0.50, 0.75, 1.0):
        train = evaluate_scores(
            grouped["2425"], anchor_strength=0.75,
            asian_constraint=strength, ou_constraint=strength,
        )
        test = evaluate_scores(
            grouped["2526"], anchor_strength=0.75,
            asian_constraint=strength, ou_constraint=strength,
        )
        print(
            f"joint={strength:>4.2f}    "
            f"{train['score_top1']:6.2%}/{test['score_top1']:6.2%}       "
            f"{train['score_logloss']:.4f}/{test['score_logloss']:.4f}       "
            f"{train['goal_logloss']:.4f}/{test['goal_logloss']:.4f}       "
            f"{train['outcome_accuracy']:6.2%}/{test['outcome_accuracy']:6.2%}"
        )

    print("\nAsian-only constraint isolation (avoids reusing the O/U price twice):")
    for strength in (0.20, 0.35, 0.50, 0.75, 1.0):
        train = evaluate_scores(
            grouped["2425"], anchor_strength=0.75,
            asian_constraint=strength,
        )
        test = evaluate_scores(
            grouped["2526"], anchor_strength=0.75,
            asian_constraint=strength,
        )
        print(
            f"asian={strength:>4.2f} score T1={train['score_top1']:6.2%}/{test['score_top1']:6.2%} "
            f"T3={train['score_top3']:6.2%}/{test['score_top3']:6.2%} "
            f"LL={train['score_logloss']:.4f}/{test['score_logloss']:.4f} "
            f"goalsLL={train['goal_logloss']:.4f}/{test['goal_logloss']:.4f}"
        )

    print("\nO/U-only constraint isolation:")
    for strength in (0.20, 0.35, 0.50, 0.75, 1.0):
        train = evaluate_scores(
            grouped["2425"], anchor_strength=0.75,
            ou_constraint=strength,
        )
        test = evaluate_scores(
            grouped["2526"], anchor_strength=0.75,
            ou_constraint=strength,
        )
        print(
            f"ou-fair={strength:>4.2f} score T1={train['score_top1']:6.2%}/{test['score_top1']:6.2%} "
            f"T3={train['score_top3']:6.2%}/{test['score_top3']:6.2%} "
            f"LL={train['score_logloss']:.4f}/{test['score_logloss']:.4f} "
            f"goalsLL={train['goal_logloss']:.4f}/{test['goal_logloss']:.4f}"
        )


if __name__ == "__main__":
    main()
