"""Guarded H2H and motivation fusion for score distributions."""

import math
from typing import Dict, List, Tuple


def _number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def apply_contextual_fusion(
    candidates: List[Tuple[Tuple[int, int], float]],
    context: Dict,
) -> Tuple[List[Tuple[Tuple[int, int], float]], Dict]:
    """Apply small sourced H2H/motivation corrections.

    Expected structured context keys:
      h2h: games, home_wins, draws, away_wins, avg_goals, quality_score
      motivation: home, away (-1..1), quality_score, source
    Free-form text is deliberately ignored.
    """
    context = context or {}
    h2h = context.get('h2h') or {}
    motivation = context.get('motivation') or {}
    outcome_weights = {'H': 1.0, 'D': 1.0, 'A': 1.0}
    goal_beta = 0.0
    evidence = []

    games = int(_number(h2h.get('games'), 0))
    h2h_quality = max(0.0, min(1.0, _number(h2h.get('quality_score'), 0.0)))
    if games >= 5 and h2h_quality >= 0.6:
        n = max(games, 1)
        empirical = {
            'H': _number(h2h.get('home_wins')) / n,
            'D': _number(h2h.get('draws')) / n,
            'A': _number(h2h.get('away_wins')) / n,
        }
        strength = min(0.08, 0.02 + games * 0.003) * h2h_quality
        for key, rate in empirical.items():
            outcome_weights[key] *= 1.0 + strength * (rate - 1 / 3) * 3
        avg_goals = _number(h2h.get('avg_goals'), 0.0)
        if 0.5 <= avg_goals <= 6.0:
            goal_beta += max(-0.05, min(0.05, (avg_goals - 2.6) * 0.035)) * h2h_quality
        evidence.append({'type': 'h2h', 'games': games, 'quality_score': h2h_quality})

    motivation_quality = max(0.0, min(1.0, _number(motivation.get('quality_score'), 0.0)))
    source = motivation.get('source')
    if source and source != 'UNAVAILABLE' and motivation_quality >= 0.7:
        home = max(-1.0, min(1.0, _number(motivation.get('home'))))
        away = max(-1.0, min(1.0, _number(motivation.get('away'))))
        delta = (home - away) * 0.06 * motivation_quality
        outcome_weights['H'] *= 1.0 + delta
        outcome_weights['A'] *= 1.0 - delta
        # High motivation on both sides often increases tempo; one-sided
        # motivation primarily changes direction rather than total goals.
        goal_beta += max(0.0, min(home, away)) * 0.025 * motivation_quality
        evidence.append({'type': 'motivation', 'source': source, 'quality_score': motivation_quality})

    if not evidence or not candidates:
        return candidates, {'applied': False, 'reason': 'no_qualified_structured_context'}

    def outcome(score):
        return 'H' if score[0] > score[1] else 'A' if score[0] < score[1] else 'D'

    adjusted = []
    for score, probability in candidates:
        factor = outcome_weights[outcome(score)] * math.exp(goal_beta * sum(score))
        adjusted.append((score, max(0.0, float(probability)) * factor))
    total = sum(probability for _, probability in adjusted)
    if total <= 0:
        return candidates, {'applied': False, 'reason': 'zero_probability'}
    adjusted = [(score, probability / total) for score, probability in adjusted]
    adjusted.sort(key=lambda item: -item[1])
    return adjusted, {
        'applied': True,
        'outcome_weights': {key: round(value, 4) for key, value in outcome_weights.items()},
        'goal_beta': round(goal_beta, 4),
        'evidence': evidence,
        'guards': {'max_direction_adjustment': 0.12, 'max_goal_beta': 0.075},
    }
