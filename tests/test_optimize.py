"""准确率优化：联赛画像、走势、置信度"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.football as football


def test_league_profile_serie_a():
    lp = football.resolve_league_profile('意甲第32轮')
    assert lp['avg_goal'] < 1.35
    assert lp['low_score'] > 1.0


def test_euro_momentum():
    series = [
        [1.40, 4.0, 7.0, 93.0, 't3'],
        [1.50, 3.9, 6.5, 92.5, 't2'],
        [1.65, 3.7, 5.8, 92.0, 't1'],
    ]
    m = football.analyze_euro_momentum(series)
    assert m['shift_supremacy'] > 0


def test_confidence_low_on_conflict():
    asian = {'implied_supremacy': 1.0}
    euro = {
        'implied_supremacy': -0.9,
        'kelly': {'spread': 1.5},
        'implied_lambdas': {'home': 2.2, 'away': 0.6},
    }
    total = {'implied_total': 2.5}
    team = {
        'attack_home': 2.5, 'defense_home': 0.8, 'attack_away': 0.8, 'defense_away': 0.4,
        'league_profile': football.LEAGUE_PROFILES['default'],
    }
    c = football.compute_prediction_confidence(asian, euro, total, team)
    assert c['score'] < football.CONFIDENCE_LOW_THRESHOLD
    assert c['recommend_count'] == 1


def test_lambda_refine_improves_or_equal():
    targets = (0.55, 0.25, 0.20)
    start = football.estimate_lambdas(0.4, 2.5)
    refined = football._fit_lambda_refine(
        start, 0.4, 2.5, targets, 0.0,
        ou_targets=football._ou_total_distribution(2.5),
    )
    e0 = football._lambda_fit_error(start, 0.4, 2.5, targets, 0.0)
    e1 = football._lambda_fit_error(refined, 0.4, 2.5, targets, 0.0)
    assert e1 <= e0 + 1e-6


if __name__ == '__main__':
    test_league_profile_serie_a()
    test_euro_momentum()
    test_confidence_low_on_conflict()
    test_lambda_refine_improves_or_equal()
    print('✓ 优化模块测试全部通过')
