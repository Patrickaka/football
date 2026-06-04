"""
模型单元测试：净胜球反推、大小球、λ 拟合、冷热、球队强度
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import football


def _mini_asian(hcap=0.75, hp=0.56):
    ap = 1.0 - hp
    return {
        'handicap': hcap,
        'open_handicap': hcap,
        'open_prob': {'home_recv': hp, 'away_give': ap},
        'close_prob': {'home_recv': hp, 'away_give': ap},
        'favor': 'home',
        'diff_range': [0.5, 1.5],
    }


def _mini_euro():
    return {
        'close': {'home': 0.55, 'draw': 0.25, 'away': 0.20},
        'open': {'home': 0.52, 'draw': 0.26, 'away': 0.22},
    }


def _mini_total():
    return {
        'close_line': 2.5,
        'open_line': 2.5,
        'close_prob': {'over': 0.56, 'under': 0.44},
        'implied_total': 2.55,
        'expected_goals': [2, 4],
        'lean': 'over',
    }


def test_implied_total_monotonic():
    t_low = football.implied_total_goals(2.5, 0.42)
    t_high = football.implied_total_goals(2.5, 0.58)
    assert t_low < t_high < 3.2, (t_low, t_high)


def test_asian_supremacy_differs_from_handicap_line():
    """反推净胜球不应机械等于让球盘数值"""
    sup = football.asian_implied_supremacy(0.75, 0.62, 0.38, 2.5)
    assert abs(sup - 0.75) > 0.04, sup


def test_fit_lambdas_matches_euro_margins():
    p_h, p_d, p_a = 0.52, 0.26, 0.22
    sup = football.blend_market_supremacy(0.35, 0.48)
    lam_h, lam_a, _, rho = football.fit_lambdas_from_markets(
        sup, 2.5, 0.54, p_h, p_d, p_a
    )
    matrix = football.build_score_matrix(lam_h, lam_a, rho=rho)
    margins = football._matrix_margins(matrix)
    assert abs(margins['home'] - p_h) < 0.06, margins
    assert abs(margins['draw'] - p_d) < 0.06, margins
    assert abs(margins['away'] - p_a) < 0.06, margins


def test_predict_scores_top1_reasonable():
    candidates, lh, la, meta = football.predict_scores(
        _mini_asian(), _mini_euro(), _mini_total()
    )
    assert lh > la, (lh, la)
    assert meta['supremacy_blended'] is not None
    top = candidates[0][0]
    assert top[0] >= top[1], top


def test_score_heat_labels():
    cold = football.score_heat_label(4, 0, 0.12)
    hot = football.score_heat_label(1, 1, 0.04)
    assert cold[0] == 'cold'
    assert hot[0] == 'hot'


def test_parse_team_strength_from_html():
    def block(team, gf, ga):
        return (
            f'<span class="dz-l zhu">{team} 2 :1</span>'
            f'近10场战绩<span class="ying">6胜</span><span class="ping">3平</span>'
            f'<span class="shu">1负</span><span class="mar_left20">'
            f'进<span class="ying">{gf}球</span>失<span class="shu">{ga}球</span></span>'
        )

    snippet = block('丹麦', 28, 9) + block('民主刚果', 12, 4)
    football.fetch = lambda *a, **k: snippet
    st = football.fetch_team_strength('x', '丹麦', '民主刚果')
    assert st is not None
    assert st['attack_home'] > st['attack_away']


if __name__ == '__main__':
    test_implied_total_monotonic()
    test_asian_supremacy_differs_from_handicap_line()
    test_fit_lambdas_matches_euro_margins()
    test_predict_scores_top1_reasonable()
    test_score_heat_labels()
    test_parse_team_strength_from_html()
    print('✓ 模型测试全部通过')
