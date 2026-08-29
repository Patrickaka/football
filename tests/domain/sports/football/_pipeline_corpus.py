# -*- coding: utf-8 -*-
"""analyze_match 两个抽出段的语料：候选比分、盘口、以及组装用的四十个部件。

与黄金生成器共用，测试和黄金看的是同一批输入。
"""
from src.domain.sports.football.risk import _evaluate_risk_level

CANDIDATES = [((1, 0), 0.18), ((2, 1), 0.22), ((1, 1), 0.20), ((0, 0), 0.12),
              ((2, 0), 0.14), ((0, 1), 0.08), ((3, 1), 0.06)]
TOTALS = [{'line': 2.5, 'over': 1.9, 'under': 1.9, 'close': {'line': 2.5}},
          {'line': 3.25, 'over': 2.0, 'under': 1.8, 'close': {'line': 3.25}},
          {'line': 1.75, 'over': 1.75, 'under': 2.1, 'close': {'line': 1.75}},
          {}, None]
# `euro['close']` 是 `analyze_euro` 去水后的**概率**，不是赔率——
# 下游 `_anchor_score_candidates_to_1x2` 直接把它当概率用（判据 10：读真实值）
EUROS = [{'H': 2.0, 'D': 3.4, 'A': 4.0,
          'close': {'home': 0.522, 'draw': 0.269, 'away': 0.209}},
         {'H': 1.3, 'D': 5.5, 'A': 9.0,
          'close': {'home': 0.745, 'draw': 0.176, 'away': 0.079}},
         {'H': 6.0, 'D': 4.2, 'A': 1.5,
          'close': {'home': 0.150, 'draw': 0.214, 'away': 0.636}},
         {}, None]
ASIANS = [{'handicap': -0.5, 'home_odds': 0.95, 'away_odds': 0.95},
          {'handicap': 1.0, 'home_odds': 0.9, 'away_odds': 1.0}, {}, None]
PROFILES = [{}, None,
            {'applied': True, 'goal_beta': 0.05, 'sample_count': 120,
             'outcome_weights': {'H': 1.05, 'D': 0.95, 'A': 1.0}},
            {'applied': False, 'reason': 'too_few'}]

DIST = {'0': 0.10, '1': 0.24, '2': 0.27, '3': 0.20, '4+': 0.19}
BASE_PARTS = dict(
    asian={'handicap': -0.5, 'home_odds': 0.95, 'away_odds': 0.95,
           'water_gap': 0.02, 'change': {}},
    calibration_effect={'applied': True, 'shift': 0.01},
    candidates=list(CANDIDATES),
    confidence={'score': 0.62, 'level': 'medium', 'label': '中', 'notes': [],
                'recommend_count': 3},
    dixon_coles_result={'applied': True, 'rho': -0.05},
    euro={'H': 2.0, 'D': 3.4, 'A': 4.0,
          'close': {'home': 0.522, 'draw': 0.269, 'away': 0.209},
          'implied_supremacy': 0.4, 'implied_lambdas': {'home': 1.5, 'away': 1.1},
          'kelly': {'home': 1.0, 'draw': 1.0, 'away': 1.0}},
    euro_asian_dev={'deviation': 0.03},
    goal_count_result={'recommend': ['2'], 'distribution': dict(DIST)},
    goal_dist_after_calibration=dict(DIST),
    goal_dist_before_calibration=dict(DIST),
    half_full_time={'胜/胜': 0.2, '平/胜': 0.15},
    joint_anomaly={'level': 'none'},
    lam_away=1.1, lam_home=1.5,
    league_profile={'avg_goals': 2.7, 'home_advantage': 0.3, 'league': '英超'},
    live_context={'lineup': {}},
    live_context_quality={'quality_score': 0.6, 'official_bet_allowed': True,
                          'blockers': [], 'checks': {}},
    lottery={'probabilities': {'H': 0.5, 'D': 0.28, 'A': 0.22},
             'prediction': 'H', 'spf': {'candidate': '胜', 'probability': 0.5}},
    market_change_result={'applied': False},
    match={'match_id': 'm1', 'home': '主队', 'away': '客队', 'league': '英超',
           'match_time': '2026-08-29 20:00', 'num': '周三001'},
    meta={'score_goal_anchor': {'applied': True}},
    ml_result=None,
    model_status={'ml_enabled': False},
    model_weights={'market': 0.5, 'team': 0.3, 'elo': 0.2, 'ml': 0.0},
    probability_rank=[('H', 0.5), ('D', 0.28), ('A', 0.22)],
    production_spf_policy={},
    recommend=[{'score': '2-1', 'probability': 0.22}],
    recommend_rank=[('2-1', 0.22)],
    # 用真实的 `_evaluate_risk_level` 产物，别手捏——
    # 手捏的版本缺 `description`，组装第一步就 KeyError，
    # 差分「零差异」也只是新旧两边同样报错（判据 23）
    risk=_evaluate_risk_level(
        {'handicap': -0.5, 'home_odds': 0.95, 'away_odds': 0.95},
        {'H': 2.0, 'D': 3.4, 'A': 4.0,
         'close': {'home': 0.522, 'draw': 0.269, 'away': 0.209}},
        {'line': 2.5, 'over': 1.9, 'under': 1.9},
        {'detected': False}, {'score': 0.62, 'level': 'medium'}, {}),
    settlement={'applied': True},
    similar_market_detail={},
    similar_market_result={},
    single_odds={},
    steam_result={'detected': False},
    team={'home_attack': 1.4, 'away_attack': 1.1},
    top_scores=[{'score': '2-1', 'probability': 0.22}],
    total={'line': 2.5, 'over': 1.9, 'under': 1.9, 'close': {'line': 2.5}},
    upset={'level': 'low'},
    value_bets=[],
)

