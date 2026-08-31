# -*- coding: utf-8 -*-
"""生成 football elo / upset 的黄金语料条目。

覆盖清单**对着领域层的导出逐个核过**（F-5 的教训：覆盖报告只能报它见过的函数）。
"""
import itertools
import pathlib

from src.domain.sports.football import elo, scoring_model as sm, upset
from tests.domain.golden import describe_exception
from scripts.gen_football_modeling_golden import REAL, STRENGTH

TEAM_NAMES = ['曼联', '  曼联 FC ', 'Manchester United', '曼联(英)', '', '   ', None,
              '队' * 60, '皇家马德里 B队', 'AC米兰', 'St. Pauli', '拜仁慕尼黑\n',
              '<b>阿森纳</b>', '1860慕尼黑']
RATINGS = (1000, 1200, 1500, 1600, 1800, 2100, 2400)
LEAGUE_TYPES = ('友谊赛', '联赛', '杯赛', '洲际杯', '世界杯', '未知', '', None)

CANDIDATES = [((h, a), p) for (h, a), p in
              sorted(sm.build_score_matrix(1.5, 1.1, 7, -0.11).items(),
                     key=lambda kv: -kv[1])[:10]]
ANOMALIES = [None, {}, {'deviation': 0.20}, {'deviation': 0.40}, {'deviation': 0.60}]
STEAMS = [None, {}, {'summary': {'dominant_signal': 'steam_drop'}},
          {'summary': {'dominant_signal': 'trap'}},
          {'summary': {'dominant_signal': 'normal'}}]
TEAMS = [None, {}, STRENGTH,
         dict(STRENGTH, home_recent={'form_pts': 1.0}, away_recent={'form_pts': 2.5}),
         dict(STRENGTH, home_recent={'form_pts': 2.5}, away_recent={'form_pts': 1.0})]


def entries():
    for name in TEAM_NAMES:
        yield f'sanitize:{name!r}', elo.sanitize_team_name(name)
    for a, b in itertools.product(RATINGS, RATINGS):
        yield f'expected:{a}/{b}', elo.expected_score(a, b)
        yield f'goals_expected:{a}/{b}', elo.elo_to_goals_expected(a, b)
    for rating in RATINGS:
        yield f'strength:{rating}', elo.elo_to_strength_factor(rating)
    for league_type in LEAGUE_TYPES:
        try:
            yield f'k_factor:{league_type!r}', elo.k_factor(league_type)
        except Exception as exc:
            yield f'k_factor:{league_type!r}', describe_exception(exc)
        try:
            yield f'league_weight:{league_type!r}', elo.league_weight(league_type)
        except Exception as exc:
            yield f'league_weight:{league_type!r}', describe_exception(exc)

    # 爆冷：真实三件套 × 五组球队近况 × 五种异常 × 五种资金流
    for i, (asian, euro, total) in enumerate(REAL[:12]):
        yield f'upset_risk:{i}', upset._evaluate_upset_risk(asian, euro)
        for ti, team in enumerate(TEAMS):
            yield (f'upset_profile:{i}/{ti}',
                   upset._evaluate_upset_profile(asian, euro, team, total))
            for ai, anomaly in enumerate(ANOMALIES):
                for si, steam in enumerate(STEAMS):
                    yield (f'upset_profile:{i}/{ti}/{ai}/{si}',
                           upset._evaluate_upset_profile(asian, euro, team, total,
                                                         anomaly, steam))
                    yield (f'upset_assess:{i}/{ti}/{ai}/{si}',
                           upset.assess_football_upset(asian, euro, team, CANDIDATES,
                                                       total, anomaly, steam))
