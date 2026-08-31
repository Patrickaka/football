# -*- coding: utf-8 -*-
"""生成 value_betting / 聚类融合 / 推荐挑选的黄金语料。"""
import itertools

from src.domain.sports.football import risk as _risk, scoring, scoring_model as sm, value
from tests.domain.golden import describe_exception
from scripts.gen_football_modeling_golden import REAL, STRENGTH

PRED = {'H': 0.45, 'D': 0.28, 'A': 0.27}
ODDS = {'H': 2.0, 'D': 3.4, 'A': 3.8}
LEAGUE = {'avg_goal': 1.52, 'home_boost': 1.08, 'low_score': 0.88,
          'draw_mult': 0.95, 'name': '英超'}
MATRIX = sm.build_score_matrix(1.5, 1.1, 7, -0.11)
CANDIDATES = sorted(MATRIX.items(), key=lambda kv: -kv[1])[:14]
PRIORS = [None, {}, {'H': 0.5, 'D': 0.25, 'A': 0.25}, {'H': 0.2, 'D': 0.3, 'A': 0.5}]


def _y(fn, label, *a, **kw):
    try:
        yield f'{fn.__name__}:{label}', fn(*a, **kw)
    except Exception as exc:
        yield f'{fn.__name__}:{label}', describe_exception(exc)


def entries():
    for prob, odds in itertools.product((0.05, 0.2, 0.45, 0.8, 0.99),
                                        (1.01, 1.5, 2.5, 5.0, 50.0)):
        yield f'value:{prob}/{odds}', value.calculate_value(prob, odds)
        yield f'ev:{prob}/{odds}', value.calculate_ev(prob, odds)
    for weight in (0.0, 0.3, 1.0):
        yield f'adjust:{weight}', value.adjust_by_value(dict(PRED), ODDS, weight)
    for threshold in (0.0, 0.02, 0.2):
        yield f'bets:{threshold}', value.identify_value_bets(dict(PRED), ODDS, threshold)
    for i, bad in enumerate(({}, {'H': 0.0, 'D': 0.0, 'A': 0.0})):
        yield f'adjust_bad:{i}', value.adjust_by_value(bad, ODDS)
        yield f'bets_bad:{i}', value.identify_value_bets(bad, ODDS)

    for hcap, total in itertools.product((-1.0, 0.0, 0.5), (2.25, 2.5, 3.0)):
        for weight in (0.0, 0.3, 1.0):
            for pi, prior in enumerate(PRIORS):
                yield (f'fuse:{hcap}/{total}/{weight}/{pi}',
                       value.fuse_with_prior(dict(PRED), hcap, total, weight, prior))

    for i, (asian, euro, total) in enumerate(REAL[:8]):
        confidence = _risk.compute_prediction_confidence(asian, euro, total, STRENGTH)
        for n in (1, 3, 5):
            for team in (None, STRENGTH):
                for sim in (None, {'count': 30, 'confidence': 0.7, 'avg_distance': 0.2}):
                    yield (f'pick:{i}/{n}/{bool(team)}/{bool(sim)}',
                           scoring._pick_recommendations(
                               CANDIDATES, asian, euro, total, n, 12,
                               confidence, LEAGUE, team, sim))
                    # **注入聚类先验**：适配层传的是读存储的函数，这里传固定值
                    yield (f'pick_prior:{i}/{n}/{bool(team)}/{bool(sim)}',
                           scoring._pick_recommendations(
                               CANDIDATES, asian, euro, total, n, 12,
                               confidence, LEAGUE, team, sim,
                               market_prior_fn=lambda h, t: {'H': 0.5, 'D': 0.25, 'A': 0.25}))
