# -*- coding: utf-8 -*-
"""生成 ML 契约层的黄金语料。

**判据 20b 是这一批的硬约束**：黄金里**只有「我们自己算的」**——
动态 rho、泊松、DC 的 τ 因子、特征契约、按时间切分。
带 numpy/catboost/xgboost/sklearn 的函数一个都没进来
（3-17b 在 lottery3d 上因为 `requirements.txt` 用 `>=`、CI 装的版本比本地新，
把库算出来的数钉进黄金直接红了 5 条）。

判断方法：问「换一个版本的 catboost，这个值还一样吗」。
"""
import itertools

from src.domain.sports.football import ml_contract as mc

LEAGUES = ['意甲', '意乙', '葡超', '希腊超', '阿甲', '欧冠', '欧联', '世界杯', '英超', '', None]
TOTAL_LINES = [None, 2.0, 2.25, 2.5, 3.0, 3.5]
HANDICAPS = [None, -1.5, -0.25, 0.0, 0.25, 1.5]
DIST = {0: 0.12, 1: 0.26, 2: 0.27, 3: 0.19, 4: 0.10, 5: 0.04, 6: 0.02}
ROWS = [{'date': f'2026-0{m}-01', 'x': i} for m, i in zip((1, 2, 3, 4, 5, 6, 7, 8), range(8))]


def entries():
    for league, line, handicap in itertools.product(LEAGUES, TOTAL_LINES, HANDICAPS):
        yield (f'dc_rho:{league}/{line}/{handicap}',
               mc.get_dc_rho(league, line, handicap))
    for k in range(8):
        for lam in (0.5, 1.5, 2.6, 4.0):
            yield f'poisson:{k}/{lam}', mc.poisson_pmf(k, lam)
    for rho in (0.0, 0.05, 0.1, -0.1):
        for h, a in itertools.product(range(3), range(3)):
            yield f'dc_tau:{rho}/{h}{a}', mc.dixon_coles_adjustment(rho, 1.5, 1.1, h, a)
            yield f'dc_prob:{rho}/{h}{a}', mc.dixon_coles_score_prob(h, a, 1.5, 1.1, rho)
    for n in (1, 2, 3, 5):
        yield f'goal_rec:{n}', mc.recommend_goal_counts_from_dist(DIST, n)
    yield 'goal_dist', mc.get_goal_count_distribution_from_dist(DIST)
    yield 'goal_dist_empty', mc.get_goal_count_distribution_from_dist({})

    names = mc.get_feature_names()
    yield 'feature_names', names
    yield 'feature_defaults', mc.get_feature_defaults()
    for label, payload in (('empty', {}), ('full', {n: 0.0 for n in names}),
                           ('partial', {n: 0.0 for n in names[:5]}),
                           ('extra', {**{n: 0.0 for n in names}, 'extra': 1.0})):
        yield f'validate:{label}', mc.validate_features(payload)
        yield f'audit:{label}', mc.audit_feature_payload(payload)
    for name in names[:6] + ['不存在的特征']:
        yield f'describe:{name}', mc.get_feature_description(name)

    for ratio in (0.0, 0.5, 0.8, 1.0):
        yield f'split:{ratio}', mc.split_by_time(list(ROWS), ratio)
    yield 'split:empty', mc.split_by_time([], 0.8)
