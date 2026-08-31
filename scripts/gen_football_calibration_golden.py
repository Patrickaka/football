# -*- coding: utf-8 -*-
"""生成 football 校准族的黄金语料条目。

**保序回归返回的是闭包**——黄金里存的是它在一组探针概率上的输出，
不是函数对象（比函数对象等于比内存地址）。

签名都按真实契约核对过：`_get_bucket_key` 是
`(league, total_line, asian, expected_total)`——第三第四个参数很容易写反，
迁移时就在适配层写反过一次，双跑差分抓到了。
"""
import itertools
import pathlib

from src.domain.sports.football import calibration as cal
from src.domain.sports.football import calibration_buckets as cb
from src.domain.sports.football import scoring_model as sm
from tests.domain.golden import describe_exception

PROB_PAIRS = [
    [(0.1, 0), (0.3, 0), (0.5, 1), (0.7, 1), (0.9, 1)] * 4,
    [(0.05, 0), (0.15, 0), (0.25, 1), (0.35, 0), (0.45, 1), (0.55, 1), (0.95, 1)] * 3,
    [(0.5, 1)] * 10,
    [(0.5, 0)] * 10,
    [],
    [(0.2, 0)],
]
PROBE = [0.0, 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98, 1.0]

CALIBRATION_DATA = [
    None, {},
    {'platt_params': (1.0, 0.0)},
    {'platt_params': (2.0, -0.5)},
    {'platt_params': (0.5, 1.0), 'isotonic': [(0.1, 0.05), (0.5, 0.55), (0.9, 0.95)]},
    {'platt_params': (1.0, 0.0), 'league_profile': {'avg_goal': 1.52, 'draw_mult': 0.95},
     'asian_handicap': 0.5, 'total_line': 2.5},
    {'draw_rate': 0.30, 'goal_rate': 1.1, 'score_rate': 0.9},
]

MATRICES = {f'{lh}-{la}': sm.build_score_matrix(lh, la, 7, -0.11)
            for lh, la in ((1.5, 1.1), (2.4, 0.8), (0.9, 0.9))}

ALIAS_MAP = {'曼联': ['曼彻斯特联', '红魔'], '国安': ['北京国安'], '皇马': ['皇家马德里']}
TEAM_NAMES = ('曼联', '曼彻斯特联', '红魔', '北京国安', '国安', '皇家马德里',
              '切尔西', '', None, '  曼联  ', '北京国安足球俱乐部', '联')

HISTORY = [
    {'actual_home': 1, 'actual_away': 0,
     'predicted_probs': {(1, 0): 0.12, (0, 0): 0.10, (2, 1): 0.08}},
    {'actual_home': 2, 'actual_away': 2,
     'predicted_probs': {(2, 2): 0.05, (1, 1): 0.11, (0, 1): 0.09}},
]


def entries():
    for i, pairs in enumerate(PROB_PAIRS):
        yield f'platt_fit:{i}', cal.fit_platt_scaling(pairs)
        fn = cal.isotonic_regression_calibration(pairs)
        yield f'isotonic:{i}', [round(fn(x), 12) for x in PROBE]
    for A, B in ((1.0, 0.0), (2.0, -0.5), (0.5, 1.0), (-1.0, 0.0)):
        for p in (0.0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0):
            yield f'platt_apply:{A}/{B}/{p}', cal.calibrate_with_platt(
                {(0, 0): p, (1, 0): 1 - p}, {'platt_params': (A, B)})

    for mk, matrix in sorted(MATRICES.items()):
        for ci, data in enumerate(CALIBRATION_DATA):
            yield f'platt_matrix:{mk}/{ci}', cal.calibrate_with_platt(matrix, data)
            yield f'hierarchical:{mk}/{ci}', cal.hierarchical_calibration(matrix, data)
            for method in ('platt', 'isotonic', 'none', 'unknown', 'hierarchical'):
                yield (f'calibrate:{mk}/{method}/{ci}',
                       cal.calibrate_probabilities(matrix, method, data))
        yield f'hierarchical_default:{mk}', cal.hierarchical_calibration(matrix)

    for ci, data in enumerate(CALIBRATION_DATA):
        for p_draw in (0.10, 0.22, 0.28, 0.40):
            try:
                yield f'draw_factor:{ci}/{p_draw}', cal._get_draw_calibration_factor(data, p_draw)
            except Exception as exc:
                yield f'draw_factor:{ci}/{p_draw}', describe_exception(exc)
        for total_goals in (0, 1, 2, 3, 5):
            try:
                yield (f'goal_factor:{ci}/{total_goals}',
                       cal._get_goal_calibration_factor(data, total_goals,
                                                        {0: 0.2, 1: 0.3, 2: 0.3, 3: 0.2}))
            except Exception as exc:
                yield f'goal_factor:{ci}/{total_goals}', describe_exception(exc)
        for h, a in ((0, 0), (1, 0), (1, 1), (2, 1), (3, 3)):
            try:
                yield (f'score_factor:{ci}/{h}{a}',
                       cal._get_score_calibration_factor(data, h, a, 0.1))
            except Exception as exc:
                yield f'score_factor:{ci}/{h}{a}', describe_exception(exc)

    for name in TEAM_NAMES:
        yield f'alias:{name!r}', cal.resolve_team_alias(name, ALIAS_MAP)
        yield f'alias_none:{name!r}', cal.resolve_team_alias(name, None)
    yield 'platt_pairs', cal.platt_pairs_from_history(HISTORY)
    yield 'platt_pairs_empty', cal.platt_pairs_from_history([])

    for league, total_line, expected, asian in itertools.product(
            ('英超', '瑞典超', ''), (2.0, 2.5, 2.75, 3.25), (1.9, 2.6, 3.4),
            (-1.5, -0.25, 0.0, 0.5, 1.5)):
        yield (f'bucket:{league}/{total_line}/{expected}/{asian}',
               cb.goal_bucket_key(league, total_line, asian=asian, expected_total=expected))
    for league, total_line in itertools.product(('英超', ''), (2.0, 2.5)):
        yield f'bucket_default:{league}/{total_line}', cb.goal_bucket_key(league, total_line)
    for score, league, total_line, asian, level in itertools.product(
            ('1-0', '0-0', '2-2'), ('英超', ''), (2.0, 2.75), (-0.5, 0.0, 1.0), (0, 1, 2)):
        yield (f'bayes_bucket:{score}/{league}/{total_line}/{asian}/{level}',
               cb.bayesian_bucket_key(score, league, total_line, asian, level))

    for i, db in enumerate(({}, None, 'x',
                            {'k': {'calibration_factors': {'1': 0.9, '2': 1.1, 'x': 1.0}}},
                            {'k': {'predicted_distributions': [{'0': 0.2, '1': 0.8}, 'bad']}},
                            {'k': 'notadict'})):
        import copy
        yield f'restore_keys:{i}', cb._restore_goal_keys(copy.deepcopy(db))
    for i, mapping in enumerate(({'1': 0.5, '2': 0.5}, {1: 0.5, 2: 0.5}, {'a': 1.0}, {})):
        yield f'int_keyed:{i}', cb._int_keyed(mapping)

    # 校准因子计算——**第一版差分漏了它**，于是一个 NameError（少 import
    # defaultdict）一路溜到全量测试才暴露。覆盖报告只能报它见过的函数。
    import copy
    buckets = []
    for n in (0, 3, 4, 8, 30, 200):
        for weighted in (False, True):
            bucket = {
                'sample_count': n,
                'actual_goals': ([1, 2, 2, 3, 0, 2, 1, 4] * 30)[:n],
                'predicted_distributions': [{0: 0.15, 1: 0.3, 2: 0.3, 3: 0.15, 4: 0.1}] * n,
            }
            if weighted:
                bucket['sample_weights'] = [0.5] * n
            buckets.append((f'{n}/{weighted}', bucket))
    buckets.append(('empty', {'sample_count': 0, 'actual_goals': [],
                              'predicted_distributions': []}))
    for n in (8, 30):
        buckets.append((f'weighted_count/{n}', {
            'sample_count': n, 'weighted_sample_count': n * 0.4,
            'actual_goals': ([1, 2, 2, 3, 0, 2, 1, 4] * 30)[:n],
            'predicted_distributions': [{0: 0.15, 1: 0.3, 2: 0.3, 3: 0.15, 4: 0.1}] * n,
            'sample_weights': [0.4] * n,
        }))
    for label, bucket in buckets:
        work = copy.deepcopy(bucket)
        cb.compute_calibration_factors(work)
        yield f'factors:{label}', work.get('calibration_factors')

    for di, dist in enumerate(({0: 0.2, 1: 0.5, 2: 0.3}, {0: 1.0}, {}, {0: 0.0, 1: 0.0})):
        for fi, factors in enumerate(({}, None, {1: 1.5}, {1: 1.5, 3: 2.0}, {0: 0.0, 1: 0.0})):
            yield f'apply_goal:{di}/{fi}', cb.apply_goal_calibration(dist, factors)
