# -*- coding: utf-8 -*-
"""生成 football scoring 的黄金语料。

覆盖清单**对着领域层导出的 34 个函数逐个核过**——`_score_entry` 与
`apply_market_change_prior` 第一版漏掉了，覆盖报告的「没覆盖」一栏抓到的。
"""
import itertools

from src.domain.sports.football import scoring as new
from src.domain.sports.football import scoring_model as sm
from scripts.gen_football_modeling_golden import REAL, STRENGTH


def _y(fn, label, *a, **kw):
    try:
        yield f'{fn}:{label}', getattr(new, fn)(*a, **kw)
    except Exception as exc:
        yield f'{fn}:{label}', f'{type(exc).__name__}: {exc}'


def entries():
    MATRICES = {f'{lh}-{la}': sm.build_score_matrix(lh, la, 7, -0.11)
                for lh,la in ((1.5,1.1),(2.4,0.8),(0.9,0.9))}
    def top(m, k=12): return sorted(m.items(), key=lambda kv:-kv[1])[:k]
    LP = {'avg_goal':1.52,'home_boost':1.08,'low_score':0.88,'draw_mult':0.95,'name':'英超'}
    EURO_ODDS = {'home':2.0,'draw':3.4,'away':3.8}

    for mk, m in MATRICES.items():
        cands = top(m)
        dist = sm._ou_total_distribution(2.6, 6)
        yield from _y('_normalize_goal_dist', mk, dist)
        for i,(a,e,t) in enumerate(REAL[:6]):
            yield from _y('_anchor_goal_dist_to_total_line', f'{mk}/{i}', dist, t)
            yield from _y('_goal_over_under_from_line', f'{mk}/{i}', dist, t)
            yield from _y('_anchor_score_candidates_to_1x2', f'{mk}/{i}', m, e['close'])
            yield from _y('_anchor_score_candidates_to_goal_mean', f'{mk}/{i}', m, t['implied_total'])
            yield from _y('_adjust_score_probs_with_total_movement', f'{mk}/{i}', m, t)
            yield from _y('_adjust_goal_dist_with_total_movement', f'{mk}/{i}', dist, t)
            yield from _y('_joint_market_state', f'{mk}/{i}', a, e, t)
            yield from _y('_assess_market_data_quality', f'{mk}/{i}', a, e, t)
            yield from _y('_total_market_tempo_signal', f'{mk}/{i}', t)
            yield from _y('_score_total_movement_factor', f'{mk}/{i}', 1, 0, t)
            yield from _y('_apply_joint_market_state', f'{mk}/{i}', cands, a, e, t)
            yield from _y('_alignment_score', f'{mk}/{i}', 1, 0, a, e, t)
            yield from _y('_recommend_reasons', f'{mk}/{i}', 1, 0, a, e, t)
            yield from _y('_recommend_reasons', f'{mk}/{i}/team', 1, 0, a, e, t, STRENGTH)
            yield from _y('_candidate_result_support', f'{mk}/{i}', cands)
            yield from _y('_candidate_result_support', f'{mk}/{i}/4', cands, 4)
            ec = e['close']
            yield from _y('fit_lambdas_from_markets', f'{mk}/{i}',
                a.get('implied_supremacy', 0.3), t['close_line'], t['close_prob']['over'],
                ec['home'], ec['draw'], ec['away'], t['open_line'], STRENGTH, None, LP,
                a['handicap'], a['open_handicap'])
            for n_ in (1, 3, 5):
                scored = [(c, p, {'h': c[0], 'a': c[1]}) for c, p in cands]
                picked = [{'home': c[0], 'away': c[1], 'prob': p} for c, p in cands[:5]]
                yield from _y('_diversify_score_recommendations', f'{mk}/{i}/{n_}',
                    picked, cands, n_, a.get('favor', 'home'), 0, 2)
                yield from _y('_diversify_score_recommendations', f'{mk}/{i}/{n_}/short',
                    picked[:2], cands, n_, a.get('favor', 'home'), 0, 2)
        for line in (2.0, 2.5, 3.25):
            yield from _y('_score_total_line_factor', f'{mk}/{line}', 1, 0, line)
            yield from _y('_common_score_overheat_factor', f'{mk}/{line}', 1, 0, 0.12, line)
        for p_over in (0.35, 0.5, 0.68):
            yield from _y('_implied_total_mean', f'{mk}/{p_over}', 2.5, p_over)

    for h,a_ in itertools.product(range(4), range(4)):
        yield from _y('_score_result_code', f'{h}{a_}', h, a_)
        yield from _y('_get_score_cluster', f'{h}{a_}', h, a_)
        yield from _y('score_pattern', f'{h}{a_}', h, a_)
        yield from _y('_estimate_score_odds', f'{h}{a_}', h, a_, EURO_ODDS)
        yield from _y('score_implied_prob_from_euro', f'{h}{a_}', h, a_, EURO_ODDS)
        yield from _y('_baseline_freq', f'{h}{a_}', h, a_)
        yield from _y('score_heat_label', f'{h}{a_}', h, a_, 0.12, LP)
        yield from _y('score_heat_label', f'{h}{a_}/euro', h, a_, 0.12, LP, EURO_ODDS)
        yield from _y('_score_entry', f'{h}{a_}', h, a_, 0.12)
    for p in (0.02,0.1,0.3,0.6,0.95):
        yield from _y('_heat_filter_weight', str(p), p)
    for c in ('C0','C1','C2','C3','unknown'):
        yield from _y('_get_cluster_name', c, c)
    HF_DIST = {'distribution': {'HH': 0.3, 'DH': 0.2, 'AA': 0.5}}
    HF_ROWS = {'probs': [{'code': 'HH', 'raw_prob': 0.3}, {'code': 'DH', 'probability': 20.0},
                         {'code': 'AA', 'raw_prob': 0.5}, {'no_code': 1}]}
    for name, hf in (('dist', HF_DIST), ('rows', HF_ROWS), ('none', None), ('empty', {})):
        yield from _y('_half_full_probs_to_dict', name, hf)
    rows = [{'half': 'H', 'full': 'H', 'probability': 30.0, 'code': 'HH'},
            {'half': 'D', 'full': 'H', 'probability': 20.0, 'code': 'DH'},
            {'half': 'A', 'full': 'A', 'probability': 50.0, 'code': 'AA'}]
    for i,(a,e,t) in enumerate(REAL[:4]):
        yield from _y('_adjust_half_full_with_score_context', str(i), rows, top(MATRICES['1.5-1.1']))
        yield from _y('_adjust_half_full_with_market_context', str(i), rows, a, e, t)
        score_probs = {f'{h}-{aa}': p for (h,aa),p in top(MATRICES['1.5-1.1'])}
        for w in (0.0, 0.08, 0.3):
            yield from _y('apply_market_change_prior', f'{i}/{w}', score_probs, a, t, w)

