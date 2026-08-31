# -*- coding: utf-8 -*-
"""生成 football modeling 的黄金语料条目。

**每个函数的签名都按真实契约核对过**——F-3 与 F-4 各踩过一次「语料喂错形状、
两边抛同样的错、差分零差异其实什么也没测」（判据 8）。生成器最后会自查
「有没有函数一次都没产出有效值」。

**MCMC 用注入的固定随机源**（判据 16）：不注入的话黄金不可复现。
"""
import itertools
import json
import pathlib
import random

from src.domain.sports.football import bayes, lambdas, markets, risk, scoring_model
from tests.domain.golden import describe_exception

ROOT = pathlib.Path(__file__).resolve().parents[1]
MK = json.loads((ROOT / 'tests/fixtures/football_markets_corpus.json').read_text(encoding='utf-8'))

LAMS = [(1.5, 1.1), (2.4, 0.8), (0.7, 2.1), (1.2, 1.2), (3.0, 0.4), (0.9, 0.9), (2.0, 1.6)]
RHOS = [0.0, -0.06, -0.11, -0.16, 0.05, 0.12, -0.3, 0.3]
PROFILES = [None,
            {'avg_goal': 1.42, 'home_boost': 1.06, 'low_score': 0.92, 'draw_mult': 1.0},
            {'avg_goal': 1.52, 'home_boost': 1.08, 'low_score': 0.88, 'draw_mult': 0.95,
             'name': '英超'}]

STRENGTH = {
    'attack_home': 1.6, 'defense_home': 1.0, 'attack_away': 1.1, 'defense_away': 1.3,
    'form_diff': 0.4, 'momentum_supremacy': 0.05,
    'home_recent': {'games': 10, 'gf': 18, 'ga': 9, 'form_pts': 2.0, 'attack': 1.8, 'defense': 0.9},
    'away_recent': {'games': 10, 'gf': 11, 'ga': 13, 'form_pts': 1.2, 'attack': 1.1, 'defense': 1.3},
    'home_venue': {'games': 5, 'gf': 12, 'ga': 3, 'form_pts': 2.6, 'attack': 2.4, 'defense': 0.6},
    'away_venue': {'games': 5, 'gf': 4, 'ga': 7, 'form_pts': 1.0, 'attack': 0.8, 'defense': 1.4},
    'elo_home': 1580, 'elo_away': 1490, 'elo_xg_home': 1.6, 'elo_xg_away': 1.1,
}


class SeededRandom:
    """固定随机源：MCMC 的黄金靠它才可复现（判据 16）"""

    def __init__(self, seed):
        self._r = random.Random(seed)

    def random(self):
        return self._r.random()


def _cover_probs(prob_dict, side):
    """亚盘概率的键名随让球方向变，取值要兼容三种形态"""
    if side == 'home':
        return prob_dict.get('home_give') or prob_dict.get('home_recv') or prob_dict.get('home')
    return prob_dict.get('away_recv') or prob_dict.get('away_give') or prob_dict.get('away')


def real_market_triples():
    """用 F-2 的领域层从真实赔率语料构造 asian / euro / total 三件套

    **必须补上 `implied_supremacy` / `implied_lambdas`**：`analyze_asian` /
    `analyze_euro` 不产出这两个键，是 `pipeline` 事后塞进去的（线上缓存里
    确实有）。第一版语料漏了这步，于是 `compute_prediction_confidence` 的
    三条扣分分支一条都走不到——122 条样本的 score **恒等于 0.880**，
    把 `CONFIDENCE_HIGH_THRESHOLD` 从 0.72 改成 0.4 都测不出来（判据 23）。
    """
    out = []
    for i in range(min(len(MK['asian']), len(MK['euro']), len(MK['total']))):
        try:
            a = markets.analyze_asian(MK['asian'][i])
            e = markets.analyze_euro(MK['euro'][i])
            t = markets.analyze_total(MK['total'][i])
        except Exception:
            continue
        ph = _cover_probs(a['close_prob'], 'home')
        pa = _cover_probs(a['close_prob'], 'away')
        a['implied_supremacy'] = lambdas.asian_implied_supremacy(
            a['handicap'], ph, pa, t['implied_total'])
        ec = e['close']
        e['implied_supremacy'] = lambdas.euro_implied_supremacy(
            ec['home'], ec['draw'], ec['away'], t['implied_total'])
        eh, ea = lambdas.euro_implied_lambdas(ec['home'], ec['draw'], ec['away'],
                                              t['implied_total'])
        e['implied_lambdas'] = {'home': eh, 'away': ea}
        out.append((a, e, t))
    return out


REAL = real_market_triples()


def entries():
    # ---- scoring_model ----
    for lh, la in LAMS:
        for rho in RHOS:
            for dist in ('poisson', 'negative_binomial'):
                yield (f'matrix:{lh}/{la}/{rho}/{dist}',
                       scoring_model.build_score_matrix(lh, la, 7, rho, dist))
            for h, a in itertools.product(range(5), range(5)):
                yield f'dc_tau:{h}{a}/{lh}/{la}/{rho}', scoring_model._dc_tau(h, a, lh, la, rho)
        m = scoring_model.build_score_matrix(lh, la, 7, -0.11)
        yield f'margins:{lh}/{la}', scoring_model._matrix_margins(m)
        for mk in (4, 6, 8):
            yield f'total_margins:{lh}/{la}/{mk}', scoring_model._matrix_total_margins(m, mk)
        for k in range(7):
            yield f'ou_dist:{lh}/{la}/{k}', scoring_model._ou_total_distribution(lh + la, k)
        for pd in (0.20, 0.25, 0.28, 0.32):
            yield f'dc_rho:{lh}/{la}/{pd}', scoring_model._estimate_dc_rho(lh, la, pd)
        for euro in ((0.45, 0.28, 0.27), (0.7, 0.2, 0.1), (0.2, 0.3, 0.5)):
            yield f'to_euro:{lh}/{la}/{euro[0]}', scoring_model.calibrate_to_euro(m, *euro)
    for k in range(6):
        for r, pp in ((2.0, 0.4), (5.0, 0.6), (0.5, 0.9), (1.0, 0.0), (1.0, 1.0), (-1.0, 0.5)):
            yield f'nb_pmf:{k}/{r}/{pp}', scoring_model._negative_binomial_pmf(k, r, pp)
    for mean, var in ((1.5, 1.8), (1.5, 1.5), (1.5, 1.2), (2.0, 4.0), (0.0, 1.0)):
        try:
            yield f'nb_params:{mean}/{var}', scoring_model._nb_params_from_mean_var(mean, var)
        except Exception as exc:
            yield f'nb_params:{mean}/{var}', describe_exception(exc)
    for lp in PROFILES:
        yield f'nb_overdisp:{lp}', scoring_model._estimate_nb_overdispersion(lp)
    for h, a in itertools.product(range(4), range(4)):
        yield f'outcome:{h}{a}', scoring_model._outcome(h, a)
        yield f'result_label:{h}{a}', scoring_model._result_label(h, a)
    for x in (-5.0, -1.0, 0.0, 1.0, 5.0, 100.0, -100.0):
        yield f'sigmoid:{x}', scoring_model._sigmoid(x)

    # ---- lambdas ----
    for i, (a, e, t) in enumerate(REAL):
        hcap = a['handicap']
        ph = _cover_probs(a['close_prob'], 'home')
        pa = _cover_probs(a['close_prob'], 'away')
        oph = _cover_probs(a['open_prob'], 'home')
        opa = _cover_probs(a['open_prob'], 'away')
        yield (f'asian_sup:{i}',
               lambdas.asian_implied_supremacy(hcap, ph, pa, t['implied_total']))
        yield (f'asian_sup_open:{i}',
               lambdas.asian_implied_supremacy(hcap, ph, pa, t['implied_total'],
                                               a['open_handicap'], oph, opa))
        ec = e['close']
        yield f'euro_sup:{i}', lambdas.euro_implied_supremacy(ec['home'], ec['draw'], ec['away'])
        yield (f'euro_sup_hint:{i}',
               lambdas.euro_implied_supremacy(ec['home'], ec['draw'], ec['away'],
                                              t['implied_total']))
        yield (f'euro_lams:{i}',
               lambdas.euro_implied_lambdas(ec['home'], ec['draw'], ec['away'],
                                            t['implied_total']))
        for lam in ((1.5, 1.1), (2.0, 1.0)):
            yield (f'hcap_adj:{i}/{lam[0]}',
                   lambdas.apply_handicap_change_adjustment(lam[0], lam[1],
                                                            a['open_handicap'], hcap))
            yield (f'hcap_adj_time:{i}/{lam[0]}',
                   lambdas.apply_handicap_change_adjustment(lam[0], lam[1], a['open_handicap'],
                                                            hcap, '08-27 10:00', '08-28 12:00'))
            yield (f'total_adj:{i}/{lam[0]}',
                   lambdas.apply_total_line_change_adjustment(lam[0], lam[1],
                                                              t['open_line'], t['close_line']))
            yield (f'total_adj_time:{i}/{lam[0]}',
                   lambdas.apply_total_line_change_adjustment(lam[0], lam[1], t['open_line'],
                                                              t['close_line'],
                                                              '08-27 10:00', '08-28 12:00'))
    for sa, se in itertools.product((-1.5, -0.5, 0.0, 0.5, 1.5), (-1.2, 0.0, 1.2)):
        yield f'blend_sup:{sa}/{se}', lambdas.blend_market_supremacy(sa, se)
    for tt, sup in itertools.product((2.0, 2.6, 3.2), (-1.0, 0.0, 1.0)):
        yield f'market_lams:{tt}/{sup}', lambdas.market_implied_lambdas(tt, sup)
    for tm in ('2026-08-28 12:00', '08-28 12:00', '', None, 'bad', '12:00'):
        yield f'parse_time:{tm!r}', lambdas._parse_time(tm)
    for market in ((1.5, 1.1), (2.0, 1.0), (0.8, 2.2)):
        for team in (None, (1.4, 1.2), (2.2, 0.7)):
            for elo in (None, (1.6, 1.0)):
                yield (f'blend_lams:{market}/{team}/{elo}',
                       lambdas.blend_lambdas_with_market(market, team, elo))
    for sup in (-1.0, 0.0, 1.0):
        for tl in (2.0, 2.75, 3.5):
            yield f'est_lams:{sup}/{tl}', lambdas.estimate_lambdas(sup, tl)
            yield f'est_lams_min:{sup}/{tl}', lambdas.estimate_lambdas(sup, tl, 0.3)
    targets = (0.45, 0.28, 0.27)
    for lh, la in LAMS:
        for rho in (0.0, -0.11):
            ou = dict(scoring_model._matrix_total_margins(
                scoring_model.build_score_matrix(1.6, 1.2, rho=rho)))
            yield (f'fit_err:{lh}/{la}/{rho}',
                   lambdas._lambda_fit_error((lh, la), 0.4, 2.6, targets, rho))
            yield (f'fit_err_ou:{lh}/{la}/{rho}',
                   lambdas._lambda_fit_error((lh, la), 0.4, 2.6, targets, rho, ou))
            yield (f'fit_err_team:{lh}/{la}/{rho}',
                   lambdas._lambda_fit_error((lh, la), 0.4, 2.6, targets, rho, None, (1.4, 1.2)))
            yield (f'fit_err_both:{lh}/{la}/{rho}',
                   lambdas._lambda_fit_error((lh, la), 0.4, 2.6, targets, rho, ou, (1.4, 1.2)))
    for lp in PROFILES:
        for tt in (2.2, 2.6, 3.1):
            yield f'team_lams:{lp}/{tt}', lambdas.team_poisson_lambdas(STRENGTH, tt, lp)
            yield (f'team_lams_inline:{lp}/{tt}',
                   lambdas.team_poisson_lambdas(dict(STRENGTH, league_profile=lp), tt, None))
    try:
        yield 'team_lams:empty', lambdas.team_poisson_lambdas({}, 2.6, None)
    except Exception as exc:   # 空 strength 会抛——把异常型态也钉住
        yield 'team_lams:empty', describe_exception(exc)

    # ---- risk ----
    for lh, la in LAMS:
        m = scoring_model.build_score_matrix(lh, la, 7, -0.11)
        top = sorted(m.items(), key=lambda kv: -kv[1])[:12]
        for n in (1, 3, 5, 8):
            for thr in (0.0, 0.5, 1.0):
                yield (f'diverse:{lh}/{la}/{n}/{thr}',
                       risk.diverse_score_selection(top, n, thr))
    for i, (a, e, t) in enumerate(REAL):
        yield f'confidence:{i}', risk.compute_prediction_confidence(a, e, t)
        conf = risk.compute_prediction_confidence(a, e, t, STRENGTH)
        yield f'confidence_team:{i}', conf
        for si, steam in enumerate((None, {}, {'detected': True, 'direction': 'home',
                                               'strength': 0.6}, {'detected': False})):
            for mi, sim in enumerate((None, {}, {'count': 30, 'confidence': 0.7,
                                                 'avg_distance': 0.2})):
                yield (f'risk_level:{i}/{si}/{mi}',
                       risk._evaluate_risk_level(a, e, t, steam, conf, sim))

    # ---- bayes ----
    for seed in (7, 42):
        for lh, la in LAMS[:3]:
            yield (f'mcmc:{seed}/{lh}/{la}',
                   bayes._mcmc_sample_lambdas(targets, 2.6, 0.4, None, None,
                                              n_samples=40, burn_in=10, rng=SeededRandom(seed)))
    for lp in PROFILES[:2]:
        yield (f'bayes_scores:{lp}',
               bayes.bayesian_predict_scores(targets, 2.6, 0.4, lp, STRENGTH,
                                             rng=SeededRandom(3)))
    for lp in PROFILES:
        yield f'gamma_prior:{lp}', bayes._gamma_prior_params(lp, STRENGTH)
        yield f'gamma_prior_nost:{lp}', bayes._gamma_prior_params(lp, None)
    yield 'rho_prior', bayes._rho_prior_params()
    for pd in (0.10, 0.22, 0.28, 0.40):
        yield f'draw_bounds:{pd}', bayes._draw_probability_bounds(pd)
        for hcap in (-1.0, 0.0, 0.5, 1.5):
            for rates in ((0.25, 0.25, 0.25), (0.35, 0.15, 0.28), (0.1, 0.4, 0.3)):
                yield (f'heuristic_draw:{pd}/{hcap}/{rates[0]}',
                       bayes._heuristic_draw_calibration(0.45, pd, 1 - 0.45 - pd, hcap, *rates))
    for ph, pd, pa in ((0.45, 0.28, 0.27), (0.7, 0.15, 0.15), (0.2, 0.3, 0.5)):
        for hcap in (-1.5, -0.25, 0.0, 0.75, 2.0):
            yield (f'redistribute:{pd}/{hcap}',
                   bayes._redistribute_draw_probability(ph, pd, pa, hcap))
            yield (f'calib_draw:{pd}/{hcap}',
                   bayes.calibrate_draw_probability(ph, pd, pa, hcap))
            yield (f'calib_draw_rates:{pd}/{hcap}',
                   bayes.calibrate_draw_probability(ph, pd, pa, hcap, 0.32, 0.18, 0.27))
    for i, (a, e, t) in enumerate(REAL[:8]):
        for lp in PROFILES:
            yield f'residual_feat:{i}/{lp}', bayes._build_residual_features(a, e, t, STRENGTH, lp)
            yield (f'residual_feat_noteam:{i}/{lp}',
                   bayes._build_residual_features(a, e, t, None, lp))
    feats = bayes._build_residual_features(REAL[0][0], REAL[0][1], REAL[0][2], STRENGTH, None)
    for lh, la in LAMS[:3]:
        m = scoring_model.build_score_matrix(lh, la, 7, -0.11)
        yield f'residual_corr:{lh}/{la}', bayes.apply_residual_correction(m, feats)
        yield f'residual_corr_nofeat:{lh}/{la}', bayes.apply_residual_correction(m, None)
