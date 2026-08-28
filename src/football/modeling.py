# -*- coding: utf-8 -*-
"""足球比分建模的适配层：喂配置，算法在 `domain/sports/football/`

领域层拆成四块：
- `scoring_model`：泊松 / 负二项 / Dixon-Coles / 比分矩阵与边际
- `lambdas`：期望进球的市场隐含、球队推算、盘口变化调整、拟合与融合
- `risk`：置信度、比分多样化、风险评级
- `bayes`：MCMC 采样、残差修正、平局校准（随机源由调用方注入）

45 个导出名与迁移前逐个一致（存量测试按名字导入）。
"""

from ..common.logger import setup_logger
from ..domain.sports.football import bayes as _bayes
from ..domain.sports.football import lambdas as _lam
from ..domain.sports.football import risk as _risk
from ..domain.sports.football import scoring_model as _sm

log = setup_logger('football')

# ---- scoring_model ----
_negative_binomial_pmf = _sm._negative_binomial_pmf
_nb_params_from_mean_var = _sm._nb_params_from_mean_var
_estimate_nb_overdispersion = _sm._estimate_nb_overdispersion
_dc_tau = _sm._dc_tau
build_score_matrix = _sm.build_score_matrix
_matrix_margins = _sm._matrix_margins
_matrix_total_margins = _sm._matrix_total_margins
_ou_total_distribution = _sm._ou_total_distribution
_estimate_dc_rho = _sm._estimate_dc_rho
calibrate_to_euro = _sm.calibrate_to_euro
_outcome = _sm._outcome
_result_label = _sm._result_label
_sigmoid = _sm._sigmoid

# ---- lambdas ----
_asian_payout_home = _lam._asian_payout_home
_asian_cover_prob = _lam._asian_cover_prob
asian_implied_supremacy = _lam.asian_implied_supremacy
euro_implied_supremacy = _lam.euro_implied_supremacy
euro_implied_lambdas = _lam.euro_implied_lambdas
blend_market_supremacy = _lam.blend_market_supremacy
market_implied_lambdas = _lam.market_implied_lambdas
_parse_time = _lam._parse_time
apply_handicap_change_adjustment = _lam.apply_handicap_change_adjustment
apply_total_line_change_adjustment = _lam.apply_total_line_change_adjustment
blend_lambdas_with_market = _lam.blend_lambdas_with_market
team_poisson_lambdas = _lam.team_poisson_lambdas
_lambda_fit_error = _lam._lambda_fit_error
_fit_lambda_refine = _lam._fit_lambda_refine
_fit_lambda_grid = _lam._fit_lambda_grid
estimate_lambdas = _lam.estimate_lambdas

# ---- risk ----
compute_prediction_confidence = _risk.compute_prediction_confidence
diverse_score_selection = _risk.diverse_score_selection
_evaluate_risk_level = _risk._evaluate_risk_level

# ---- bayes ----
_gamma_prior_params = _bayes._gamma_prior_params
_rho_prior_params = _bayes._rho_prior_params
_log_posterior = _bayes._log_posterior
_mcmc_sample_lambdas = _bayes._mcmc_sample_lambdas
bayesian_predict_scores = _bayes.bayesian_predict_scores
_build_residual_features = _bayes._build_residual_features
_train_residual_model = _bayes._train_residual_model
apply_residual_correction = _bayes.apply_residual_correction
_train_draw_calibration_model = _bayes._train_draw_calibration_model
calibrate_draw_probability = _bayes.calibrate_draw_probability
_draw_probability_bounds = _bayes._draw_probability_bounds
_redistribute_draw_probability = _bayes._redistribute_draw_probability
_heuristic_draw_calibration = _bayes._heuristic_draw_calibration
