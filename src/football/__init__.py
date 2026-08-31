# -*- coding: utf-8 -*-
"""
足球比分预测模块 - 动态抓取赔率数据（数据来源: odds.500.com）

包结构（薄 facade，业务逻辑在子模块）：
- config      常量配置、联赛画像、可选子系统延迟导入
- fetching    HTTP抓取/缓存/限流、比赛列表
- parsing     赔率页解析、公司赔率、球队实力
- markets     亚盘/欧赔/大小球/凯利分析
- modeling    泊松/贝叶斯建模、λ拟合、比分矩阵
- calibrating Platt/isotonic/分层校准
- upset       爆冷识别
- scoring     比分候选、半全场、推荐挑选
- pipeline    analyze_match 编排
- cli         命令行入口
- elo/ml/result_sync/bayes_report 等原有独立模块不变
"""

from .config import (
    FOOTBALL_PREDICTION_LOGIC_VERSION, LOTTERY_OFFICIAL_ODDS_WEIGHT, SCORE_1X2_MARKET_ANCHOR_STRENGTH, ACTIONABLE_1X2_MIN_PROBABILITY, ACTIONABLE_1X2_MIN_MARGIN,
    get_elo_system, elo_to_goals_expected, elo_to_strength_factor, ELO_AVAILABLE, similar_market_match,
    SIMILAR_MARKET_AVAILABLE, steam_move_detector, integrate_steam_signal, STEAM_MOVE_AVAILABLE, calibrate_predictions,
    get_calibrator, BAYESIAN_CALIBRATION_AVAILABLE, get_cache, set_cache, invalidate_cache,
    clear_all_cache, CACHE_AVAILABLE, get_team_elo, get_elo_difference, DYNAMIC_ELO_AVAILABLE,
    adjust_by_value, identify_value_bets, calculate_value, calculate_ev, VALUE_BETTING_AVAILABLE,
    get_dynamic_weights, fuse_predictions, DYNAMIC_WEIGHTS_AVAILABLE, fuse_poisson_with_prior, get_market_prior,
    MARKET_CLUSTERING_AVAILABLE, BASE, INDEX_URL, INDEX_URLS, MATCH_LIST_CACHE_PATH,
    _MATCH_LIST_STATUS, HEADERS, MIN_AVG_NUMBERS, HANDICAP_TREND_EPS, WATER_TREND_EPS,
    EURO_PROB_TREND_EPS, TOTAL_LEAN_THRESHOLD, KELLY_BIAS_EPS, MAX_GOALS, CLOSE_BLEND_WEIGHT,
    LAMBDA_COARSE_STEP, LAMBDA_FINE_STEP, LAMBDA_FINE_RADIUS, FIT_W_1X2, FIT_W_TOTAL,
    FIT_W_SUPREMACY, FIT_W_OU_DIST, FIT_W_TEAM, SUP_ASIAN_WEIGHT, SUP_EURO_WEIGHT,
    AVG_LEAGUE_GOAL, HOME_VENUE_ATTACK_BOOST, SUPREMACY_CONFLICT_GAP, MOMENTUM_SUPREMACY_WEIGHT, DISPERSION_WINDOW,
    LAMBDA_REFINE_STEPS, LAMBDA_REFINE_STEP0, CONFIDENCE_LOW_THRESHOLD, CONFIDENCE_HIGH_THRESHOLD, LEAGUE_PROFILES,
    HEAT_RATIO_HOT, HEAT_RATIO_COLD, HEAT_FILTER_PENALTY, COLD_FILTER_BONUS, HANDICAP_CHANGE_LAMBDA_BOOST,
    TOTAL_LINE_CHANGE_LAMBDA_BOOST, LAMBDA_WEIGHT_MARKET, LAMBDA_WEIGHT_TEAM, LAMBDA_WEIGHT_ELO, SCORE_BASELINE_FREQ,
    ODDS_PAGES, OUZHI_JSON_URL,
)
from .fetching import (
    FETCH_PAGE_TTL, FETCH_MAX_CONCURRENCY, FETCH_RATE_LIMIT, FETCH_RETRY_ATTEMPTS, FETCH_THROTTLE_SECONDS,
    FETCH_THROTTLE_CEILING, RATE_LIMIT_STATUSES, _FETCH_CACHE_LIMIT, _FETCH_URL_LOCK_LIMIT, _fetch_cache,
    _fetch_url_locks, _fetch_cache_lock, _fetch_semaphore, _fetch_throttle_lock, _fetch_throttle_until,
    _fetch_rate_lock, _fetch_next_slot, _fetch_cache_get, _fetch_cache_set, _fetch_url_lock,
    clear_fetch_cache, _enter_fetch_throttle, _await_fetch_throttle, _await_rate_slot, fetch,
    _fetch_raw, _fetch_once, fetch_json, _fetch_match_list_remote, _save_match_list_cache,
    _load_match_list_cache, get_match_list_status, _zgzcw_schedule_fallback, fetch_match_list, search_match,
)
from .parsing import (
    get_close_total_line, parse_handicap, parse_total_line, parse_lottery_handicap, _lottery_odds_probabilities,
    _blend_lottery_probabilities, _spf_selection_profile, lottery_market_probabilities, _apply_lottery_market_availability, _html_to_text,
    _extract_avg, _handicap_text_to_num, _extract_company_odds, _fetch_avg_page, calculate_bookmaker_consensus,
    fetch_single_company_odds, fetch_yazhi, _extract_handicap_from_segment, _parse_odds_value, fetch_ouzhi,
    fetch_daxiao, RECENT_FORM_PAT, _team_in_context, get_live_league_profile, resolve_league_profile,
    _parse_recent_form, fetch_team_strength, _blend_close_open,
)
from .markets import (
    remove_vig, _analyze_handicap_trend, calculate_implied_total, analyze_asian, _return_rate_from_odds,
    kelly_index_triple, _kelly_outcome_label, _linear_regression_slope, analyze_kelly, analyze_kelly_trend,
    analyze_euro_momentum, fetch_ouzhi_company, compute_dispersion, compute_joint_anomaly, euro_to_handicap_implied,
    compute_euro_asian_deviation, analyze_euro, analyze_total, _poisson_pmf, _poisson_tail_over,
    implied_total_goals,
)
from .modeling import (
    _negative_binomial_pmf, _nb_params_from_mean_var, _estimate_nb_overdispersion, _build_residual_features, _train_residual_model,
    apply_residual_correction, _train_draw_calibration_model, calibrate_draw_probability, _draw_probability_bounds, _redistribute_draw_probability,
    _heuristic_draw_calibration, _gamma_prior_params, _rho_prior_params, _log_posterior, _mcmc_sample_lambdas,
    bayesian_predict_scores, _outcome, market_implied_lambdas, _parse_time, apply_handicap_change_adjustment,
    apply_total_line_change_adjustment, blend_lambdas_with_market, diverse_score_selection, _dc_tau, _matrix_margins,
    _asian_payout_home, _asian_cover_prob, asian_implied_supremacy, euro_implied_supremacy, euro_implied_lambdas,
    blend_market_supremacy, compute_prediction_confidence, team_poisson_lambdas, _ou_total_distribution, _matrix_total_margins,
    estimate_lambdas, _lambda_fit_error, _fit_lambda_refine, _fit_lambda_grid, _estimate_dc_rho,
    build_score_matrix, calibrate_to_euro, _sigmoid, _evaluate_risk_level, _result_label,
)
from .calibrating import (
    LEAGUE_CALIBRATION_CACHE, normalize_team_name, fetch_league_historical_data, train_league_platt_params, get_league_calibration_data,
    recalibrate_league, clear_calibration_cache, list_calibrated_leagues, fit_platt_scaling, calibrate_with_platt,
    isotonic_regression_calibration, calibrate_probabilities, hierarchical_calibration, _get_draw_calibration_factor, _get_goal_calibration_factor,
    _get_score_calibration_factor,
)
from .upset import (
    _evaluate_upset_profile, _evaluate_upset_risk, assess_football_upset,
)
from .scoring import (
    perturb_parameters, ensemble_predict_scores, fit_lambdas_from_markets, _baseline_freq, score_implied_prob_from_euro,
    score_heat_label, _heat_filter_weight, calculate_half_full_time_probs, _half_full_probs_to_dict, predict_scores,
    _estimate_score_odds, _score_entry, _alignment_score, _recommend_reasons, apply_market_change_prior,
    SCORE_CLUSTERS, _get_score_cluster, _get_cluster_name, score_pattern, _score_total_line_factor,
    _common_score_overheat_factor, _total_market_tempo_signal, _joint_market_state, _apply_joint_market_state, _score_total_movement_factor,
    _adjust_score_probs_with_total_movement, _anchor_score_candidates_to_1x2, _anchor_score_candidates_to_goal_mean, _normalize_goal_dist, _implied_total_mean,
    _anchor_goal_dist_to_total_line, _goal_over_under_from_line, _adjust_goal_dist_with_total_movement, _score_result_code, _candidate_result_support,
    _adjust_half_full_with_score_context, _adjust_half_full_with_market_context, _assess_market_data_quality, _pick_recommendations, _diversify_score_recommendations,
)
from .pipeline import (
    _cached_prediction_logic_version, _is_prediction_cache_current, _is_lottery_cache_current, build_match_analysis, analyze_match,
)
from .cli import (
    main, _heat_tag, render_cli,
)
