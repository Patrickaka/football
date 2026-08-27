# -*- coding: utf-8 -*-
"""
福彩3D预测器 V3.1+（标准库版，准确率优化）

包结构（薄 facade，业务逻辑在子模块）：
- config     常量配置与可调权重
- fetching   历史数据抓取
- features   基础特征（形态/遗漏/马尔可夫/热冷/和值/对频/斜率）
- scoring    评分与选号（窗口权重/数字评分/组三组六/直选排名）
- fusion     规则+ML融合与策略推荐
- records    线上预测记录与结算
- backtest   回测/随机基线/置换检验/权重搜索
- prediction run_prediction 预测入口与 CLI
- ml         机器学习预测器
"""

from .config import (
    URL, RECENT_WINDOWS, RECENT_WINDOW, WINDOW_BACKTEST_TRIALS, WINDOW_WEIGHT_PRIOR,
    EXP_DECAY, BACKTEST_TRIALS, PERMUTATION_SHUFFLES, RECENT_SUM_SPAN_SHIFT, W_HOT_GLOBAL,
    W_HOT_POS, W_MISS_HIGH, W_MISS_MID, W_MARKOV, W_MARKOV2,
    MARKOV_MAX_SCORE, W_LAST_APPEAR, W_NEIGHBOR, W_ROAD_MATCH, W_DANMA_HIT,
    W_KILL_PENALTY, W_CONSECUTIVE, W_POS_REPEAT, W_RATIO_MATCH, RANDOM_POS_REPEAT,
    RANDOM_DIGIT_REUSE, SUM_SOFT_SIGMA, SPAN_SOFT_SIGMA, W_FORM_PRIOR, W_TRIPLET_POS,
    W_TRIPLET_GLOBAL, ZHXUAN_POS_TOPK, EXPLORATION_RATE, DANMA_TOP_POOL, DANMA_RANDOM_RATE,
    RECOMMEND_GROUPS, ZHIXUAN_TOP3, ZU6_POOL_SIZE, ZU6_FOUR_SIZE, RANDOM_NOISE,
    RECENT_WINDOW_REBOUND, REBOUND_BONUS, REBOUND_THRESHOLD, HOT_RATIO, WARM_RATIO,
    FEATURE_FLAGS, COLD_RATIO, HOT_WINDOW, SUM_TREND_WINDOW, SUM_TREND_ADJUST,
    MISS_CYCLE_WINDOW, MISS_OVER_RATIO_THRESHOLD, MISS_OVER_BONUS, PAIR_FREQ_WINDOWS, PAIR_HIGH_FREQ_THRESHOLD,
    PAIR_BONUS, SLOPE_MIN_CHAIN, SLOPE_MAX_CHAIN, W_SLOPE_MATCH, POS_NAMES_3D,
    FORM_SWITCH_WEIGHT, ZU6_STREAK_THRESHOLD, ZU3_STREAK_THRESHOLD, SUM_INTERVAL_WINDOW, SUM_INTERVAL_WIDTH,
    SUM_INTERVAL_BONUS, SUM_EXTREME_PENALTY, RECENT_RECOMMEND_WINDOW, RECENT_RECOMMEND_PENALTY, RECENT_RECOMMEND_CONSECUTIVE_PENALTY,
    ZU6_RECENT_WINDOW, ZU6_RECENT_PENALTY, ZU6_RECENT_DECAY, ZU3_PRESENCE_WINDOWS, ZU3_MIN_SAMPLES,
    ZU3_PAIRS_COUNT, ZU3_TIER_SIZES, ZU6_USE_KILL, ZU6_CANDIDATE_SIZE, ZU6_PRESENCE_WINDOWS,
    W_ZU6_HOT, W_ZU6_MISS, W_ZU6_POS, W_ZU6_PAIR, W_ZU6_BLEND,
    WINDOW_WEIGHTS_KV_KEY, PREDICTOR_VERSION, ML_MODEL_VERSION, MIN_DATA_PERIODS_FOR_ML_FUSION, ML_CACHE_MAX_AGE_SECONDS,
    ONLINE_PREDICTION_FILE, DIVERSITY_WEIGHT, SERVED_POOL_CANDIDATE_SIZE, CORRELATION_THRESHOLD, CORRELATION_PENALTY,
    FEATURE_EVAL_PERIOD, FEATURE_MIN_CONTRIBUTION, FEATURE_DOWNGRADE_FACTOR, MARKOV_LAPLACE_ALPHA, TUNABLE_WEIGHTS,
    WEIGHT_SEARCH_RANGES, default_weights, patch_weights,
)
from .fetching import (
    _fetch_data_internal, fetch_data,
)
from .features import (
    calc_span, miss_value, neighbor, road, exp_weighted_counts,
    build_markov, build_markov2, markov_prob_smoothed, gaussian_score, _recent_slice,
    odd_even_key, big_small_key, ratio_label, has_consecutive_digits, _slope_step,
    _detect_position_slope_chain, _cross_period_slope_signals, analyze_slope_patterns, slope_triplet_bonus, backtest_slope_patterns,
    entropy_model, rebound_model, classify_digits_by_hot, sum_trend_model, average_miss_cycle,
    miss_cycle_bonus, pair_frequency, high_freq_pairs, pair_bonus, form_switch_bonus,
    sum_interval_bonus, recent_recommend_penalty, max_digit_overlap, classify_form, FORM_LABELS,
    THEORY_FORM_P, form_miss, _form_recent_p,
)
from .scoring import (
    backtest_dan_kill, backtest_form_prediction, backtest_sum_span_interval, select_diverse_pool, position_repeat_count,
    _clamp, _empty_lag1, analyze_lag1_dynamics, ensemble_lag1_dynamics, derive_dynamic_weights,
    analyze_patterns, ensemble_patterns, analyze_sum_span, ensemble_sum_span, digit_scores,
    ensemble_digit_scores, zu6_digit_scores, position_digit_scores, ensemble_position_digit_scores, _window_weights_cache,
    _window_weights_cache_time, _window_weights_cache_numbers_hash, default_window_weights, load_persisted_window_weights, save_persisted_window_weights,
    refresh_persisted_window_weights, resolve_window_weights, compute_window_weights, analyze_form_probability, recommend_form_bet,
    zu3_digit_presence, zu3_pair_scores, zu3_combos_from_pair, zu3_zu_notes_from_pair, pick_zu3_pairs,
    zu3_coverage_tiers, pick_dan_tuo_kill, pick_zu6_four, zu6_notes_from_digits, TICKET_PRICE,
    build_zu6_coverage_tiers, build_zu6_primary, evaluate_zu6_pool_recent, _zu6_four_payload, _zu6_four_balance_score,
    build_zu6_four_variants, _effective_digit_score, pick_zu6_pool, _blend_dan_score,
    _triplet_digit_base, triplet_weight, triplet_weight_detail, build_detail_list, select_danma,
    _position_constrained_pool, _merge_rank_pools, rank_triplets, _meta_from_raw, build_ranking_meta,
    evaluate_strategy_admission,
)
from .fusion import (
    fuse_rule_ml, load_recent_rule_performance, load_latest_ml_performance, is_ml_eligible_from_backtest, save_strategy_records,
    settle_strategy_records, generate_strategy_recommendations, select_strategy_mode, recommend_budget_level, auto_recommend_count,
)
from .records import (
    load_recent_3d_recommendations, recent_zu6_digit_penalty, load_recent_zu6_four, save_recent_zu6_four, save_recent_3d_recommendations,
    load_online_predictions, save_online_prediction, settle_prediction, settle_pending_online_predictions, calculate_online_stats,
    recommendation_stability, get_stability_level, adjust_exploration_rate,
)
from .backtest import (
    backtest, random_baseline_backtest, permutation_test, backtest_objective, evaluate_weights,
    _sample_random_weights, _mutate_weights, search_weights, print_search_report,
)
from .prediction import (
    _prediction_cache, _cache_time, _is_today_cache, clear_cache, _transition_for_api,
    assess_data_quality, is_ml_prediction_cache_valid, run_prediction, print_report, main,
)
