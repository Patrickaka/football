# -*- coding: utf-8 -*-
"""
快乐8预测模块

包结构（薄 facade，业务逻辑在子模块）：
- config     常量配置、玩法定义、策略状态
- strategies 策略可用性判定与解析
- stats      超几何分布/显著性校正/验证评分
- candidates 候选池生成
- records    记录读写、快照结算读取、策略健康
- analyzer   KL8Analyzer 主分析器
- snapshots  run_prediction 入口、缓存、策略激活
- backtest   KL8RollingBacktest 滚动回测
- validation 策略验证编排
- fetch/scheduler/diagnostics 数据抓取与定时任务（原有独立模块）
"""

from .config import (
    KL8_PREDICTOR_VERSION, VERIFY_ONLY_MODE, KL8_NUM_RANGE, KL8_DRAW_COUNT, KL8_DEFAULT_HISTORY,
    KL8_EXPECTED_GAP, KL8_MIN_PREDICTION_PERIODS, BACKTEST_MIN_OOS_PERIODS, BACKTEST_FINAL_TEST_PERIODS, BACKTEST_TRAIN_PERIODS,
    BACKTEST_TOTAL_REQUIRED_PERIODS, BACKTEST_PERMUTATION_COUNT, BACKTEST_STABILITY_WINDOWS, BACKTEST_STABILITY_THRESHOLD, SELECT_CONFIG,
    SELECT_TYPES, SELECT_PLAY_KEYS, FUSHI_CONFIG, FUSHI_PLAY_KEYS, MULTI_SLIP_CONFIG,
    FEATURE_CONFIG, MODEL_CONFIG, ACTIVE_STRATEGIES, REFERENCE_STRATEGY, VALIDATION_CANDIDATES,
    CANDIDATE_STRATEGIES, ABLATION_FEATURES, STRATEGY_TRIAL_RESULTS, KL8_SNAPSHOT_DIR, KL8_SETTLEMENT_DIR,
    KL8_RECALCULATION_DIR, KL8_PRIZE_TABLE_FILE, KL8_STRATEGY_TRIAL_FILE, KL8_ACTIVE_STRATEGIES_FILE, KL8_FINAL_TEST_REPORT_FILE,
    KL8_CONFLICT_QUEUE_FILE,
)
from .strategies import (
    is_prediction_ready, has_active_signal, _strategy_is_usable, resolve_play_strategy, get_active_feature_weights,
    get_active_model_weights, FEATURE_WEIGHTS, MODEL_WEIGHTS,
)
from .stats import (
    hypergeom_pmf, hypergeom_p_ge, hypergeom_expected, _parse_play_pick_n, _play_lift,
    _prize_tier_thresholds, _hit_rate_priority_thresholds, _hit_rate_priority_score, _practical_validation_score, _play_accuracy_profile,
    benjamini_hochberg_fdr, bonferroni_correction,
)
from .candidates import (
    _clean_pick_numbers, _default_repeat_cap, _adaptive_repeat_cap, _adaptive_repeat_target, _enforce_minimum_repeats,
    _diversify_candidate_pool, _zone_spread_candidate_pool, _prize_floor_candidate_pool, _high_tier_chase_candidate_pool, _shape_targets,
    _shape_profile, _shape_penalty, _shape_balanced_candidate_pool, _score_candidate_selection, _select_final_candidate_pool,
    generate_multi_slips, _simulate_multi_slip_coverage,
)
from .records import (
    normalize_record, _checksum_numbers, _compute_next_issue, load_prize_table, _strategy_fingerprint,
    _persist_trial_results, _load_trial_results, _persist_active_strategies, _load_active_strategies, _persist_final_test_report,
    loaded_strategies, save_conflict_to_queue, list_conflict_queue, check_data_integrity, _load_last_snapshot,
    _load_recent_settlements, _summarize_settlement_window, _build_recent_settlement_performance, _build_strategy_health, _compute_prediction_changes,
)
from .analyzer import (
    KL8Analyzer, _analyzer_instance, get_kl8_analyzer, build_candidate_pool,
)
from .snapshots import (
    activate_verified_strategy, _prediction_cache, run_prediction, clear_cache, list_prediction_snapshots,
    list_exclude_recalculations, _check_settlement_exists,
)
from .backtest import (
    KL8RollingBacktest, _brier_score, _log_loss, _calibration_curve_data,
)
from .validation import (
    validate_and_activate_strategy,
)
