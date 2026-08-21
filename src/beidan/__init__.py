# -*- coding: utf-8 -*-
"""
北单（北京单场）预测模块

包结构（薄 facade，业务逻辑在子模块）：
- config      常量、请求头、okooo 会话与 WAF 状态
- modeling    概率建模（泊松/DC矩阵/λ推导/盘口解析）
- fetching    数据与赛程抓取
- markets     市场分析（亚盘/大小球/比分/进球趋势）
- schedules   玩法赔率抓取（比分/总进球/半全场）
- settling    赛果提取、历史校准与历史记录
- quality     推荐质量评估
- upset       爆冷识别
- recommending 玩法分析与推荐生成、快照、CLI
"""

from .config import (
    BEIDAN_VERSION, BEIDAN_HISTORY_KEY, BEIDAN_HISTORY_LIMIT, BASE_URL, SCHEDULE_URL,
    DC_SCHEDULE_URL, MATCH_DETAIL_URL, OKOOO_BASE, OKOOO_DANCHANG_URL, OKOOO_MATCH_URL,
    HEADERS, OKOOO_HEADERS, _okooo_session, _okooo_waf_blocked, _okooo_waf_blocked_time,
    _mark_okooo_waf_blocked, _is_okooo_waf_blocked, _init_okooo_session, BET_TYPES, MAX_GOALS,
    SCORE_SPLIT, LEAGUE_PROFILES,
)
from .modeling import (
    poisson_pmf, euro_implied_lambdas, calibrate_draw_probability, predict_scores_by_poisson, DC_RHO,
    OU_TOTAL_BLEND, TARGET_TOTAL_MIN, TARGET_TOTAL_MAX, FACTOR_MIN, FACTOR_MAX,
    STRENGTH_SPLIT, BEIDAN_STRONG_MIN_PROBABILITY, BEIDAN_STRONG_MIN_LEAD, BEIDAN_MEDIUM_MIN_PROBABILITY, BEIDAN_MEDIUM_MIN_LEAD,
    BEIDAN_HIGH_PRECISION_MIN_PROBABILITY, SCORE_OUTCOME_ANCHOR_STRENGTH, _to_euro_odds, _parse_total_line_value, _asian_line_parts,
    _asian_over_profit, implied_total_from_ou, build_dixon_coles_matrix, aggregate_goals_from_scores, anchor_score_outcomes,
    match_target_total, match_lambdas, parse_beidan_handicap, rqspf_probs_from_score_probs,
)
from .fetching import (
    fetch, fetch_json, fetch_okooo, fetch_okooo_schedule, fetch_okooo_asian_history,
    fetch_okooo_goals_history, fetch_okooo_cs_history, fetch_beidan_schedule, fetch_jczq_schedule, fetch_zqdc_schedule,
)
from .markets import (
    adjust_probs_by_asian, analyze_asian_trend, build_beidan_joint_market_state, apply_beidan_joint_market_state, build_water_market_prediction,
    build_beidan_market_admission, _beidan_market_snapshot, analyze_cs_trend, enhance_scores_with_cs, calculate_goals_factor,
    adjust_zjq_by_goals, analyze_goals_trend, calculate_asian_goal_factor, _latest_ou_market, _latest_ou_odds,
)
from .schedules import (
    fetch_beidan_bifen, fetch_beidan_zjq, fetch_beidan_bqc,
)
from .settling import (
    calculate_implied_probability, _actual_spf_from_record, _actual_zjq_from_record, _actual_bifen_from_record, _actual_rqspf_from_record,
    apply_beidan_history_calibration, _beidan_record_key, _load_beidan_history, _save_beidan_history,
)
from .quality import (
    assess_recommendation_quality, UPSET_MED_FAV_MAX, UPSET_MED_GAP_MAX, UPSET_MED_MASS_MIN, UPSET_HIGH_FAV_MAX,
    UPSET_HIGH_GAP_MAX, UPSET_HIGH_MASS_MIN, UPSET_CONFIDENT_FAV_MIN, UPSET_CONFIDENT_GAP_MIN,
)
from .upset import (
    _result_from_score, _fmt_score, assess_upset_risk, pick_upset_scores, _score_result_label,
    assess_score_consistency,
)
from .recommending import (
    build_zjq_group_recommendation, _compact_beidan_record, save_beidan_prediction_snapshot, summarize_beidan_history, _ouzhi_cache,
    fetch_ouzhi_odds, _clear_ouzhi_cache, build_beidan_match_analysis, analyze_spf, analyze_rqspf,
    build_beidan_total_goals_accuracy_gate, analyze_bifen, analyze_zjq, analyze_bqc, _candidate_beidan_dates,
    _fetch_beidan_matches_with_fallback, generate_beidan_recommendations, find_value_bets, print_recommendations, main,
)
