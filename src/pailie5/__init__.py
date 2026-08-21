#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
排列五号码分析模块 V2.0（准确率优化版）

包结构（薄 facade，业务逻辑在子模块）：
- config     配置常量、评分权重、功能开关
- caching    预测/回测磁盘缓存
- features   频率/遗漏/马尔可夫/多窗口评分与持久化
- pool       推荐池生成与多样性选池
- analyzer   Pailie5Analyzer 主分析器
- prediction run_prediction 预测入口
"""

from .config import (
    DATA_FILE, EXP_DECAY, FEATURE_FLAGS, HISTORY_URL, NUMBERS,
    RECENT_WINDOW, RECENT_WINDOWS, RECOMMEND_GROUPS,
)
from .caching import _is_today_cache
from .features import (
    analyze_ratio_pattern, apply_recent_recommend_penalty,
    average_miss_cycle_5, big_small_key_5, build_markov2_pos,
    build_markov_pos, default_window_weights, digit_scores_single_window,
    ensemble_digit_scores_multi_window, ensemble_sum_span_5,
    exp_weighted_counts, gaussian_score, has_consecutive_digits_5,
    load_recent_recommend, load_window_weights, markov_prob_smoothed,
    miss_value_pos, odd_even_key_5, pick_dan_kill, save_recent_recommend,
    save_window_weights,
)
from .pool import (
    backtest_window_weights, generate_pool, triplet_score_5,
)
from .analyzer import Pailie5Analyzer, get_pailie5_analyzer
from .prediction import clear_cache, run_prediction
