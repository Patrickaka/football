#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大乐透分析器 - 融合排名模型、特征贡献度、动态权重调整、周期识别、多模型集成投票
============================================================================

大乐透规则：
- 前区：从01-35中选择5个号码
- 后区：从01-12中选择2个号码

包结构（薄 facade，业务逻辑在子模块）：
- config     常量配置、特征权重、随机基准
- analyzer   LotteryAnalyzer 主分析器
- fusion     规则+ML融合预测
- records    线上预测记录与奖级结算
- prediction run_prediction 预测入口
- ml         机器学习预测器
"""

from .config import (
    BACK_FEATURE_WEIGHTS, BACK_NUMBERS, FEATURE_BACKTEST_TRIALS,
    FEATURE_WEIGHTS, FRONT_NUMBERS, FULL_HISTORY_FETCH_COUNT,
    GAP_TIGHTEN_FACTOR, LOTTERY_PREDICTOR_VERSION, MARKOV2_WEIGHT,
    MAX_CONSECUTIVE_IN_RECOMMEND, MIN_FULL_HISTORY_ISSUES,
    MIN_REAL_HISTORY_FOR_RANKING, ML_BACKTEST_TRIALS, ODD_PARITY_TOLERANCE,
    RANDOM_BASELINE, ROLLING_BACKTEST_TRIALS, SIZE_BALANCE_RANGE,
    TIME_DECAY_FACTOR, ZONE_COVERAGE_MIN,
)
from .analyzer import LotteryAnalyzer, get_lottery_analyzer
from .fusion import compute_fusion_weights, fuse_rule_ml
from .records import (
    DALETOU_PREDICTIONS_KEY, _next_issue, calculate_online_stats,
    dlt_prize_tier, load_online_predictions, save_online_prediction,
    settle_predictions,
)
from .prediction import (
    _is_today_cache, _needs_full_history_bootstrap, clear_cache,
    run_prediction,
)
