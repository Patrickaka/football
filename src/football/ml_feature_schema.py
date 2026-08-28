#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML特征字段定义
==============

功能：
1. 定义所有ML特征的字段名、类型、默认值
2. 特征版本管理
3. 特征验证和标准化
"""

from typing import Dict, List, Any

from ..domain.sports.football import ml_contract as _mlc

# 纯计算转发给领域层
get_feature_names = _mlc.get_feature_names
get_feature_defaults = _mlc.get_feature_defaults
validate_features = _mlc.validate_features
audit_feature_payload = _mlc.audit_feature_payload
get_feature_description = _mlc.get_feature_description


# ==================== 特征版本 ====================
FEATURE_VERSION = "v2"


# ==================== 特征字段定义 ====================

# ELO特征
ELO_FEATURES = [
    {'name': 'elo_home', 'type': 'float', 'default': 1500.0, 'description': '主队ELO评分'},
    {'name': 'elo_away', 'type': 'float', 'default': 1500.0, 'description': '客队ELO评分'},
    {'name': 'elo_diff', 'type': 'float', 'default': 0.0, 'description': 'ELO差值'},
]

# 近期状态特征（5场）
RECENT_FORM_5_FEATURES = [
    {'name': 'home_attack_5', 'type': 'float', 'default': 1.5, 'description': '主队最近5场进攻效率'},
    {'name': 'home_defense_5', 'type': 'float', 'default': 1.5, 'description': '主队最近5场防守效率'},
    {'name': 'away_attack_5', 'type': 'float', 'default': 1.5, 'description': '客队最近5场进攻效率'},
    {'name': 'away_defense_5', 'type': 'float', 'default': 1.5, 'description': '客队最近5场防守效率'},
    {'name': 'home_form_points_5', 'type': 'float', 'default': 1.5, 'description': '主队最近5场场均积分'},
    {'name': 'away_form_points_5', 'type': 'float', 'default': 1.5, 'description': '客队最近5场场均积分'},
    {'name': 'home_win_rate_5', 'type': 'float', 'default': 0.33, 'description': '主队最近5场胜率'},
    {'name': 'away_win_rate_5', 'type': 'float', 'default': 0.33, 'description': '客队最近5场胜率'},
    {'name': 'home_draw_rate_5', 'type': 'float', 'default': 0.33, 'description': '主队最近5场平局率'},
    {'name': 'away_draw_rate_5', 'type': 'float', 'default': 0.33, 'description': '客队最近5场平局率'},
]

# 近期状态特征（10场）
RECENT_FORM_10_FEATURES = [
    {'name': 'home_win_rate_10', 'type': 'float', 'default': 0.33, 'description': '主队最近10场胜率'},
    {'name': 'away_win_rate_10', 'type': 'float', 'default': 0.33, 'description': '客队最近10场胜率'},
    {'name': 'home_goals_for_10', 'type': 'float', 'default': 1.5, 'description': '主队最近10场场均进球'},
    {'name': 'home_goals_against_10', 'type': 'float', 'default': 1.5, 'description': '主队最近10场场均失球'},
    {'name': 'away_goals_for_10', 'type': 'float', 'default': 1.5, 'description': '客队最近10场场均进球'},
    {'name': 'away_goals_against_10', 'type': 'float', 'default': 1.5, 'description': '客队最近10场场均失球'},
]

# 主客场拆分特征
HOME_AWAY_FEATURES = [
    {'name': 'home_h_goals_for_5', 'type': 'float', 'default': 1.5, 'description': '主队最近5个主场进球/场'},
    {'name': 'home_h_goals_against_5', 'type': 'float', 'default': 1.5, 'description': '主队最近5个主场失球/场'},
    {'name': 'away_a_goals_for_5', 'type': 'float', 'default': 1.5, 'description': '客队最近5个客场进球/场'},
    {'name': 'away_a_goals_against_5', 'type': 'float', 'default': 1.5, 'description': '客队最近5个客场失球/场'},
]

# 样本量特征
SAMPLE_COUNT_FEATURES = [
    {'name': 'home_matches_count', 'type': 'int', 'default': 0, 'description': '主队可用历史比赛数'},
    {'name': 'away_matches_count', 'type': 'int', 'default': 0, 'description': '客队可用历史比赛数'},
]

# 欧赔特征
EURO_FEATURES = [
    {'name': 'euro_home_prob', 'type': 'float', 'default': 0.333, 'description': '欧赔去水后主胜概率'},
    {'name': 'euro_draw_prob', 'type': 'float', 'default': 0.333, 'description': '欧赔去水后平局概率'},
    {'name': 'euro_away_prob', 'type': 'float', 'default': 0.334, 'description': '欧赔去水后客胜概率'},
]

# 亚盘特征
ASIAN_FEATURES = [
    {'name': 'asian_handicap', 'type': 'float', 'default': 0.0, 'description': '亚盘让球（主队视角）'},
    {'name': 'asian_home_prob', 'type': 'float', 'default': 0.5, 'description': '亚盘去水后主队概率'},
    {'name': 'asian_away_prob', 'type': 'float', 'default': 0.5, 'description': '亚盘去水后客队概率'},
]

# 大小球特征
TOTAL_FEATURES = [
    {'name': 'total_line', 'type': 'float', 'default': 2.5, 'description': '大小球盘口线'},
    {'name': 'over_prob', 'type': 'float', 'default': 0.5, 'description': '大球概率'},
    {'name': 'under_prob', 'type': 'float', 'default': 0.5, 'description': '小球概率'},
]

# 联赛特征
LEAGUE_FEATURES = [
    {'name': 'league_avg_goals_100', 'type': 'float', 'default': 2.7, 'description': '联赛最近100场平均进球'},
    {'name': 'league_draw_rate_100', 'type': 'float', 'default': 0.26, 'description': '联赛最近100场平局率'},
    {'name': 'is_home_favorite', 'type': 'int', 'default': 1, 'description': '是否主队热门（1是0否）'},
]

# 缺失标记特征
MISSING_FLAG_FEATURES = [
    {'name': 'has_asian_odds', 'type': 'int', 'default': 1, 'description': '是否有亚盘数据'},
    {'name': 'has_total_odds', 'type': 'int', 'default': 1, 'description': '是否有大小球数据'},
    {'name': 'has_euro_odds', 'type': 'int', 'default': 1, 'description': '是否有欧赔数据'},
]


# ==================== 特征合并 ====================

ALL_FEATURES = (
    ELO_FEATURES +
    RECENT_FORM_5_FEATURES +
    RECENT_FORM_10_FEATURES +
    HOME_AWAY_FEATURES +
    SAMPLE_COUNT_FEATURES +
    EURO_FEATURES +
    ASIAN_FEATURES +
    TOTAL_FEATURES +
    LEAGUE_FEATURES +
    MISSING_FLAG_FEATURES
)








