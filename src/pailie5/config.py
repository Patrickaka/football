#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""排列五模块配置：数据源、评分权重、功能开关、推荐参数"""

from ..common.paths import data_path

DATA_FILE = data_path('pailie5_history.json')
HISTORY_URL = 'https://www.8300.cn/kjhhis/5/200.html'
NUMBERS = list(range(0, 10))  # 0-9

# 多窗口配置（参考3D）
RECENT_WINDOWS = (30, 60, 90, 150)
RECENT_WINDOW = 150  # 展示用最大窗口

# 指数衰减系数（越近期权重越高）
EXP_DECAY = 0.97

# ==================== 评分权重 ====================
# 全局热号权重
W_HOT_GLOBAL = 2.5
# 位置热号权重
W_HOT_POS = 3.0
# 遗漏奖励（高遗漏号码）
W_MISS_HIGH = 1.5   # 遗漏超过平均2倍
W_MISS_MID = 0.8    # 遗漏超过平均1.5倍
# 马尔可夫权重
W_MARKOV = 4.5
W_MARKOV2 = 1.2
MARKOV_MAX_SCORE = 5.0   # 马尔可夫得分上限
MARKOV_LAPLACE_ALPHA = 1.0  # 拉普拉斯平滑系数
# 和值/跨度软约束
SUM_SOFT_SIGMA = 4.0
SPAN_SOFT_SIGMA = 1.8
# 上期同号奖励
W_LAST_APPEAR = 1.5
# 连号奖励
W_CONSECUTIVE = 1.2
# 位置级数字评分奖励（位置热号超出全局热号的部分）
W_POS_SPECIFIC = 1.5
# 重复数字惩罚（避免推荐池过度集中于单个数字）
W_REPEAT = 2.0
# 不重复数字奖励（鼓励数字多样性）
W_DISTINCT = 11.0
W_POS_REPEAT_PENALTY = 1.0
# 奇偶比/大小比匹配奖励
W_RATIO_MATCH = 1.5
# 杀码惩罚（软约束）
W_KILL_PENALTY = 5.0
# 胆码奖励
W_DANMA_HIT = 3.5

# ==================== 功能开关 ====================
FEATURE_FLAGS = {
    "hot": True,           # 热号得分
    "miss": True,          # 遗漏加分
    "markov": True,        # 马尔可夫转移
    "sum_span": True,      # 和值跨度
    "consecutive": True,   # 连号奖励
    "lag1_repeat": True,   # 上期同位重复
    "ratio": True,         # 奇偶比/大小比
}

# ==================== 推荐配置 ====================
RECOMMEND_GROUPS = 30      # 推荐注数

# 推荐去重配置
RECENT_RECOMMEND_WINDOW = 5          # 最近N期推荐历史
RECENT_RECOMMEND_PENALTY = 2.0       # 重复推荐惩罚
RECENT_RECOMMEND_CONSECUTIVE_PENALTY = 4.0  # 连续推荐额外惩罚

# 多样性配置
DIVERSITY_WEIGHT = 0.5
CORRELATION_THRESHOLD = 3   # 数字重合阈值（5位号码，重合3个才算高相关）
CORRELATION_PENALTY = 1.0
COVERAGE_WEIGHT = 3.0       # 覆盖未选数字的奖励权重

# KV 存储键
RECENT_RECOMMEND_KV_KEY = 'pailie5_recent_recommend'
WINDOW_WEIGHTS_KV_KEY = 'pailie5_window_weights'
