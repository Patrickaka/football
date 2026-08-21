# -*- coding: utf-8 -*-
"""快乐8常量配置、玩法定义、策略状态"""

import math
import json
import time
import hashlib
import uuid
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple
from itertools import combinations
from pathlib import Path

from src.common.paths import data_path
from src.common.repositories import doc_store
from src.common.logger import setup_logger

log = setup_logger('kl8')


KL8_PREDICTOR_VERSION = "kl8-v10.10-select6-primary-accuracy"

# ─── v9.2: 只显示已验证策略模式 ───
# 正式策略优先；尚未通过验证时继续输出玩法专属的动态参考策略，供持续
# 结算和样本外观察。页面会明确标记“未验证参考”，不能当成稳定优势。
VERIFY_ONLY_MODE = False

# ─── 快乐8常量 ───
KL8_NUM_RANGE = 80       # 号码范围 1-80
KL8_DRAW_COUNT = 20      # 每期开出20个号码
KL8_DEFAULT_HISTORY = 250  # 默认使用最近250期
KL8_EXPECTED_GAP = (KL8_NUM_RANGE - KL8_DRAW_COUNT) / KL8_DRAW_COUNT  # = 3.0
KL8_MIN_PREDICTION_PERIODS = 50

# ─── 回测常量 ───
BACKTEST_MIN_OOS_PERIODS = 300   # 最小样本外期数（v9.2: 验证集300期）
BACKTEST_FINAL_TEST_PERIODS = 200  # 最终封存测试期数
BACKTEST_TRAIN_PERIODS = 300  # v9.2: 训练集固定300期
BACKTEST_TOTAL_REQUIRED_PERIODS = 800  # v9.2: 总共需要800期
BACKTEST_PERMUTATION_COUNT = 1000  # 置换检验次数
BACKTEST_STABILITY_WINDOWS = 4     # 稳定性检查窗口数
BACKTEST_STABILITY_THRESHOLD = 3   # 至少3/4窗口Lift>0

# ─── 选型配置：选3~选10各选多少号码 ───
SELECT_CONFIG = {
    3: {'pick': 3, 'top_n': 10,  'desc': '选3'},
    4: {'pick': 4, 'top_n': 12,  'desc': '选4'},
    5: {'pick': 5, 'top_n': 15,  'desc': '选5'},
    6: {'pick': 6, 'top_n': 15,  'desc': '选6'},
    7: {'pick': 7, 'top_n': 18,  'desc': '选7'},
    8: {'pick': 8, 'top_n': 20,  'desc': '选8'},
    9: {'pick': 9, 'top_n': 22,  'desc': '选9'},
    10: {'pick': 10, 'top_n': 24, 'desc': '选10'},
}
SELECT_TYPES = tuple(sorted(SELECT_CONFIG))
SELECT_PLAY_KEYS = tuple(f'select_{st}' for st in SELECT_TYPES)
FUSHI_CONFIG = {
    'fu_shi_4': {
        'desc': '选4复式7码',
        'base_pick': 4,
        'pool_size': 7,
        'numbers_field': 'top4_numbers',
        'scores_field': 'top4_scores',
        'pool_label': '7个核心号码',
        'prize_key': 'select_4',
    },
    'fu_shi_7': {
        'desc': '选5复式8码',
        'base_pick': 5,
        'pool_size': 8,
        'numbers_field': 'top8_numbers',
        'scores_field': 'top8_scores',
        'pool_label': '8个核心号码',
        'prize_key': 'fu_shi_7',
    },
    'fu_shi_10_11': {
        'desc': '选10复式11码',
        'base_pick': 10,
        'pool_size': 11,
        'numbers_field': 'top11_numbers',
        'scores_field': 'top11_scores',
        'pool_label': '11个核心号码',
        'prize_key': 'select_10',
    },
}
FUSHI_PLAY_KEYS = tuple(FUSHI_CONFIG)

# ─── 多注覆盖方案（v10：提升组合层面命中4+概率）───
# 原理：单注选6命中4约2.7%/期、5约0.3%/期（纯随机，模型无预测edge）。
# 通过生成多组、彼此差异化的选号集合（权重扰动+覆盖惩罚），让组合覆盖更多号码，
# 从而提升“至少一组命中4+”的概率。这是覆盖率杠杆，不改变单注期望命中（仍为1.5），
# 仅提升“组合层面”的命中率——即以更多注数换取更高的组合中奖概率。
# 多注覆盖配置。
# pick_size: 每组输出的号码个数(默认=玩法选号数)。对用户“选5复式”玩法，
#   每组当 6 码打选5复式=C(6,5)=6注。多组覆盖提升“组合中5”概率。
#   实测结论(见 backtest_kl8_coverage.py)：6码堆组数性价比(率/百元≈2.37)恒高于扩7码(≈1.50)；
#   成本随组数线性、命中率线性、性价比在8~16组达满档。这是覆盖面杠杆，不改变单注期望命中，
#   以更多注数换取更高组合中奖概率。v9.4 起由 8×6 最便宜档上调为 12×6 均衡档
#   (144元/期，组合中5约3.0%、中4+约23.5%)，进一步压低"全军覆没"频率。
MULTI_SLIP_CONFIG = {}  # v10.1: 仅输出单组号码，关闭多注覆盖。

# ─── 特征开关配置（v5：所有特征默认停用，需回测验证才能启用）───
# 按玩法分开评估: 每个特征可以有per-play-type的enabled状态
FEATURE_CONFIG = {
    'frequency':        {'enabled': True, 'weight': 0.2,   'desc': '频率偏离度(hot模式:热号加分)'},
    'gap':              {'enabled': False, 'weight': 0.0,   'desc': '遗漏偏离度 -- 仅展示指标,不参与预测'},
    'position_residual': {'enabled': True, 'weight': 0.2,   'desc': '区内残差(剔除全局频率后的区位偏移)'},
    'road_residual':    {'enabled': True, 'weight': 0.15,   'desc': '路内残差(剔除全局频率后的路数偏移)'},
    'sum':              {'enabled': False, 'weight': 0.0,   'desc': '和值特征 -- 停用'},
    'zone':             {'enabled': False, 'weight': 0.0,   'desc': '区位近期开出率 -- 停用'},
    'repeat':           {'enabled': False, 'weight': 0.0,   'desc': '重号特征(3个候选方向: neutral/avoid/follow)'},
    'adjacent':         {'enabled': True, 'weight': 0.15,   'desc': '邻号特征:相邻号码平均频率'},
    'odd_even':         {'enabled': True, 'weight': 0.1,   'desc': '奇偶特征:所在奇偶组频率偏离'},
    'big_small':        {'enabled': True, 'weight': 0.05,   'desc': '大小特征:所在大小组频率偏离'},
}

# ─── 投票模型权重（v6：停用，等策略注册表接管）───
FEATURE_CONFIG['seeded_random'] = {
    'enabled': False,
    'weight': 0.0,
    'desc': 'deterministic random baseline for validation controls',
}

MODEL_CONFIG = {
    'bayesian': {'enabled': False, 'weight': 0.0, 'desc': '停用: 倾向热号,与排名频率冷号方向相反'},
    'rank':     {'enabled': True, 'weight': 1.0, 'desc': '排名模型(使用hot模式)'},
    'markov':   {'enabled': False, 'weight': 0.0, 'desc': '停用: 低号码偏差,未出现号码全0.25'},
}

# ─── v6 策略注册表（按玩法分别配置，取代全局FEATURE_CONFIG的预测权重）───
# 每个玩法有独立的 strategy_id、feature_weights、model_weights、window_size
# 预测、快照、结算、回测都必须记录 strategy_id
# 当前所有策略默认空（无信号），需回测验证后才能填入具体配置
ACTIVE_STRATEGIES = {
    'select_3': {
        'strategy_id': '',           # 空=无验证策略
        'feature_weights': {},       # 空=不启用任何特征
        'model_weights': {},         # 空=不启用任何模型
        'window_size': 0,            # 0=无固定窗口
    },
    'select_4': {
        'strategy_id': '',
        'feature_weights': {},
        'model_weights': {},
        'window_size': 0,
    },
    'select_5': {
        'strategy_id': '',
        'feature_weights': {},
        'model_weights': {},
        'window_size': 0,
    },
    'select_6': {
        'strategy_id': '',
        'feature_weights': {},
        'model_weights': {},
        'window_size': 0,
    },
    'select_7': {
        'strategy_id': '',
        'feature_weights': {},
        'model_weights': {},
        'window_size': 0,
    },
    'select_8': {
        'strategy_id': '',
        'feature_weights': {},
        'model_weights': {},
        'window_size': 0,
    },
    'select_9': {
        'strategy_id': '',
        'feature_weights': {},
        'model_weights': {},
        'window_size': 0,
    },
    'select_10': {
        'strategy_id': '',
        'feature_weights': {},
        'model_weights': {},
        'window_size': 0,
    },
    'fu_shi_4': {
        'strategy_id': '',
        'feature_weights': {},
        'model_weights': {},
        'window_size': 0,
    },
    'fu_shi_7': {
        'strategy_id': '',
        'feature_weights': {},
        'model_weights': {},
        'window_size': 0,
    },
    'fu_shi_10_11': {
        'strategy_id': '',
        'feature_weights': {},
        'model_weights': {},
        'window_size': 0,
    },
}

# ─── 默认参考策略（v7.1新增）───
# 回测未通过时，自动降级到此策略：基础统计排序，但明确标记为"未验证参考"
from copy import deepcopy

REFERENCE_STRATEGY = {
    'strategy_id': 'reference_heuristic_v2',
    'feature_weights': {
        'frequency': 0.45,
        'gap': 0.20,
        'trend': 0.20,
        'pair_cooccurrence': 0.10,
        'position_residual': 0.05,
        'position_residual_cross': 0.0,
        'road_residual': 0.0,
        'repeat': 0.0,
        'odd_even': 0.0,
        'big_small': 0.0,
    },
    'model_weights': {
        'rank': 1.0,
        'bayesian': 0.0,
        'markov': 0.0,
    },
    'window_size': 100,
    'prediction_mode': 'reference_unvalidated',
    'is_validated': False,
}

# ─── v9.2: 固定验证候选策略池 ───
# 不允许临时无限加参数，避免"碰巧回测好看"
# 800期时不加500窗口（训练段只有300期，500窗口在早期回测用不满）
# 验证候选和日常候选分开：VALIDATION_CANDIDATES 只用于锦标赛

VALIDATION_CANDIDATES = {
    'freq_50': {
        'strategy_id': 'candidate_freq_50',
        'feature_weights': {'frequency': 1.0, 'position_residual': 0.0, 'road_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 50,
        'repeat_direction': 'neutral',
    },
    'freq_100': {
        'strategy_id': 'candidate_freq_100',
        'feature_weights': {'frequency': 1.0, 'position_residual': 0.0, 'road_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 100,
        'repeat_direction': 'neutral',
    },
    'freq_150': {
        'strategy_id': 'candidate_freq_150',
        'feature_weights': {'frequency': 1.0, 'position_residual': 0.0, 'road_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 150,
        'repeat_direction': 'neutral',
    },
    'freq_250': {
        'strategy_id': 'candidate_freq_250',
        'feature_weights': {'frequency': 1.0, 'position_residual': 0.0, 'road_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 250,
        'repeat_direction': 'neutral',
    },
    'repeat_follow_100': {
        'strategy_id': 'candidate_repeat_follow_100',
        'feature_weights': {'frequency': 0.60, 'position_residual': 0.0, 'road_residual': 0.0, 'repeat': 0.15, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 100,
        'repeat_direction': 'follow',
        'repeat_follow_score': 0.90,
        'repeat_non_follow_score': 0.50,
        'pool_max_last_numbers': 7,
    },
    'position_100': {
        'strategy_id': 'candidate_position_100',
        'feature_weights': {'frequency': 0.70, 'position_residual': 0.30, 'road_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 100,
        'repeat_direction': 'neutral',
    },
    'road_100': {
        'strategy_id': 'candidate_road_100',
        'feature_weights': {'frequency': 0.75, 'road_residual': 0.25, 'position_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 100,
        'repeat_direction': 'neutral',
    },
}

# Broaden the fixed validation slate without changing activation gates.
# These variants let the tournament test gap, mixed features, repeat-follow
# windows and candidate-pool post-processing instead of relying on one shape.
VALIDATION_CANDIDATES.update({
    'gap_50': {
        'strategy_id': 'candidate_gap_50',
        'feature_weights': {'frequency': 0.0, 'gap': 1.0, 'position_residual': 0.0, 'road_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 50,
        'repeat_direction': 'neutral',
    },
    'gap_100': {
        'strategy_id': 'candidate_gap_100',
        'feature_weights': {'frequency': 0.0, 'gap': 1.0, 'position_residual': 0.0, 'road_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 100,
        'repeat_direction': 'neutral',
    },
    'freq_gap_75': {
        'strategy_id': 'candidate_freq_gap_75',
        'feature_weights': {'frequency': 0.55, 'gap': 0.45, 'position_residual': 0.0, 'road_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 75,
        'repeat_direction': 'neutral',
    },
    'freq_gap_150': {
        'strategy_id': 'candidate_freq_gap_150',
        'feature_weights': {'frequency': 0.55, 'gap': 0.45, 'position_residual': 0.0, 'road_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 150,
        'repeat_direction': 'neutral',
    },
    'repeat_follow_50': {
        'strategy_id': 'candidate_repeat_follow_50',
        'feature_weights': {'frequency': 0.60, 'gap': 0.0, 'position_residual': 0.0, 'road_residual': 0.0, 'repeat': 0.15, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 50,
        'repeat_direction': 'follow',
        'repeat_follow_score': 0.90,
        'repeat_non_follow_score': 0.50,
    },
    'repeat_follow_150': {
        'strategy_id': 'candidate_repeat_follow_150',
        'feature_weights': {'frequency': 0.60, 'gap': 0.0, 'position_residual': 0.0, 'road_residual': 0.0, 'repeat': 0.15, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 150,
        'repeat_direction': 'follow',
        'repeat_follow_score': 0.90,
        'repeat_non_follow_score': 0.50,
    },
    'position_repeat_follow_100': {
        'strategy_id': 'candidate_position_repeat_follow_100',
        'feature_weights': {'frequency': 0.50, 'gap': 0.0, 'position_residual': 0.25, 'road_residual': 0.0, 'repeat': 0.15, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 100,
        'repeat_direction': 'follow',
        'repeat_follow_score': 0.90,
        'repeat_non_follow_score': 0.50,
    },
    'road_repeat_follow_100': {
        'strategy_id': 'candidate_road_repeat_follow_100',
        'feature_weights': {'frequency': 0.50, 'gap': 0.0, 'position_residual': 0.0, 'road_residual': 0.25, 'repeat': 0.15, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 100,
        'repeat_direction': 'follow',
        'repeat_follow_score': 0.90,
        'repeat_non_follow_score': 0.50,
    },
    'repeat_follow_100_no_diversify': {
        'strategy_id': 'candidate_repeat_follow_100_no_diversify',
        'feature_weights': {'frequency': 0.60, 'gap': 0.0, 'position_residual': 0.0, 'road_residual': 0.0, 'repeat': 0.15, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 100,
        'repeat_direction': 'follow',
        'repeat_follow_score': 0.90,
        'repeat_non_follow_score': 0.50,
        'pool_diversify': False,
    },
    'repeat_follow_100_repeat_cap3': {
        'strategy_id': 'candidate_repeat_follow_100_repeat_cap3',
        'feature_weights': {'frequency': 0.60, 'gap': 0.0, 'position_residual': 0.0, 'road_residual': 0.0, 'repeat': 0.15, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 100,
        'repeat_direction': 'follow',
        'repeat_follow_score': 0.90,
        'repeat_non_follow_score': 0.50,
        'pool_max_last_numbers': 3,
    },
})

VALIDATION_CANDIDATES.update({
    'trend_100': {
        'strategy_id': 'candidate_trend_100',
        'feature_weights': {'frequency': 0.50, 'trend': 0.30, 'gap': 0.20, 'position_residual': 0.0, 'road_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 100,
        'repeat_direction': 'neutral',
    },
    'trend_follow_100': {
        'strategy_id': 'candidate_trend_follow_100',
        'feature_weights': {'frequency': 0.40, 'trend': 0.30, 'gap': 0.15, 'repeat': 0.15, 'position_residual': 0.0, 'road_residual': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 100,
        'repeat_direction': 'follow',
        'repeat_follow_score': 0.90,
        'repeat_non_follow_score': 0.50,
    },
    'trend_cross_100': {
        'strategy_id': 'candidate_trend_cross_100',
        'feature_weights': {'frequency': 0.40, 'trend': 0.25, 'position_residual': 0.15, 'position_residual_cross': 0.10, 'gap': 0.10, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 100,
        'repeat_direction': 'neutral',
    },
    'pair_cooc_100': {
        'strategy_id': 'candidate_pair_cooc_100',
        'feature_weights': {'frequency': 0.50, 'gap': 0.30, 'pair_cooccurrence': 0.20, 'trend': 0.0, 'position_residual': 0.0, 'road_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 100,
        'repeat_direction': 'neutral',
    },
    'trend_pair_100': {
        'strategy_id': 'candidate_trend_pair_100',
        'feature_weights': {'frequency': 0.40, 'trend': 0.25, 'gap': 0.15, 'pair_cooccurrence': 0.20, 'position_residual': 0.0, 'road_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 100,
        'repeat_direction': 'neutral',
    },
    'trend_follow_pair_100': {
        'strategy_id': 'candidate_trend_follow_pair_100',
        'feature_weights': {'frequency': 0.35, 'trend': 0.25, 'gap': 0.10, 'repeat': 0.10, 'pair_cooccurrence': 0.20, 'position_residual': 0.0, 'road_residual': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 100,
        'repeat_direction': 'follow',
        'repeat_follow_score': 0.90,
        'repeat_non_follow_score': 0.50,
    },
})

VALIDATION_CANDIDATES.update({
    'hot_adjacent_100': {
        'strategy_id': 'candidate_hot_adjacent_100',
        'feature_weights': {'frequency': 0.25, 'adjacent': 0.20, 'position_residual': 0.20, 'road_residual': 0.15, 'odd_even': 0.10, 'big_small': 0.05, 'repeat': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 100,
        'repeat_direction': 'neutral',
        'frequency_mode': 'hot',
    },
    'hot_adjacent_150': {
        'strategy_id': 'candidate_hot_adjacent_150',
        'feature_weights': {'frequency': 0.25, 'adjacent': 0.20, 'position_residual': 0.20, 'road_residual': 0.15, 'odd_even': 0.10, 'big_small': 0.05, 'repeat': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 150,
        'repeat_direction': 'neutral',
        'frequency_mode': 'hot',
    },
    'hot_full_100': {
        'strategy_id': 'candidate_hot_full_100',
        'feature_weights': {'frequency': 0.20, 'adjacent': 0.15, 'position_residual': 0.20, 'road_residual': 0.15, 'odd_even': 0.10, 'big_small': 0.05, 'trend': 0.10, 'pair_cooccurrence': 0.05, 'repeat': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 100,
        'repeat_direction': 'neutral',
        'frequency_mode': 'hot',
    },
})

VALIDATION_CANDIDATES.update({
    'select5_balanced_75': {
        'strategy_id': 'candidate_select5_balanced_75',
        'feature_weights': {'frequency': 0.45, 'gap': 0.25, 'trend': 0.15, 'position_residual': 0.10, 'pair_cooccurrence': 0.05, 'repeat': 0.0, 'road_residual': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 75,
        'repeat_direction': 'neutral',
        'pool_max_last_numbers': 2,
        'final_selection_mode': 'best_variant',
    },
    'select5_low_repeat_100': {
        'strategy_id': 'candidate_select5_low_repeat_100',
        'feature_weights': {'frequency': 0.50, 'gap': 0.20, 'position_residual': 0.15, 'trend': 0.10, 'pair_cooccurrence': 0.05, 'repeat': 0.0, 'road_residual': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 100,
        'repeat_direction': 'neutral',
        'pool_max_last_numbers': 1,
        'final_selection_mode': 'low_repeat',
    },
    'select5_hot_zone_150': {
        'strategy_id': 'candidate_select5_hot_zone_150',
        'feature_weights': {'frequency': 0.35, 'adjacent': 0.20, 'position_residual': 0.20, 'road_residual': 0.10, 'trend': 0.10, 'pair_cooccurrence': 0.05, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 150,
        'repeat_direction': 'neutral',
        'frequency_mode': 'hot',
        'pool_max_last_numbers': 2,
        'final_selection_mode': 'zone_spread',
    },
    'select6_balanced_100': {
        'strategy_id': 'candidate_select6_balanced_100',
        'feature_weights': {'frequency': 0.40, 'gap': 0.20, 'trend': 0.15, 'pair_cooccurrence': 0.15, 'position_residual': 0.10, 'repeat': 0.0, 'road_residual': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 100,
        'repeat_direction': 'neutral',
        'pool_max_last_numbers': 3,
        'final_selection_mode': 'best_variant',
    },
    'select6_repeat_follow_75': {
        'strategy_id': 'candidate_select6_repeat_follow_75',
        'feature_weights': {'frequency': 0.45, 'gap': 0.15, 'trend': 0.15, 'pair_cooccurrence': 0.10, 'repeat': 0.15, 'position_residual': 0.0, 'road_residual': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 75,
        'repeat_direction': 'follow',
        'repeat_follow_score': 0.90,
        'repeat_non_follow_score': 0.50,
        'pool_max_last_numbers': 4,
        'final_selection_mode': 'repeat_follow',
    },
    'select6_hot_balanced_150': {
        'strategy_id': 'candidate_select6_hot_balanced_150',
        'feature_weights': {'frequency': 0.30, 'adjacent': 0.20, 'position_residual': 0.15, 'road_residual': 0.10, 'trend': 0.15, 'pair_cooccurrence': 0.10, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 150,
        'repeat_direction': 'neutral',
        'frequency_mode': 'hot',
        'pool_max_last_numbers': 3,
        'final_selection_mode': 'balanced',
    },
})

# Deterministic random and shape-only controls. These are not intended to be
# predictive; they make the validation slate compare heuristic signals against
# neutral pools that obey the same final-selection rules.
VALIDATION_CANDIDATES.update({
    'random_shape_50': {
        'strategy_id': 'candidate_random_shape_50',
        'feature_weights': {'seeded_random': 1.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 50,
        'repeat_direction': 'neutral',
        'frequency_mode': 'neutral',
        'pool_max_last_numbers': None,
        'final_selection_mode': 'shape_balanced',
    },
    'random_prize_floor_100': {
        'strategy_id': 'candidate_random_prize_floor_100',
        'feature_weights': {'seeded_random': 1.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 100,
        'repeat_direction': 'neutral',
        'frequency_mode': 'neutral',
        'pool_max_last_numbers': None,
        'final_selection_mode': 'prize_floor',
    },
    'random_low_repeat_100': {
        'strategy_id': 'candidate_random_low_repeat_100',
        'feature_weights': {'seeded_random': 1.0},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 100,
        'repeat_direction': 'neutral',
        'frequency_mode': 'neutral',
        'pool_max_last_numbers': 2,
        'final_selection_mode': 'low_repeat',
    },
})

# 跨期带出 + 重号筛选的预注册候选。窗口和重号上限固定为少量档位，
# 由walk-forward验证比较，不根据最终测试结果临时调参。
VALIDATION_CANDIDATES.update({
    'transition_repeat_75_cap2': {
        'strategy_id': 'candidate_transition_repeat_75_cap2',
        'feature_weights': {'frequency': 0.25, 'gap': 0.15, 'trend': 0.10, 'next_transition': 0.30, 'repeat': 0.05, 'position_residual': 0.10, 'pair_cooccurrence': 0.05},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 75,
        'repeat_direction': 'follow',
        'pool_max_last_numbers': 2,
        'final_selection_mode': 'shape_balanced',
    },
    'transition_repeat_100_cap3': {
        'strategy_id': 'candidate_transition_repeat_100_cap3',
        'feature_weights': {'frequency': 0.20, 'gap': 0.15, 'trend': 0.10, 'next_transition': 0.30, 'repeat': 0.05, 'position_residual': 0.10, 'position_residual_cross': 0.05, 'pair_cooccurrence': 0.05},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 100,
        'repeat_direction': 'follow',
        'pool_max_last_numbers': 3,
        'final_selection_mode': 'shape_balanced',
    },
    'transition_repeat_150_cap4': {
        'strategy_id': 'candidate_transition_repeat_150_cap4',
        'feature_weights': {'frequency': 0.20, 'gap': 0.10, 'trend': 0.10, 'next_transition': 0.35, 'repeat': 0.05, 'position_residual': 0.10, 'road_residual': 0.05, 'pair_cooccurrence': 0.05},
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 150,
        'repeat_direction': 'follow',
        'pool_max_last_numbers': 4,
        'final_selection_mode': 'shape_balanced',
    },
})

# v9.2: CANDIDATE_STRATEGIES 保留为 VALIDATION_CANDIDATES 的别名（向后兼容）
CANDIDATE_STRATEGIES = VALIDATION_CANDIDATES

# ─── v9: 独立消融试验表（不再依赖当前启用权重）───
# 消融回测不再只测 FEATURE_CONFIG 中 weight>0 的特征
# 而是使用这张独立试验表，确保残差、重号、路数策略都参与完整消融
ABLATION_FEATURES = {
    'frequency': 1.0,
    'position_residual': 1.0,
    'road_residual': 1.0,
    'repeat_follow': 0.15,
    'trend': 0.30,
    'position_residual_cross': 0.10,
    'pair_cooccurrence': 0.20,
    'next_transition': 0.30,
    'adjacent': 0.15,
    'odd_even': 0.10,
    'big_small': 0.05,
    'seeded_random': 1.0,
}

# ─── 策略试验结果记录表（v8新增）───
# 所有候选策略的回测结果统一记录于此，最终做全量FDR校正
# 格式: [{'strategy_id': ..., 'play_type': ..., 'raw_p_value': ..., 'validation_lift': ..., ...}]
STRATEGY_TRIAL_RESULTS = []


KL8_SNAPSHOT_DIR = data_path('kl8_snapshots')
KL8_SETTLEMENT_DIR = data_path('kl8_settlements')
KL8_RECALCULATION_DIR = data_path('kl8_recalculations')
KL8_PRIZE_TABLE_FILE = data_path('kl8_prize_table.json')

# ─── v9: 策略试验与激活持久化 ───
KL8_STRATEGY_TRIAL_FILE = data_path('kl8_strategy_trials.json')
KL8_ACTIVE_STRATEGIES_FILE = data_path('kl8_active_strategies.json')
KL8_FINAL_TEST_REPORT_FILE = data_path('kl8_final_test_report.json')

# ─── 冲突审核队列 ───
KL8_CONFLICT_QUEUE_FILE = data_path('kl8_conflict_queue.json')


# ─── 预测就绪检查（v6: 基于ACTIVE_STRATEGIES判断）───

