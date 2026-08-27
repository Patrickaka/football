# -*- coding: utf-8 -*-
"""福彩3D规则+ML融合与策略推荐"""

import json
import math
import os
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from contextlib import contextmanager
from itertools import combinations, product

from ..common.logger import setup_logger
from ..common.data_cache import cached_fetch
from ..common import kv_store

log = setup_logger('lottery3d')

from .config import (
    ML_MODEL_VERSION, RECOMMEND_GROUPS,
)
from .scoring import (
    triplet_weight_detail,
)

def load_recent_rule_performance():
    """加载缓存的规则模型最近表现（避免每次都跑回测）
    
    返回：
        dict: 包含 top30_rate, top3_rate, top100_rate, actual_rank_avg 等指标
    """
    try:
        perf = kv_store.load('lottery3d_rule_performance', {})
        return perf
    except Exception as e:
        log.error(f"加载规则模型表现失败: {e}")
        return {
            "top30_rate": 0.03,
            "top3_rate": 0.003,
            "top100_rate": 0.1,
            "actual_rank_avg": 500,
            "actual_rank_median": 500,
        }


def load_latest_ml_performance():
    """加载最近一次ML回测表现（用于融合权重计算）
    
    返回：
        包含top30_rate、top3_rate、actual_rank_avg等指标的字典
    """
    try:
        history = kv_store.load('lottery3d_ml_backtest_history', [])
        return history[-1] if history else {}
    except Exception as e:
        log.error(f"加载ML回测表现失败: {e}")
        return {}


def is_ml_eligible_from_backtest(period):
    """基于已保存的滚动回测结果判断ML是否符合准入条件
    
    准入条件：
        1. 存在最近的ML回测记录
        2. 回测记录未过期（模型版本、训练窗口、期号校验）
        3. Top30命中率高于随机基准(3%)
        4. 平均真实排名优于500
    
    参数：
        period: 当前期号
    
    返回：
        eligible: 是否符合准入条件
    """
    try:
        # 读取ML回测历史记录
        ml_backtest = kv_store.load('lottery3d_ml_backtest_history', [])
        
        if not ml_backtest:
            return False
        
        # 检查最近的回测结果
        recent = ml_backtest[-1] if ml_backtest else None
        if not recent:
            return False
        
        # 版本和期号校验
        record_period = recent.get('base_period')
        model_version = recent.get('model_version')
        
        # 检查回测记录是否过期（期号差异超过20期认为过期）
        if record_period and period:
            try:
                period_diff = abs(int(period) - int(record_period))
                if period_diff > 20:
                    log.info(f"ML回测记录过期（期号差异: {period_diff}期）")
                    return False
            except:
                pass
        
        # 回测写入端和准入端必须使用同一版本；此前 v7 记录被 v6 硬编码全部拒绝。
        if model_version != ML_MODEL_VERSION:
            log.info(
                f"ML模型版本不匹配（记录: {model_version}, 当前: {ML_MODEL_VERSION}）"
            )
            return False
        
        # 检查命中率是否高于基准
        top30_rate = recent.get('top30_rate', 0.0)
        actual_rank_avg = recent.get('actual_rank_avg', 1000)
        
        baseline_rate = RECOMMEND_GROUPS / 1000.0  # 3%
        
        # 准入条件：Top30命中率高于基准，且平均排名优于500
        if top30_rate > baseline_rate and actual_rank_avg < 500:
            return True
        
        return False
    except Exception as e:
        log.error(f"检查ML准入条件失败: {e}")
        return False



# ─── 领域层适配 ───
#
# 融合公式、三套策略、模式/预算/注数、策略记录的结算都在
# `domain/numeric/lottery3d/fusion.py`。**这里留着全部副作用**：
# kv 读写、时间戳、日志。

from src.domain.numeric.lottery3d import fusion as _fusion

STRATEGY_RECORDS_KEY = 'lottery3d_strategy_records'
# 策略记录只留最近这么多期。三路对比要的是趋势，不是全量档案。
STRATEGY_HISTORY_LIMIT = 200

select_strategy_mode = _fusion.select_mode
auto_recommend_count = _fusion.recommend_count


def fuse_rule_ml(rule_list, ml_list, top_n=30, rule_weight=0.55, ml_weight=0.45,
                 score=None, danma=None, kill=None, meta=None):
    """融合两份推荐。`score`/`danma`/`kill`/`meta` 只用来给 ML 独有的号码
    补一份得分拆解——四个都给齐了才补，缺一个就留 `None`，不编。"""
    can_build = all(x is not None for x in (score, danma, kill, meta))
    detail_for = None
    if can_build:
        def detail_for(num):
            return triplet_weight_detail(int(num[0]), int(num[1]), int(num[2]),
                                         score, danma, kill, meta)
    return _fusion.fuse(rule_list, ml_list, top_n, rule_weight, ml_weight, detail_for)


def generate_strategy_recommendations(rule_list, ml_list):
    """**签名比迁移前少了 `danma` 与 `kill`**：函数体从没用过它们。"""
    return _fusion.strategy_recommendations(rule_list, ml_list)


def recommend_budget_level(model_lift, recent_online_rate):
    """**签名比迁移前少了 `stability`**：函数体从没读过它，而 docstring
    专门列着它。"""
    return _fusion.budget_level(model_lift, recent_online_rate)


def save_strategy_records(period, rule_only, ml_only, fused):
    """保存三套策略记录。

    **首次发布的记录不覆盖**，已结算的更不覆盖：那是当时真的发出去的推荐，
    改了它等于篡改三路对比的原始数据。
    """
    try:
        history = kv_store.load(STRATEGY_RECORDS_KEY, [])
        existing = _fusion.find_by_period(history, period)
        if existing is not None:
            reason = '已结算' if existing.get('settled') else '已存在'
            log.info(f"策略记录{reason}，保留首次发布（期号: {period}）")
            return

        history.append(_fusion.new_strategy_record(
            period, rule_only, ml_only, fused,
            time.strftime('%Y-%m-%d %H:%M:%S')))
        kv_store.save(STRATEGY_RECORDS_KEY, history[-STRATEGY_HISTORY_LIMIT:])
        log.info(f"策略记录已保存（期号: {period}）")
    except Exception as e:
        log.error(f"保存策略记录失败: {e}")


def settle_strategy_records(periods, numbers):
    """按开奖结算三路策略记录。"""
    try:
        history = kv_store.load(STRATEGY_RECORDS_KEY, [])
        if not history:
            return
        if _fusion.settle_history(history, periods, numbers):
            kv_store.save(STRATEGY_RECORDS_KEY, history)
            log.info('三路策略记录已结算')
    except Exception as e:
        log.error(f"结算策略记录失败: {e}")
