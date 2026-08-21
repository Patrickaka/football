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

def fuse_rule_ml(rule_list, ml_list, top_n=30, rule_weight=0.55, ml_weight=0.45, score=None, danma=None, kill=None, meta=None):
    """融合规则模型和ML模型的推荐结果（支持动态权重）
    
    参数：
        rule_list: 规则模型推荐列表 [{"num": "...", "score": ..., "detail": {...}}, ...]
        ml_list: ML模型推荐列表 [{"num": "...", "model_score": ...}, ...]
        top_n: 最终推荐数量
        rule_weight: 规则模型权重（基于回测表现动态计算）
        ml_weight: ML模型权重（基于回测表现动态计算）
        score: 数字评分数组（用于构建detail）
        danma: 胆码列表
        kill: 杀码列表
        meta: 元数据
    
    返回：
        fused: 融合后的推荐列表，包含置信度标签和detail
    """
    rule_rank = {x["num"]: i for i, x in enumerate(rule_list)}
    ml_rank = {x["num"]: i for i, x in enumerate(ml_list)}
    
    # 保留规则模型的detail映射
    rule_detail = {x["num"]: x.get("detail") for x in rule_list}

    all_nums = set(rule_rank) | set(ml_rank)

    fused = []
    for num in all_nums:
        r = rule_rank.get(num, 999)
        m = ml_rank.get(num, 999)

        fuse_score = 0.0
        fuse_score += max(0, 100 - r) * rule_weight
        fuse_score += max(0, 100 - m) * ml_weight

        in_rule = num in rule_rank
        in_ml = num in ml_rank
        if in_rule and in_ml:
            fuse_score += 20
            tag = "high_confidence"
        elif in_rule:
            tag = "rule_preferred"
        elif in_ml:
            tag = "exploration"
        else:
            tag = "other"

        fused.append((fuse_score, num, tag, in_rule))

    fused.sort(reverse=True)
    
    result = []
    for fuse_score, num, tag, in_rule in fused[:top_n]:
        # 获取规则模型的detail
        detail = rule_detail.get(num)
        
        # 如果没有detail且提供了必要参数，尝试构建detail
        if detail is None and score is not None and danma is not None and kill is not None and meta is not None:
            a, b, c = int(num[0]), int(num[1]), int(num[2])
            detail = triplet_weight_detail(a, b, c, score, danma, kill, meta)
        
        result.append({
            "num": num,
            "score": round(fuse_score, 2),  # 统一字段名，兼容页面打印
            "fuse_score": round(fuse_score, 2),
            "tag": tag,
            "in_rule": num in rule_rank,
            "in_ml": num in ml_rank,
            "rule_rank": rule_rank.get(num),
            "ml_rank": ml_rank.get(num),
            "detail": detail,
        })
    
    return result


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


def save_strategy_records(period, rule_only, ml_only, fused):
    """保存三套策略记录（用于后续对比分析）
    
    参数：
        period: 期号
        rule_only: 规则模型推荐列表
        ml_only: ML模型推荐列表
        fused: 融合推荐列表
    
    注意：首次发布的策略记录不会被覆盖（即使未结算），确保策略对比的准确性
    """
    try:
        history = kv_store.load('lottery3d_strategy_records', [])
        
        # 检查是否已存在同一期的记录
        existing_record = next((h for h in history if h["period"] == period), None)
        
        if existing_record:
            # 如果已存在，检查是否已结算
            if existing_record.get("settled"):
                # 已结算，不覆盖
                log.info(f"策略记录已结算，跳过更新（期号: {period}）")
                return
            # 未结算也保留首次发布，不覆盖
            log.info(f"策略记录已存在，保留首次发布（期号: {period}）")
            return
        
        # 不存在记录，创建新记录
        record = {
            "period": period,
            "rule_only": rule_only,
            "ml_only": ml_only,
            "fused": fused,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "settled": False,
            "revision": 1,  # 首次发布版本
        }
        
        history.append(record)
        
        # 只保留最近200期记录
        history = history[-200:]
        
        kv_store.save('lottery3d_strategy_records', history)
        log.info(f"策略记录已保存（期号: {period}）")
    except Exception as e:
        log.error(f"保存策略记录失败: {e}")


def settle_strategy_records(periods, numbers):
    """结算三路策略记录（规则模型/ML模型/融合模型分别统计）
    
    参数：
        periods: 期号列表
        numbers: 号码列表
    """
    try:
        history = kv_store.load('lottery3d_strategy_records', [])
        if not history:
            return
        
        index_map = {period: i for i, period in enumerate(periods)}
        changed = False
        
        for row in history:
            if row.get("settled"):
                continue
            
            idx = index_map.get(row["period"])
            if idx is None or idx + 1 >= len(numbers):
                continue
            
            actual = "".join(map(str, numbers[idx + 1]))
            row["actual"] = actual
            row["settled"] = True
            row["draw_period"] = periods[idx + 1]
            
            # 分别统计三条策略的表现
            for name in ("rule_only", "ml_only", "fused"):
                nums = row.get(name, [])
                row[f"{name}_hit_top3"] = actual in nums[:3]
                row[f"{name}_hit_top30"] = actual in nums[:30]
                row[f"{name}_hit_top100"] = actual in nums[:100]
                row[f"{name}_rank"] = nums.index(actual) + 1 if actual in nums else 1001
            
            changed = True
        
        if changed:
            kv_store.save('lottery3d_strategy_records', history)
            log.info("三路策略记录已结算")
    except Exception as e:
        log.error(f"结算策略记录失败: {e}")


def generate_strategy_recommendations(rule_list, ml_list, danma, kill):
    """生成三套推荐策略
    
    参数：
        rule_list: 规则模型推荐列表
        ml_list: ML模型推荐列表
        danma: 胆码列表
        kill: 杀码列表
    
    返回：
        strategy_recommendations: 包含三套策略的推荐结果
    """
    rule_set = set(r["num"] for r in rule_list)
    ml_set = set(m["num"] for m in ml_list)
    
    # 保守策略：规则 + ML 交集
    conservative = [r for r in rule_list if r["num"] in ml_set][:10]
    
    # 均衡策略：规则主导，少量探索
    balanced = []
    rule_added = set()
    for r in rule_list[:20]:
        balanced.append({"num": r["num"], "score": r.get("score", 0), "source": "rule"})
        rule_added.add(r["num"])
    
    # 补充少量ML独有号码
    ml_only = [m for m in ml_list if m["num"] not in rule_added][:5]
    for m in ml_only:
        balanced.append({"num": m["num"], "score": m.get("model_score", 0), "source": "ml"})
    
    # 探索策略：ML独有 + 冷号特征
    explore = []
    ml_explore = [m for m in ml_list if m["num"] not in rule_set][:8]
    for m in ml_explore:
        explore.append({"num": m["num"], "score": m.get("model_score", 0), "source": "ml_only"})
    
    return {
        "conservative": conservative,
        "balanced": balanced[:20],
        "explore": explore[:10],
    }


def select_strategy_mode(stability, model_lift, recent_hit_rate, actual_rank_avg):
    """根据模型表现自动选择推荐模式
    
    参数：
        stability: 推荐稳定度
        model_lift: 模型相对随机基准的提升
        recent_hit_rate: 最近线上命中率
        actual_rank_avg: 真实号码平均排名
    
    返回：
        mode: 推荐模式（conservative/balanced/explore）
        reason: 选择理由
    """
    if model_lift <= 0:
        return "explore", "模型未明显优于随机基准，需要探索"

    if stability > 0.8 and recent_hit_rate < 0.03:
        return "explore", "推荐过度稳定且命中率偏低，增加探索"

    if actual_rank_avg <= 250 and model_lift > 0.01:
        return "conservative", "模型排名表现优秀，采用保守策略"

    return "balanced", "模型有提升但需保持多样性，采用均衡策略"


def recommend_budget_level(model_lift, stability, recent_online_rate):
    """根据模型表现推荐资金/注数等级
    
    参数：
        model_lift: 模型相对随机基准的提升
        stability: 推荐稳定度
        recent_online_rate: 最近线上命中率
    
    返回：
        budget_info: 资金建议信息
    """
    if model_lift <= 0:
        return {
            "level": "低",
            "suggest_count": 10,
            "reason": "模型未明显优于随机基准"
        }

    if model_lift > 0.015 and recent_online_rate >= 0.03:
        return {
            "level": "中",
            "suggest_count": 20,
            "reason": "模型近期表现略优于随机"
        }

    return {
        "level": "观察",
        "suggest_count": 10,
        "reason": "样本不足或优势不稳定"
    }


def auto_recommend_count(model_lift, rank_top100_rate, online_hit_rate):
    """根据模型表现自动调整推荐注数
    
    参数：
        model_lift: 模型相对随机基准的提升
        rank_top100_rate: Top100覆盖率
        online_hit_rate: 线上命中率
    
    返回：
        count: 推荐注数
        reason: 调整理由
    """
    if model_lift <= 0:
        return 10, "模型无明显优势，减少推荐注数"

    if rank_top100_rate >= 0.18 and online_hit_rate >= 0.03:
        return 30, "Top100覆盖率和线上命中率均良好"

    if rank_top100_rate >= 0.12:
        return 20, "Top100覆盖率尚可"

    return 15, "模型优势有限，保持适中注数"


