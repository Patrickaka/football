#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
盘口聚类模块 - 提升最大的模块
===============================

功能：
1. 将历史比赛按盘口类型聚类
2. 每个盘口建立比分先验库
3. 预测时融合泊松概率与盘口先验

这是不接数据库情况下提升最大的模块。
"""

import os
import json
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from ..common import kv_store
from ..common.logger import setup_logger

from ..domain.sports.football import market_matching as _mm
from ..domain.sports.football import value as _mm_value

log = setup_logger('football.market_clustering')

# ==================== 常量配置 ====================
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
CLUSTER_DB_FILE = os.path.join(DATA_DIR, 'market_cluster_db.json')

# 标准盘口列表
STANDARD_HANDICAPS = [-3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, 
                      -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 
                      1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]

STANDARD_TOTALS = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]

class MarketCluster:
    """
    盘口聚类
    
    将历史比赛按亚盘和大小球聚类，建立比分先验库
    """
    
    def __init__(self):
        self.clusters = {}  # {(handicap, total): {'count': N, 'scores': {score: count}}}
        self._load()
    
    def _load(self):
        """加载聚类数据库"""
        try:
            self.clusters = kv_store.load('market_cluster_db') or {}
            log.debug("已加载盘口聚类数据库: %d 个盘口组合", len(self.clusters))
        except Exception as e:
            log.warning("加载盘口聚类数据库失败: %s", e)
            self.clusters = {}

    def save(self):
        """保存聚类数据库"""
        kv_store.save('market_cluster_db', self.clusters)
    
    def _round_to_standard(self, handicap: float, total: float) -> Tuple[float, float]:
        """纯计算在领域层"""
        return _mm.round_to_standard(handicap, total)
    
    def add_match(self, handicap: float, total: float, score: str):
        """
        添加一场比赛到聚类
        
        参数：
            handicap: 亚盘让球
            total: 大小球线
            score: 比分（如 "2-1"）
        """
        std_hcap, std_total = self._round_to_standard(handicap, total)
        key = f"{std_hcap}_{std_total}"
        
        if key not in self.clusters:
            self.clusters[key] = {'count': 0, 'scores': {}}
        
        self.clusters[key]['count'] += 1
        self.clusters[key]['scores'][score] = self.clusters[key]['scores'].get(score, 0) + 1
    
    def get_prior(self, handicap: float, total: float) -> Dict[str, float]:
        """
        获取该盘口的比分先验概率
        
        参数：
            handicap: 亚盘让球
            total: 大小球线
        
        返回：
            {比分: 概率}
        """
        std_hcap, std_total = self._round_to_standard(handicap, total)
        key = f"{std_hcap}_{std_total}"
        
        if key not in self.clusters or self.clusters[key]['count'] < 10:
            # 数据不足，返回均匀分布
            return {}
        
        cluster = self.clusters[key]
        total = cluster['count']
        scores = cluster['scores']
        
        # 计算概率
        probs = {score: cnt / total for score, cnt in scores.items()}
        
        # 排序并取前20个
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])[:20]
        
        return dict(sorted_probs)
    
    def fuse_with_prior(self, poisson_probs: Dict[str, float],
                       handicap: float, total: float,
                       prior_weight: float = 0.3) -> Dict[str, float]:
        """先取先验（读聚类库），融合本身在领域层"""
        return _mm_value.fuse_with_prior(
            poisson_probs, handicap, total, prior_weight,
            prior=self.get_prior(handicap, total))

# ==================== 全局实例 ====================
_cluster = None

def get_cluster() -> MarketCluster:
    """获取全局聚类实例"""
    global _cluster
    if _cluster is None:
        _cluster = MarketCluster()
    return _cluster

def get_market_prior(handicap: float, total: float) -> Dict[str, float]:
    """获取盘口先验的便捷接口"""
    return get_cluster().get_prior(handicap, total)

def fuse_poisson_with_prior(poisson_probs: Dict[str, float],
                           handicap: float, total: float,
                           prior_weight: float = 0.3) -> Dict[str, float]:
    """融合泊松概率与盘口先验的便捷接口"""
    return get_cluster().fuse_with_prior(poisson_probs, handicap, total, prior_weight)