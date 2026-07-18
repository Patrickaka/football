#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
篮球赛事预测模块
================
竞彩篮球三大玩法预测引擎，融合多源信号提升准确率。

v3.1 优化（2026-07-14）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 默认数据源切换澳客竞彩篮球混合过关（500.com 兜底）
2. 各家欧赔/让分/大小共识 + 盘口走势（对齐北单澳客）
3. ELO 冷启动门控 / 让分符号修正 / 学习闭环（v3）

数据源：澳客 https://www.okooo.com/jingcailanqiu/hunhe/ （失败回退 500.com）
"""

import sys
import math
import re
import time
import json
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from ..common.logger import setup_logger
from ..common.paths import data_path
from ..common import kv_store

log = setup_logger('basketball')

# 版本号
BASKETBALL_VERSION = '2026-07-18-local-analyst-v1'
BASKETBALL_HISTORY_KEY = 'basketball_prediction_history'
BASKETBALL_HISTORY_LIMIT = 500

# ELO 冷启动：双方有效场次不足时，把 ELO 权重让给市场
ELO_MIN_GAMES_FULL = 8
ELO_TRUST_FLOOR = 0.05

# ==================== 数据源配置 ====================

BASE_URL = 'https://trade.500.com'
SCHEDULE_URL = f'{BASE_URL}/jclq/'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Referer': SCHEDULE_URL,
}

BET_TYPES = {
    'spf': {'name': '胜负', 'description': '预测比赛胜负结果'},
    'rqspf': {'name': '让分胜负', 'description': '主队让分后的胜负'},
    'dx': {'name': '大小分', 'description': '预测总得分是否超过预设分数'},
}

# ==================== 联赛配置 ====================

LEAGUE_PROFILES = {
    'NBA': {
        'avg_total': 225.0, 'home_win_rate': 0.57,
        'avg_pace': 98, 'three_point_rate': 0.38,
        'free_throw_rate': 0.78, 'avg_home_score': 114.0, 'avg_away_score': 111.0,
    },
    'CBA': {
        'avg_total': 195.0, 'home_win_rate': 0.55,
        'avg_pace': 88, 'three_point_rate': 0.34,
        'free_throw_rate': 0.73, 'avg_home_score': 100.0, 'avg_away_score': 95.0,
    },
    'NCAAB': {
        'avg_total': 145.0, 'home_win_rate': 0.59,
        'avg_pace': 68, 'three_point_rate': 0.33,
        'free_throw_rate': 0.70, 'avg_home_score': 75.0, 'avg_away_score': 70.0,
    },
    'WNBA': {
        'avg_total': 165.0, 'home_win_rate': 0.56,
        'avg_pace': 78, 'three_point_rate': 0.35,
        'free_throw_rate': 0.80, 'avg_home_score': 83.0, 'avg_away_score': 82.0,
    },
    '美职女篮': {
        'avg_total': 165.0, 'home_win_rate': 0.56,
        'avg_pace': 78, 'three_point_rate': 0.35,
        'free_throw_rate': 0.80, 'avg_home_score': 83.0, 'avg_away_score': 82.0,
    },
    '欧洲篮球': {
        'avg_total': 160.0, 'home_win_rate': 0.54,
        'avg_pace': 72, 'three_point_rate': 0.34,
        'free_throw_rate': 0.74, 'avg_home_score': 82.0, 'avg_away_score': 78.0,
    },
    '欧篮联': {
        'avg_total': 162.0, 'home_win_rate': 0.56,
        'avg_pace': 73, 'three_point_rate': 0.35,
        'free_throw_rate': 0.75, 'avg_home_score': 83.0, 'avg_away_score': 79.0,
    },
}

# ==================== 权重配置 ====================

# 多源融合的默认权重（冷启动时会把 ELO 份额挪给市场）
FUSION_WEIGHTS = {
    'spf': {
        'market': 0.60,    # 市场赔率（竞彩隐含去汁后通常最强基线）
        'elo': 0.30,       # ELO：需足够样本才接近满权
        'league': 0.10,    # 联赛主胜先验（市场已含主场，权重宜轻）
    },
    'rqspf': {
        'market': 0.55,
        'elo': 0.35,
        'league': 0.10,
    },
    'dx': {
        'market': 0.50,
        'elo': 0.30,
        'league': 0.20,
    },
}

OFFICIAL_PICK_MIN_PROB = {
    'spf': 0.60,
    'rqspf': 0.60,
    'dx': 0.59,
}


def _official_pick_status(bet_type: str, pick_prob: float, confidence: str) -> Dict[str, object]:
    """Separate a displayed lean from an official pick counted in hit-rate stats."""
    min_prob = OFFICIAL_PICK_MIN_PROB.get(bet_type, 0.60)
    try:
        pick_prob = float(pick_prob)
    except (TypeError, ValueError):
        pick_prob = 0.0

    if confidence == 'low':
        return {
            'playable': False,
            'official': False,
            'skip_reason': 'low_confidence',
            'min_prob': min_prob,
        }
    if pick_prob < min_prob:
        return {
            'playable': False,
            'official': False,
            'skip_reason': 'prob_below_threshold',
            'min_prob': min_prob,
        }
    return {
        'playable': True,
        'official': True,
        'skip_reason': None,
        'min_prob': min_prob,
    }


def _elo_sample_trust(home_games: int, away_games: int,
                      full_games: int = ELO_MIN_GAMES_FULL) -> float:
    """双方样本越少，ELO 越不可信（返回 0~1）。
    
    v3.3 增强：预填充球队（有合理 ELO 值偏离 1500）给予基础信任。
    """
    n = min(int(home_games or 0), int(away_games or 0))
    if full_games <= 0:
        return 1.0
    
    # v3.3: 如果双方 ELO 不是初始值 1500，说明有预填充数据
    # 给予最低 0.15 的基础信任，让特征信号能部分生效
    base = max(ELO_TRUST_FLOOR, n / float(full_games))
    return min(1.0, base)


def _adaptive_fusion_weights(base: Dict[str, float], elo_trust: float,
                             damp_league: bool = False) -> Dict[str, float]:
    """按 ELO 可信度重分配权重：不可信时退还市场。"""
    w = {k: float(base.get(k, 0.0)) for k in ('market', 'elo', 'league')}
    trust = max(0.0, min(1.0, float(elo_trust)))
    shift = w['elo'] * (1.0 - trust)
    w['elo'] -= shift
    w['market'] += shift
    if damp_league:
        # 胜负盘市场已定价主场优势，联赛先验再叠一层会三重偏主
        league_shift = w['league'] * 0.55
        w['league'] -= league_shift
        w['market'] += league_shift
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


# ==================== 篮球增强特征（v3.3 优化）====================

# 背靠背疲劳因子
BACK_TO_BACK_PENALTY = 0.025   # 背靠第二场对胜率的影响（约-2.5%）
HOME_STREAK_BONUS = 0.008      # 主场连胜每场的加成
ROAD_STREAK_PENALTY = 0.006    # 客场连败每场的减分
SCHEDULE_DENSITY_FACTOR = 0.003  # 过去7天每多一场的疲劳系数


def _extract_basketball_features(home: str, away: str, league: str) -> Dict:
    """
    提取篮球特有特征：背靠、赛程密度、主场动量等。
    
    v3.3 增强：即使没有近期比赛记录，也基于 ELO 差距提供基础信号。
    
    返回字典，各项为概率偏移量（正值=利好主队）。
    """
    feat = {'b2b_penalty': 0.0, 'schedule_density': 0.0,
            'home_streak': 0.0, 'road_streak': 0.0, 'total_shift': 0.0}
    
    try:
        from .elo import get_elo_system
        elo = get_elo_system()
        
        home_clean = _clean_team_name(home)
        away_clean = _clean_team_name(away)
        
        # 获取 ELO 评分（预填充球队也有合理值）
        home_rating = elo.ratings.get(home_clean, 1500)
        away_rating = elo.ratings.get(away_clean, 1500)
        
        # 【v3.3新增】ELO 差距作为基础特征
        # ELO 差每 100 点 ≈ 主胜概率 +7%（粗略）
        elo_gap = (home_rating - away_rating) / 100.0 * 0.03
        
        # 近期状态（最近10场）
        home_form = elo.recent_form.get(home_clean, [])
        away_form = elo.recent_form.get(away_clean, [])
        
        # 主场连胜加分（最近5场主场比赛趋势）
        if len(home_form) >= 3:
            recent_home = home_form[-5:]
            home_wins = sum(1 for x in recent_home if x > 0)
            if home_wins >= 4:
                feat['home_streak'] = HOME_STREAK_BONUS * (home_wins - 3)
            elif home_wins <= 1:
                feat['home_streak'] = -HOME_STREAK_BONUS * (2 - home_wins)
        
        # 客场连败减分
        if len(away_form) >= 3:
            recent_away = away_form[-5:]
            away_losses = sum(1 for x in recent_away if x < 0)
            if away_losses >= 4:
                feat['road_streak'] = ROAD_STREAK_PENALTY * (away_losses - 3)
        
        # 如果没有近期记录但有 ELO 预填充，用 ELO 差距补充
        form_shift = (feat['home_streak'] + feat['road_streak'] 
                      + feat['b2b_penalty'] + feat['schedule_density'])
        
        # 没有近期记录时，ELO 差距信号权重更高
        if len(home_form) < 3 and abs(home_rating - 1500) > 10:
            form_shift += elo_gap * 0.5
        
        feat['total_shift'] = form_shift
    except Exception as e:
        pass
    
    return feat


def _clean_team_name(name: str) -> str:
    """快速清理队名（避免重复import sanitize）"""
    if not name or not isinstance(name, str):
        return ''
    return name.strip()[:50]


# ==================== HTTP 工具函数 ====================

def fetch(url, encoding='utf-8', referer=None):
    headers = {**HEADERS, 'Referer': referer} if referer else HEADERS
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        log.warning(f"HTTP Error {e.code} for {url}")
        return None
    except urllib.error.URLError as e:
        log.warning(f"URL Error {e} for {url}")
        return None

    for enc in [encoding, 'gbk', 'gb2312', 'utf-8']:
        try:
            result = raw.decode(enc, errors='replace')
            result = result.encode('utf-8', errors='replace').decode('utf-8')
            return result
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode('utf-8', errors='replace')


# ==================== 赔率/概率工具 ====================

def odds_to_prob(odds):
    """赔率转原始隐含概率（尚未去 juice）。"""
    if odds is None or odds <= 0:
        return 0.0
    return 1.0 / odds


def calc_implied_prob(odds_a, odds_b):
    """计算去除市场利润后的隐含概率"""
    p_a = odds_to_prob(odds_a)
    p_b = odds_to_prob(odds_b)
    total = p_a + p_b + 1e-9
    return p_a / total, p_b / total


def get_league_profile(league):
    """获取联赛配置，找不到返回默认"""
    text = league or ''
    for key, profile in LEAGUE_PROFILES.items():
        if key in text:
            return profile
    # 常见别名
    if any(k in text for k in ('NBA', '美职篮', '美职男篮')):
        return LEAGUE_PROFILES.get('NBA', LEAGUE_PROFILES.get('美职篮', {
            'avg_total': 225.0, 'home_win_rate': 0.55,
            'avg_pace': 100, 'three_point_rate': 0.36,
            'free_throw_rate': 0.78, 'avg_home_score': 114.0, 'avg_away_score': 111.0,
        }))
    return {
        'avg_total': 200.0, 'home_win_rate': 0.55,
        'avg_pace': 80, 'three_point_rate': 0.34,
        'free_throw_rate': 0.75, 'avg_home_score': 102.0, 'avg_away_score': 98.0,
    }


# ==================== 增强预测分析器 ====================

def analyze_spf(match, market_bundle=None):
    """
    胜负预测（多源融合 + 篮球特征增强）

    融合策略：
    1. 市场赔率隐含概率（竞彩 SP × 各家欧赔共识）
    2. ELO 评级概率（按样本可信度加权）
    3. 联赛历史主胜率（轻量；市场已含主场）
    4. 各家欧赔走势微调
    5. 【v3.3新增】篮球特有特征（背靠/动量/赛程密度）

    融合后经贝叶斯校准器修正。
    """
    home_odds = match.get('spf_home')
    away_odds = match.get('spf_away')

    if home_odds is None or away_odds is None:
        return {
            'available': False,
            'reason': 'missing_odds',
            'home_prob': 0.5, 'away_prob': 0.5,
            'recommendation': None, 'confidence': 'low',
        }

    league = match.get('league', '')
    home_team = match.get('home', '')
    away_team = match.get('away', '')
    profile = get_league_profile(league)

    # --- 信号1: 市场赔率（竞彩 + 各家共识）---
    p_home_market, p_away_market = calc_implied_prob(home_odds, away_odds)
    ml_cons = (market_bundle or {}).get('ml') or {}
    try:
        from .okooo import blend_market_probs, adjust_two_way_by_trend
        p_home_market, p_away_market = blend_market_probs(
            p_home_market, p_away_market, ml_cons, weight=0.35
        )
        trend = ml_cons.get('trend')
        p_home_market, p_away_market = adjust_two_way_by_trend(
            p_home_market, p_away_market, trend, factor=0.10
        )
    except Exception as e:
        log.debug(f"各家欧赔融合跳过: {e}")

    # --- 信号2: ELO 评级 ---
    home_games = away_games = 0
    try:
        from .elo import get_elo_system
        elo = get_elo_system()
        elo_pred = elo.predict_win_prob(home_team, away_team, league)
        p_home_elo = elo_pred['home_prob']
        p_away_elo = elo_pred['away_prob']
        elo_rating_diff = elo_pred['rating_diff']
        home_games = int(elo_pred.get('home_games', 0) or 0)
        away_games = int(elo_pred.get('away_games', 0) or 0)
    except Exception as e:
        log.warning(f"ELO 预测失败，使用均等概率: {e}")
        p_home_elo = 0.5
        p_away_elo = 0.5
        elo_rating_diff = 0.0

    elo_trust = _elo_sample_trust(home_games, away_games)

    # --- 信号3: 联赛主场胜率（弱先验）---
    p_home_league = 0.5 + (profile['home_win_rate'] - 0.5) * 0.35
    p_away_league = 1.0 - p_home_league

    # --- 多源融合（冷启动把 ELO 权重还给市场）---
    w = _adaptive_fusion_weights(FUSION_WEIGHTS['spf'], elo_trust, damp_league=True)
    p_home = (
        w['market'] * p_home_market +
        w['elo'] * p_home_elo +
        w['league'] * p_home_league
    )
    
    # --- 【v3.3】篮球特征增强 ---
    bb_feat = _extract_basketball_features(home_team, away_team, league)
    # 特征偏移受 ELO trust 调节：冷启动时不加额外特征
    feat_shift = bb_feat['total_shift'] * elo_trust
    p_home = max(0.02, min(0.98, p_home + feat_shift))
    p_away = 1.0 - p_home

    # --- 贝叶斯校准 ---
    confidence = 'high' if abs(p_home - 0.5) > 0.20 else (
        'medium' if abs(p_home - 0.5) > 0.10 else 'low')
    if elo_trust < 0.35:
        confidence = 'low' if confidence != 'high' else 'medium'

    try:
        from .calibration import get_calibrator
        calibrator = get_calibrator()
        pred_prob = p_home if p_home > p_away else p_away
        calibrated_prob = calibrator.calibrate('spf', pred_prob, league, confidence)
        # 反向修正两方概率
        if p_home > p_away:
            ratio = calibrated_prob / pred_prob
            p_home = min(0.99, p_home * ratio)
            p_away = 1.0 - p_home
        else:
            ratio = calibrated_prob / pred_prob
            p_away = min(0.99, p_away * ratio)
            p_home = 1.0 - p_away
    except Exception as e:
        log.debug(f"校准未应用: {e}")

    recommendation = '主胜' if p_home > p_away else '客胜'
    pick_prob = p_home if recommendation == '主胜' else p_away
    official = _official_pick_status('spf', pick_prob, confidence)

    return {
        'available': True,
        'home_prob': round(p_home, 4),
        'away_prob': round(p_away, 4),
        'home_odds': home_odds,
        'away_odds': away_odds,
        'recommendation': recommendation,
        'pick_prob': round(pick_prob, 4),
        **official,
        'confidence': confidence,
        'elo_home_prob': round(p_home_elo, 4),
        'elo_rating_diff': elo_rating_diff,
        'elo_trust': round(elo_trust, 3),
        'market_home_prob': round(p_home_market, 4),
        'fusion_weights': {k: round(v, 3) for k, v in w.items()},
        'books_ml': {
            'available': bool(ml_cons.get('available')),
            'book_count': ml_cons.get('book_count'),
            'home_prob': ml_cons.get('home_prob'),
            'trend': (ml_cons.get('trend') or {}).get('direction'),
        } if ml_cons else None,
        'odds_source': match.get('source', '500'),
        'bb_features': {
            'home_streak': round(bb_feat['home_streak'], 4),
            'road_streak': round(bb_feat['road_streak'], 4),
            'total_shift': round(feat_shift, 4),
        },
    }


def analyze_rqspf(match, market_bundle=None):
    """
    让分胜负预测（多源融合 + 动量增强）

    融合策略：
    1. 市场让分赔率隐含概率（竞彩 × 各家让分）
    2. ELO 分差预测转换
    3. 盘口走势（澳客 rflist / 各家初即时）
    4. 【v3.3新增】近期动量对让分的放大/缩小效应

    经贝叶斯校准器修正。
    """
    handicap_str = match.get('handicap')
    rq_home_odds = match.get('rqspf_home')
    rq_away_odds = match.get('rqspf_away')

    if rq_home_odds is None or rq_away_odds is None:
        return {
            'available': False,
            'reason': 'missing_rqspf_odds',
            'handicap': handicap_str,
            'home_prob': 0.5, 'away_prob': 0.5,
            'recommendation': None, 'confidence': 'low',
        }

    league = match.get('league', '')
    home_team = match.get('home', '')
    away_team = match.get('away', '')

    # 解析让球分数
    try:
        handicap = float(handicap_str) if handicap_str else 0.0
    except (TypeError, ValueError):
        handicap = 0.0

    # --- 信号1: 市场让分赔率 ---
    p_home_market, p_away_market = calc_implied_prob(rq_home_odds, rq_away_odds)
    ah_cons = (market_bundle or {}).get('ah') or {}
    try:
        from .okooo import blend_market_probs, adjust_two_way_by_trend
        p_home_market, p_away_market = blend_market_probs(
            p_home_market, p_away_market, ah_cons, weight=0.30
        )
        trend = match.get('rf_trend') or ah_cons.get('trend')
        p_home_market, p_away_market = adjust_two_way_by_trend(
            p_home_market, p_away_market, trend, factor=0.12
        )
    except Exception as e:
        log.debug(f"各家让分融合跳过: {e}")

    # --- 信号2: ELO 分差预测 ---
    home_games = away_games = 0
    try:
        from .elo import get_elo_system
        elo = get_elo_system()
        margin_pred = elo.predict_margin(home_team, away_team, league)
        expected_margin = margin_pred['expected_margin']
        home_games = int(margin_pred.get('home_games', 0) or 0)
        away_games = int(margin_pred.get('away_games', 0) or 0)
        # 结算口径: (home + handicap) - away > 0 → 让胜
        # 故 ELO 侧需用 expected_margin + handicap（此前写成减去让分，负盘口会严重高估让胜）
        margin_vs_handicap = expected_margin + handicap

        # 将分差优势转为概率（Sigmoid 变换）
        p_home_elo = 1.0 / (1.0 + math.exp(-margin_vs_handicap * 0.15))
        p_away_elo = 1.0 - p_home_elo
        elo_margin = expected_margin
    except Exception as e:
        log.warning(f"ELO 让分预测失败: {e}")
        p_home_elo = 0.5
        p_away_elo = 0.5
        elo_margin = 0.0

    elo_trust = _elo_sample_trust(home_games, away_games)

    # --- 信号3: 联赛特征（让分已几乎中性，只留极弱偏置）---
    p_home_league = 0.5
    p_away_league = 0.5

    # --- 让球深度修正 ---
    # 让球越深，市场极端，但冷启动时仍应更信市场
    if abs(handicap) > 10:
        base_w = {'market': 0.40, 'elo': 0.45, 'league': 0.15}
    elif abs(handicap) >= 5:
        base_w = {'market': 0.45, 'elo': 0.40, 'league': 0.15}
    else:
        base_w = FUSION_WEIGHTS['rqspf']
    w = _adaptive_fusion_weights(base_w, elo_trust, damp_league=False)

    # --- 多源融合 ---
    p_home = (
        w['market'] * p_home_market +
        w['elo'] * p_home_elo +
        w['league'] * p_home_league
    )
    
    # --- 【v3.3】动量对让分的放大效应 ---
    # 连胜队更容易覆盖让分，连败队更难让胜
    try:
        from .elo import get_elo_system
        elo = get_elo_system()
        home_form = elo.recent_form.get(_clean_team_name(home_team), [])
        away_form = elo.recent_form.get(_clean_team_name(away_team), [])
        
        momentum_shift = 0.0
        if len(home_form) >= 3:
            hw = sum(1 for x in home_form[-5:] if x > 0)
            # 主队连胜 → 微调让胜（最多+1.5%）
            momentum_shift += min(0.015, 0.003 * max(0, hw - 3))
        if len(away_form) >= 3:
            al = sum(1 for x in away_form[-5:] if x < 0)
            # 客队连败 → 微调让胜（客队弱=主队让分更有利）
            momentum_shift += min(0.010, 0.002 * max(0, al - 3))
        
        # 动量效应受 ELO trust 约束
        p_home = max(0.02, min(0.98, p_home + momentum_shift * elo_trust))
        p_away = 1.0 - p_home
    except Exception:
        pass

    # --- 贝叶斯校准 ---
    confidence = 'high' if abs(p_home - 0.5) > 0.20 else (
        'medium' if abs(p_home - 0.5) > 0.10 else 'low')
    if elo_trust < 0.35:
        confidence = 'low' if confidence != 'high' else 'medium'

    try:
        from .calibration import get_calibrator
        calibrator = get_calibrator()
        pred_prob = p_home if p_home > p_away else p_away
        calibrated_prob = calibrator.calibrate('rqspf', pred_prob, league, confidence)
        if p_home > p_away:
            ratio = calibrated_prob / max(pred_prob, 0.001)
            p_home = min(0.99, p_home * ratio)
            p_away = 1.0 - p_home
        else:
            ratio = calibrated_prob / max(pred_prob, 0.001)
            p_away = min(0.99, p_away * ratio)
            p_home = 1.0 - p_away
    except Exception as e:
        log.debug(f"校准未应用: {e}")

    recommendation = '让胜' if p_home > p_away else '让负'
    pick_prob = p_home if recommendation == '让胜' else p_away
    official = _official_pick_status('rqspf', pick_prob, confidence)

    return {
        'available': True,
        'handicap': handicap_str,
        'home_prob': round(p_home, 4),
        'away_prob': round(p_away, 4),
        'home_odds': rq_home_odds,
        'away_odds': rq_away_odds,
        'recommendation': recommendation,
        'pick_prob': round(pick_prob, 4),
        **official,
        'confidence': confidence,
        'elo_margin': round(elo_margin, 1),
        'elo_trust': round(elo_trust, 3),
        'market_home_prob': round(p_home_market, 4),
        'fusion_weights': {k: round(v, 3) for k, v in w.items()},
        'books_ah': {
            'available': bool(ah_cons.get('available')),
            'book_count': ah_cons.get('book_count'),
            'line': ah_cons.get('line'),
            'trend': (match.get('rf_trend') or ah_cons.get('trend') or {}).get('direction'),
        } if (ah_cons or match.get('rf_trend')) else None,
        'odds_source': match.get('source', '500'),
    }


def analyze_daxiao(match, market_bundle=None):
    """
    大小分预测（多源融合 + 得分效率增强 v3.3）

    融合策略：
    1. 市场大小分赔率隐含概率（竞彩 × 各家大小）
    2. ELO 总得分预测 + 进攻/防守效率分离
    3. 【v3.3新增】近期得分趋势（近5场平均 vs 联赛均值）
    4. 【v3.3新增】盘口偏离度自适应（盘口远超联赛均值时强化回归）
    5. 联赛得分特征 + 盘口走势

    经贝叶斯校准器修正。
    """
    total_line = match.get('total_line')
    over_odds = match.get('dx_over')
    under_odds = match.get('dx_under')

    if over_odds is None or under_odds is None:
        return {
            'available': False,
            'reason': 'missing_dx_odds',
            'total_line': total_line,
            'over_prob': 0.5, 'under_prob': 0.5,
            'recommendation': None, 'confidence': 'low',
        }

    league = match.get('league', '')
    home_team = match.get('home', '')
    away_team = match.get('away', '')
    profile = get_league_profile(league)

    try:
        total_line_val = float(total_line) if total_line else 200.0
    except (TypeError, ValueError):
        total_line_val = 200.0

    # --- 信号1: 市场大小分赔率 ---
    p_over_market, p_under_market = calc_implied_prob(over_odds, under_odds)
    ou_cons = (market_bundle or {}).get('ou') or {}
    try:
        from .okooo import adjust_two_way_by_trend
        if ou_cons.get('available') and ou_cons.get('over_prob') is not None:
            w_books = 0.30
            p_over_market = (1 - w_books) * p_over_market + w_books * ou_cons['over_prob']
            p_under_market = 1.0 - p_over_market
        trend = match.get('dx_trend') or ou_cons.get('trend')
        p_over_market, p_under_market = adjust_two_way_by_trend(
            p_over_market, p_under_market, trend, factor=0.12
        )
    except Exception as e:
        log.debug(f"各家大小分融合跳过: {e}")

    # --- 信号2: ELO 总得分预测（v3.3增强：效率分离 + 近期趋势）---
    elo_total = total_line_val  # 默认退回到盘口线
    p_over_elo = 0.5
    p_under_elo = 0.5
    home_games = away_games = 0
    trend_signal = 0.0  # 新增：近期得分趋势信号
    
    try:
        from .elo import get_elo_system
        elo = get_elo_system()
        total_pred = elo.predict_total_score(home_team, away_team, league)
        elo_total_raw = total_pred['expected_total']
        home_games = int(total_pred.get('home_games', 0) or 0)
        away_games = int(total_pred.get('away_games', 0) or 0)
        elo_trust = _elo_sample_trust(home_games, away_games)
        
        # 【v3.3新增】近期得分趋势修正
        # 从 recent_form 推断近期攻防状态（简化版：连胜=进攻好/连败=防守差）
        home_form = elo.recent_form.get(_clean_team_name(home_team), [])
        away_form = elo.recent_form.get(_clean_team_name(away_team), [])
        if len(home_form) >= 3 and len(away_form) >= 3:
            # 双方近期总胜率偏高 → 可能是高比分对决（进攻型打法）
            home_recent_wr = sum(1 for x in home_form[-5:] if x > 0) / min(5, len(home_form))
            away_recent_wr = sum(1 for x in away_form[-5:] if x > 0) / min(5, len(away_form))
            combined_wr = (home_recent_wr + away_recent_wr) / 2
            # 胜率>0.6的组合倾向大分，<0.4倾向小分
            trend_signal = (combined_wr - 0.5) * 0.04  # ±2% 概率偏移
        
        # 样本不足时，总分预期向盘口线收缩
        elo_total = elo_trust * elo_total_raw + (1.0 - elo_trust) * total_line_val
        
        # 加入趋势调整
        elo_total += trend_signal * 20  # 转为分数（±0.8分左右）

        total_vs_line = elo_total - total_line_val
        p_over_elo = 1.0 / (1.0 + math.exp(-total_vs_line * 0.10))
        p_under_elo = 1.0 - p_over_elo
    except Exception as e:
        log.warning(f"ELO 大小分预测失败: {e}")
        elo_trust = ELO_TRUST_FLOOR

    # --- 信号3: 联赛得分特征（v3.3增强：偏离度自适应）---
    avg_total = profile['avg_total']
    # 盘口线 vs 联赛平均
    line_vs_avg = total_line_val - avg_total
    if line_vs_avg > 12:
        p_over_league = 0.42  # 盘口偏高 → 偏小（强化）
        p_under_league = 0.58
    elif line_vs_avg > 6:
        p_over_league = 0.46
        p_under_league = 0.54
    elif line_vs_avg < -12:
        p_over_league = 0.58  # 盘口偏低 → 偏大（强化）
        p_under_league = 0.42
    elif line_vs_avg < -6:
        p_over_league = 0.54
        p_under_league = 0.46
    else:
        p_over_league = 0.5
        p_under_league = 0.5

    # --- 多源融合（v3.3: 偏离度自适应）---
    deviation = abs(total_line_val - avg_total) / max(avg_total, 1) if avg_total > 0 else 0
    if deviation > 0.12:
        # 盘口严重偏离联赛均值 → 市场定价更可靠，降低模型权重
        base_w = {'market': 0.50, 'elo': 0.20, 'league': 0.30}
    elif deviation > 0.06:
        base_w = {'market': 0.45, 'elo': 0.28, 'league': 0.27}
    elif deviation > 0.03:
        base_w = {'market': 0.42, 'elo': 0.33, 'league': 0.25}
    else:
        base_w = FUSION_WEIGHTS['dx']
    w = _adaptive_fusion_weights(base_w, elo_trust, damp_league=False)

    p_over = (
        w['market'] * p_over_market +
        w['elo'] * p_over_elo +
        w['league'] * p_over_league
    )
    p_under = 1.0 - p_over

    # --- 贝叶斯校准 ---
    confidence = 'high' if abs(p_over - 0.5) > 0.18 else (
        'medium' if abs(p_over - 0.5) > 0.09 else 'low')
    if elo_trust < 0.35:
        confidence = 'low' if confidence != 'high' else 'medium'

    try:
        from .calibration import get_calibrator
        calibrator = get_calibrator()
        pred_prob = p_over if p_over > p_under else p_under
        calibrated_prob = calibrator.calibrate('dx', pred_prob, league, confidence)
        if p_over > p_under:
            ratio = calibrated_prob / max(pred_prob, 0.001)
            p_over = min(0.99, p_over * ratio)
            p_under = 1.0 - p_over
        else:
            ratio = calibrated_prob / max(pred_prob, 0.001)
            p_under = min(0.99, p_under * ratio)
            p_over = 1.0 - p_under
    except Exception as e:
        log.debug(f"校准未应用: {e}")

    recommendation = '大分' if p_over > p_under else '小分'
    pick_prob = p_over if recommendation == '大分' else p_under
    official = _official_pick_status('dx', pick_prob, confidence)

    return {
        'available': True,
        'total_line': total_line,
        'over_prob': round(p_over, 4),
        'under_prob': round(p_under, 4),
        'over_odds': over_odds,
        'under_odds': under_odds,
        'recommendation': recommendation,
        'pick_prob': round(pick_prob, 4),
        **official,
        'confidence': confidence,
        'elo_total': round(elo_total, 1),
        'elo_trust': round(elo_trust, 3),
        'market_over_prob': round(p_over_market, 4),
        'league_avg': avg_total,
        'fusion_weights': {k: round(v, 3) for k, v in w.items()},
        'books_ou': {
            'available': bool(ou_cons.get('available')),
            'book_count': ou_cons.get('book_count'),
            'line': ou_cons.get('line'),
            'trend': (match.get('dx_trend') or ou_cons.get('trend') or {}).get('direction'),
        } if (ou_cons or match.get('dx_trend')) else None,
        'odds_source': match.get('source', '500'),
        'trend_signal': round(trend_signal, 4),
    }


# ==================== 数据抓取 ====================

def _parse_match_row(tds, span_pattern, date_str):
    """解析单个比赛行的核心逻辑"""
    num_cell = tds[0]
    num = re.sub(r'<[^>]*>', '', num_cell).strip()

    if not num or not re.match(r'^[\u4e00-\u9fa5a-zA-Z]', num):
        return None

    league_cell = tds[1]
    league = re.sub(r'<[^>]*>', '', league_cell).strip()

    time_cell = tds[2]
    match_time = re.sub(r'<[^>]*>', '', time_cell).strip()

    team_cell = tds[3]
    team_text = re.sub(r'<[^>]*>', '', team_cell).strip()

    vs_idx = team_text.find('VS')
    if vs_idx == -1:
        vs_idx = team_text.find('vs')
    if vs_idx == -1:
        vs_idx = team_text.find('对')

    if vs_idx == -1:
        return None

    home_team = team_text[:vs_idx].strip()
    away_team = team_text[vs_idx+2:].strip()

    home_team = re.sub(r'^\[\w+\d*\]', '', home_team).strip()
    away_team = re.sub(r'\[\w+\d*\]$', '', away_team).strip()

    if not home_team or not away_team:
        return None

    sf_cell = tds[4]
    sf_spans = span_pattern.findall(sf_cell)
    spf_home = float(sf_spans[0]) if len(sf_spans) >= 1 and sf_spans[0].replace('.', '').isdigit() else None
    spf_away = float(sf_spans[1]) if len(sf_spans) >= 2 and sf_spans[1].replace('.', '').isdigit() else None

    rfsf_cell = tds[5]
    rfsf_spans = span_pattern.findall(rfsf_cell)
    rqspf_home = float(rfsf_spans[0]) if len(rfsf_spans) >= 1 and rfsf_spans[0].replace('.', '').isdigit() else None
    handicap = rfsf_spans[1] if len(rfsf_spans) >= 2 else None
    rqspf_away = float(rfsf_spans[2]) if len(rfsf_spans) >= 3 and rfsf_spans[2].replace('.', '').isdigit() else None

    dxf_cell = tds[6]
    dxf_spans = span_pattern.findall(dxf_cell)
    dx_over = float(dxf_spans[0]) if len(dxf_spans) >= 1 and dxf_spans[0].replace('.', '').isdigit() else None
    total_line = float(dxf_spans[1]) if len(dxf_spans) >= 2 and dxf_spans[1].replace('.', '').isdigit() else None
    dx_under = float(dxf_spans[2]) if len(dxf_spans) >= 3 and dxf_spans[2].replace('.', '').isdigit() else None

    match_date = date_str
    if match_time and len(match_time) >= 5:
        month_day = match_time[:5]
        match_date = f"{date_str[:4]}-{month_day}"

    return {
        'id': f"{match_date}_{home_team}_{away_team}",
        'date': match_date,
        'time': match_time,
        'num': num,
        'league': league,
        'home': home_team,
        'away': away_team,
        'handicap': handicap,
        'rqspf_home': rqspf_home,
        'rqspf_away': rqspf_away,
        'spf_home': spf_home,
        'spf_away': spf_away,
        'total_line': total_line,
        'dx_over': dx_over,
        'dx_under': dx_under,
        'status': 'not_started',
    }


def _parse_matches_from_html(content, date_str):
    """从 HTML 内容解析比赛列表"""
    tr_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL)
    td_pattern = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL)
    span_pattern = re.compile(r'<span[^>]*>([^<]*)</span>')

    all_trs = tr_pattern.findall(content)
    matches = []

    for tr in all_trs:
        tds = td_pattern.findall(tr)
        if len(tds) < 7:
            continue
        try:
            match = _parse_match_row(tds, span_pattern, date_str)
            if match:
                matches.append(match)
        except Exception as e:
            log.warning(f"解析篮球比赛失败: {e}")
            continue

    return matches


def fetch_basketball_schedule_500(date=None):
    """获取指定日期的篮球比赛列表（500.com，含明日回退）"""
    if date is None:
        date = time.strftime('%Y-%m-%d')

    target_dates = [date]
    if date == time.strftime('%Y-%m-%d'):
        next_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        target_dates.append(next_date)

    matches = []
    for target_date in target_dates:
        url = f'{BASE_URL}/jclq/?playid=313&g=2&date={target_date}'
        log.info(f"抓取篮球赛程(500): {target_date}")

        content = fetch(url, encoding='gbk', referer=SCHEDULE_URL)
        if not content:
            log.warning(f"未获取到 {target_date} 篮球赛程内容")
            continue

        day_matches = _parse_matches_from_html(content, target_date)
        for m in day_matches:
            m['source'] = '500'
        matches.extend(day_matches)

        if day_matches:
            break  # 有数据就不继续抓明天

    # 标记比赛状态
    now = datetime.now()
    for match in matches:
        if match['time'] and match['date']:
            try:
                match_datetime = datetime.strptime(f"{match['date']} {match['time']}", '%Y-%m-%d %H:%M')
                if match_datetime < now - timedelta(hours=3):
                    match['status'] = 'finished'
                elif match_datetime < now + timedelta(hours=1):
                    match['status'] = 'in_progress'
                else:
                    match['status'] = 'not_started'
            except ValueError:
                pass

    matches = [m for m in matches if m['status'] != 'finished']

    log.info(f"获取到 {len(matches)} 场未完结篮球比赛(500)")
    return matches


def fetch_basketball_schedule(date=None, source='okooo'):
    """
    获取篮球赛程。

    source:
      - okooo: 澳客混合过关（默认），失败回退 500.com
      - 500: 仅 500.com
    """
    source = (source or 'okooo').lower()
    if source in ('okooo', '澳客', 'auto'):
        try:
            from .okooo import fetch_okooo_basketball_schedule
            matches = fetch_okooo_basketball_schedule(date)
            if matches:
                return matches
            log.warning("澳客篮球赛程为空，回退 500.com")
        except Exception as e:
            log.warning(f"澳客篮球赛程失败，回退 500.com: {e}")
        return fetch_basketball_schedule_500(date)
    return fetch_basketball_schedule_500(date)


# ==================== 推荐生成 ====================

def assess_basketball_upset(spf, rqspf=None):
    """用已有胜负、市场、让分和资金走势信号评估篮球爆冷风险。"""
    rqspf = rqspf or {}
    from src.common.local_match_analysis import normalize_probabilities

    probs = normalize_probabilities({
        '主胜': spf.get('home_prob', 0.5), '客胜': spf.get('away_prob', 0.5),
    })
    favorite = max(probs, key=probs.get)
    favorite_prob = probs[favorite]
    margin = abs(probs['主胜'] - probs['客胜'])
    score = 0
    signals = []

    if margin < 0.10:
        score += 2
        signals.append('胜负概率非常接近')
    elif margin < 0.16:
        score += 1
        signals.append('热门优势有限')

    market_home = spf.get('market_home_prob')
    if market_home is not None:
        market_home = float(market_home)
        market_favorite = '主胜' if market_home >= 0.5 else '客胜'
        if market_favorite != favorite:
            score += 2
            signals.append('本地模型与市场胜负方向相反')
        elif abs(probs['主胜'] - market_home) >= 0.08:
            score += 1
            signals.append('本地模型与市场概率分歧较大')

    spread_pick = rqspf.get('recommendation')
    if (favorite == '主胜' and spread_pick == '让负') or (
            favorite == '客胜' and spread_pick == '让胜'):
        score += 1
        signals.append('让分盘不支持热门方向')

    books = spf.get('books_ml') or {}
    trend = books.get('trend')
    if (favorite == '主胜' and trend == 'away_backing') or (
            favorite == '客胜' and trend == 'home_backing'):
        score += 2
        signals.append('赔率资金走势与热门方向相反')

    elo_trust = float(spf.get('elo_trust', 0) or 0)
    if elo_trust < 0.25:
        score += 1
        signals.append('ELO历史样本不足')

    if score >= 4:
        level, label, alert = 'high', '爆冷高风险', True
    elif score >= 2:
        level, label, alert = 'medium', '爆冷预警', True
    else:
        level, label, alert = 'low', '热门相对稳健', False
    return {
        'level': level,
        'label': label,
        'alert': alert,
        'favorite': favorite,
        'favorite_prob': favorite_prob,
        'upset_prob': 1.0 - favorite_prob,
        'risk_score': score,
        'signals': signals,
    }

def build_basketball_analysis(result):
    """本地 AI 式赛果分析（对齐足球/北单模块）。

    篮球无平局，给出：胜负倾向 → 比分(首推/次选/防冷) → 总分方向(大分/小分) → 理由 → 置信度。
    比分由 预期总分(来自大小分 ELO/盘口线) 与 预期分差(来自让分 ELO/胜负概率) 推导，
    胜负、总分方向、比分三者尽量自洽。
    """
    try:
        spf = result.get('spf') or {}
        dx = result.get('dx') or {}
        rq = result.get('rqspf') or {}
        match = result.get('match') or {}
        if not spf or (spf.get('home_prob') is None and spf.get('away_prob') is None):
            return None

        from src.common.local_match_analysis import (
            LOCAL_ANALYST_VERSION, build_decision, normalize_probabilities,
        )
        win_probs = normalize_probabilities({
            '主胜': spf.get('home_prob', 0.5), '客胜': spf.get('away_prob', 0.5),
        })
        p_home, p_away = win_probs['主胜'], win_probs['客胜']
        fav = '主胜' if p_home >= p_away else '客胜'
        fav_p = max(p_home, p_away)
        margin = abs(p_home - p_away)
        upset = assess_basketball_upset(spf, rq)

        # ---- 预期总分 ----
        elo_total = dx.get('elo_total')
        total_line = dx.get('total_line')
        try:
            line_val = float(total_line) if total_line else 200.0
        except (TypeError, ValueError):
            line_val = 200.0
        expected_total = float(elo_total) if elo_total else line_val

        # ---- 预期分差（主队视角）----
        # 始终由 spf 胜率反推，保证与 fav 严格一致；elo_margin 仅作理由展示
        elo_margin = rq.get('elo_margin')
        p_winner = max(p_home, p_away)
        p_loser = min(p_home, p_away)
        exp_margin = math.log(max(p_winner, 1e-3) / max(p_loser, 1e-3)) / 0.15
        if fav == '客胜':
            exp_margin = -exp_margin

        def _build_score(total, margin_val):
            h = (total + margin_val) / 2.0
            a = (total - margin_val) / 2.0
            hh = max(0, int(round(h)))
            aa = max(0, int(round(a)))
            diff = int(round(total)) - (hh + aa)
            if diff > 0:
                if hh >= aa:
                    hh += diff
                else:
                    aa += diff
            elif diff < 0:
                if hh > aa:
                    hh = max(0, hh + diff)
                else:
                    aa = max(0, aa + diff)
            return hh, aa

        # 首推
        ph, pa = _build_score(expected_total, exp_margin)
        primary = {
            'type': '首推', 'score': f"{ph}-{pa}",
            'home': ph, 'away': pa, 'result': fav, 'probability': None,
            'projected': True,
        }

        # 次选：同胜负方，总分随大小分方向偏移
        dx_rec = dx.get('recommendation')
        offset = 4 if dx_rec == '大分' else (-4 if dx_rec == '小分' else 0)
        sh, sa = _build_score(expected_total + offset, exp_margin)
        secondary = {
            'type': '次选', 'score': f"{sh}-{sa}",
            'home': sh, 'away': sa, 'result': fav, 'probability': None,
            'projected': True,
        }

        # 防冷：反向方小胜（约 3 分）
        if fav == '主胜':
            uh = max(0, int(round((expected_total - 3) / 2)))
            ua = max(0, int(round(expected_total)) - uh)
            upick = {'type': '防冷', 'score': f"{uh}-{ua}", 'home': uh, 'away': ua, 'result': '客胜', 'probability': p_away}
        else:
            ua = max(0, int(round((expected_total - 3) / 2)))
            uh = max(0, int(round(expected_total)) - ua)
            upick = {'type': '防冷', 'score': f"{uh}-{ua}", 'home': uh, 'away': ua, 'result': '主胜', 'probability': p_home}

        # 只有胜负接近时才给防冷，避免强信号比赛也固定输出反向比分。
        score_picks = [primary, secondary]
        if upset['alert']:
            upick['probability'] = None
            upick['projected'] = True
            score_picks.append(upick)

        # ---- 总分方向 ----
        over_p = dx.get('over_prob')
        under_p = dx.get('under_prob')
        ou_dir = '大分' if (over_p or 0) >= (under_p or 0) else '小分'
        ou_p = max(over_p or 0, under_p or 0)
        total_read = {
            'expected': expected_total,
            'line': line_val,
            'over_prob': over_p,
            'under_prob': under_p,
            'direction': ou_dir,
            'direction_prob': ou_p,
        }

        # ---- 理由 ----
        reasons = []
        reasons.append(f"模型隐含分差 {exp_margin:+.1f}（主队视角）")
        bb = spf.get('bb_features') or {}
        if bb.get('home_streak') is not None:
            reasons.append(f"主队近况连胜倾向 {bb.get('home_streak'):+.2f}、客队背靠 {bb.get('road_streak'):+.2f}")
        if dx.get('league_avg'):
            reasons.append(f"联赛场均总分 {dx.get('league_avg')}，盘口线 {line_val}")
        if spf.get('confidence'):
            reasons.append(f"胜负置信度：{spf.get('confidence')}")
        upset_alert = upset['alert']
        if upset_alert:
            reasons.append(f"⚠️ {upset['label']}：{'；'.join(upset['signals'])}，关注反向比分")

        if fav == '主胜':
            verdict = f"{match.get('home', '主队')}胜面最高 {fav_p:.0%}，优势{'明显' if margin >= 0.12 else '有限'}"
        else:
            verdict = f"{match.get('away', '客队')}胜面最高 {fav_p:.0%}，优势{'明显' if margin >= 0.12 else '有限'}"

        decision = build_decision(
            win_probs, confidence=spf.get('confidence'),
            upset_alert=upset_alert, min_single=0.58, min_margin=0.12,
        )
        return {
            'analysis_model': LOCAL_ANALYST_VERSION,
            'verdict': verdict,
            'favorite': fav,
            'favorite_prob': fav_p,
            'margin': margin,
            'win_prob': {'home': p_home, 'away': p_away},
            'score_picks': score_picks,
            'total': total_read,
            'reasons': reasons,
            'confidence': spf.get('confidence'),
            'upset_alert': upset_alert,
            'upset': upset,
            'decision': decision,
        }
    except Exception as e:
        log.warning(f"build_basketball_analysis 失败: {e}")
        return None


def generate_basketball_recommendations(date=None, bet_types=None, source='okooo'):
    """
    生成篮球预测推荐

    参数:
        date: 日期字符串，默认今天
        bet_types: 预测类型列表，默认 ['spf', 'rqspf', 'dx']
        source: 数据源，默认 'okooo'（可 '500'）
    """
    if date is None:
        date = time.strftime('%Y-%m-%d')

    if bet_types is None:
        bet_types = ['spf', 'rqspf', 'dx']

    matches = fetch_basketball_schedule(date, source=source)
    actual_source = (matches[0].get('source') if matches else source) or source

    # 澳客：并行抓取各家赔率
    market_cache = {}
    if actual_source == 'okooo' and matches:
        try:
            from .okooo import prefetch_market_bundles
            market_cache = prefetch_market_bundles(
                [m.get('okooo_id') or m.get('id') for m in matches]
            )
        except Exception as e:
            log.warning(f"各家赔率预取失败: {e}")

    results = []
    for match in matches:
        mid = str(match.get('okooo_id') or match.get('id') or '')
        market_bundle = market_cache.get(mid)
        result = {
            'match': match,
            'spf': None,
            'rqspf': None,
            'dx': None,
            'market_bundle': {
                'ml': (market_bundle or {}).get('ml', {}).get('available'),
                'ah': (market_bundle or {}).get('ah', {}).get('available'),
                'ou': (market_bundle or {}).get('ou', {}).get('available'),
            } if market_bundle else None,
        }

        if 'spf' in bet_types:
            result['spf'] = analyze_spf(match, market_bundle)

        if 'rqspf' in bet_types:
            result['rqspf'] = analyze_rqspf(match, market_bundle)

        if 'dx' in bet_types:
            result['dx'] = analyze_daxiao(match, market_bundle)

        analysis = build_basketball_analysis(result)
        if analysis:
            result['analysis'] = analysis

        results.append(result)

    # 保存预测记录
    try:
        from .records import save_predictions
        save_predictions(date, results, version=BASKETBALL_VERSION)
    except Exception as e:
        log.warning(f"保存预测记录失败: {e}")

    return {
        'date': date,
        'count': len(results),
        'results': results,
        'version': BASKETBALL_VERSION,
        'source': actual_source,
    }



# ==================== 价值投注筛选 ====================

def find_value_bets(results, threshold=0.05):
    """
    筛选价值投注机会

    要求模型相对市场有足够 edge，且 ELO 样本不是完全冷启动
    （否则 edge 多是噪声）。
    """
    value_bets = []
    for r in results:
        match = r['match']

        # SPF 价值投注
        if r['spf'] and r['spf']['available']:
            spf = r['spf']
            if float(spf.get('elo_trust', 0) or 0) < 0.25 and spf.get('confidence') == 'low':
                pass
            else:
                model_home_p = spf.get('home_prob', 0.5)
                market_home_p = spf.get('market_home_prob', model_home_p)
                model_pick = model_home_p if spf['recommendation'] == '主胜' else (1 - model_home_p)
                market_pick = market_home_p if spf['recommendation'] == '主胜' else (1 - market_home_p)
                edge = model_pick - market_pick

                if edge > threshold:
                    value_bets.append({
                        'type': '胜负',
                        'match': f"{match['home']} vs {match['away']}",
                        'recommendation': spf['recommendation'],
                        'edge': round(edge, 4),
                        'prob': round(model_pick, 4),
                        'confidence': spf.get('confidence', ''),
                        'elo_trust': spf.get('elo_trust'),
                    })

        # RQSPF 价值投注
        if r['rqspf'] and r['rqspf']['available']:
            rqspf = r['rqspf']
            if float(rqspf.get('elo_trust', 0) or 0) < 0.25 and rqspf.get('confidence') == 'low':
                pass
            else:
                model_home_p = rqspf.get('home_prob', 0.5)
                market_home_p = rqspf.get('market_home_prob', model_home_p)
                model_pick = model_home_p if rqspf['recommendation'] == '让胜' else (1 - model_home_p)
                market_pick = market_home_p if rqspf['recommendation'] == '让胜' else (1 - market_home_p)
                edge = model_pick - market_pick

                if edge > threshold:
                    value_bets.append({
                        'type': '让分胜负',
                        'match': f"{match['home']} vs {match['away']} ({match.get('handicap', '')})",
                        'recommendation': rqspf['recommendation'],
                        'edge': round(edge, 4),
                        'prob': round(model_pick, 4),
                        'confidence': rqspf.get('confidence', ''),
                        'elo_trust': rqspf.get('elo_trust'),
                    })

        # DX 价值投注
        if r['dx'] and r['dx']['available']:
            dx = r['dx']
            if float(dx.get('elo_trust', 0) or 0) < 0.25 and dx.get('confidence') == 'low':
                pass
            else:
                model_over_p = dx.get('over_prob', 0.5)
                market_over_p = dx.get('market_over_prob', model_over_p)
                model_pick = model_over_p if dx['recommendation'] == '大分' else (1 - model_over_p)
                market_pick = market_over_p if dx['recommendation'] == '大分' else (1 - market_over_p)
                edge = model_pick - market_pick

                if edge > threshold:
                    value_bets.append({
                        'type': '大小分',
                        'match': f"{match['home']} vs {match['away']} (总分{match.get('total_line', '')})",
                        'recommendation': dx['recommendation'],
                        'edge': round(edge, 4),
                        'prob': round(model_pick, 4),
                        'confidence': dx.get('confidence', ''),
                        'elo_trust': dx.get('elo_trust'),
                    })

    value_bets.sort(key=lambda x: -x['edge'])
    return value_bets[:20]


# ==================== 历史记录与统计 ====================

def summarize_basketball_history(limit=50):
    """获取篮球预测历史摘要"""
    try:
        from .records import get_predictions, get_prediction_stats
        predictions = get_predictions(limit=limit)
        stats = get_prediction_stats()
        return {
            'predictions': predictions,
            'stats': stats,
            'total': len(predictions),
        }
    except Exception as e:
        log.warning(f"获取历史记录失败: {e}")
        history = kv_store.load(BASKETBALL_HISTORY_KEY, [])
        if isinstance(history, list):
            return history[-limit:]
        return []
