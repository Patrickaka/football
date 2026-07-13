#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
篮球 ELO 实力评分系统
======================
专为篮球设计的 ELO 评分，核心改进点：

1. **胜负分差修正 (Margin of Victory)**:
   篮球比分差距远大于足球，ELO 更新需考虑分差。
   分差越大，K 因子放大越多，但有上限避免噪声。

2. **主客场优势**: 篮球主场优势约 +60 ELO（高于足球的 +50）

3. **联赛差异化 K 因子**: NBA/CBA/国际赛不同权重

4. **连胜/连败动量修正**: 近期状态纳入预测

与 football/elo.py 同架构，持久化通过 kv_store（MySQL + JSON 降级）。
"""

import re
import math
import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime

from ..common import kv_store

logger = logging.getLogger(__name__)

# ==================== ELO 配置 ====================

BB_ELO_KEY = 'basketball_elo_ratings'
INITIAL_ELO = 1500
HOME_ADVANTAGE = 65  # 篮球主场优势更大
ELO_MIN = 1000
ELO_MAX = 2200

# K 因子配置（篮球比赛重要性）
BB_K_FACTORS = {
    'NBA': 28,
    'CBA': 25,
    'NCAAB': 30,       # 大学篮球波动更大
    'WNBA': 24,
    '美职女篮': 24,
    '欧洲篮球': 22,
    '欧篮联': 26,
    '国际赛': 30,
    '友谊赛': 18,
    '常规赛': 24,
    '季后赛': 35,       # 季后赛权重更高
    '总决赛': 40,
}

# 联赛平均得分（用于分差归一化）
LEAGUE_AVG_SCORE = {
    'NBA': 112,
    'CBA': 100,
    'NCAAB': 72,
    'WNBA': 82,
    '美职女篮': 82,
    '欧洲篮球': 80,
    '欧篮联': 80,
}

# 联赛权重系数（影响 K 因子）
BB_LEAGUE_WEIGHTS = {
    'NBA': 1.15,
    'CBA': 0.90,
    'NCAAB': 1.00,
    'WNBA': 0.85,
    '美职女篮': 0.85,
    '欧洲篮球': 0.95,
    '欧篮联': 1.05,
    '国际赛': 1.10,
    '友谊赛': 0.70,
}


def _sanitize_team_name(team_name: str) -> Optional[str]:
    """清理并验证球队名称"""
    if team_name is None:
        return None
    if not isinstance(team_name, str):
        team_name = str(team_name)
    team_name = team_name.strip()
    if not team_name:
        return None
    if len(team_name) > 100:
        team_name = team_name[:100]
    team_name = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', team_name)
    try:
        team_name.encode('utf-8').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    team_name = re.sub(r'[<>:"/\\|?*]', '_', team_name)
    return team_name if team_name else None


def _get_k_factor(league: str) -> float:
    """获取 K 因子"""
    if not isinstance(league, str):
        league = str(league) if league else ''
    for key, val in BB_K_FACTORS.items():
        if key in league:
            return val
    return 24.0  # 默认


def _get_league_weight(league: str) -> float:
    """获取联赛权重系数"""
    if not isinstance(league, str):
        league = str(league) if league else ''
    for key, val in BB_LEAGUE_WEIGHTS.items():
        if key in league:
            return val
    return 1.0


def _mov_multiplier(margin: float, league: str = '') -> float:
    """
    胜负分差修正因子 (Margin of Victory Multiplier)

    篮球的分差信息量比足球大得多：
    - 1-5分: 接近，修正小
    - 6-15分: 正常差距
    - 15+分: 大胜/大败，修正有上限

    公式: ln(1 + margin) * 校准系数，确保递减增长
    """
    abs_margin = abs(margin)
    if abs_margin < 1:
        return 1.0

    # 对数增长，避免大分差过度修正
    multiplier = math.log(1 + abs_margin) / math.log(1 + 5)  # 以5分为基准1.0

    # 联赛得分水平归一化：高得分联赛(NBA)的分差价值需打折
    avg_score = LEAGUE_AVG_SCORE.get(league, 90)
    normalization = 90.0 / avg_score  # NBA~0.80, CBA~0.90
    multiplier *= normalization

    # 限制在 [1.0, 2.5] 范围
    return max(1.0, min(2.5, multiplier))


class BasketballELORatingSystem:
    """
    篮球 ELO 评分系统

    特性：
    - 胜负分差修正
    - 主客场优势
    - 联赛差异化
    - 近期状态追踪
    """

    def __init__(self):
        self.ratings: Dict[str, float] = {}
        self.history: Dict[str, list] = {}
        self.recent_form: Dict[str, list] = {}  # 近5场胜负记录
        self._load()

    def _load(self):
        """从 kv_store 加载 ELO 数据"""
        try:
            data = kv_store.load(BB_ELO_KEY, {})
            if not isinstance(data, dict):
                data = {}
            self.ratings = data.get('ratings', {})
            self.history = data.get('history', {})
            self.recent_form = data.get('recent_form', {})

            if not isinstance(self.ratings, dict):
                self.ratings = {}
            if not isinstance(self.history, dict):
                self.history = {}
            if not isinstance(self.recent_form, dict):
                self.recent_form = {}

            self._clean_data()
            logger.info(f"篮球 ELO 已加载 {len(self.ratings)} 支球队")
        except Exception as e:
            logger.error(f"加载篮球 ELO 失败: {e}")
            self.ratings = {}
            self.history = {}
            self.recent_form = {}

    def _clean_data(self):
        """清理无效数据"""
        invalid = []
        for key, val in self.ratings.items():
            clean = _sanitize_team_name(key)
            if clean is None or clean != key:
                invalid.append(key)
                continue
            if not isinstance(val, (int, float)):
                invalid.append(key)
                continue
            if val < ELO_MIN - 100 or val > ELO_MAX + 100:
                self.ratings[key] = INITIAL_ELO
        for key in invalid:
            self.ratings.pop(key, None)
            self.history.pop(key, None)
            self.recent_form.pop(key, None)

    def _save(self):
        """保存到 kv_store"""
        try:
            kv_store.save(BB_ELO_KEY, {
                'ratings': self.ratings,
                'history': self.history,
                'recent_form': self.recent_form,
                'updated_at': datetime.now().isoformat(),
            })
        except Exception as e:
            logger.error(f"保存篮球 ELO 失败: {e}")

    def get_rating(self, team: str) -> float:
        """获取球队 ELO 评分"""
        clean = _sanitize_team_name(team)
        if clean is None:
            return INITIAL_ELO
        if clean not in self.ratings:
            self._init_team(clean)
        return self.ratings.get(clean, INITIAL_ELO)

    def _init_team(self, team: str):
        """初始化新球队"""
        self.ratings[team] = INITIAL_ELO
        self.history[team] = [{
            'rating': INITIAL_ELO,
            'date': datetime.now().isoformat(),
            'event': 'initialized'
        }]
        self.recent_form[team] = []
        self._save()

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """计算 A 队预期胜率"""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _form_factor(self, team: str) -> float:
        """
        近期状态因子

        基于近5场胜率：
        - 5胜: +0.04 (4%概率加成)
        - 0胜: -0.04
        - 混合: 线性插值
        """
        form = self.recent_form.get(team, [])
        if len(form) < 3:
            return 0.0
        recent = form[-5:]
        wins = sum(1 for x in recent if x > 0)
        win_rate = wins / len(recent)
        # 状态因子：偏离0.5越多，修正越大，但限制在±0.04
        return (win_rate - 0.5) * 0.08

    def update_ratings(self, home: str, away: str,
                       home_score: int, away_score: int,
                       league: str = 'NBA') -> Tuple[float, float]:
        """
        更新 ELO 评分

        参数:
            home: 主队
            away: 客队
            home_score: 主队得分
            away_score: 客队得分
            league: 联赛

        返回:
            (主队新评分, 客队新评分)
        """
        home_clean = _sanitize_team_name(home)
        away_clean = _sanitize_team_name(away)
        if home_clean is None or away_clean is None:
            return INITIAL_ELO, INITIAL_ELO

        try:
            home_score = int(home_score)
            away_score = int(away_score)
        except (TypeError, ValueError):
            return self.get_rating(home_clean), self.get_rating(away_clean)

        if home_score < 0 or away_score < 0:
            return self.get_rating(home_clean), self.get_rating(away_clean)

        home_rating = self.get_rating(home_clean)
        away_rating = self.get_rating(away_clean)

        # 主场优势
        home_eff = home_rating + HOME_ADVANTAGE

        # 预期胜率
        expected_home = self._expected_score(home_eff, away_rating)
        expected_away = 1.0 - expected_home

        # 实际结果
        if home_score > away_score:
            actual_home = 1.0
            actual_away = 0.0
        else:
            actual_home = 0.0
            actual_away = 1.0

        # 分差修正
        margin = home_score - away_score
        mov_mult = _mov_multiplier(margin, league)

        # K 因子
        k = _get_k_factor(league)
        league_w = _get_league_weight(league)

        # 近期状态修正（微调预期）
        home_form = self._form_factor(home_clean)
        away_form = self._form_factor(away_clean)
        expected_home_adj = max(0.05, min(0.95, expected_home + home_form - away_form))

        # 更新评分
        delta_home = k * league_w * mov_mult * (actual_home - expected_home_adj)
        delta_away = k * league_w * mov_mult * (actual_away - (1.0 - expected_home_adj))

        new_home = max(ELO_MIN, min(ELO_MAX, home_rating + delta_home))
        new_away = max(ELO_MIN, min(ELO_MAX, away_rating + delta_away))

        self.ratings[home_clean] = round(new_home, 2)
        self.ratings[away_clean] = round(new_away, 2)

        # 更新近期状态
        home_result = 1 if home_score > away_score else -1
        away_result = -home_result
        for team, result in [(home_clean, home_result), (away_clean, away_result)]:
            form = self.recent_form.setdefault(team, [])
            form.append(result)
            if len(form) > 10:
                self.recent_form[team] = form[-10:]

        # 记录历史
        now = datetime.now().isoformat()
        self.history.setdefault(home_clean, []).append({
            'rating': self.ratings[home_clean],
            'date': now,
            'event': f'vs {away_clean} {home_score}-{away_score}',
            'change': round(delta_home, 2)
        })
        self.history.setdefault(away_clean, []).append({
            'rating': self.ratings[away_clean],
            'date': now,
            'event': f'vs {home_clean} {away_score}-{home_score}',
            'change': round(delta_away, 2)
        })

        # 保留最近100条
        for team in [home_clean, away_clean]:
            if len(self.history[team]) > 100:
                self.history[team] = self.history[team][-100:]

        self._save()

        logger.info(
            f"篮球 ELO 更新: {home_clean} {home_rating:.1f}→{self.ratings[home_clean]:.1f}, "
            f"{away_clean} {away_rating:.1f}→{self.ratings[away_clean]:.1f} "
            f"({league}, {home_score}-{away_score}, MOV×{mov_mult:.2f})"
        )

        return self.ratings[home_clean], self.ratings[away_clean]

    def predict_win_prob(self, home: str, away: str,
                         league: str = 'NBA') -> Dict[str, float]:
        """
        预测胜负概率

        返回:
            {
                'home_prob': 主队胜率,
                'away_prob': 客队胜率,
                'home_rating': 主队ELO,
                'away_rating': 客队ELO,
                'rating_diff': 评分差,
                'home_form': 主队近期状态,
                'away_form': 客队近期状态,
            }
        """
        home_clean = _sanitize_team_name(home)
        away_clean = _sanitize_team_name(away)

        if home_clean is None or away_clean is None:
            return {
                'home_prob': 0.5, 'away_prob': 0.5,
                'home_rating': INITIAL_ELO, 'away_rating': INITIAL_ELO,
                'rating_diff': 0, 'home_form': 0.0, 'away_form': 0.0,
            }

        home_rating = self.get_rating(home_clean)
        away_rating = self.get_rating(away_clean)

        home_eff = home_rating + HOME_ADVANTAGE
        expected_home = self._expected_score(home_eff, away_rating)

        # 近期状态修正
        home_form = self._form_factor(home_clean)
        away_form = self._form_factor(away_clean)
        home_prob = max(0.01, min(0.99, expected_home + home_form - away_form))

        return {
            'home_prob': round(home_prob, 4),
            'away_prob': round(1.0 - home_prob, 4),
            'home_rating': round(home_rating, 2),
            'away_rating': round(away_rating, 2),
            'rating_diff': round(home_rating - away_rating, 2),
            'home_form': round(home_form, 4),
            'away_form': round(away_form, 4),
        }

    def predict_total_score(self, home: str, away: str,
                            league: str = 'NBA') -> Dict[str, float]:
        """
        预测比赛总得分（用于大小分）

        基于 ELO 差距和联赛平均得分估算：
        - 双方 ELO 越高 → 得分越多
        - 防守强的对手会压制得分
        - 联赛基准得分

        返回:
            {
                'expected_total': 预期总得分,
                'expected_home_score': 主队预期得分,
                'expected_away_score': 客队预期得分,
            }
        """
        home_clean = _sanitize_team_name(home)
        away_clean = _sanitize_team_name(away)

        home_rating = self.get_rating(home_clean or home)
        away_rating = self.get_rating(away_clean or away)

        # avg_score 是单队场均得分（如 NBA ≈ 112）
        avg_score = LEAGUE_AVG_SCORE.get(league, 90)

        # ELO 高于平均 → 得分能力更强（每100 ELO 差约2分）
        home_offense = avg_score + (home_rating - INITIAL_ELO) * 0.02
        away_offense = avg_score + (away_rating - INITIAL_ELO) * 0.02

        # 防守修正：对手 ELO 越高，自己得分越少（每100 ELO 差约-1分）
        home_def_adj = (away_rating - INITIAL_ELO) * 0.01
        away_def_adj = (home_rating - INITIAL_ELO) * 0.01

        # 确保最小区间合理
        min_score = max(40, avg_score * 0.55)
        expected_home = max(min_score, home_offense - home_def_adj)
        expected_away = max(min_score, away_offense - away_def_adj)

        expected_total = expected_home + expected_away

        return {
            'expected_total': round(expected_total, 1),
            'expected_home_score': round(expected_home, 1),
            'expected_away_score': round(expected_away, 1),
        }

    def predict_margin(self, home: str, away: str,
                       league: str = 'NBA') -> Dict[str, float]:
        """
        预测让分盘口下的分差

        返回:
            {
                'expected_margin': 预期主队净胜分(正=主胜, 负=客胜),
                'home_win_prob': 主队胜率,
            }
        """
        prob_data = self.predict_win_prob(home, away, league)
        home_rating = prob_data['home_rating']
        away_rating = prob_data['away_rating']

        # ELO 差距 → 预期分差
        # 经验公式: 每 100 ELO 差 ≈ 6-7 分
        rating_diff = (home_rating + HOME_ADVANTAGE) - away_rating
        expected_margin = rating_diff * 0.065

        # 近期状态修正
        form_margin = (prob_data['home_form'] - prob_data['away_form']) * 20
        expected_margin += form_margin

        return {
            'expected_margin': round(expected_margin, 1),
            'home_win_prob': prob_data['home_prob'],
        }

    def get_team_info(self, team: str) -> Dict:
        """获取球队完整信息"""
        clean = _sanitize_team_name(team)
        if clean is None:
            return {'rating': INITIAL_ELO, 'form': [], 'history': []}
        return {
            'rating': self.get_rating(clean),
            'form': self.recent_form.get(clean, []),
            'history': self.history.get(clean, [])[-10:],
        }

    def get_top_teams(self, limit: int = 20) -> List[Dict]:
        """获取评分排名"""
        sorted_teams = sorted(self.ratings.items(), key=lambda x: -x[1])
        return [{'team': t, 'rating': round(r, 2)} for t, r in sorted_teams[:limit]]


# ==================== 全局实例 ====================

_elo_system: Optional[BasketballELORatingSystem] = None


def get_elo_system() -> BasketballELORatingSystem:
    """获取全局 ELO 系统实例（单例）"""
    global _elo_system
    if _elo_system is None:
        _elo_system = BasketballELORatingSystem()
    return _elo_system


def reload_elo_system():
    """重新加载 ELO 系统（数据更新后调用）"""
    global _elo_system
    _elo_system = BasketballELORatingSystem()
    return _elo_system
