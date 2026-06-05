#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
球队 ELO 实力评分系统
用于衡量球队长期真实实力，参与进球期望值和比分预测计算

容错处理：
- JSON 损坏处理
- 文件不存在处理
- 球队名称为空处理
- 球队名称编码异常处理
- 新球队首次出现自动初始化

ELO 算法说明：
- 初始评分：1500
- K因子：根据比赛重要性动态调整
- 主场优势：+50分
- 联赛系数：不同联赛有不同权重
"""

import os
import json
import logging
import re
from typing import Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# ELO 配置
from ..common.paths import data_path
ELO_FILE = data_path('elo_ratings.json')
INITIAL_ELO = 1500
HOME_ADVANTAGE = 50

# K因子配置（根据比赛重要性）
K_FACTORS = {
    '友谊赛': 20,
    '联赛': 25,
    '杯赛': 30,
    '洲际杯': 35,
    '世界杯': 40,
}

# 联赛权重系数
LEAGUE_WEIGHTS = {
    '英超': 1.1,
    '西甲': 1.1,
    '德甲': 1.05,
    '意甲': 1.05,
    '法甲': 1.0,
    '中超': 0.8,
    '欧冠': 1.2,
    '欧联杯': 1.1,
    '世界杯': 1.3,
    '欧洲杯': 1.25,
    '友谊赛': 0.7,
}


class ELORatingSystem:
    """
    ELO 评分系统类
    具备完善的容错处理能力
    """
    
    def __init__(self, elo_file: str = ELO_FILE):
        """
        初始化 ELO 系统
        
        参数:
            elo_file: ELO 数据存储文件路径
        """
        self.elo_file = elo_file
        self.ratings: Dict[str, float] = {}
        self.history: Dict[str, list] = {}  # 存储每个球队的历史评分记录
        self._load_ratings()
    
    def _sanitize_team_name(self, team_name: str) -> Optional[str]:
        """
        清理并验证球队名称
        
        参数:
            team_name: 原始球队名称
        
        返回:
            清理后的球队名称，无效则返回 None
        """
        if team_name is None:
            logger.warning("球队名称为 None")
            return None
        
        # 转换为字符串
        if not isinstance(team_name, str):
            team_name = str(team_name)
        
        # 去除空白字符
        team_name = team_name.strip()
        
        # 检查是否为空
        if not team_name:
            logger.warning("球队名称为空")
            return None
        
        # 检查长度
        if len(team_name) > 100:
            logger.warning(f"球队名称过长: {len(team_name)} 字符")
            team_name = team_name[:100]
        
        # 检查是否包含非法字符（只允许中文、英文、数字和常见符号）
        # 移除不可打印字符
        team_name = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', team_name)
        
        # 验证编码（尝试编码为 UTF-8）
        try:
            team_name.encode('utf-8').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            logger.error(f"球队名称编码异常: {e}")
            return None
        
        # 替换特殊字符
        team_name = re.sub(r'[<>:"/\\|?*]', '_', team_name)
        
        return team_name if team_name else None
    
    def _load_ratings(self):
        """
        从文件加载 ELO 评分（容错版本）
        """
        # 文件不存在时初始化空数据
        if not os.path.exists(self.elo_file):
            logger.info(f"ELO 文件不存在，将创建新数据: {self.elo_file}")
            self.ratings = {}
            self.history = {}
            return
        
        try:
            with open(self.elo_file, 'r', encoding='utf-8') as f:
                # 读取文件内容
                content = f.read()
                
                # 检查内容是否为空
                if not content.strip():
                    logger.warning("ELO 文件为空，初始化空数据")
                    self.ratings = {}
                    self.history = {}
                    return
                
                # 解析 JSON
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 解析失败（文件损坏）: {e}")
                    # 尝试修复或重建
                    self._recover_from_corrupted_file(content)
                    return
                
                # 验证数据结构
                if not isinstance(data, dict):
                    logger.error("ELO 文件数据结构错误，应为字典")
                    self.ratings = {}
                    self.history = {}
                    return
                
                # 提取数据
                self.ratings = data.get('ratings', {})
                self.history = data.get('history', {})
                
                # 验证 ratings 格式
                if not isinstance(self.ratings, dict):
                    logger.error("ratings 字段格式错误，应为字典")
                    self.ratings = {}
                
                # 验证 history 格式
                if not isinstance(self.history, dict):
                    logger.error("history 字段格式错误，应为字典")
                    self.history = {}
                
                # 清理无效数据
                self._clean_invalid_data()
                
                logger.info(f"已加载 {len(self.ratings)} 支球队的 ELO 评分")
                
        except OSError as e:
            logger.error(f"读取 ELO 文件失败: {e}")
            self.ratings = {}
            self.history = {}
        except Exception as e:
            logger.error(f"加载 ELO 评分时发生未知错误: {e}")
            self.ratings = {}
            self.history = {}
    
    def _recover_from_corrupted_file(self, content: str):
        """
        尝试从损坏的 JSON 文件中恢复数据
        
        参数:
            content: 文件内容
        """
        logger.info("尝试从损坏的 JSON 文件中恢复数据")
        
        try:
            # 尝试提取有效的 JSON 片段
            # 查找第一个 { 和最后一个 }
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = content[start_idx:end_idx+1]
                try:
                    data = json.loads(json_str)
                    self.ratings = data.get('ratings', {})
                    self.history = data.get('history', {})
                    logger.info("成功从损坏文件中恢复部分数据")
                    # 立即保存修复后的数据
                    self._save_ratings()
                    return
                except json.JSONDecodeError:
                    logger.warning("无法从损坏文件中提取有效 JSON")
            
            # 无法恢复，初始化空数据
            logger.warning("无法恢复损坏的文件，初始化空数据")
            self.ratings = {}
            self.history = {}
            
        except Exception as e:
            logger.error(f"恢复损坏文件失败: {e}")
            self.ratings = {}
            self.history = {}
    
    def _clean_invalid_data(self):
        """
        清理无效数据
        """
        # 清理 ratings 中的无效值
        invalid_keys = []
        for key, value in self.ratings.items():
            # 验证球队名称
            clean_key = self._sanitize_team_name(key)
            if clean_key is None or clean_key != key:
                invalid_keys.append(key)
                continue
            
            # 验证评分值
            if not isinstance(value, (int, float)) or not isinstance(value, (int, float)):
                invalid_keys.append(key)
                continue
            
            # 验证评分范围（合理范围 1000-2000）
            if value < 1000 or value > 2000:
                logger.warning(f"球队 {key} 的评分 {value} 超出合理范围，重置为初始值")
                self.ratings[key] = INITIAL_ELO
        
        # 删除无效条目
        for key in invalid_keys:
            logger.warning(f"删除无效球队数据: {key}")
            del self.ratings[key]
        
        # 清理 history 中的无效数据
        for team in list(self.history.keys()):
            clean_team = self._sanitize_team_name(team)
            if clean_team is None or clean_team != team:
                del self.history[team]
                continue
            
            if not isinstance(self.history[team], list):
                self.history[team] = []
                continue
            
            # 清理无效历史记录
            self.history[team] = [
                record for record in self.history[team]
                if isinstance(record, dict) and 'rating' in record and 'date' in record
            ]
    
    def _save_ratings(self):
        """
        保存 ELO 评分到文件（容错版本）
        """
        data = {
            'ratings': self.ratings,
            'history': self.history,
            'updated_at': __import__('datetime').datetime.now().isoformat()
        }
        
        try:
            # 先写入临时文件
            temp_file = self.elo_file + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 原子替换原文件
            if os.path.exists(self.elo_file):
                os.remove(self.elo_file)
            os.rename(temp_file, self.elo_file)
            
            logger.debug(f"ELO 评分已保存")
            
        except OSError as e:
            logger.error(f"保存 ELO 评分失败: {e}")
            # 尝试清理临时文件
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        except Exception as e:
            logger.error(f"保存 ELO 评分时发生未知错误: {e}")
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    def get_rating(self, team_name: str) -> float:
        """
        获取球队 ELO 评分（容错版本）
        
        参数:
            team_name: 球队名称
        
        返回:
            ELO 评分值（如果球队无效返回初始评分）
        """
        # 清理球队名称
        clean_name = self._sanitize_team_name(team_name)
        
        if clean_name is None:
            logger.warning(f"无法获取无效球队的评分: {team_name}")
            return INITIAL_ELO
        
        # 缺失球队自动初始化
        if clean_name not in self.ratings:
            self._initialize_team(clean_name)
        
        return self.ratings.get(clean_name, INITIAL_ELO)
    
    def _initialize_team(self, team_name: str):
        """
        初始化新球队的 ELO 评分（容错版本）
        
        参数:
            team_name: 球队名称（已清理）
        """
        try:
            self.ratings[team_name] = INITIAL_ELO
            self.history[team_name] = [{
                'rating': INITIAL_ELO,
                'date': __import__('datetime').datetime.now().isoformat(),
                'event': 'initialized'
            }]
            logger.info(f"初始化新球队 ELO: {team_name} = {INITIAL_ELO}")
            self._save_ratings()
        except Exception as e:
            logger.error(f"初始化球队失败: {team_name}, 错误: {e}")
    
    def _get_expected_score(self, rating1: float, rating2: float) -> float:
        """
        计算预期得分概率
        
        参数:
            rating1: 球队1评分
            rating2: 球队2评分
        
        返回:
            球队1的预期得分概率
        """
        return 1 / (1 + 10 ** ((rating2 - rating1) / 400))
    
    def _get_k_factor(self, league_type: str) -> int:
        """
        获取 K 因子
        
        参数:
            league_type: 联赛类型
        
        返回:
            K 因子值
        """
        if not isinstance(league_type, str):
            league_type = str(league_type)
        
        for key in K_FACTORS:
            if key in league_type:
                return K_FACTORS[key]
        return K_FACTORS['联赛']  # 默认值
    
    def _get_league_weight(self, league_type: str) -> float:
        """
        获取联赛权重系数
        
        参数:
            league_type: 联赛类型
        
        返回:
            权重系数
        """
        if not isinstance(league_type, str):
            league_type = str(league_type)
        
        for key in LEAGUE_WEIGHTS:
            if key in league_type:
                return LEAGUE_WEIGHTS[key]
        return 1.0  # 默认值
    
    def update_ratings(self, home_team: str, away_team: str, 
                       home_score: int, away_score: int, 
                       league_type: str = '联赛') -> Tuple[float, float]:
        """
        更新球队 ELO 评分（容错版本）
        
        参数:
            home_team: 主队名称
            away_team: 客队名称
            home_score: 主队进球数
            away_score: 客队进球数
            league_type: 联赛类型
        
        返回:
            (主队新评分, 客队新评分)
        """
        # 清理球队名称
        home_clean = self._sanitize_team_name(home_team)
        away_clean = self._sanitize_team_name(away_team)
        
        if home_clean is None:
            logger.error(f"主队名称无效: {home_team}")
            return INITIAL_ELO, INITIAL_ELO
        
        if away_clean is None:
            logger.error(f"客队名称无效: {away_team}")
            return INITIAL_ELO, INITIAL_ELO
        
        # 验证比分
        if not isinstance(home_score, int) or home_score < 0:
            logger.warning(f"主队比分无效: {home_score}，设为 0")
            home_score = 0
        
        if not isinstance(away_score, int) or away_score < 0:
            logger.warning(f"客队比分无效: {away_score}，设为 0")
            away_score = 0
        
        try:
            # 获取当前评分
            home_rating = self.get_rating(home_clean)
            away_rating = self.get_rating(away_clean)
            
            # 考虑主场优势
            home_rating_with_advantage = home_rating + HOME_ADVANTAGE
            
            # 计算预期得分
            expected_home = self._get_expected_score(home_rating_with_advantage, away_rating)
            expected_away = 1 - expected_home
            
            # 确定实际得分
            if home_score > away_score:
                actual_home = 1.0
                actual_away = 0.0
            elif home_score == away_score:
                actual_home = 0.5
                actual_away = 0.5
            else:
                actual_home = 0.0
                actual_away = 1.0
            
            # 获取 K 因子和联赛权重
            k_factor = self._get_k_factor(league_type)
            league_weight = self._get_league_weight(league_type)
            
            # 计算新评分
            new_home_rating = home_rating + k_factor * league_weight * (actual_home - expected_home)
            new_away_rating = away_rating + k_factor * league_weight * (actual_away - expected_away)
            
            # 限制评分范围
            new_home_rating = max(1000, min(2000, new_home_rating))
            new_away_rating = max(1000, min(2000, new_away_rating))
            
            # 更新评分
            self.ratings[home_clean] = round(new_home_rating, 2)
            self.ratings[away_clean] = round(new_away_rating, 2)
            
            # 记录历史
            now = __import__('datetime').datetime.now().isoformat()
            self.history[home_clean].append({
                'rating': self.ratings[home_clean],
                'date': now,
                'event': f'vs {away_clean} {home_score}-{away_score}',
                'change': round(new_home_rating - home_rating, 2)
            })
            self.history[away_clean].append({
                'rating': self.ratings[away_clean],
                'date': now,
                'event': f'vs {home_clean} {away_score}-{home_score}',
                'change': round(new_away_rating - away_rating, 2)
            })
            
            # 保留最近 100 条历史记录
            if len(self.history[home_clean]) > 100:
                self.history[home_clean] = self.history[home_clean][-100:]
            if len(self.history[away_clean]) > 100:
                self.history[away_clean] = self.history[away_clean][-100:]
            
            # 保存到文件
            self._save_ratings()
            
            logger.info(f"ELO 更新: {home_clean} {home_rating:.2f}→{self.ratings[home_clean]:.2f}, "
                        f"{away_clean} {away_rating:.2f}→{self.ratings[away_clean]:.2f} "
                        f"(联赛: {league_type}, 赛果: {home_score}-{away_score})")
            
            return self.ratings[home_clean], self.ratings[away_clean]
        
        except Exception as e:
            logger.error(f"更新 ELO 评分失败: {e}")
            return INITIAL_ELO, INITIAL_ELO
    
    def predict_match(self, home_team: str, away_team: str, 
                      league_type: str = '联赛') -> Dict[str, float]:
        """
        预测比赛结果概率（容错版本）
        
        参数:
            home_team: 主队名称
            away_team: 客队名称
            league_type: 联赛类型
        
        返回:
            包含主胜、平局、客胜概率的字典
        """
        # 清理球队名称
        home_clean = self._sanitize_team_name(home_team)
        away_clean = self._sanitize_team_name(away_team)
        
        if home_clean is None or away_clean is None:
            logger.warning(f"无法预测无效球队的比赛: {home_team} vs {away_team}")
            return {
                'home': 0.333,
                'draw': 0.334,
                'away': 0.333,
                'home_rating': INITIAL_ELO,
                'away_rating': INITIAL_ELO,
                'rating_diff': 0
            }
        
        try:
            home_rating = self.get_rating(home_clean)
            away_rating = self.get_rating(away_clean)
            
            # 考虑主场优势
            home_rating_with_advantage = home_rating + HOME_ADVANTAGE
            
            # 计算预期得分
            expected_home = self._get_expected_score(home_rating_with_advantage, away_rating)
            
            # 转换为概率
            rating_diff = home_rating - away_rating
            draw_prob = max(0.1, min(0.4, 0.25 - 0.0005 * abs(rating_diff)))
            
            home_win_prob = expected_home * (1 - draw_prob)
            away_win_prob = (1 - expected_home) * (1 - draw_prob)
            
            return {
                'home': round(home_win_prob, 4),
                'draw': round(draw_prob, 4),
                'away': round(away_win_prob, 4),
                'home_rating': home_rating,
                'away_rating': away_rating,
                'rating_diff': round(rating_diff, 2)
            }
        
        except Exception as e:
            logger.error(f"预测比赛失败: {e}")
            return {
                'home': 0.333,
                'draw': 0.334,
                'away': 0.333,
                'home_rating': INITIAL_ELO,
                'away_rating': INITIAL_ELO,
                'rating_diff': 0
            }
    
    def get_team_history(self, team_name: str, limit: int = 20) -> list:
        """
        获取球队历史评分记录（容错版本）
        
        参数:
            team_name: 球队名称
            limit: 返回记录条数
        
        返回:
            历史记录列表
        """
        # 清理球队名称
        clean_name = self._sanitize_team_name(team_name)
        
        if clean_name is None:
            logger.warning(f"无法获取无效球队的历史记录: {team_name}")
            return []
        
        if clean_name not in self.history:
            return []
        
        # 验证 limit
        if not isinstance(limit, int) or limit <= 0:
            limit = 20
        
        return self.history[clean_name][-limit:]
    
    def get_top_teams(self, limit: int = 10) -> list:
        """
        获取评分最高的球队（容错版本）
        
        参数:
            limit: 返回球队数量
        
        返回:
            球队评分列表（降序）
        """
        try:
            # 验证 limit
            if not isinstance(limit, int) or limit <= 0:
                limit = 10
            
            sorted_teams = sorted(self.ratings.items(), key=lambda x: -x[1])
            return [{
                'team': team,
                'rating': round(rating, 2)
            } for team, rating in sorted_teams[:limit]]
        
        except Exception as e:
            logger.error(f"获取排名失败: {e}")
            return []
    
    def get_team_count(self) -> int:
        """
        获取球队数量
        
        返回:
            球队总数
        """
        return len(self.ratings)
    
    def reset_rating(self, team_name: str):
        """
        重置球队评分（容错版本）
        
        参数:
            team_name: 球队名称
        """
        # 清理球队名称
        clean_name = self._sanitize_team_name(team_name)
        
        if clean_name is None:
            logger.warning(f"无法重置无效球队的评分: {team_name}")
            return
        
        if clean_name in self.ratings:
            try:
                self.ratings[clean_name] = INITIAL_ELO
                self.history[clean_name] = [{
                    'rating': INITIAL_ELO,
                    'date': __import__('datetime').datetime.now().isoformat(),
                    'event': 'reset'
                }]
                self._save_ratings()
                logger.info(f"重置球队 ELO: {clean_name} = {INITIAL_ELO}")
            except Exception as e:
                logger.error(f"重置球队评分失败: {e}")
    
    def remove_team(self, team_name: str):
        """
        删除球队（容错版本）
        
        参数:
            team_name: 球队名称
        """
        # 清理球队名称
        clean_name = self._sanitize_team_name(team_name)
        
        if clean_name is None:
            logger.warning(f"无法删除无效球队: {team_name}")
            return
        
        if clean_name in self.ratings:
            try:
                del self.ratings[clean_name]
                if clean_name in self.history:
                    del self.history[clean_name]
                self._save_ratings()
                logger.info(f"删除球队: {clean_name}")
            except Exception as e:
                logger.error(f"删除球队失败: {e}")


# 全局 ELO 系统实例
_elo_system = None


def get_elo_system() -> ELORatingSystem:
    """
    获取全局 ELO 系统实例（单例模式，容错版本）
    
    返回:
        ELO 评分系统实例
    """
    global _elo_system
    
    if _elo_system is None:
        try:
            _elo_system = ELORatingSystem()
        except Exception as e:
            logger.error(f"创建 ELO 系统实例失败: {e}")
            # 返回一个空的系统作为降级方案
            _elo_system = ELORatingSystem()
    
    return _elo_system


def elo_to_goals_expected(elo_rating: float, opponent_elo: float) -> float:
    """
    将 ELO 评分转换为进球期望值 (xG)（容错版本）
    
    参数:
        elo_rating: 球队 ELO 评分
        opponent_elo: 对手 ELO 评分
    
    返回:
        进球期望值
    """
    try:
        # 验证输入
        if not isinstance(elo_rating, (int, float)):
            elo_rating = float(elo_rating) if elo_rating else INITIAL_ELO
        
        if not isinstance(opponent_elo, (int, float)):
            opponent_elo = float(opponent_elo) if opponent_elo else INITIAL_ELO
        
        # ELO 差距与进球期望的关系
        rating_diff = elo_rating - opponent_elo
        base_xg = 1.5  # 平均进球期望
        
        # 每 100 ELO 差距约对应 0.3 进球差异
        xg = base_xg + (rating_diff / 100) * 0.3
        
        # 限制范围
        return max(0.2, min(5.0, xg))
    
    except Exception as e:
        logger.error(f"计算 xG 失败: {e}")
        return 1.5


def elo_to_strength_factor(elo_rating: float, league_avg_elo: float = 1500) -> float:
    """
    将 ELO 评分转换为实力因子（容错版本）
    
    参数:
        elo_rating: 球队 ELO 评分
        league_avg_elo: 联赛平均 ELO（默认1500）
    
    返回:
        实力因子（1.0 为平均水平）
    """
    try:
        # 验证输入
        if not isinstance(elo_rating, (int, float)):
            elo_rating = float(elo_rating) if elo_rating else INITIAL_ELO
        
        if not isinstance(league_avg_elo, (int, float)):
            league_avg_elo = float(league_avg_elo) if league_avg_elo else INITIAL_ELO
        
        diff = elo_rating - league_avg_elo
        return 1.0 + (diff / 500)
    
    except Exception as e:
        logger.error(f"计算实力因子失败: {e}")
        return 1.0


# ==================== 测试函数 ====================

def test_error_handling():
    """
    测试容错处理功能
    """
    print("="*60)
    print("ELO 容错处理测试")
    print("="*60)
    
    # 测试 1: 损坏的 JSON 文件
    print("\n1. 测试损坏的 JSON 文件处理")
    if os.path.exists(ELO_FILE):
        os.rename(ELO_FILE, ELO_FILE + '.bak')
    
    # 创建损坏的文件
    with open(ELO_FILE, 'w', encoding='utf-8') as f:
        f.write('{"ratings": { "曼联": 1500, "利物浦":')  # 不完整的 JSON
    
    elo = ELORatingSystem()
    print(f"   损坏文件处理后球队数: {elo.get_team_count()}")
    
    # 测试 2: 空球队名称
    print("\n2. 测试空球队名称")
    result = elo.get_rating("")
    print(f"   空名称评分: {result}")
    
    # 测试 3: None 球队名称
    print("\n3. 测试 None 球队名称")
    result = elo.get_rating(None)
    print(f"   None 名称评分: {result}")
    
    # 测试 4: 编码异常
    print("\n4. 测试编码异常球队名称")
    try:
        bad_name = '球队\x00名称'  # 包含空字符
        result = elo.get_rating(bad_name)
        print(f"   异常名称评分: {result}")
    except Exception as e:
        print(f"   异常名称处理: 捕获异常 {type(e).__name__}")
    
    # 测试 5: 新球队首次出现
    print("\n5. 测试新球队首次出现")
    result = elo.get_rating("新球队测试")
    print(f"   新球队评分: {result}")
    print(f"   球队总数: {elo.get_team_count()}")
    
    # 测试 6: 无效比分
    print("\n6. 测试无效比分")
    h, a = elo.update_ratings("曼联", "利物浦", -1, "abc", "英超")
    print(f"   无效比分处理后: 曼联={h}, 利物浦={a}")
    
    # 测试 7: 恢复备份
    if os.path.exists(ELO_FILE + '.bak'):
        os.remove(ELO_FILE)
        os.rename(ELO_FILE + '.bak', ELO_FILE)
        print("\n7. 已恢复原始数据")
    
    print("\n" + "="*60)
    print("容错测试完成！")
    print("="*60)


if __name__ == '__main__':
    # 运行测试
    test_error_handling()