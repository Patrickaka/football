# -*- coding: utf-8 -*-
"""概率校准：Platt 缩放、保序回归、分层校准与三类校准因子。

纯计算——不读全局配置、不碰存储与时钟。联赛画像、历史样本、别名表都由
调用方传入；`src/football/calibrating.py` 负责取这些东西。
"""

import logging
import math

from .scoring_model import _matrix_margins, _sigmoid

log = logging.getLogger('domain.football.calibration')


def resolve_team_alias(name, alias_map):
    """按别名表把队名归一到标准名。

    这是 `normalize_team_name` 的纯计算部分——别名表从 `kv_store` 取，
    取的动作留在适配层。

    **匹配是三级的，而且第三级很宽**：先全等标准名或落在别名列表里，
    再做**双向包含**（`alias in name` 或 `name in alias`）。
    第三级会把「国安」匹配到任何含「国安」的标准名，也会把「北京国安足球俱乐部」
    匹配到别名「国安」。行为原样保留，用例把这个宽度钉住了。
    """
    if not name:
        return name
    name = name.strip()
    if not alias_map:
        return name
    for standard_name, aliases in alias_map.items():
        if name in aliases or name == standard_name:
            return standard_name
        for alias in aliases:
            if alias in name or name in alias:
                return standard_name
    return name


def platt_pairs_from_history(historical_data):
    """把历史比赛摊成 `(预测概率, 是否命中)` 对，供 Platt 拟合

    每场比赛贡献**整张比分矩阵**的格数——一场就是几十对。
    """
    pairs = []
    for match in historical_data:
        actual_score = (match['actual_home'], match['actual_away'])
        for (h, a), prob in match['predicted_probs'].items():
            pairs.append((prob, 1 if (h, a) == actual_score else 0))
    return pairs


def fit_platt_scaling(prob_pairs):
    """
    拟合 Platt 缩放参数。
    
    参数:
        prob_pairs: 列表，每个元素为 (model_prob, actual_outcome)
                    model_prob: 模型输出概率
                    actual_outcome: 实际结果（1=发生, 0=未发生）
    
    返回:
        (A, B): Platt 缩放参数，校准后概率 = sigmoid(A * p + B)
    """
    if len(prob_pairs) < 10:
        return (1.0, 0.0)  # 数据不足，返回恒等变换
    
    # 初始化参数
    A, B = 1.0, 0.0
    max_iter = 100
    learning_rate = 0.1
    
    for _ in range(max_iter):
        grad_A, grad_B = 0.0, 0.0
        for p, y in prob_pairs:
            sig = _sigmoid(A * p + B)
            grad_A += (sig - y) * p
            grad_B += (sig - y)
        
        A -= learning_rate * grad_A / len(prob_pairs)
        B -= learning_rate * grad_B / len(prob_pairs)
    
    return (A, B)


def calibrate_with_platt(matrix, calibration_data):
    """
    使用 Platt 缩放校准概率矩阵。
    
    参数:
        matrix: 原始概率矩阵 {(h, a): prob}
        calibration_data: 历史校准数据，包含 Platt 参数
    
    返回:
        校准后的概率矩阵
    """
    if not calibration_data or 'platt_params' not in calibration_data:
        return matrix
    
    A, B = calibration_data['platt_params']
    calibrated = {}
    for (h, a), prob in matrix.items():
        calibrated[(h, a)] = _sigmoid(A * prob + B)
    
    # 归一化
    total = sum(calibrated.values())
    if total > 0:
        calibrated = {cell: prob / total for cell, prob in calibrated.items()}
    
    return calibrated


def isotonic_regression_calibration(prob_pairs):
    """
    等渗回归校准（非参数方法）。
    
    参数:
        prob_pairs: 列表，每个元素为 (model_prob, actual_outcome)
    
    返回:
        校准函数，输入模型概率，输出校准后概率
    """
    if len(prob_pairs) < 5:
        return lambda p: p  # 数据不足，返回恒等函数
    
    # 按模型概率排序
    prob_pairs.sort(key=lambda x: x[0])
    
    n = len(prob_pairs)
    # 使用 PAV 算法（Pool Adjacent Violators）
    # 简化版本：分组并计算每组的平均实际概率
    groups = []
    current_group = [prob_pairs[0]]
    
    for i in range(1, n):
        current_mean = sum(p[1] for p in current_group) / len(current_group)
        next_mean = sum(p[1] for p in prob_pairs[i:i+1]) / 1
        
        if next_mean >= current_mean:
            current_group.append(prob_pairs[i])
        else:
            groups.append(current_group)
            current_group = [prob_pairs[i]]
    
    if current_group:
        groups.append(current_group)
    
    # 创建校准映射
    calib_map = {}
    for group in groups:
        mean_prob = sum(p[0] for p in group) / len(group)
        mean_outcome = sum(p[1] for p in group) / len(group)
        calib_map[mean_prob] = mean_outcome
    
    # 线性插值函数
    def calibrate(p):
        if not calib_map:
            return p
        
        sorted_probs = sorted(calib_map.keys())
        
        if p <= sorted_probs[0]:
            return calib_map[sorted_probs[0]]
        if p >= sorted_probs[-1]:
            return calib_map[sorted_probs[-1]]
        
        # 找到相邻的两个点
        for i in range(len(sorted_probs) - 1):
            if sorted_probs[i] <= p <= sorted_probs[i + 1]:
                # 线性插值
                t = (p - sorted_probs[i]) / (sorted_probs[i + 1] - sorted_probs[i])
                return (1 - t) * calib_map[sorted_probs[i]] + t * calib_map[sorted_probs[i + 1]]
        
        return p
    
    return calibrate


def calibrate_probabilities(matrix, method='platt', calibration_data=None):
    """
    概率校准主函数。
    
    参数:
        matrix: 原始概率矩阵 {(h, a): prob}
        method: 校准方法，'platt'、'isotonic' 或 'hierarchical'
        calibration_data: 历史校准数据
    
    返回:
        校准后的概率矩阵
    """
    if method == 'hierarchical':
        return hierarchical_calibration(matrix, calibration_data)
    elif method == 'platt':
        return calibrate_with_platt(matrix, calibration_data)
    elif method == 'isotonic':
        if calibration_data and 'prob_pairs' in calibration_data:
            calib_func = isotonic_regression_calibration(calibration_data['prob_pairs'])
            calibrated = {(h, a): calib_func(prob) for (h, a), prob in matrix.items()}
            total = sum(calibrated.values())
            if total > 0:
                return {cell: prob / total for cell, prob in calibrated.items()}
        return matrix
    else:
        return matrix


def hierarchical_calibration(matrix, calibration_data=None):
    """
    三级分层概率校准：比分层 → 进球层 → 胜平负层

    结构：
        比分层: 针对每个具体比分进行校准
        进球层: 针对总进球数进行校准（低进球、中等进球、高进球）
        胜平负层: 针对主胜/平局/客胜进行校准

    最终概率 = score_prob × draw_factor × goal_factor × score_factor

    参数:
        matrix: 原始概率矩阵 {(h, a): prob}
        calibration_data: 历史校准数据

    返回:
        校准后的概率矩阵
    """
    if not calibration_data:
        return matrix

    calibrated = {}

    # 1. 计算基础概率和边际概率
    margins = _matrix_margins(matrix)
    p_home = margins['home']
    p_draw = margins['draw']
    p_away = margins['away']

    # 2. 计算总进球分布
    goal_dist = {}
    for (h, a), prob in matrix.items():
        total = h + a
        goal_dist[total] = goal_dist.get(total, 0) + prob

    # 3. 获取校准因子
    # 胜平负校准因子
    draw_factor = _get_draw_calibration_factor(calibration_data, p_draw)

    for (h, a), prob in matrix.items():
        total_goals = h + a

        # 进球层校准因子（低进球<3, 中进球3-5, 高进球>5）
        goal_factor = _get_goal_calibration_factor(calibration_data, total_goals, goal_dist)

        # 比分层校准因子
        score_factor = _get_score_calibration_factor(calibration_data, h, a, prob)

        # 应用三级校准
        calibrated[(h, a)] = prob * draw_factor * goal_factor * score_factor

    # 归一化
    total = sum(calibrated.values())
    if total > 0:
        calibrated = {cell: prob / total for cell, prob in calibrated.items()}

    log.info("三级分层校准完成")
    return calibrated


def _get_draw_calibration_factor(calibration_data, p_draw):
    """
    获取平局校准因子。

    参数:
        calibration_data: 历史校准数据
        p_draw: 当前预测的平局概率

    返回:
        平局校准因子
    """
    draw_calib = calibration_data.get('draw_calibration', {})

    # 基于历史数据计算校准因子
    # 如果历史平局概率被低估/高估，调整因子
    expected_draw = draw_calib.get('expected_draw_rate', 0.25)
    actual_draw = draw_calib.get('actual_draw_rate', 0.25)

    if actual_draw == 0:
        return 1.0

    # 校准因子 = 实际平局率 / 期望平局率
    # 但需要平滑处理，避免极端值
    base_factor = actual_draw / expected_draw

    # 应用概率相关的调整
    # 如果预测概率偏离历史均值，适当调整
    draw_prob_factor = 1.0
    if p_draw > 0:
        # 如果预测平局概率高于平均，适当下调（防止过度自信）
        if p_draw > expected_draw:
            draw_prob_factor = 0.95 + (p_draw - expected_draw) * 0.1

    return min(max(base_factor * draw_prob_factor, 0.7), 1.3)


def _get_goal_calibration_factor(calibration_data, total_goals, goal_dist):
    """
    获取进球数校准因子。

    参数:
        calibration_data: 历史校准数据
        total_goals: 总进球数
        goal_dist: 当前预测的进球分布

    返回:
        进球数校准因子
    """
    goal_calib = calibration_data.get('goal_calibration', {})

    # 按进球数分组
    if total_goals <= 2:
        group = 'low'
    elif total_goals <= 5:
        group = 'medium'
    else:
        group = 'high'

    # 获取历史校准数据
    expected_rate = goal_calib.get(f'{group}_expected', 0.33)
    actual_rate = goal_calib.get(f'{group}_actual', 0.33)

    if actual_rate == 0 or expected_rate == 0:
        return 1.0

    # 基础校准因子
    base_factor = actual_rate / expected_rate

    # 当前预测的该组概率
    current_prob = sum(prob for g, prob in goal_dist.items()
                      if (group == 'low' and g <= 2) or
                         (group == 'medium' and 3 <= g <= 5) or
                         (group == 'high' and g > 5))

    # 如果预测概率与历史差异较大，适当调整
    prob_factor = 1.0
    if current_prob > 0 and expected_rate > 0:
        prob_ratio = current_prob / expected_rate
        # 温和调整，避免过度校正
        prob_factor = 0.9 + (prob_ratio - 1) * 0.2

    return min(max(base_factor * prob_factor, 0.7), 1.4)


def _get_score_calibration_factor(calibration_data, h, a, prob):
    """
    获取具体比分校准因子。

    参数:
        calibration_data: 历史校准数据
        h: 主队进球数
        a: 客队进球数
        prob: 当前预测的该比分概率

    返回:
        比分校准因子
    """
    score_calib = calibration_data.get('score_calibration', {})

    # 获取该比分的历史校准数据
    score_key = f"{h}-{a}"
    score_data = score_calib.get(score_key, {})

    expected_prob = score_data.get('expected', 0.02)
    actual_prob = score_data.get('actual', 0.02)

    if actual_prob == 0 or expected_prob == 0:
        return 1.0

    # 基础校准因子
    base_factor = actual_prob / expected_prob

    # 考虑比分类型（常见比分 vs 冷门比分）
    # 常见比分需要更保守的校准
    is_common = (h, a) in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2)]
    if is_common:
        # 常见比分，校准因子趋向于1
        base_factor = 0.95 + (base_factor - 0.95) * 0.5

    # 概率置信度调整
    # 低概率比分需要更强的校准
    confidence_factor = 1.0
    if prob < 0.02:
        # 低概率事件，增加校准强度
        confidence_factor = 1.1
    elif prob > 0.1:
        # 高概率事件，减弱校准强度
        confidence_factor = 0.95

    return min(max(base_factor * confidence_factor, 0.6), 1.5)
