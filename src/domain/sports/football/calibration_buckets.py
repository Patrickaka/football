# -*- coding: utf-8 -*-
"""校准分桶：桶键、整数键还原、因子计算与分布应用。

纯计算——**存储留在适配层**（判据 16）。两个校准器（进球数、贝叶斯）的
桶键函数在迁移前是「零次使用 self 的实例方法」，本质就是纯函数。

**整数键还原是防腐层，不能省**（判据 26/27）：kv_store 走 JSON，
`{2: 0.9}` 存一轮回来是 `{'2': 0.9}`，而 `factors.get(2)` 查不到 `'2'`
会让每个因子悄悄回落成 1.0——分布原样返回、日志干净、一切正常。
`40398d1` 就是这么踩的。
"""

import logging
from collections import defaultdict

log = logging.getLogger('domain.football.calibration_buckets')

MIN_ACTIVATION_SAMPLES = 4
POOLING_K = 12.0


def _restore_goal_keys(db):
    """把整个 db 里所有「进球数 → 概率」映射的键还原成 int。

    只动 `calibration_factors` 与 `predicted_distributions`——分桶键
    （`'瑞典超_2.75_+0.50_2.50'`）本来就是字符串，不能碰。
    """
    if not isinstance(db, dict):
        return {}
    for bucket in db.values():
        if not isinstance(bucket, dict):
            continue
        factors = bucket.get('calibration_factors')
        if isinstance(factors, dict):
            bucket['calibration_factors'] = _int_keyed(factors)
        dists = bucket.get('predicted_distributions')
        if isinstance(dists, list):
            bucket['predicted_distributions'] = [
                _int_keyed(dist) if isinstance(dist, dict) else dist
                for dist in dists
            ]
    return db


def _int_keyed(mapping):
    """键转 int。**转不动的原样留下**——宁可留一个查不到的键，
    也不能把它悄悄丢掉：丢掉之后概率就不再归一，而那不会报错。
    """
    restored = {}
    for key, value in mapping.items():
        try:
            restored[int(key)] = value
        except (TypeError, ValueError):
            restored[key] = value
    return restored


def goal_bucket_key(league: str, total_line: float, 
                    asian: float = 0.0, expected_total: float = 2.5) -> str:
    """
    生成分桶键
    
    分桶维度：
    - 联赛名
    - 大小球盘口（四舍五入到0.25）
    - 让球盘口（四舍五入到0.5，带符号）
    - 预测总进球（四舍五入到0.5）
    
    参数：
        league: 联赛名称
        total_line: 大小球盘口线
        asian: 亚盘让球
        expected_total: 模型预测总进球数
    
    返回：
        分桶键字符串
    """
    # 大小球盘口按0.25分桶
    bucketed_line = round(total_line * 4) / 4
    # 让球盘口按0.5分桶，带符号（+/-）
    bucketed_asian = round(asian * 2) / 2
    # 预测总进球按0.5分桶
    bucketed_expected = round(expected_total * 2) / 2
    
    return f"{league}_{bucketed_line:.2f}_{bucketed_asian:+.2f}_{bucketed_expected:.2f}"


def bayesian_bucket_key(score: str, league: str, total_line: float, asian: float, level: int) -> str:
    """
    生成分桶键
    
    参数：
        score: 比分
        league: 联赛名称
        total_line: 大小球盘口
        asian: 让球盘口
        level: 层级（1=最细，3=最粗）
    
    返回：
        分桶键字符串
    """
    if level == 1:
        # 层级1：比分 + 联赛 + 大小球 + 让球
        bucketed_line = round(total_line * 4) / 4
        bucketed_asian = round(asian * 2) / 2
        return f"{score}_{league}_{bucketed_line:.2f}_{bucketed_asian:+.2f}"
    elif level == 2:
        # 层级2：比分 + 大小球 + 让球
        bucketed_line = round(total_line * 4) / 4
        bucketed_asian = round(asian * 2) / 2
        return f"{score}_all_{bucketed_line:.2f}_{bucketed_asian:+.2f}"
    else:
        # 层级3：仅比分
        return f"{score}_all_all_all"


def compute_calibration_factors(bucket):
    """
    更新分桶的校准因子
    
    计算每个进球数的实际频率与预测频率的比值，作为校准因子。
    """
    sample_count = bucket.get('weighted_sample_count', bucket['sample_count'])

    if sample_count < MIN_ACTIVATION_SAMPLES:
        # 样本过少，噪声太大，不计算校准因子
        bucket['calibration_factors'] = {}
        return
    
    # 计算实际频率分布
    weights = bucket.get('sample_weights') or [1.0] * len(bucket['actual_goals'])
    if len(weights) < len(bucket['actual_goals']):
        weights = weights + [1.0] * (len(bucket['actual_goals']) - len(weights))

    actual_dist = defaultdict(float)
    for goals, weight in zip(bucket['actual_goals'], weights):
        actual_dist[goals] += weight
    
    # 归一化
    actual_total = sum(actual_dist.values())
    actual_dist = {k: v / actual_total for k, v in actual_dist.items()}
    
    # 计算平均预测分布
    pred_dist = defaultdict(float)
    for dist, weight in zip(bucket['predicted_distributions'], weights):
        for goals, prob in dist.items():
            pred_dist[goals] += prob * weight
    
    # 归一化
    pred_total = sum(pred_dist.values())
    if pred_total > 0:
        pred_dist = {k: v / pred_total for k, v in pred_dist.items()}
    else:
        bucket['calibration_factors'] = {}
        return
    
    # 计算校准因子：实际频率 / 预测频率
    # 添加平滑处理，避免除零和极端值；并按样本量做部分池化收缩，
    # 样本少时因子向 1.0 收缩（弱校准），样本多时逐步逼近观测频率比。
    calibration_factors = {}
    all_goals = set(actual_dist.keys()) | set(pred_dist.keys())

    shrink_weight = sample_count / (sample_count + POOLING_K)

    for goals in all_goals:
        actual_prob = actual_dist.get(goals, 0.01)  # 平滑
        pred_prob = pred_dist.get(goals, 0.01)      # 平滑

        # 原始校准因子，限制在合理范围
        factor = actual_prob / pred_prob
        factor = max(0.5, min(2.0, factor))  # 限制在0.5~2.0之间

        # 部分池化：向 1.0（不校准）收缩，收缩程度随样本量增大而减弱
        factor = 1.0 + (factor - 1.0) * shrink_weight

        calibration_factors[goals] = factor

    bucket['calibration_factors'] = calibration_factors


def apply_goal_calibration(goal_dist, factors):
    """把校准因子应用到进球数分布并归一化。

    因子取不到的进球数按 1.0 处理；**并集**取自分布与因子两边——
    因子里有而分布里没有的进球数会被带进来（原始概率 0，乘完仍是 0，
    但键会出现在输出里）。行为原样保留。
    """
    if not factors:
        return goal_dist

    calibrated = {}
    for goals in set(goal_dist.keys()) | set(factors.keys()):
        calibrated[goals] = goal_dist.get(goals, 0.0) * factors.get(goals, 1.0)

    total = sum(calibrated.values())
    if total > 0:
        return {k: v / total for k, v in sorted(calibrated.items())}
    return calibrated
