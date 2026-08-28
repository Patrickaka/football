"""把比分矩阵拟合到亚盘与大小球的**公平价**上。

**这是整条链里唯一让不同玩法互相自洽的地方。** 胜平负、让球、比分、总进球
各自算各自的话，同一场比赛会给出互相矛盾的推荐——让球盘说主队稳，
比分推荐却全是平局。做法是让它们共用一张比分矩阵，而这张矩阵被约束到
「按公平赔率下注，期望收益为零」。

用的是**指数倾斜**（exponential tilting）：给每个比分乘上 `exp(θ·特征值)`，
再归一化。θ 由二分搜索确定，使该特征的期望恰好为 0。这在所有满足约束的
分布里选出与原分布 KL 距离最小的那个——**换句话说，只改必须改的部分**。

**开盘到临场的走势不当作第二个独立预测。** 终盘价已经把那段信息吸收了，
再叠一次等于把同一条证据用两遍；走势只用来判断可靠性与冲突
（见 `joint_state` 的 `agreement_factor`）。

这一层不读配置、不碰时钟——十几个权重与门槛全部由调用方传入。
"""
import math
import re
from collections import defaultdict

# 亚盘方向映射成 [-1, 1] 的信号。**主队方向给满、客队方向打折**：
# 北单盘口以主队为基准报价，客队那侧的水位变化常常只是跟随
DIRECTION_WEIGHTS = {'home_backing': 1.0, 'away_laying': 0.65,
                     'away_backing': -1.0, 'home_laying': -0.65}
TEMPO_WEIGHTS = {'over_backing': 1.0, 'under_laying': 0.65,
                 'under_backing': -1.0, 'over_laying': -0.65}

# 指数倾斜的搜索区间与轮数。θ 落在 ±12 之外意味着约束本身无解
THETA_LOW, THETA_HIGH, THETA_ROUNDS = -12.0, 12.0, 40
# exp 的指数夹紧范围。**不夹会溢出**——θ·特征值 在极端比分上能到几百
EXP_CLAMP = 20.0
# 分桶时的浮点取整位数：特征值相同的比分合并成一个桶，减少二分的计算量
BUCKET_PRECISION = 12


def _clamp(value, low=-1.0, high=1.0):
    return max(low, min(high, value))


def joint_state(asian_trend, goals_trend, asian_history, goals_history,
                strength_divisor=0.12, strength_floor=0.25,
                handicap_divisor=0.5, line_divisor=0.5,
                direction_blend=0.65, tempo_blend=0.55,
                conflict_threshold=-0.12, conflict_damping=0.40):
    """把亚盘与大小球的走势合成一个方向信号和一个节奏信号。

    每个信号有两个来源：**水位的动向**与**盘口线本身的移动**。
    两者按 `direction_blend` / `tempo_blend` 加权——线的移动权重略低，
    因为它是离散的（半球一跳），单次变动的信息量不如水位连续。

    `conflict` 是水位与线**指向相反**的情形：水位在压大球而线却往下调。
    那说明两个信号在打架，此时把节奏信号衰减到 `conflict_damping`——
    **不是取反也不是忽略**，而是承认「这场看不清」。
    """
    direction = DIRECTION_WEIGHTS.get(asian_trend.get('direction'), 0.0)
    try:
        strength = float(asian_trend.get('strength', 0.0)) / strength_divisor
        direction *= _clamp(strength, strength_floor, 1.0)
    except (TypeError, ValueError):
        direction = 0.0

    handicap_signal = _signal_from_move(
        asian_history, lambda entry: entry.get('handicap'), handicap_divisor)
    direction = direction_blend * direction + (1 - direction_blend) * handicap_signal

    water_tempo = TEMPO_WEIGHTS.get(goals_trend.get('direction'), 0.0)
    line_signal = _signal_from_move(
        goals_history, lambda entry: _first_number(entry.get('line')), line_divisor)
    conflict = water_tempo * line_signal < conflict_threshold
    tempo = tempo_blend * water_tempo + (1 - tempo_blend) * line_signal
    if conflict:
        tempo *= conflict_damping

    return {
        'direction_signal': _clamp(direction),
        'tempo_signal': _clamp(tempo),
        'handicap_signal': handicap_signal,
        'line_signal': line_signal,
        'asian_trend': asian_trend,
        'goals_trend': goals_trend,
        'conflict': conflict,
        'agreement_factor': conflict_damping if conflict else 1.0,
    }


def _signal_from_move(history, extract, divisor):
    """首尾两期之间的变化，归一化到 [-1, 1]。取不到就返回 0。

    **用首尾而不是逐期平均**：盘口线是阶梯式跳变的，中间的往返在这里
    没有意义，真正重要的是「从开盘到现在挪了多少」。
    """
    if not history or len(history) < 2:
        return 0.0
    try:
        first, last = extract(history[0]), extract(history[-1])
        return _clamp((float(last) - float(first)) / divisor)
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _first_number(value):
    match = re.search(r'[\d.]+', str(value))
    return match.group() if match else None


def tilt_to_fair_price(distribution, feature, strength):
    """指数倾斜：把分布调到「按公平赔率下注期望收益为零」。

    `feature(key)` 给出每个结果下的单位收益。找 θ 使 `E[feature] = 0`，
    然后**只施加 `strength` 那一部分**——完全对齐等于把模型丢掉，
    只信市场；不对齐则等于无视已经成交的价格。

    **目标落在支撑集之外时不动**（返回 `target_outside_support`）：
    所有结果都是正收益或都是负收益的话，任何 θ 都到不了 0，
    强行搜索只会把分布推到边界上。
    """
    values = {key: feature(key) for key in distribution}
    buckets = defaultdict(float)
    for key, probability in distribution.items():
        buckets[round(values[key], BUCKET_PRECISION)] += probability

    def expectation(theta):
        weighted = [(value, probability * math.exp(_clamp(theta * value,
                                                          -EXP_CLAMP, EXP_CLAMP)))
                    for value, probability in buckets.items()]
        denominator = sum(probability for _, probability in weighted)
        return sum(value * probability for value, probability in weighted) / denominator

    low, high = THETA_LOW, THETA_HIGH
    if expectation(low) > 0 or expectation(high) < 0:
        return distribution, {'applied': False, 'reason': 'target_outside_support'}

    before = expectation(0.0)
    for _ in range(THETA_ROUNDS):
        mid = (low + high) / 2.0
        if expectation(mid) < 0:
            low = mid
        else:
            high = mid
    theta = strength * (low + high) / 2.0

    adjusted = {key: probability * math.exp(_clamp(theta * values[key],
                                                   -EXP_CLAMP, EXP_CLAMP))
                for key, probability in distribution.items()}
    denominator = sum(adjusted.values())
    adjusted = {key: probability / denominator for key, probability in adjusted.items()}
    return adjusted, {
        'applied': True,
        'theta': round(theta, 5),
        'fair_profit_before': round(before, 5),
        'fair_profit_after': round(
            sum(adjusted[key] * values[key] for key in adjusted), 5),
    }


def fair_odds(price_one, price_two):
    """两侧报价去水后，一侧的公平赔率。任一侧缺失返回 `None`。"""
    if not price_one or not price_two:
        return None
    inverse_one, inverse_two = 1.0 / price_one, 1.0 / price_two
    return (inverse_one + inverse_two) / inverse_one


def normalise_matrix(score_probs):
    """把比分分布规整成 `{(主, 客): 概率}` 并归一化。

    解析不了的键跳过、负概率夹到 0——**都不报错**：这份分布来自上游的
    模型输出，个别脏值不该让整场预测失败。全空时由调用方判断。
    """
    matrix = {}
    for score, probability in score_probs.items():
        try:
            key = int(score[0]), int(score[1])
            matrix[key] = matrix.get(key, 0.0) + max(0.0, float(probability))
        except (TypeError, ValueError, IndexError):
            continue
    total = sum(matrix.values())
    if total <= 0:
        return None
    return {key: probability / total for key, probability in matrix.items()}


def summarise_shift(before, after):
    """约束前后的两个可观测量：期望总进球与主胜概率。

    **报出来是为了让人能判断约束有没有推过头。** 只说「已应用」看不出
    幅度，而这两个量正是各玩法推荐真正依赖的。
    """
    return {
        'expected_goals_before': sum(sum(key) * p for key, p in before.items()),
        'expected_goals_after': sum(sum(key) * p for key, p in after.items()),
        'home_win_before': sum(p for (home, away), p in before.items() if home > away),
        'home_win_after': sum(p for (home, away), p in after.items() if home > away),
    }
