"""ML 训练集的构造与集成加权：正负样本、时间衰减、按期切分、加权融合。

**正例只有一条，负例要靠采样。** 每期开出一注，没开出的有 999 注——全都拿来
当负例的话，正负比 1:999，模型学到「全都别选」就能拿到 99.9% 的准确率。
所以每期只抽一部分负例，而**怎么抽决定了模型看到的是什么世界**：随便抽会
让负例集中在和值 10~17 那一带（组合数最多的地方），模型于是把「和值偏」
当成了「不会开」。按和值分层抽就是为了避开这件事。

**验证集必须按期号切，不能按样本切。** 同一期的正例与它的负例共享全部历史
统计量，切在期中间等于把答案漏给验证集——命中率会好看，而好看得毫无道理。

`model_score` 只能用来排序。训练集是 1:30 的采样比例，不是真实的 1:999，
所以输出的数不是概率，拿它当中奖概率会差一个数量级。
"""
import math
from itertools import product
from typing import NamedTuple

from src.domain.numeric.lottery3d.space import DIGIT_SPACE, POSITIONS

# 数据不足时返回的空训练集。**不是 None**：迁移前返回四个 None，而调用方的
# 守卫写的是 `if result is None or len(result) < 4`——四元组既不是 None，
# 长度也正好是 4，那道守卫从来没拦下过任何东西，越过它之后才在 len(None)
# 上炸掉。空的具名结果让 `if not samples.rows` 这种判断有东西可依据。
class TrainingSet(NamedTuple):
    rows: list      # 特征矩阵
    labels: list    # 1 = 这期开出的那注，0 = 抽出来的负例
    weights: list   # 样本权重（时间衰减）
    groups: list    # 期号分组，按期切验证集时用

    def __bool__(self):
        return bool(self.rows)


EMPTY = TrainingSet([], [], [], [])

POSITIVE, NEGATIVE = 1, 0


class SamplingSettings(NamedTuple):
    """训练集构造的可调项。**全部由调用方传入**——它们是配置，不是领域知识。"""

    neg_samples: int        # 每期抽多少负例
    min_history: int        # 前多少期只做历史，不产出样本
    feature_window: int     # 造特征时往回看多少期
    decay_tiers: tuple      # ((期数上界, 权重), ...)，按顺序命中第一档
    base_weight: float      # 落在所有档之外的旧样本权重

    def universe(self):
        """全部候选号码，按 (百, 十, 个) 的字典序。

        **顺序是契约的一部分**：负例在各层内是原地打乱后取前 N 个，
        换一个初始顺序就换了一份训练集，而结果照样跑得出来。
        """
        return all_combos()

    @property
    def universe_size(self):
        return DIGIT_SPACE.size ** POSITIONS


# 验证集占最后这个比例的期数
VALIDATION_SPLIT = 0.8
# 少于这么多期就不切验证集：切出来的那几期算不出有意义的分数
MIN_PERIODS_FOR_SPLIT = 10
# 验证集至少要有这么多条样本
MIN_VALIDATION_ROWS = 5
NEUTRAL_SCORE = 0.5


def time_decay_weight(periods_ago, tiers, base_weight):
    """越近的期权重越高。`tiers` 形如 ((30, 2.0), (60, 1.4))，按顺序命中第一档。

    **加权不是因为近期更容易预测**，而是因为号码的分布特征（冷热、形态偏好）
    会随时间漂移，旧样本描述的是另一个分布。
    """
    for boundary, weight in tiers:
        if periods_ago <= boundary:
            return weight
    return base_weight


def stratified_quota(group_sizes, total_samples, universe_size, rng):
    """按各层大小分配采样配额，配额不足的部分随机补给还有余量的层。

    **每层至少给 1**（只要总额够）：和值 0 那一层只有一个组合，按比例分配会
    得到 0，于是极端和值永远不进训练集，模型就学不到「它们也会开」。
    """
    quota = {}
    remaining = total_samples
    for group, size in group_sizes.items():
        share = max(1, int(total_samples * size / universe_size))
        granted = min(share, size, remaining)
        quota[group] = granted
        remaining -= granted

    # 剩余配额随机补给，不按固定顺序——固定顺序会让某几层长期吃到额外样本
    order = list(group_sizes)
    rng.shuffle(order)
    for group in order:
        if remaining <= 0:
            break
        spare = min(remaining, group_sizes[group] - quota.get(group, 0))
        quota[group] = quota.get(group, 0) + spare
        remaining -= spare
    return quota


def build_training_samples(numbers, make_engineer, settings, rng):
    """滚动构造训练集：每期一条正例 + 若干分层抽样的负例。

    `make_engineer(history)` 由调用方给——造特征器要读一堆配置，那是配置层
    的事。这里只负责「哪些号码进训练集、各自多重」。
    """
    # 这道守卫与下面 `range` 的边界重合——历史恰好等于下限时，
    # `range(min_history, min_history)` 本来就是空的。留着是为了让「数据不足」
    # 在代码里有个明确的位置，而不是靠 range 的边界行为隐含。
    # **变异验证挡不住它**（把 `<=` 改成 `<` 行为完全相同），这是已知的等价变异
    if len(numbers) <= settings.min_history:
        return EMPTY

    rows, labels, weights, groups = [], [], [], []
    for index in range(settings.min_history, len(numbers)):
        start = max(0, index - settings.feature_window)
        engineer = make_engineer(numbers[start:index])
        actual = tuple(numbers[index])
        weight = time_decay_weight(len(numbers) - index,
                                   settings.decay_tiers, settings.base_weight)

        rows.append(engineer.build_features(actual))
        labels.append(POSITIVE)
        weights.append(weight)
        groups.append(index)

        for combo in _negative_samples(actual, settings, rng):
            rows.append(engineer.build_features(combo))
            labels.append(NEGATIVE)
            weights.append(weight)
            groups.append(index)

    return TrainingSet(rows, labels, weights, groups)


def _negative_samples(actual, settings, rng):
    """这一期的负例：按和值分层，各层内随机抽。"""
    by_sum = {}
    for combo in settings.universe():
        if combo == actual:
            continue
        by_sum.setdefault(sum(combo), []).append(combo)

    sizes = {group: len(combos) for group, combos in by_sum.items()}
    # 分母排除掉实际开奖的那一注——它不是负例候选
    quota = stratified_quota(sizes, settings.neg_samples,
                             settings.universe_size - 1, rng)

    taken = 0
    for group, share in quota.items():
        if share <= 0:
            continue
        combos = by_sum[group]
        rng.shuffle(combos)
        for combo in combos[:share]:
            if taken >= settings.neg_samples:
                break
            taken += 1
            yield combo


def split_by_period(groups, rows_count):
    """按期号切出 (训练下标, 验证下标)。切不动时返回 None。

    **按期切而不是按样本切**：同一期的正例和它的负例共享全部历史统计量，
    从中间切开等于让验证集看到训练集的答案。这类泄漏不会报错，
    只会让验证分数好得没有道理。
    """
    if not groups or len(groups) != rows_count:
        return None
    periods = sorted(set(groups))
    if len(periods) < MIN_PERIODS_FOR_SPLIT:
        return None

    boundary = periods[int(len(periods) * VALIDATION_SPLIT)]
    train = [index for index, period in enumerate(groups) if period <= boundary]
    valid = [index for index, period in enumerate(groups) if period > boundary]
    if len(valid) < MIN_VALIDATION_ROWS:
        return None
    return train, valid


def fallback_split(rows_count):
    """没有期号时退回按样本切。**这是降级，不是等价方案**——
    同一期的样本会被劈开，验证分数因此偏高。
    """
    boundary = max(MIN_PERIODS_FOR_SPLIT, int(rows_count * VALIDATION_SPLIT))
    if rows_count - boundary < MIN_VALIDATION_ROWS:
        return None
    return list(range(boundary)), list(range(boundary, rows_count))


def validation_score(labels, probabilities):
    """把 LogLoss 折成「越大越好」的分数，用来给各模型分配融合权重。

    直接用 LogLoss 的话，权重要取倒数，而 loss 接近 0 时倒数会炸；
    `1/(1+loss)` 落在 (0, 1] 且单调，拿来当权重刚好。
    """
    if not labels:
        return NEUTRAL_SCORE
    epsilon = 1e-15
    loss = 0.0
    for truth, probability in zip(labels, probabilities):
        probability = max(epsilon, min(1 - epsilon, probability))
        loss -= truth * math.log(probability) + (1 - truth) * math.log(1 - probability)
    return 1.0 / (1.0 + loss / len(labels))


def blend_weights(scores):
    """按验证得分分配融合权重。全为 0 时均分——**不能让权重全 0**，
    那样融合结果恒为 0，一千注排名完全随机而毫无迹象。
    """
    total = sum(scores)
    if not scores:
        return []
    if total == 0:
        return [1.0 / len(scores)] * len(scores)
    return [score / total for score in scores]


def blend_predictions(predictions, weights):
    """按权重加权平均各模型的预测。"""
    if not predictions:
        return []
    return [sum(weight * row[index] for row, weight in zip(predictions, weights))
            for index in range(len(predictions[0]))]


def all_combos():
    """全部候选号码，按 (百, 十, 个) 的字典序。

    与 `SamplingSettings.universe()` 是同一个序列——采样和预测必须枚举出
    同一份候选，否则「训练时见过的号码」和「预测时打分的号码」对不上。
    """
    return product(DIGIT_SPACE.numbers(), repeat=POSITIONS)
