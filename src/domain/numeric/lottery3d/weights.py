"""评分与排名的可调参数。

迁移前这些是四十来个散在 `config.py` 里的模块级全局量，评分函数直接读它们。
收成两个不可变容器，是为了让「哪些量是调参对象」有一处能一眼看全的清单——
散着的时候，回答「改这个数会影响什么」只能靠 grep。

**每一个数都会改变最终推荐哪些号，而改错了不会报错。** 所以这里不设默认值：
调用方必须显式给全，漏掉一个是 `TypeError`，不是悄悄用了个 0。

分成两层是因为它们的作用点不同：`DigitWeights` 决定单个数字的分，
`TripletWeights` 决定一注三个数字合起来的分。前者变了后者一定跟着变，
反过来不成立。
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Baselines:
    """公平摇奖下的理论基线，用来判断观测值「偏高还是偏低」。

    自适应缩放全都是「实测 ÷ 基线」。**基线写错不会报错**，只会让所有缩放
    系统性偏一个常数倍——所以它和权重一样由调用方显式给，不在领域层内联。
    """

    position_repeat: float   # 某一位与上期相同的概率
    digit_reuse: float       # 上期任一数字在本期再次出现的概率


@dataclass(frozen=True)
class DigitWeights:
    """给单个数字打分用的权重。"""

    hot_global: float        # 全局热号加分
    hot_position: float      # 分位热号加分
    markov: float            # 一阶转移
    markov2: float           # 二阶转移
    # 转移分的上限。没有它，一个样本极少的转移经平滑后仍可能给出极高的分——
    # 那不是信号强，是分母太小。
    markov_max: float
    markov_alpha: float      # 转移概率的拉普拉斯平滑系数
    miss_high: float         # 长期遗漏加分
    miss_mid: float          # 中等遗漏加分
    last_appear: float       # 上期出现过的数字
    neighbor: float          # 上期号码的邻号
    road_match: float        # 与上期同路
    decay: float             # 指数加权的衰减系数


@dataclass(frozen=True)
class TripletWeights:
    """给一注（三个数字）打分用的权重。"""

    danma_hit: float          # 命中胆码
    kill_penalty: float       # 命中杀码
    sum_sigma: float          # 和值高斯打分的宽度
    span_sigma: float         # 跨度高斯打分的宽度
    consecutive: float        # 含连号
    position_repeat: float    # 与上期同位重复
    ratio_match: float        # 奇偶比/大小比与近期吻合
    slope_match: float        # 与斜连关注码吻合
    pair_bonus: float         # 命中高频对子
    form_prior: float         # 形态先验（组六/组三/豹子的理论占比）
    triplet_position: float   # 三位基础分里分位评分的比重
    triplet_global: float     # 三位基础分里全局评分的比重
    diversity: float          # 选池时每多覆盖一个数字的奖励
    correlation_penalty: float    # 与已选注重合过多的惩罚
    correlation_threshold: int    # 重合几个数字算「过多」
    noise: float              # Top50 随机扰动的幅度
    exploration_rate: float   # 探索模式触发概率
    danma_top_pool: int       # 胆码从前几名里选
    danma_random_rate: float  # 胆码随机选的概率


# 和值/跨度的高斯打分系数。写死在这里而不是进 `TripletWeights`：它们不是
# 调参对象，是「和值比跨度更重要」这个判断，改它等于改模型结构。
SUM_GAUSSIAN_WEIGHT = 8.0
SPAN_GAUSSIAN_WEIGHT = 5.0
# 命中热门和值/跨度/和值尾数的固定加分。同上，属结构不属调参。
HOT_SUM_BONUS = 2.0
HOT_SPAN_BONUS = 1.5
HOT_SUM_TAIL_BONUS = 1.0
