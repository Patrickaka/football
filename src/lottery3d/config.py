# -*- coding: utf-8 -*-
"""福彩3D常量配置与可调权重"""

import sys
from contextlib import contextmanager

from ..common.logger import setup_logger

log = setup_logger('lottery3d')

# stdout 的编码由**进程入口**设置（`main.py`），不在库模块顶层做。
# 放这里的代价：pytest 一旦有任何测试导入到本模块，就会把它的捕获流换掉，
# 之后每个用例的 setup/teardown 都报 UnicodeDecodeError——**从那一刻起
# 整套测试全红，而且看不出跟谁有关**。
#
# `try/except` 与 `hasattr` 挡不住这件事：`reconfigure` 在 pytest 下**会成功**，
# 成功本身就是问题。同样的坑 `src/football/config.py` 与
# `src/webapp/settings.py` 各踩过一次。

# 拉取约 2000 期历史（≈6 年）。更长的历史让频率/转移/和值估计与回测显著更稳定，
# 减少 200 期小样本下的噪声。预测路径仍按窗口（≤90 期）截取，故不影响线上速度。
URL = "https://www.8300.cn/kjhhis/3/2000.html"

RECENT_WINDOWS = (30, 45, 60, 90)
RECENT_WINDOW = 90  # 展示用最大窗口
# 窗口权重回测参数（保守收缩）
WINDOW_BACKTEST_TRIALS = 100  # 增加回测期数，减少偶然因素影响
WINDOW_WEIGHT_PRIOR = 5.0  # 增大先验权重，平滑窗口权重波动
EXP_DECAY = 0.96
BACKTEST_TRIALS = 80
PERMUTATION_SHUFFLES = 200  # 置换检验打乱次数，评估命中率是否显著优于随机（建议线上至少200，离线验证1000）
# 和值/跨度近期偏移参数
RECENT_SUM_SPAN_SHIFT = 0.0  # 关闭近期5期偏移，避免追涨杀跌

# 缓存配置

W_HOT_GLOBAL = 2.5   # 原 4.0；降低热号全局权重，减少同一号码长期霸榜
W_HOT_POS = 3.0     # 原 5.0；降低分位热号权重，让转移概率有更多发言权
# 冷号遗漏加分：W_MISS_HIGH 对待极高遗漏值（≥20 期），W_MISS_MID 对待中等遗漏值（≥12 期）
# 实盘保守版本：大幅降权冷号遗漏奖励，避免追冷
W_MISS_HIGH = 1.5   # 遗漏 20 期加 1.5*(1+20/20)=3分（大幅降权）
W_MISS_MID = 1.0    # 中等遗漏值加分（降权）
W_MARKOV = 5.0       # 原 8.0；降低马尔可夫转移权重，避免过度依赖最近一期
W_MARKOV2 = 1.5      # 原 4.0；降低二阶马尔可夫转移权重
MARKOV_MAX_SCORE = 6.0  # 马尔可夫转移得分上限，避免主导推荐结果
W_LAST_APPEAR = 2.5
W_NEIGHBOR = 2.0
W_ROAD_MATCH = 1.5
W_DANMA_HIT = 4.0
W_KILL_PENALTY = 4.0  # 杀码软降权（原6.0过强，易误伤开奖数字）
W_CONSECUTIVE = 1.5   # 含相邻连号（如 12、67）
W_POS_REPEAT = 1.2    # 与上期同位重复（直选复刻），每码；实际强度由 lag1 动态缩放
W_RATIO_MATCH = 1.8   # 奇偶比 / 大小比与近期热门匹配
# 随机基准：单位置复刻 10%；指定数字在下一期三码中出现 ≈ 27.1%
RANDOM_POS_REPEAT = 0.10
RANDOM_DIGIT_REUSE = 1 - (9 / 10) ** 3
SUM_SOFT_SIGMA = 3.2
SPAN_SOFT_SIGMA = 1.4
# 形态先验：按真实形态概率(组六72%/组三27%/豹子1%)给分，使推荐池形态分布贴合真实开奖。
# 实测原推荐组六偏少(64% vs 真实71%)、组三偏多，此项把组六占比拉回~72%。
W_FORM_PRIOR = 6.0

# 直选组合：分位评分为主、全局评分为辅（直选带顺序，分位信号更关键）
W_TRIPLET_POS = 1.0
W_TRIPLET_GLOBAL = 0.30
ZHXUAN_POS_TOPK = 5  # Top3/Top5 分位候选池每位置取前 N 码

# 探索机制：推荐时有一定概率从候选池中随机选择
# 稳定基础版：关闭探索机制
EXPLORATION_RATE = 0.0  # 关闭探索，始终选择最高分号码

# 动态胆码机制：70%概率选 Top2，30%概率从 Top6 中随机选 2 个
# 稳定基础版：关闭随机胆码
DANMA_TOP_POOL = 6  # 胆码候选池大小
DANMA_RANDOM_RATE = 0.0  # 关闭随机选择胆码

# 推荐注数（直选为带顺序的三位数）
RECOMMEND_GROUPS = 30  # 推荐池扩大至 30 注
ZHIXUAN_TOP3 = 3
ZU6_POOL_SIZE = 6
ZU6_FOUR_SIZE = 4

# Top50 随机扰动：避免同分号长期霸榜
# 稳定基础版：关闭随机噪声
RANDOM_NOISE = 0.0  # 关闭随机噪声

# 近期回补模型：统计最近 30 期，严重欠账的号码额外加分
RECENT_WINDOW_REBOUND = 30  # 回补统计窗口
REBOUND_BONUS = 0.5  # 严重欠账号码加分（实盘保守版本：大幅降权）
REBOUND_THRESHOLD = 0.5  # 欠账阈值（实际值/理论值 < 0.5 认为严重欠账）

# 冷热平衡模型：推荐池号码类型比例
HOT_RATIO = 0.40   # 热号比例 40%
WARM_RATIO = 0.40  # 温号比例 40%

# 特征开关：用于消融测试
# 稳定基础版配置：关闭形态切换、冷热平衡、数字配对和遗漏（待消融验证）
FEATURE_FLAGS = {
    "hot": True,           # 热号得分
    "miss": False,         # 遗漏加分（关闭：待消融回测验证有效性）
    "markov": True,        # 马尔可夫转移
    "neighbor": False,     # 邻号加分（关闭：待消融回测验证有效性）
    "road": False,         # 012 路匹配（关闭：待消融回测验证有效性）
    "sum_span": True,      # 和值跨度
    "pair": True,          # 数字配对（high_pairs 已在 meta 预计算）
    "form_switch": False,  # 形态切换（关闭：避免赌徒谬误）
    "cold_hot_balance": False,  # 冷热平衡（关闭：避免干扰模型排序）
    "consecutive": True,   # 连号奖励（开启：待消融回测验证有效性）
    "lag1_repeat": True,   # 上期同位重复、全重复、同集合惩罚（开启：待消融回测验证有效性）
    "ratio": True,         # 奇偶比、大小比奖励（开启：待消融回测验证有效性）
    "slope": True,         # 斜连走势（同位/跨期，辅助加分）
}
COLD_RATIO = 0.20  # 冷号比例 20%
HOT_WINDOW = 20    # 冷热判断窗口

# 和值趋势模型：统计最近 20 期和值趋势，动态调整和值中心
# 实盘版本：关闭和值趋势调整（避免过度约束）
SUM_TREND_WINDOW = 20  # 和值趋势统计窗口
SUM_TREND_ADJUST = 0.0  # 和值中心调整幅度（关闭）

# 遗漏周期模型：计算平均遗漏周期和超期倍率
MISS_CYCLE_WINDOW = 200  # 统计平均遗漏周期的窗口大小
MISS_OVER_RATIO_THRESHOLD = 2.5  # 超期倍率阈值
MISS_OVER_BONUS = 1.0  # 超期额外加分（实盘保守版本：降权）

# 数字配对模型：统计数字对出现频率
PAIR_FREQ_WINDOWS = [50, 100, 200]  # 统计窗口
PAIR_HIGH_FREQ_THRESHOLD = 0.15  # 高频对子阈值（出现频率 > 15%）
PAIR_BONUS = 2.5  # 高频对子加分

# 斜连走势：同位连续 ±1、跨期百→十→个对角（样本少，仅轻量加分）
SLOPE_MIN_CHAIN = 3
SLOPE_MAX_CHAIN = 6
W_SLOPE_MATCH = 1.2
POS_NAMES_3D = ("百", "十", "个")

# 组三组六切换模型：根据连续出现调整权重
# 实盘版本：关闭组三组六强制切换（典型赌徒谬误，连续出现不代表下期一定要切换）
FORM_SWITCH_WEIGHT = 0.0  # 切换奖励权重（关闭）
ZU6_STREAK_THRESHOLD = 8  # 组六连续出现阈值
ZU3_STREAK_THRESHOLD = 4  # 组三连续出现阈值

# 和值区间回归模型：预测区间而非具体和值
# 实盘版本：关闭和值区间回归奖励（和值适合软排序，不适合硬约束）
SUM_INTERVAL_WINDOW = 5  # 计算中心的窗口大小
SUM_INTERVAL_WIDTH = 3  # 区间宽度（中心 ± width）
SUM_INTERVAL_BONUS = 0.0  # 区间内加分（关闭）
SUM_EXTREME_PENALTY = 0.0  # 极端区间降权（关闭）

# 最近N期去重机制：对重复推荐降权，使每日推荐池在天与天之间轮换。
# 3D 为公平摇奖，任意 30 注互异组合命中率恒为 3%，故轮换不改变命中期望(实测900期均落在
# 3%±1.1% 噪声带内)，但能把日间重复度从 ~54% 降到 ~14%。惩罚值取较大以形成"近窗内基本不重复"。
RECENT_RECOMMEND_WINDOW = 5  # 最近推荐窗口大小
RECENT_RECOMMEND_PENALTY = 5.0  # 日间轮换惩罚（略降，减少对 Top30 命中率的干扰）
RECENT_RECOMMEND_CONSECUTIVE_PENALTY = 16.0  # 连续推荐的号码惩罚

# 组六四码逐期轮换（默认关闭：轮换不为命中率服务，会主动换掉相对高分码）
ZU6_RECENT_WINDOW = 5
ZU6_RECENT_PENALTY = 0.0
ZU6_RECENT_DECAY = 0.6

# 组三专用推荐（v4.9 新增，v4.10 改高效口径）：
# 福彩3D 规则：组选3 一注 = 3 码含一重复位 → 3 种排列（如 225 → 225/252/522），单注 2 元、奖金 346 元；
# 单选 一注 = 1 种排列，2 元、奖金 1040 元。对子 {a,b} 的全部 6 种排列：
#   组选三 2 注（"225"+"552"，4 元）即可完整覆盖，与 6 注单选（12 元）中奖概率完全相同（EV 相同）。
# 数字选择与组六同理无 edge（组三条件下数字均匀），模型只输出"近期组三开奖中出现率较高"
# 的对子，并诚实标注概率：任取 K 组条件命中 = K/C(10,2) = K/45，与选哪些码无关。
ZU3_PRESENCE_WINDOWS = (25,)      # 与组六 presence 同思路：近25期组三开奖去重后数字出现率
ZU3_MIN_SAMPLES = 10              # 组三样本不足时扩大到 60 期
ZU3_PAIRS_COUNT = 4               # 组三推荐组数（四组）
ZU3_TIER_SIZES = (4, 8, 12, 20)   # 组三覆盖档位：K 组对子 → 组选三 2K 注/4K 元，条件命中 K/45

# 组六选码：不用杀码降权（杀码误伤开奖数字的风险大于收益）
ZU6_USE_KILL = False
ZU6_CANDIDATE_SIZE = 8

# 组六四码只预测“数字是否进入开奖号集合”，不预测位置。滚动回测显示，
# 用每期去重后的边际出现率比通用的分位/遗漏/马尔可夫混合分更稳定。
# 两个相邻窗口等权融合，降低单一窗口偶然波动；较短窗口用于同分时破平。
# 逐期样本外验证（开发 611 个组六期、验证 667 个组六期）显示，单独使用
# 最近 25 期的“数字是否出现”频率，比 25/40 双窗平均更稳定：验证命中
# 9.15% vs 8.25%，且开发集同方向（8.67% vs 8.02%）。不叠加 40 期窗，
# 避免较旧样本把近期边际频率信号稀释。
ZU6_PRESENCE_WINDOWS = (25,)

# 组六专用单码评分权重
W_ZU6_HOT = 3.0
W_ZU6_MISS = 1.5
W_ZU6_POS = 1.2
W_ZU6_PAIR = 2.0
W_ZU6_BLEND = 1.5

# 窗口权重持久化键
WINDOW_WEIGHTS_KV_KEY = "lottery3d_window_weights"

# 预测版本号
PREDICTOR_VERSION = "3d-v4.11-fast-six-cover"
ML_MODEL_VERSION = "ml-v7"
MIN_DATA_PERIODS_FOR_ML_FUSION = 300
ML_CACHE_MAX_AGE_SECONDS = 36 * 3600

# 线上实盘记录文件
ONLINE_PREDICTION_FILE = "data/lottery3d_online_predictions.json"

# 推荐池多样性控制：最大化数字覆盖率
DIVERSITY_WEIGHT = 1.5  # 多样性权重
SERVED_POOL_CANDIDATE_SIZE = 150  # 贪心选池候选范围


# 推荐号码去相关：减少高度相关推荐
CORRELATION_THRESHOLD = 2  # 重合数字阈值
CORRELATION_PENALTY = 3.0  # 相关惩罚分数

# 自动淘汰失效特征：定期评估特征贡献
FEATURE_EVAL_PERIOD = 30  # 特征评估周期（期数）
FEATURE_MIN_CONTRIBUTION = 0.01  # 最小贡献率阈值（1%）
FEATURE_DOWNGRADE_FACTOR = 0.5  # 降权因子

# 马尔可夫转移：拉普拉斯平滑系数 α（加法平滑，α=1 即标准 Laplace）
MARKOV_LAPLACE_ALPHA = 1.0

# 可调评分权重（供 search_weights 搜索）
# 精简版本：只保留已启用特征的权重，避免浪费计算在无效参数上
TUNABLE_WEIGHTS = (
    "W_HOT_GLOBAL",
    "W_HOT_POS",
    "W_MARKOV",
    "W_MARKOV2",
    "W_DANMA_HIT",
    "W_KILL_PENALTY",
    "SUM_SOFT_SIGMA",
    "SPAN_SOFT_SIGMA",
)

# 随机搜索时各参数相对默认值的缩放范围 (low, high)
WEIGHT_SEARCH_RANGES = {
    "W_HOT_GLOBAL": (0.5, 2.0),
    "W_HOT_POS": (0.5, 2.0),
    "W_MISS_MID": (0.4, 2.5),
    "W_MARKOV": (0.5, 2.0),
    "W_MARKOV2": (0.3, 1.5),
    "W_LAST_APPEAR": (0.3, 2.5),
    "W_NEIGHBOR": (0.3, 2.5),
    "W_ROAD_MATCH": (0.0, 3.0),
    "W_DANMA_HIT": (0.5, 2.5),
    "W_KILL_PENALTY": (3.0, 12.0),
    "SUM_SOFT_SIGMA": (2.0, 5.0),
    "SPAN_SOFT_SIGMA": (0.8, 2.2),
}


def default_weights():
    """当前默认评分权重快照"""
    return {k: globals()[k] for k in TUNABLE_WEIGHTS}


@contextmanager
def patch_weights(weights):
    """临时覆盖模块级权重，供回测/搜索使用"""
    saved = {k: globals()[k] for k in TUNABLE_WEIGHTS}
    for k in TUNABLE_WEIGHTS:
        if k in weights:
            globals()[k] = weights[k]
    try:
        yield
    finally:
        for k, v in saved.items():
            globals()[k] = v


