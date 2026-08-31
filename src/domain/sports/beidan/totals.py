"""大小球：贴水与欧赔的换算、由盘口反推隐含总进球、目标总进球的融合。

**这一层是整条链的源头**：算出来的总进球决定了两队的 λ，λ 决定比分矩阵，
比分矩阵决定所有玩法的推荐。它偏 0.3 个球，下游全部跟着偏。

不读配置——联赛均值、融合权重、上下界一律由调用方传入。
"""
import re

from src.domain.sports.beidan import handicap as _handicap
from src.domain.sports.beidan.scoring_model import poisson_pmf

# 贴水与欧赔的分界。亚洲盘的贴水值域约 0.5~1.3，欧赔最低也在 1.01 以上，
# 两者在 1.2 附近才会混淆。**判错的代价是隐含概率差一倍**：
# 把贴水 0.85 当欧赔用，1/0.85 = 1.18 会算出大于 1 的概率。
WATER_MAX = 1.2
# 二分搜索的区间与轮数。0.5~9.0 覆盖了任何真实赛事的总进球
SEARCH_LOW, SEARCH_HIGH, SEARCH_ROUNDS = 0.5, 9.0, 60
# 期望收益求和的上界。15 球以上在这个搜索区间里质量可以忽略
MAX_GOALS_FOR_PROFIT = 16


def to_euro_odds(value):
    """亚洲盘贴水转欧赔；已经是欧赔的原样返回。解析不了返回 `None`。

    中国足彩网/bet365 的大小球常给贴水格式（大球 0.83 / 小球 1.0），
    欧赔 = 贴水 + 1.0。**直接拿贴水当欧赔套 1/odds 会严重高估概率。**
    """
    if value is None:
        return None
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    if value <= 0:
        return None
    if value <= WATER_MAX:
        return value + 1.0
    return value


def parse_line_value(value, default=2.5):
    """把盘口线解析成数值，分盘（`2.5/3`）取两条线的中点。

    这里的中点只用于「这场大概几个球」的粗略定位；真正的结算要用
    `handicap.line_parts` 拆成两条线分别算——**两者不能互相替代**。
    """
    if isinstance(value, (int, float)):
        return float(value)
    numbers = re.findall(r'\d+(?:\.\d+)?', str(value or ''))
    if not numbers:
        return default
    parsed = [float(number) for number in numbers[:2]]
    return sum(parsed) / len(parsed)


def implied_total(over_odds, under_odds, line=2.5):
    """由大小球盘口反推隐含总进球（泊松假设）。任一侧缺失返回 `None`。

    做法是二分搜索：找一个总进球均值，使得**按公平赔率买大球的期望收益为 0**。
    公平赔率先由去水后的大球概率算出，所以这个 λ 反映的是市场认为的中心值，
    而不是赔率本身的高低。
    """
    if not over_odds or not under_odds:
        return None
    over_euro = to_euro_odds(over_odds)
    under_euro = to_euro_odds(under_odds)
    if not over_euro or not under_euro:
        return None
    try:
        prob_over_raw = 1.0 / over_euro
        prob_under_raw = 1.0 / under_euro
    except (ValueError, TypeError):
        return None
    total_raw = prob_over_raw + prob_under_raw
    if total_raw <= 0:
        return None

    prob_over = prob_over_raw / total_raw
    fair_over_odds = 1.0 / max(prob_over, 1e-9)
    total_line = parse_line_value(line)

    def expected_profit(mean):
        return sum(
            poisson_pmf(goals, mean)
            * _handicap.over_profit(goals, total_line, fair_over_odds)
            for goals in range(MAX_GOALS_FOR_PROFIT)
        )

    low, high = SEARCH_LOW, SEARCH_HIGH
    for _ in range(SEARCH_ROUNDS):
        mid = (low + high) / 2
        if expected_profit(mid) < 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def target_total(profile_avg_goals, over_odds=None, under_odds=None,
                 asian_factor=1.0, goals_factor=1.0, line=2.5,
                 blend=0.6, factor_low=0.85, factor_high=1.15,
                 total_low=1.8, total_high=3.6):
    """比赛的目标总进球：盘口隐含值与联赛均值融合，再叠加两个趋势因子。

    **联赛均值占大头（blend 是盘口那一侧的权重）。** 盘口会因为一条噪声报价
    大幅跳动，而联赛均值是几百场攒出来的——让盘口主导，个别错报就能把整场
    预测带偏。

    两个因子先夹到 `[factor_low, factor_high]`（软约束），最后的总进球再夹到
    `[total_low, total_high]`（硬约束）。**两道都要**：因子是乘性的，
    两个都到上界就是 1.32 倍，只靠软约束仍会推出 4 球以上的目标。

    `profile_avg_goals` 由调用方从联赛档案里取——**这一层不认识联赛名**。
    """
    implied = implied_total(over_odds, under_odds, line)
    if implied:
        target = blend * implied + (1 - blend) * profile_avg_goals
    else:
        target = profile_avg_goals

    try:
        asian_factor = max(factor_low, min(factor_high, float(asian_factor)))
        goals_factor = max(factor_low, min(factor_high, float(goals_factor)))
    except (ValueError, TypeError):
        # 因子解析不了就当没有趋势，而不是让整场预测崩掉
        asian_factor, goals_factor = 1.0, 1.0

    target = target * asian_factor * goals_factor
    return max(total_low, min(total_high, target))
