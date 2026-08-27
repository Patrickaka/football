"""亚洲盘口：让球解析、结算线拆分、让球胜平负换算。

**分盘（`0.25` / `0.75`）会拆成两条结算线，各承担一半本金。** 这是亚盘与
欧盘最大的结构差异：欧盘一场只有一个结果，亚盘的一注可能一半赢一半走水。
把它当成单条线处理，隐含总进球会算偏——`_asian_over_profit` 就是为此存在的。

这一层只做盘口的算术，不认识联赛、不读配置。
"""
import math
import re

# 亚盘的最小刻度是四分之一球。解析时先归到这个刻度上，
# 否则 2.7 这种脏数据会落进两条线之间，`fraction` 判断跟着失准。
LINE_STEP = 0.25

WIN, PUSH, LOSE = '让胜', '让平', '让负'
TOP_SCORES_KEPT = 5


def parse(handicap):
    """把盘口解析成带符号的浮点数。解析不出来返回 `None`。

    **返回 `None` 而不是 0.0**：0.0 是「平手盘」这个真实盘口，
    用它表示「没有盘口」会让两种完全不同的情况混成一个值。
    """
    if handicap is None:
        return None
    if isinstance(handicap, (int, float)):
        return float(handicap)

    text = str(handicap).strip()
    if not text:
        return None
    # 全角括号来自页面复制，`(-1)` 与 `（-1）` 是同一个盘口
    text = text.replace('（', '(').replace('）', ')')
    match = re.search(r'[-+]?\d+(?:\.\d+)?', text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def line_parts(line):
    """一条亚盘线实际对应的结算线：整数与半球一条，分盘两条。

    `2.25` 结算在 2 和 2.5 两条线上，各一半本金；`2.75` 结算在 2.5 和 3。
    """
    line = round(float(line) / LINE_STEP) * LINE_STEP
    fraction = round(line - math.floor(line), 2)
    if fraction == 0.25:
        return (math.floor(line), math.floor(line) + 0.5)
    if fraction == 0.75:
        return (math.floor(line) + 0.5, math.floor(line) + 1.0)
    return (line,)


def over_profit(goals, line, decimal_odds):
    """买大球在给定进球数下的**平均**收益（本金为 1）。

    分盘时两条结算线各算一次再平均——这正是「一半赢一半走水」的算术表达。
    走水记 0（退本金），不是记赢。
    """
    profits = []
    for settlement_line in line_parts(line):
        if goals > settlement_line:
            profits.append(decimal_odds - 1.0)
        elif goals == settlement_line:
            profits.append(0.0)
        else:
            profits.append(-1.0)
    return sum(profits) / len(profits)


def rqspf_from_scores(score_probs, handicap):
    """让球胜平负的概率，以及贡献最大的几个比分。

    让球后的净胜球 = 主队进球 + 盘口 - 客队进球。**盘口带符号**，
    主队让球时它是负数，所以这里是加不是减。
    """
    handicap_value = parse(handicap)
    if handicap_value is None:
        return {}, {'available': False, 'reason': 'missing_handicap'}

    probs = {WIN: 0.0, PUSH: 0.0, LOSE: 0.0}
    scored = []
    for (home_goals, away_goals), prob in score_probs.items():
        margin = home_goals + handicap_value - away_goals
        label = WIN if margin > 0 else (LOSE if margin < 0 else PUSH)
        probs[label] += prob
        scored.append({
            'score': f"{home_goals}-{away_goals}",
            'handicap_score': f"{home_goals + handicap_value:g}-{away_goals}",
            'result': label,
            'probability': prob,
        })

    total = sum(probs.values())
    if total > 0:
        probs = {key: value / total for key, value in probs.items()}

    scored.sort(key=lambda item: -item['probability'])
    return probs, {
        'available': True,
        'handicap': handicap_value,
        'top_scores': scored[:TOP_SCORES_KEPT],
    }
