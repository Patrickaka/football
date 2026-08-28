"""盘口走势：亚盘、大小球、比分三条线的方向、强度与派生因子。

**这一层量的是「市场往哪边动」，不是「哪边会赢」。** 赔率降说明钱压过来了，
但庄家降赔也可能只是在平衡两边的敞口——所以每个因子都是**弱修正**，
调整幅度控制在 15% 到 30% 之间，不足以单独翻转一个结论。

迁移前这些阈值全是函数体里的裸数字（`0.02`、`0.03`、`0.05`、`0.15`、
`1.2`、`0.85`…），既没有名字也没有出处，改一个要先读懂整段代码。
现在一律由调用方传入。
"""
HOME_WIN, DRAW, AWAY_WIN = '胜', '平', '负'

STABLE = 'stable'
HOME_BACKING, HOME_LAYING = 'home_backing', 'home_laying'
AWAY_BACKING, AWAY_LAYING = 'away_backing', 'away_laying'
OVER_BACKING, OVER_LAYING = 'over_backing', 'over_laying'
UNDER_BACKING, UNDER_LAYING = 'under_backing', 'under_laying'

# 总进球的低分桶与高分桶。**键是字符串**，与 `scoring_model.aggregate_goals`
# 保持一致——那边的 '7+' 本来就不是数字
LOW_GOAL_BUCKETS = ('0', '1', '2')
HIGH_GOAL_BUCKETS = ('3', '4', '5', '6', '7+')


def _paired_changes(history, window, first_key, second_key):
    """相邻两期的赔率变化。缺任一侧的那一对直接跳过——**不补 0**：
    补 0 会把「没有数据」算成「没有变化」，而后者是一个真实的信号。
    """
    recent = history[-window:] if history else []
    first, second = [], []
    for index in range(1, len(recent)):
        previous, current = recent[index - 1], recent[index]
        if previous.get(first_key) and current.get(first_key):
            first.append(current[first_key] - previous[first_key])
        if previous.get(second_key) and current.get(second_key):
            second.append(current[second_key] - previous[second_key])
    return first, second


def _mean(values):
    return sum(values) / len(values) if values else 0


def adjust_probs_by_asian(home_prob, draw_prob, away_prob, asian_history,
                          window=5, move_threshold=0.02, factor=0.15,
                          counter_ratio=0.5):
    """按亚盘水位的走向微调 1X2，再归一化。

    降赔的一方加权、升赔的一方减权，**反方向只给一半的力度**
    （`counter_ratio`）：一边的水位在动不代表另一边同样确定，
    对称调整会把一条信息用成两条。

    平局概率不参与调整——亚盘让球盘本来就不表达平局。
    """
    if not asian_history or len(asian_history) < 2:
        return home_prob, draw_prob, away_prob

    home_changes, away_changes = _paired_changes(
        asian_history, window, 'home_odds', 'away_odds')
    home_trend, away_trend = _mean(home_changes), _mean(away_changes)

    if home_trend > move_threshold:
        home_prob *= (1 - factor)
        away_prob *= (1 + factor * counter_ratio)
    elif home_trend < -move_threshold:
        home_prob *= (1 + factor)
        away_prob *= (1 - factor * counter_ratio)

    if away_trend > move_threshold:
        away_prob *= (1 - factor)
        home_prob *= (1 + factor * counter_ratio)
    elif away_trend < -move_threshold:
        away_prob *= (1 + factor)
        home_prob *= (1 - factor * counter_ratio)

    total = home_prob + draw_prob + away_prob
    if total > 0:
        return home_prob / total, draw_prob / total, away_prob / total
    return home_prob, draw_prob, away_prob


def analyze_asian(asian_history, window=5, direction_threshold=0.03):
    """亚盘走势的方向与强度。

    **方向按主队水位优先判定**：主队降赔就是 `home_backing`，
    只有主队没动才看客队。两边同时动的情况在真实盘口里很少，
    真出现时以主队为准是有意的——北单的盘口以主队为基准报价。
    """
    if not asian_history:
        return {'direction': STABLE, 'strength': 0}
    recent = asian_history[-window:]
    if len(recent) < 2:
        return {'direction': STABLE, 'strength': 0}

    home_changes, away_changes = _paired_changes(
        asian_history, window, 'home_odds', 'away_odds')
    home_change, away_change = _mean(home_changes), _mean(away_changes)

    if home_change < -direction_threshold:
        direction = HOME_BACKING
    elif home_change > direction_threshold:
        direction = HOME_LAYING
    elif away_change < -direction_threshold:
        direction = AWAY_BACKING
    elif away_change > direction_threshold:
        direction = AWAY_LAYING
    else:
        direction = STABLE

    return {
        'direction': direction,
        'strength': round(abs(home_change) + abs(away_change), 4),
        'avg_home_change': round(home_change, 4),
        'avg_away_change': round(away_change, 4),
    }


def analyze_goals(goals_history, window=10, direction_threshold=0.05):
    """大小球走势。门槛比亚盘松（0.05 对 0.03）——大小球的水位波动本来就大。"""
    if not goals_history or len(goals_history) < 2:
        return {'direction': STABLE, 'strength': 0}

    over_changes, under_changes = _paired_changes(
        goals_history, window, 'over_odds', 'under_odds')
    over_change, under_change = _mean(over_changes), _mean(under_changes)

    if over_change < -direction_threshold:
        direction = OVER_BACKING
    elif over_change > direction_threshold:
        direction = OVER_LAYING
    elif under_change < -direction_threshold:
        direction = UNDER_BACKING
    elif under_change > direction_threshold:
        direction = UNDER_LAYING
    else:
        direction = STABLE

    return {
        'direction': direction,
        'strength': round(abs(over_change) + abs(under_change), 4),
        'avg_over_change': round(over_change, 4),
        'avg_under_change': round(under_change, 4),
    }


def analyze_correct_score(cs_history, window=10, move_threshold=0.1, kept=5):
    """比分盘的热门榜：按当前赔率从低到高排，赔率低的就是被押得多的。

    `direction` 只有 `active` 与 `stable` 两种——**比分盘没有方向可言**，
    它有几十个选项，说「往哪边动」没有意义，只能说「动没动」。
    """
    if not cs_history or len(cs_history) < 2:
        return {'direction': STABLE, 'strength': 0, 'hot_scores': []}

    by_score = {}
    for entry in cs_history[-window:]:
        score, odds = entry.get('score'), entry.get('odds')
        if score and odds:
            by_score.setdefault(score, []).append(odds)

    hot = []
    for score, odds_list in by_score.items():
        move = odds_list[-1] - odds_list[0] if len(odds_list) >= 2 else 0
        hot.append({
            'score': score,
            'avg_odds': round(sum(odds_list) / len(odds_list), 2),
            'trend': ('down' if move < -move_threshold
                      else ('up' if move > move_threshold else STABLE)),
            'current_odds': odds_list[-1],
        })
    hot.sort(key=lambda item: item['current_odds'])

    return {
        'direction': 'active' if hot else STABLE,
        'strength': len(hot),
        'hot_scores': hot[:kept],
    }


def blend_scores_with_market(top_scores, cs_history, window=5,
                             new_score_discount=0.5, kept=3):
    """把模型的比分概率与比分盘的隐含概率取平均。

    **返回新的列表，不改入参**——迁移前这里直接改写传进来的字典，
    调用方拿到的和传出去的是同一个对象，读的人分不清哪一份是原始值。

    盘口上有、模型没算到的比分按 `new_score_discount` 打折补进来：
    比分盘有几十个选项，全额采信会让长尾比分挤掉模型的主推。
    """
    if not cs_history or len(cs_history) < 2:
        return list(top_scores)

    market_odds = {}
    for entry in cs_history[-window:]:
        score, odds = entry.get('score'), entry.get('odds')
        if score and odds:
            market_odds[score] = odds
    if not market_odds:
        return list(top_scores)

    blended = []
    for item in top_scores:
        if isinstance(item, dict):
            score, prob = item.get('score'), item.get('probability', 0)
        else:
            score, prob = item[0], item[1]
        if not score:
            continue
        if score in market_odds:
            blended.append((score, (prob + 1.0 / market_odds[score]) / 2, 'cs_enhanced'))
        else:
            blended.append((score, prob, 'poisson'))

    seen = {item[0] for item in blended}
    for score, odds in market_odds.items():
        if score not in seen:
            blended.append((score, (1.0 / odds) * new_score_discount, 'cs_new'))

    blended.sort(key=lambda item: -item[1])
    return [{'score': score,
             'probability': prob,
             'source': source,
             'home_goals': _goal_part(score, 0),
             'away_goals': _goal_part(score, 1)}
            for score, prob, source in blended[:kept]]


def _goal_part(score, index):
    """从 `'2-1'` 取出一侧的进球数。取不到返回 `None` 而不是 0——
    **0 是一个真实的比分**，用它表示「解析失败」会让两种情况混起来。
    """
    parts = str(score).split('-')
    if len(parts) <= index or not parts[index].isdigit():
        return None
    return int(parts[index])


def goals_factor(goals_history, window=10, lean_over=1.2, lean_under=0.85,
                 neutral=1.0, under_margin=0.5):
    """由大小球水位推一个总进球的乘性因子。

    大球水位低于小球（`avg_over < avg_under`）说明市场偏大球 → 因子上调。
    反向要多出 `under_margin` 才认——**两边不对称是有意的**：
    大球贴水天然偏低，用对称门槛会把常态误判成偏小球。
    """
    if not goals_history or len(goals_history) < 2:
        return neutral

    over_total = under_total = 0.0
    count = 0
    for entry in goals_history[-window:]:
        over, under = entry.get('over_odds'), entry.get('under_odds')
        if over and under:
            over_total += over
            under_total += under
            count += 1
    if count == 0:
        return neutral

    average_over, average_under = over_total / count, under_total / count
    if average_over < average_under:
        return lean_over
    if average_over > average_under + under_margin:
        return lean_under
    return neutral


def asian_goal_factor(asian_history, window=10,
                      tiers=((3.6, 1.3), (4.0, 1.15), (4.4, 1.0), (4.8, 0.9)),
                      floor=0.75):
    """由亚盘两侧水位之和推一个总进球因子。

    水位之和越低，说明盘口越「紧」——双方赔率都压得住，通常对应一场
    互相试探的比赛，进球预期反而高。**分档而不是连续函数**是有意的：
    水位之和的绝对值受报价习惯影响，分档比线性映射稳。
    """
    if not asian_history or len(asian_history) < 2:
        return 1.0

    total = 0.0
    count = 0
    for entry in asian_history[-window:]:
        home, away = entry.get('home_odds'), entry.get('away_odds')
        if home and away:
            total += home + away
            count += 1
    if count == 0:
        return 1.0

    average = total / count
    for bound, factor in tiers:
        if average < bound:
            return factor
    return floor


def adjust_goal_buckets(bucket_probs, goals_history, window=5,
                        trend_threshold=0.05, lift=1.2, cut=0.85,
                        low_buckets=LOW_GOAL_BUCKETS,
                        high_buckets=HIGH_GOAL_BUCKETS):
    """按大小球走势抬高或压低总进球分桶。**返回新字典，不改入参。**

    大球水位在降（钱压大球）→ 抬高高分桶、压低低分桶；反之亦然。
    只看大球那一侧的趋势——**迁移前这里还累加了小球的趋势，
    但那个值从头到尾没被读过**。
    """
    if not goals_history or len(goals_history) < 2:
        return dict(bucket_probs)

    recent = goals_history[-window:]
    over_trend = 0.0
    count = 0
    for index in range(1, len(recent)):
        previous, current = recent[index - 1], recent[index]
        if previous.get('over_odds') and current.get('over_odds'):
            over_trend += current['over_odds'] - previous['over_odds']
            count += 1
    if count == 0:
        return dict(bucket_probs)

    average = over_trend / count
    adjusted = dict(bucket_probs)
    if average < -trend_threshold:
        rise, fall = high_buckets, low_buckets
    elif average > trend_threshold:
        rise, fall = low_buckets, high_buckets
    else:
        return adjusted

    for key in rise:
        adjusted[key] = adjusted.get(key, 0) * lift
    for key in fall:
        adjusted[key] = adjusted.get(key, 0) * cut
    return adjusted
