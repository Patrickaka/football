"""融合与策略：把规则模型和 ML 模型的两份推荐合成一份，并给出买多少的建议。

**融合不是取平均，是按名次投票。** 两个模型给出的分不在同一量纲上（一个是
加权特征分、一个是模型概率），直接加权求和等于拿米和公斤相加。这里用的是
「第几名」——名次是两边都有的、可比的东西。

**三套策略是给不同风险偏好的人看的，不是三个模型。** 保守=两边都推的交集、
均衡=规则为主少量探索、探索=只有 ML 推的。它们的期望收益都一样（直选恒
1/1000），差别只在推荐池的重合度。

**预算与注数建议全都以「模型没有优势」为默认。** `model_lift <= 0` 时一律
给最低档——3D 是公平摇奖，多数时候这就是实情，让默认档位说实话比让它
好看重要。
"""

# 名次转分数的基准。第一名得 100 分、第一百名得 0 分，再往后不倒扣——
# 一个模型没推的号不该因此被罚，它只是没意见。
RANK_SCORE_BASE = 100
# 两边都推的号加一笔。**这是融合唯一真正新增的信息**：两个独立模型撞到
# 一起，比任何一边单独排得高更值得注意。
BOTH_MODELS_BONUS = 20
# 没进某个模型列表时的名次。比 RANK_SCORE_BASE 大，所以那一项贡献 0 分。
MISSING_RANK = 999

TAG_BOTH = 'high_confidence'
TAG_RULE = 'rule_preferred'
TAG_ML = 'exploration'
TAG_OTHER = 'other'


def fuse(rule_list, ml_list, top_n, rule_weight, ml_weight, detail_for=None):
    """按名次融合两份推荐。

    `detail_for(num)` 用来给 ML 独有的号码补一份得分拆解——规则模型没推过
    它，自然也没有拆解。给不出来时留 `None`，**不编一个**。
    """
    rule_rank = {item['num']: index for index, item in enumerate(rule_list)}
    ml_rank = {item['num']: index for index, item in enumerate(ml_list)}
    rule_detail = {item['num']: item.get('detail') for item in rule_list}

    scored = []
    for num in set(rule_rank) | set(ml_rank):
        in_rule, in_ml = num in rule_rank, num in ml_rank
        value = (_rank_score(rule_rank.get(num, MISSING_RANK)) * rule_weight
                 + _rank_score(ml_rank.get(num, MISSING_RANK)) * ml_weight)
        if in_rule and in_ml:
            value += BOTH_MODELS_BONUS
        scored.append((value, num, _tag(in_rule, in_ml), in_rule))

    # 并列按号码**降序**打破——与迁移前一致。方向本身无所谓，
    # 可复现才是重点：同一份输入两次运行必须给出同一个顺序。
    scored.sort(reverse=True)

    result = []
    for value, num, tag, _ in scored[:top_n]:
        detail = rule_detail.get(num)
        if detail is None and detail_for is not None:
            detail = detail_for(num)
        result.append({
            'num': num,
            # `score` 与 `fuse_score` 是同一个数。前者是页面通用字段名，
            # 后者说明它是融合分而不是模型分——两个名字都保留。
            'score': round(value, 2),
            'fuse_score': round(value, 2),
            'tag': tag,
            'in_rule': num in rule_rank,
            'in_ml': num in ml_rank,
            'rule_rank': rule_rank.get(num),
            'ml_rank': ml_rank.get(num),
            'detail': detail,
        })
    return result


def _rank_score(rank):
    """名次转分：越靠前越高，一百名之后归零而不是转负。"""
    return max(0, RANK_SCORE_BASE - rank)


def _tag(in_rule, in_ml):
    if in_rule and in_ml:
        return TAG_BOTH
    if in_rule:
        return TAG_RULE
    if in_ml:
        return TAG_ML
    return TAG_OTHER


# ─── 三套策略 ───

CONSERVATIVE_SIZE = 10
BALANCED_RULE_SIZE = 20
BALANCED_ML_SIZE = 5
BALANCED_SIZE = 20
EXPLORE_CANDIDATES = 8
EXPLORE_SIZE = 10


def strategy_recommendations(rule_list, ml_list):
    """三套给不同风险偏好的推荐。

    **迁移前这个函数还收 `danma` 与 `kill` 两个参数，函数体一个都没用。**
    删了——调用方传了却不生效，比报错危险得多。
    """
    rule_nums = {item['num'] for item in rule_list}
    ml_nums = {item['num'] for item in ml_list}

    conservative = [item for item in rule_list
                    if item['num'] in ml_nums][:CONSERVATIVE_SIZE]

    balanced = [{'num': item['num'], 'score': item.get('score', 0), 'source': 'rule'}
                for item in rule_list[:BALANCED_RULE_SIZE]]
    added = {item['num'] for item in rule_list[:BALANCED_RULE_SIZE]}
    balanced += [{'num': item['num'], 'score': item.get('model_score', 0), 'source': 'ml'}
                 for item in ml_list if item['num'] not in added][:BALANCED_ML_SIZE]

    explore = [{'num': item['num'], 'score': item.get('model_score', 0),
                'source': 'ml_only'}
               for item in ml_list if item['num'] not in rule_nums][:EXPLORE_CANDIDATES]

    return {'conservative': conservative,
            'balanced': balanced[:BALANCED_SIZE],
            'explore': explore[:EXPLORE_SIZE]}


# ─── 模式、预算、注数 ───

# 「排名优秀」的门槛。1000 注随机排的期望名次是 500，250 是它的一半。
EXCELLENT_RANK = 250
# 认定「确有提升」所需的最小 lift。比它小的差距在几十期样本上分辨不出来。
MEANINGFUL_LIFT = 0.01
STRONG_LIFT = 0.015
# 随机基准命中率：Top30 覆盖 1000 注里的 30 注。
BASELINE_RATE = 0.03
# 稳定度过高的界，与 `records.HIGH_STABILITY` 是同一个概念的同一个值。
HIGH_STABILITY = 0.8


def select_mode(stability, model_lift, recent_hit_rate, actual_rank_avg):
    """自动选推荐模式，返回 (模式, 理由)。

    **理由和模式一样重要**：模式只是三个词，人要看到「为什么是它」才知道
    该不该照做。
    """
    if model_lift <= 0:
        return 'explore', '模型未明显优于随机基准，需要探索'
    if stability > HIGH_STABILITY and recent_hit_rate < BASELINE_RATE:
        return 'explore', '推荐过度稳定且命中率偏低，增加探索'
    if actual_rank_avg <= EXCELLENT_RANK and model_lift > MEANINGFUL_LIFT:
        return 'conservative', '模型排名表现优秀，采用保守策略'
    return 'balanced', '模型有提升但需保持多样性，采用均衡策略'


def budget_level(model_lift, recent_online_rate):
    """资金/注数等级。

    **迁移前这个函数还收一个 `stability`，函数体从没读过它**，而 docstring
    专门列着它。删了。
    """
    if model_lift <= 0:
        return {'level': '低', 'suggest_count': 10, 'reason': '模型未明显优于随机基准'}
    if model_lift > STRONG_LIFT and recent_online_rate >= BASELINE_RATE:
        return {'level': '中', 'suggest_count': 20, 'reason': '模型近期表现略优于随机'}
    return {'level': '观察', 'suggest_count': 10, 'reason': '样本不足或优势不稳定'}


# Top100 覆盖率的两道门槛。随机基准是 10%，所以 0.12 只是「略好」、
# 0.18 才算「明显好」。
DECENT_TOP100 = 0.12
GOOD_TOP100 = 0.18


def recommend_count(model_lift, rank_top100_rate, online_hit_rate):
    """自动调整推荐注数，返回 (注数, 理由)。"""
    if model_lift <= 0:
        return 10, '模型无明显优势，减少推荐注数'
    if rank_top100_rate >= GOOD_TOP100 and online_hit_rate >= BASELINE_RATE:
        return 30, 'Top100覆盖率和线上命中率均良好'
    if rank_top100_rate >= DECENT_TOP100:
        return 20, 'Top100覆盖率尚可'
    return 15, '模型优势有限，保持适中注数'


# ─── 三路策略记录的结算 ───

STRATEGY_LANES = ('rule_only', 'ml_only', 'fused')
HIT_TIERS = (('hit_top3', 3), ('hit_top30', 30), ('hit_top100', 100))
# 没进列表时记的名次。比 1000 大，所以它在平均排名里明确表示「没排上」，
# 而不是碰巧排在最后一名。
UNRANKED = 1001


def settle_row(row, actual, draw_period):
    """结算一条策略记录：三条赛道**各自**统计。

    分开统计是这份记录存在的全部意义——把三路混在一起算命中率，就无从判断
    融合到底有没有比单独的规则模型好。
    """
    row['actual'] = actual
    row['settled'] = True
    row['draw_period'] = draw_period
    for lane in STRATEGY_LANES:
        nums = row.get(lane, [])
        for field, size in HIT_TIERS:
            row[f'{lane}_{field}'] = actual in nums[:size]
        row[f'{lane}_rank'] = nums.index(actual) + 1 if actual in nums else UNRANKED
    return row


def settle_history(history, periods, numbers):
    """结算全部待结算的策略记录，返回是否有改动。

    与预测记录同一条规则：**一条记录预测的是它自己期号的下一期**。
    """
    index_of = {period: index for index, period in enumerate(periods)}
    changed = False
    for row in history:
        if row.get('settled'):
            continue
        index = index_of.get(row['period'])
        if index is None or index + 1 >= len(numbers):
            continue
        settle_row(row, ''.join(map(str, numbers[index + 1])), periods[index + 1])
        changed = True
    return changed


FIRST_REVISION = 1


def new_strategy_record(period, rule_only, ml_only, fused, created_at):
    return {'period': period, 'rule_only': rule_only, 'ml_only': ml_only,
            'fused': fused, 'created_at': created_at, 'settled': False,
            'revision': FIRST_REVISION}


def find_by_period(history, period):
    return next((row for row in history if row['period'] == period), None)
