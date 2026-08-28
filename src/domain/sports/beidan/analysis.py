"""北单的赛果解读与组装：把一份比分分布翻译成人能读的结论。

这一层不产生新的预测信号。`scoring_model` 算出的那张比分概率表是唯一的
事实来源，**胜平负、进球数、比分推荐全部从它边际化得出**——三者因此天然
自洽。迁移前这段话写在函数的 docstring 里，而实现散落在 194 行里，中间
夹着四个没有名字的裸数字。

依赖 `src/common/local_match_analysis`：那是足球、篮球、北单共用的一套
「单选 / 双选 / 观望」判定，只 import `math.isfinite`，没有状态也没有 IO。
三个项目共用同一套判定是有意的——**同一个概率在不同页面上不该给出不同的
建议**，各写一份迟早会漂（判据 11）。
"""
from datetime import datetime, timedelta

from src.common.local_match_analysis import (
    LOCAL_ANALYST_VERSION, build_decision, build_score_strategy,
    normalize_probabilities, pick_high_score_scenario,
)
from src.domain.sports.beidan import settlement, upset as upset_rules

# 大小球的分界线。2.5 是行业惯例的中位盘，**取它是为了让「大球/小球」这个词
# 与用户在盘口上看到的那条线对得上**，不是模型算出来的。
TOTAL_LINE = 2.5
# 进球偏大到什么程度才值得额外给一注「大比分」。0.52 只比五五开高一点点——
# 这一注是补充覆盖，不是主推，门槛定高了它就永远不出现。
HIGH_SCORE_MIN_OVER = 0.52
# 首选领先次选多少算「优势明显」。只影响结论句的措辞，不影响任何计算。
CLEAR_EDGE_MARGIN = 0.12
# 上游没给置信度时的兜底。0.5 = 「不知道」，而不是「一半把握」。
DEFAULT_CONFIDENCE = 0.5
TOP_GOALS_KEPT = 3

_FAVORITE_CN = {'home': '胜', 'draw': '平', 'away': '负'}

# 总进球的三个组。**`'2'` 同时属于小球组与中位组**，三组之和大于 1——
# 这是有意的重叠：它们是三种可选的投注方式，不是一个划分。
ZJQ_GROUPS = (
    ('small', '小球组', ('0', '1', '2')),
    ('middle', '中位组', ('2', '3')),
    ('big', '大球组', ('3', '4', '5', '6', '7+')),
)


def _as_score_matrix(score_probs):
    """比分分布有两种形态：元组键字典与 JSON 安全的三元组列表。

    列表形态是落盘用的（JSON 不允许元组键），读回来时长度不为 3 的项直接
    丢掉——那是写坏的记录，补一个默认值只会让它混进统计里。
    """
    if not isinstance(score_probs, list):
        return score_probs or {}
    matrix = {}
    for item in score_probs:
        if len(item) == 3:
            home, away, probability = item
            matrix[(int(home), int(away))] = float(probability)
    return matrix


def _outcome_marginals(candidates):
    """从比分分布边际化出胜平负，同时按总进球数归并。"""
    win = draw = lose = 0.0
    goals = {}
    for (home, away), probability in candidates:
        if home > away:
            win += probability
        elif home == away:
            draw += probability
        else:
            lose += probability
        total = home + away
        goals[total] = goals.get(total, 0.0) + probability
    mass = win + draw + lose
    if mass > 0:
        win, draw, lose = win / mass, draw / mass, lose / mass
    return (win, draw, lose), goals


def _pick_secondary(ranked, primary_score):
    """次选要与首推**在某个维度上不同**，否则两注等于一注。

    三个维度任一不同即可：胜负方向、是否平局、总进球数。`2-1` 与 `3-2` 都是
    主胜，但进球数不同，仍算有效的次选——覆盖的是「赢几个」这条不确定性。
    实在找不到就退回概率第二高的那个。
    """
    home, away = primary_score
    for score, probability in ranked[1:]:
        candidate_home, candidate_away = score
        if ((candidate_home > candidate_away) != (home > away)
                or (candidate_home == candidate_away) != (home == away)
                or (candidate_home + candidate_away) != (home + away)):
            return score, probability
    return ranked[1] if len(ranked) > 1 else None


def _score_pick(kind, score, probability, **extra):
    home, away = score
    return {
        'type': kind,
        'score': f'{home}-{away}',
        'home': home, 'away': away,
        'result': upset_rules.result_from_score(score),
        'probability': probability,
        **extra,
    }


def _upset_pick(upset, matrix):
    """爆冷候选来自 `upset`，而它的比分是字符串——解析不出来就没有这一注。

    吞掉解析失败而不是抛：防冷是**附加**的一注，它缺席不该让整份分析没有。
    """
    candidates = upset.get('candidates') or []
    if not (upset.get('alert') and candidates):
        return None
    try:
        home, away = (int(part) for part in candidates[0]['score'].split('-'))
    except (ValueError, KeyError):
        return None
    return _score_pick('防冷', (home, away), matrix.get((home, away), 0.0))


def _goals_reading(goals, total_line, top_kept):
    over = sum(mass for count, mass in goals.items() if count > total_line)
    under = sum(mass for count, mass in goals.items() if count < total_line)
    ranked = sorted(goals.items(), key=lambda item: -item[1])
    top_two = ranked[:2]
    interval = (f'{top_two[0][0]}-{top_two[1][0]}球' if len(top_two) > 1
                else f'{top_two[0][0]}球')
    direction, direction_prob = ('大球', over) if over >= under else ('小球', under)
    return {
        'expected': sum(count * mass for count, mass in goals.items()),
        'line': total_line,
        'over_prob': over,
        'under_prob': under,
        'direction': direction,
        'direction_prob': direction_prob,
        'most_likely_interval': interval,
        'top_goals': [{'goals': count, 'probability': mass}
                      for count, mass in ranked[:top_kept]],
    }


def _reasons(spf_result, lambdas, upset, favorite_prob):
    """给人看的理由叙述。**这一段不参与任何计算**，所以没有任何断言盯着它
    ——判据 12 说的正是这里，漏一条没人会发现。"""
    lam_home, lam_away = lambdas
    odds = spf_result.get('odds') or {}
    asian_trend = spf_result.get('asian_trend')
    reasons = []
    if lam_home is not None and lam_away is not None:
        bias = ('主队进攻占优' if lam_home > lam_away else
                ('客队进攻占优' if lam_away > lam_home else '双方攻防均衡'))
        reasons.append(f'模型预期进球 主{lam_home:.1f}/客{lam_away:.1f}，{bias}')
    if odds:
        reasons.append(f"欧赔 胜{odds.get('胜')}/平{odds.get('平')}/负{odds.get('负')}")
    if asian_trend and asian_trend.get('direction'):
        reasons.append(f"亚盘走势：{asian_trend.get('direction')}")
    if upset.get('alert'):
        reasons.append(
            f"⚠️ 爆冷预警（{upset.get('label')}）：热门{upset.get('favorite')}"
            f"仅{favorite_prob:.0%}，关注反向比分")
    elif upset.get('confident'):
        reasons.append(
            f"✅ 热门稳胆：{upset.get('favorite')}{favorite_prob:.0%} 且领先次选 "
            f"{upset.get('gap'):.0%}，真实冷门率约 30%")
    return reasons


def _verdict(spf_result, favorite, favorite_prob, draw_prob, margin, clear_edge):
    edge = '明显' if margin >= clear_edge else '有限'
    if favorite == 'home':
        return f"{spf_result.get('home', '主队')}胜面最高 {favorite_prob:.0%}，优势{edge}"
    if favorite == 'away':
        return f"{spf_result.get('away', '客队')}胜面最高 {favorite_prob:.0%}，优势{edge}"
    return f'平局概率 {draw_prob:.0%}，双方势均力敌'


def build_match_analysis(spf_result, min_single, min_margin,
                         total_line=TOTAL_LINE,
                         high_score_min_over=HIGH_SCORE_MIN_OVER,
                         clear_edge_margin=CLEAR_EDGE_MARGIN,
                         default_confidence=DEFAULT_CONFIDENCE,
                         top_goals_kept=TOP_GOALS_KEPT):
    """把一份胜平负分析结果翻译成完整的赛果解读。输入不可用时返回 `None`。

    **不吞异常**：迁移前整个函数体裹在一个 `except Exception` 里，任何缺陷
    都会变成一句 warning 加一个 `None`，与「这场数据不全」长得一模一样
    （判据 6）。现在由适配层决定要不要兜——领域层只管算不出来时返回 `None`。
    """
    if not spf_result or spf_result.get('error'):
        return None
    matrix = _as_score_matrix(spf_result.get('score_probs'))
    if not matrix:
        return None

    upset = spf_result.get('upset') or {}
    candidates = list(matrix.items())
    (win, draw, lose), goals = _outcome_marginals(candidates)

    probabilities = normalize_probabilities(
        {'home': win, 'draw': draw, 'away': lose})
    win, draw, lose = (probabilities['home'], probabilities['draw'],
                       probabilities['away'])
    favorite = max(probabilities, key=probabilities.get)
    favorite_prob = probabilities[favorite]
    margin = favorite_prob - sorted(probabilities.values(), reverse=True)[1]

    ranked = sorted(candidates, key=lambda item: -item[1])
    primary_score, primary_prob = ranked[0]
    score_picks = [_score_pick('首推', primary_score, primary_prob)]
    secondary = _pick_secondary(ranked, primary_score)
    if secondary:
        score_picks.append(_score_pick('次选', secondary[0], secondary[1]))
    upset_pick = _upset_pick(upset, matrix)
    if upset_pick:
        score_picks.append(upset_pick)

    goals_read = _goals_reading(goals, total_line, top_goals_kept)
    high_scenario = pick_high_score_scenario(candidates)
    if goals_read['over_prob'] >= high_score_min_over and high_scenario:
        score = high_scenario['score']
        if not any(pick['home'] == score[0] and pick['away'] == score[1]
                   for pick in score_picks):
            score_picks.append(_score_pick(
                '大比分', score, high_scenario['probability'],
                scenario_probability=high_scenario['tail_probability']))
    goals_read['high_score_probability'] = (
        high_scenario['tail_probability'] if high_scenario else 0.0)

    quality_level = (spf_result.get('quality') or {}).get('level')
    # 「分歧较大」与「低」在决策层是同一件事：都表示 top1 不可信。
    # 合并发生在这里而不是 quality 里——quality 要区分它们，决策不需要。
    confidence_level = ('low' if quality_level in ('low', 'split')
                        else quality_level)
    alert = bool(upset.get('alert'))
    return {
        'analysis_model': LOCAL_ANALYST_VERSION,
        'verdict': _verdict(spf_result, favorite, favorite_prob, draw, margin,
                            clear_edge_margin),
        'favorite': _FAVORITE_CN.get(favorite, '胜'),
        'favorite_prob': favorite_prob,
        'margin': margin,
        'wdl': {'胜': win, '平': draw, '负': lose},
        'score_picks': score_picks,
        'goals': goals_read,
        'reasons': _reasons(
            spf_result,
            (spf_result.get('lambda_home'), spf_result.get('lambda_away')),
            upset, favorite_prob),
        'confidence': spf_result.get('confidence') or default_confidence,
        'risk_level': quality_level,
        'upset_alert': alert,
        'decision': build_decision(probabilities, confidence=confidence_level,
                                   upset_alert=alert, min_single=min_single,
                                   min_margin=min_margin),
        'score_strategy': build_score_strategy(candidates,
                                               confidence=confidence_level,
                                               upset_alert=alert),
    }


def zjq_groups(zjq_probs, definitions=ZJQ_GROUPS):
    """把总进球的八个档位归并成三个可投注的组，按概率从高到低排。"""
    if not zjq_probs:
        return {'groups': [], 'primary': None}

    groups = [{
        'key': key,
        'label': label,
        'options': list(options),
        'probability': round(
            sum(float(zjq_probs.get(option, 0) or 0) for option in options), 6),
        'advice': f"{label} {'/'.join(options)}",
    } for key, label, options in definitions]

    groups.sort(key=lambda group: -group['probability'])
    return {'groups': groups, 'primary': groups[0] if groups else None}


def bqc(match, bqc_odds):
    """半全场：九种组合的去水概率，外加半场与全场两个边际。

    **没有 `probabilities` 时也不写 `error`**（与胜平负那几个不同）——
    原样保留自迁移前，调用方靠键是否存在来判断。
    """
    result = {
        'match_id': match['id'], 'num': match['num'],
        'home': match['home'], 'away': match['away'],
        'league': match['league'], 'time': match['time'], 'type': 'bqc',
    }
    if match['id'] not in bqc_odds:
        result['error'] = '半全场数据不可用'
        return result

    odds = bqc_odds[match['id']]
    probabilities = settlement.implied_probability(odds)
    if not probabilities:
        return result

    ranked = sorted(probabilities.items(), key=lambda item: -item[1])
    half, full = {}, {}
    for combination, probability in probabilities.items():
        half[combination[0]] = half.get(combination[0], 0) + probability
        full[combination[1]] = full.get(combination[1], 0) + probability
    result.update({
        'odds': odds,
        'probabilities': probabilities,
        'top3': ranked[:3],
        'prediction': ranked[0][0],
        'confidence': ranked[0][1],
        'half_probabilities': half,
        'full_probabilities': full,
    })
    return result


def total_goals_gate_inputs(section, line_water, model_over_from=3, max_goals=7):
    """把北单的港水大小球换算成准入门槛要的两组输入。

    返回 `(市场概率, 模型大球, 模型小球)`。港水报的是**净赢**，加 1 才是欧赔
    ——漏掉这个 1，两边的隐含概率会同时偏高，比值却看起来正常。
    """
    over_water, under_water = line_water
    market = {}
    try:
        over_decimal = 1.0 + float(over_water)
        under_decimal = 1.0 + float(under_water)
        inverse = {'over': 1.0 / over_decimal, 'under': 1.0 / under_decimal}
        total = sum(inverse.values())
        if over_decimal > 1.0 and under_decimal > 1.0 and total > 0:
            market = {key: value / total for key, value in inverse.items()}
    except (TypeError, ValueError, ZeroDivisionError):
        market = {}

    model_over = model_under = 0.0
    for key, value in ((section or {}).get('probabilities') or {}).items():
        try:
            goals = max_goals if str(key) == f'{max_goals}+' else int(key)
            probability = float(value)
        except (TypeError, ValueError):
            continue
        if goals >= model_over_from:
            model_over += probability
        else:
            model_under += probability
    return market, model_over, model_under


def value_bets(recommendations, threshold=0.05,
               bet_types=('spf', 'rqspf', 'zjq'),
               skipped_options={'zjq': ('7+',)},
               lenient_odds=('zjq',)):
    """挑出模型概率高于市场隐含概率的注，按优势从大到小排。

    `skipped_options` 里的档位不参与：`'7+'` 是个开区间，它的赔率对应的是
    「7 球及以上」，而模型给的是同一个开区间的概率——**两者口径相同，
    但这一档的赔率长期虚高**，算出来的优势是假的。

    `lenient_odds` 记的是一处不对称：总进球那支用 `.get` 取赔率，另外两支
    直接下标——`probabilities` 在而 `odds` 不在时前者跳过、后者 KeyError。
    原样保留自迁移前（判据 17 说的「一半严格一半放任」），把它写成参数是为了
    让这处不对称有个名字，而不是继续藏在两段几乎相同的代码里。
    """
    picks = []
    for match in recommendations:
        for bet_type in bet_types:
            if bet_type not in match:
                continue
            section = match[bet_type]
            if 'probabilities' not in section:
                continue
            odds = (section.get('odds') or {} if bet_type in lenient_odds
                    else section['odds'])
            skipped = skipped_options.get(bet_type, ())
            for option, probability in section['probabilities'].items():
                if option in skipped:
                    continue
                price = odds.get(option)
                if not (price and price > 0):
                    continue
                implied = 1 / price
                edge = probability - implied
                if edge > threshold:
                    picks.append({
                        'num': match['num'], 'home': match['home'],
                        'away': match['away'], 'type': bet_type,
                        'option': option, 'probability': probability,
                        'odd': price, 'implied_probability': implied,
                        'edge': edge,
                    })
    return sorted(picks, key=lambda pick: -pick['edge'])


def candidate_dates(date, allow_fallback=True, days=2):
    """当天没有赛程时往后顺延几天。日期解析不了就只用原样那一个。"""
    dates = [date]
    if not allow_fallback:
        return dates
    try:
        base = datetime.strptime(date, '%Y-%m-%d')
    except (TypeError, ValueError):
        return dates
    for offset in range(1, days + 1):
        candidate = base + timedelta(days=offset)
        dates.append(f'{candidate.year:04d}-{candidate.month:02d}-'
                     f'{candidate.day:02d}')
    return dates
