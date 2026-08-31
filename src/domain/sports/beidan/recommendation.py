"""北单四种玩法的推荐组装：从欧赔到一注可读的建议。

**这一层没有算法，只有顺序。** 四条流水线共用同一个骨架——欧赔 → 去水 →
一组 λ → 一张比分表 → 各自的边际。玩法之间不该出现「胜平负说主胜、
总进球说 0 球」这种自相矛盾，唯一可靠的做法是让它们全部出自同一张表。

外部依赖全部以参数进来，一件不留：

- `ouzhi`：欧赔，由调用方抓好。迁移前是函数体里直接 `fetch_ouzhi_odds`。
- `calibrate(probs, bet_type, league) -> (probs, meta)`：历史校准要读已结算
  快照，那是存储。默认 `identity_calibration`，什么也不做。
- `model` / `market`：两组**已经配好配置**的操作。比分模型要知道联赛档案、
  Dixon-Coles 的 rho、最大进球数、锚定强度；盘口层要知道十几个阈值。
  那些都是配置，不该出现在领域层（判据 10、16），所以由适配层配好后注入。

`model` 需要：`predict_scores` `calibrate_draw` `league_profile`
`target_total` `lambdas` `score_matrix` `anchor_outcomes` `aggregate_goals`
`rqspf_from_scores`
`market` 需要：`latest_total` `apply_joint` `analyze_asian` `analyze_goals`
`analyze_correct_score` `blend_scores_with_market` `asian_goal_factor`
`goals_factor` `adjust_goal_buckets`

两组都是鸭子类型，拼错一个属性要到运行时才炸——`test_recommendation.py`
里有一道按 AST 比对的守卫，把这一层用到的名字与适配层提供的名字对齐。
"""
from src.domain.sports.beidan import (
    analysis, quality as quality_rules, settlement, upset as upset_rules,
)

# 模型与市场的融合权重。**比分与总进球给的不一样**：比分市场只挂出十来个
# 热门比分、深度很浅，市场那一份的信息量本就更少；总进球八个档位全都有价，
# 所以给到 0.45。两组数字迁移前都是函数体里的裸数字。
BIFEN_MODEL_WEIGHT, BIFEN_MARKET_WEIGHT = 0.6, 0.4
ZJQ_MODEL_WEIGHT, ZJQ_MARKET_WEIGHT = 0.55, 0.45

# 欧赔某一档缺失时的兜底概率。三个数加起来正好是 1，所以「三档全缺」等价于
# 「毫无信息」。实际上**只有单档缺失时它才真的起作用**，而那种情况下
# 另外两档的去水概率已经归一过了，兜底值会让三者之和大于 1——
# 紧接着的归一会把它压回去。原样保留自迁移前。
FALLBACK_1X2 = {'胜': 0.33, '平': 0.33, '负': 0.34}

UPSET_CANDIDATES_KEPT = 2
TOP_KEPT = 3
# 「大球」指三球及以上——`over25` 这个名字来自 2.5 的盘口口径，
# 而档位是离散的，所以它列举档位而不是比大小。
OVER_25_BUCKETS = ('3', '4', '5', '6', '7+')


def identity_calibration(probabilities, bet_type, league=None):
    """不做历史校准。领域层的默认值——校准要读已结算历史，那是适配层的事。"""
    return probabilities, {'applied': False, 'reason': 'no_calibrator'}


def _header(match, bet_type, extra=()):
    header = {
        'match_id': match['id'], 'num': match['num'],
        'home': match['home'], 'away': match['away'],
        'league': match['league'], 'time': match['time'],
    }
    for key in extra:
        header[key] = match[key]
    header['type'] = bet_type
    return header


def _euro_odds(match, ouzhi):
    """欧赔的两个来源：单独抓的那份，与赛程里自带的那份。

    返回 `(赔率, 来源标记)`；两处都没有时返回 `(None, None)`。
    赛程自带的那份要三个都在才用——**缺一个就整组不要**，
    补一个默认赔率会让去水后的概率悄悄偏向缺的那一档。
    """
    if ouzhi:
        return ouzhi, None
    prices = (match.get('spf_sp'), match.get('spf_s'), match.get('spf_f'))
    if all(prices):
        return {'home': prices[0], 'draw': prices[1], 'away': prices[2]}, 'zgzcw_main'
    return None, None


def _normalised_1x2(odds_data):
    """欧赔取倒数再归一。

    这里不走 `settlement.implied_probability`——那个会把非正赔率整档丢掉，
    而这条路径要的是「三档恒在」。两者在干净赔率上同解，脏赔率上不同：
    这里会 `ZeroDivisionError`，那边会少一档。原样保留自迁移前。
    """
    home, draw, away = (1 / odds_data['home'], 1 / odds_data['draw'],
                        1 / odds_data['away'])
    total = home + draw + away
    return home / total, draw / total, away / total


def _seeded(probabilities):
    """去水概率补齐三档。缺哪一档用哪一档的兜底值。"""
    return {key: probabilities.get(key, default)
            for key, default in FALLBACK_1X2.items()}


def _blend(model_probs, market_probs, model_weight, market_weight):
    """按权重融合两组概率。

    **两边的键必须是同一种类型**，否则并集会把它们并列成两批互不相干的档位，
    各自只拿到自己那一半权重。比分那一路正是这样（见 `bifen` 的注释）。
    """
    return {key: model_probs.get(key, 0.0) * model_weight
                 + market_probs.get(key, 0.0) * market_weight
            for key in set(model_probs) | set(market_probs)}


def _normalised(probabilities):
    total = sum(probabilities.values())
    if total <= 0:
        return probabilities
    return {key: value / total for key, value in probabilities.items()}


def _goal_factors(market, asian_data, goals_data):
    """盘口与大小球各给一个进球倾向因子。没有历史时是 1.0（不改变任何东西）。"""
    asian = (market.asian_goal_factor(asian_data['history'])
             if asian_data and asian_data.get('history') else 1.0)
    goals = (market.goals_factor(goals_data['history'])
             if goals_data and goals_data.get('history') else 1.0)
    return asian, goals


def _top_scores(score_probs):
    return [{'score': f'{home}-{away}', 'probability': probability,
             'home_goals': home, 'away_goals': away}
            for (home, away), probability
            in sorted(score_probs.items(), key=lambda item: -item[1])[:TOP_KEPT]]


def spf(match, ouzhi, model, market, asian_data=None, cs_data=None,
        goals_data=None, calibrate=identity_calibration,
        upset_candidates=UPSET_CANDIDATES_KEPT):
    """胜平负。**最终概率是比分矩阵的边际，不是去水后的欧赔。**

    去水概率先喂进比分模型、被联合盘口状态修正，再从修正后的矩阵反算胜平负
    ——所以这一注与比分、总进球三者必然自洽。

    **历史校准的结果在这里被覆盖掉了**：它作用在喂进模型的那一版上，
    `probabilities` 随后由矩阵边际重写。所以校准的影响是间接的，走的是
    「它改变了模型的输入」这条路，而不是「它改了输出」。原样保留自迁移前，
    但这与让球那一路不同（那边校准的结果直接留在输出里），值得知道。
    """
    result = _header(match, 'spf')
    odds_data, source = _euro_odds(match, ouzhi)
    if odds_data is None:
        result['error'] = '欧赔数据不可用'
        return result
    if source:
        result['odds_source'] = source

    odds = {'胜': odds_data['home'], '平': odds_data['draw'],
            '负': odds_data['away']}
    probabilities = settlement.implied_probability(odds)
    if not probabilities:
        return result

    result['odds'] = odds
    result['margin'] = sum(1.0 / value for value in odds.values()) - 1.0
    seeds = _seeded(probabilities)
    model_probs = _normalised(seeds)
    model_probs, result['history_calibration'] = calibrate(
        model_probs, 'spf', match.get('league'))
    result['probabilities'] = model_probs
    result['raw_probabilities'] = probabilities
    result['prediction'] = max(model_probs, key=model_probs.get)
    result['confidence'] = model_probs[result['prediction']]

    total_line, total_over, total_under = market.latest_total(goals_data)
    prediction = model.predict_scores(
        model_probs.get('胜', seeds['胜']), model_probs.get('平', seeds['平']),
        model_probs.get('负', seeds['负']),
        league=match['league'], handicap=match.get('handicap', 0),
        total_over_odds=total_over, total_under_odds=total_under,
        total_line=total_line)
    joint_scores, joint_meta = market.apply_joint(
        prediction.get('score_probs'), asian_data, goals_data)
    prediction['score_probs'] = joint_scores
    prediction['top3'] = _top_scores(joint_scores)
    result['joint_market_state'] = joint_meta
    result['asian_adjusted'] = bool(joint_meta.get('applied'))

    model_probs = {
        '胜': sum(p for (h, a), p in joint_scores.items() if h > a),
        '平': sum(p for (h, a), p in joint_scores.items() if h == a),
        '负': sum(p for (h, a), p in joint_scores.items() if h < a),
    }
    result['probabilities'] = model_probs
    result['prediction'] = max(model_probs, key=model_probs.get)
    result['confidence'] = model_probs[result['prediction']]

    if cs_data and cs_data.get('history'):
        prediction = market.blend_scores_with_market(prediction, cs_data['history'])
        result['cs_adjusted'] = True

    result['scores'] = prediction['top3']
    result['score_probs'] = [
        [home, away, round(probability, 6)]
        for (home, away), probability in (prediction.get('score_probs') or {}).items()]
    result['lambda_home'] = prediction['lambda_home']
    result['lambda_away'] = prediction['lambda_away']
    result['target_total'] = prediction.get('target_total')
    result['total_line'] = total_line
    result['outcome_anchor'] = prediction.get('outcome_anchor')

    upset = upset_rules.assess_risk(model_probs)
    upset['candidates'] = upset_rules.pick_scores(
        prediction.get('score_probs'), upset.get('favorite'),
        top_n=upset_candidates)
    upset['chalk_result'] = result['prediction']
    result['upset'] = upset

    context = {}
    if asian_data and asian_data.get('history'):
        result['asian_trend'] = market.analyze_asian(asian_data['history'])
    if result.get('asian_trend'):
        context['asian_direction'] = result['asian_trend'].get('direction')
    result['score_consistency'] = upset_rules.score_consistency(
        result['scores'], result['prediction'])
    context['score_consistency'] = result['score_consistency']
    result['quality'] = quality_rules.assess(
        model_probs, prediction=result['prediction'], context=context)

    if cs_data and cs_data.get('history'):
        result['cs_trend'] = market.analyze_correct_score(cs_data['history'])
    return result


def _official_rqspf_odds(match):
    """官方让球赔率优先用现成的那份，没有就从三个单价拼。

    三个价**都要大于 1.0** 才认——赔率不可能不到 1（那意味着赢了还赔本金），
    小于等于 1 的值只可能是占位符或者抓错了列。
    """
    official = match.get('rqspf_odds') or match.get('lottery_rqspf_odds')
    if official:
        return official
    try:
        prices = [float(match.get(key) or 0.0)
                  for key in ('rqspf_sp', 'rqspf_s', 'rqspf_f')]
    except (TypeError, ValueError):
        return None
    if all(price > 1.0 for price in prices):
        return dict(zip(('让胜', '让平', '让负'), prices))
    return None


def rqspf(match, ouzhi, handicap_value, model, market, asian_data=None,
          goals_data=None, calibrate=identity_calibration):
    """让球胜平负。让球值由调用方解析后传入（页面文本形如 `'(-1)'`）。

    与胜平负的差别不只是多一个让球值：**这里的概率没有再被矩阵边际覆盖**
    ——`rqspf_from_scores` 本来就是从矩阵算出来的，所以校准的结果直接留在
    输出里。同一个 `calibrate` 在两条路上的可见度不同。
    """
    result = _header(match, 'rqspf', extra=('handicap',))
    if handicap_value is None:
        result['error'] = '让球值不可用，无法计算让球胜平负'
        return result

    odds_data, source = _euro_odds(match, ouzhi)
    if odds_data is None:
        result['error'] = '欧赔数据不可用，无法计算让球胜平负'
        return result
    if source:
        result['odds_source'] = source

    odds = {'胜': odds_data['home'], '平': odds_data['draw'],
            '负': odds_data['away']}
    probabilities = settlement.implied_probability(odds)
    if not probabilities:
        result['error'] = '欧赔概率不可用，无法计算让球胜平负'
        return result

    seeds = _seeded(probabilities)
    total_line, total_over, total_under = market.latest_total(goals_data)
    prediction = model.predict_scores(
        seeds['胜'], seeds['平'], seeds['负'],
        league=match.get('league', ''), handicap=handicap_value,
        total_over_odds=total_over, total_under_odds=total_under,
        total_line=total_line)
    prediction['score_probs'], result['joint_market_state'] = market.apply_joint(
        prediction['score_probs'], asian_data, goals_data)
    result['asian_adjusted'] = bool(result['joint_market_state'].get('applied'))

    rq_probs, rq_meta = model.rqspf_from_scores(
        prediction['score_probs'], handicap_value)
    if not rq_probs:
        result['error'] = '让球胜平负概率计算失败'
        result['rqspf_meta'] = rq_meta
        return result

    rq_probs, result['history_calibration'] = calibrate(
        rq_probs, 'rqspf', match.get('league'))

    result['spf_odds'] = odds
    official = _official_rqspf_odds(match)
    result['odds'] = official or {}
    result['official_odds_available'] = bool(official)
    result['raw_spf_probabilities'] = probabilities
    result['probabilities'] = rq_probs
    result['prediction'] = max(rq_probs, key=rq_probs.get)
    result['confidence'] = rq_probs[result['prediction']]
    result['lambda_home'] = prediction['lambda_home']
    result['lambda_away'] = prediction['lambda_away']
    result['target_total'] = prediction.get('target_total')
    result['total_line'] = total_line
    result['outcome_anchor'] = prediction.get('outcome_anchor')
    result['rqspf_meta'] = rq_meta

    context = {}
    if asian_data and asian_data.get('history'):
        result['asian_trend'] = market.analyze_asian(asian_data['history'])
        if result.get('asian_trend'):
            context['asian_direction'] = result['asian_trend'].get('direction')
    result['quality'] = quality_rules.assess(
        rq_probs, prediction=result['prediction'], context=context)
    result['scores'] = [
        {'score': item['score'], 'handicap_score': item['handicap_score'],
         'result': item['result'], 'probability': item['probability']}
        for item in rq_meta.get('top_scores', [])]
    return result


def bifen(match, ouzhi, model, market, market_odds=None, asian_data=None,
          goals_data=None, calibrate=identity_calibration,
          model_weight=BIFEN_MODEL_WEIGHT, market_weight=BIFEN_MARKET_WEIGHT,
          upset_candidates=UPSET_CANDIDATES_KEPT):
    """比分。模型矩阵与市场报价融合后归一。

    **两套键在这里不是同一种类型**：模型矩阵的键是 `(主, 客)` 元组，
    市场报价的键是 `'1-0'` 这样的字符串。融合取的是两者的并集，于是它们
    并列成两批档位——模型的每一格只留下 60%、市场的每一格只留下 40%，
    谁也没有真的融合进谁。归一之后总和仍是 1.0，而 `top3` 几乎总是被市场
    那几个报过价的比分占满，模型对榜首没有任何贡献。

    **原样保留自迁移前**：改掉它会改变用户看到的比分推荐，那是一次产品决策，
    不该是迁移的副产品（判据 19）。市场报价缺席时这条路不走；线上默认请求的
    三个玩法里没有比分，要靠 `types=bifen` 才触发。用例把现状钉住了。
    """
    result = _header(match, 'bifen')
    odds_data, source = _euro_odds(match, ouzhi)
    if odds_data is None:
        result['error'] = '欧赔数据不可用，无法计算比分'
        return result
    if source:
        result['odds_source'] = source

    home, draw, away = _normalised_1x2(odds_data)
    profile = model.league_profile(match.get('league'))
    p_home, p_draw, p_away = model.calibrate_draw(
        home, draw, away, match.get('handicap', 0),
        league_draw_rate=profile['draw_rate'])

    total_line, total_over, total_under = market.latest_total(goals_data)
    asian_factor, goals_factor = _goal_factors(market, asian_data, goals_data)
    target_total = model.target_total(
        match['league'], total_over, total_under, asian_factor, goals_factor,
        total_line=total_line)
    lam_home, lam_away = model.lambdas(p_home, p_draw, p_away, target_total)
    matrix = model.score_matrix(lam_home, lam_away)
    matrix, result['outcome_anchor'] = model.anchor_outcomes(
        matrix, {'胜': p_home, '平': p_draw, '负': p_away})

    # 这里的 `if market_odds` 是**等价变异的来源**：改成一律
    # `implied_probability(market_odds or {})` 之后全语料 667 条一字不差
    # ——空报价过去也是空概率，下一行照样短路（判据 9b）。
    # 留着是因为「没有报价」与「报价全是脏值」是两件事，短路让这一点显式。
    market_probs = (settlement.implied_probability(market_odds)
                    if market_odds else None)
    if market_probs:
        blended = _blend(matrix, market_probs, model_weight, market_weight)
        result['market_adjusted'] = True
        result['odds'] = market_odds
    else:
        blended = dict(matrix)
        result['market_adjusted'] = False
    result['model_based'] = True

    blended, result['history_calibration'] = calibrate(
        blended, 'bifen', match.get('league'))
    blended, result['joint_market_state'] = market.apply_joint(
        blended, asian_data, goals_data)
    blended = _normalised(blended)

    result['probabilities'] = blended
    result['lambda_home'] = lam_home
    result['lambda_away'] = lam_away
    result['target_total'] = target_total

    ranked = sorted(blended.items(), key=lambda item: -item[1])
    result['top3'] = ranked[:TOP_KEPT]
    result['prediction'] = ranked[0][0] if ranked else None
    result['confidence'] = ranked[0][1] if ranked else 0.0
    result['quality'] = quality_rules.assess(
        blended, prediction=result['prediction'], context={})

    upset = upset_rules.assess_risk({'胜': p_home, '平': p_draw, '负': p_away})
    upset['candidates'] = upset_rules.pick_scores(
        blended, upset.get('favorite'), top_n=upset_candidates)
    upset['chalk_result'] = (upset_rules.result_from_score(result['prediction'])
                             if result['prediction'] else None)
    result['upset'] = upset
    return result


def zjq(match, ouzhi, model, market, market_odds=None, asian_data=None,
        goals_data=None, calibrate=identity_calibration,
        model_weight=ZJQ_MODEL_WEIGHT, market_weight=ZJQ_MARKET_WEIGHT,
        over_buckets=OVER_25_BUCKETS):
    """总进球。与比分共用同一组 λ，所以两者的分布天然相加为一。

    这一路的两套键**都是字符串档位**（`'0'`~`'6'` 与 `'7+'`），所以融合是
    真的融合——与比分那一路的差别正在这里。
    """
    result = _header(match, 'zjq')
    odds_data, source = _euro_odds(match, ouzhi)
    if odds_data is None:
        result['error'] = '欧赔数据不可用，无法计算总进球'
        return result
    if source:
        result['odds_source'] = source

    home, draw, away = _normalised_1x2(odds_data)
    total_line, total_over, total_under = market.latest_total(goals_data)
    asian_factor, goals_factor = _goal_factors(market, asian_data, goals_data)
    target_total = model.target_total(
        match.get('league'), total_over, total_under, asian_factor, goals_factor,
        total_line=total_line)
    lam_home, lam_away = model.lambdas(home, draw, away, target_total)
    matrix = model.score_matrix(lam_home, lam_away)
    matrix, result['outcome_anchor'] = model.anchor_outcomes(
        matrix, {'胜': home, '平': draw, '负': away})
    matrix, result['joint_market_state'] = market.apply_joint(
        matrix, asian_data, goals_data)
    probabilities = model.aggregate_goals(matrix)

    if goals_data and goals_data.get('history'):
        probabilities = market.adjust_goal_buckets(probabilities,
                                                   goals_data['history'])
        result['goals_adjusted'] = True

    if market_odds:
        market_probs = settlement.implied_probability(market_odds)
        if market_probs:
            probabilities = _blend(probabilities, market_probs,
                                   model_weight, market_weight)
            result['odds'] = market_odds
            result['market_probabilities'] = market_probs
            result['market_adjusted'] = True

    probabilities = _normalised(probabilities)
    probabilities, result['history_calibration'] = calibrate(
        probabilities, 'zjq', match.get('league'))

    result['probabilities'] = probabilities
    result['mu_home'] = lam_home
    result['mu_away'] = lam_away

    ranked = sorted(probabilities.items(), key=lambda item: -item[1])
    result['top3'] = ranked[:TOP_KEPT]
    result['prediction'] = ranked[0][0]
    result['confidence'] = ranked[0][1]
    result['quality'] = quality_rules.assess(
        probabilities, prediction=result['prediction'], context={})
    result['goal_groups'] = analysis.zjq_groups(probabilities)

    over = sum(probabilities.get(bucket, 0) for bucket in over_buckets)
    result['over25_prob'] = over
    result['under25_prob'] = 1 - over

    if goals_data and goals_data.get('history'):
        result['goals_trend'] = market.analyze_goals(goals_data['history'])
    return result
