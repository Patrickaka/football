# -*- coding: utf-8 -*-
"""竞彩玩法概率：由比分分布 + 官方赔率算胜平负 / 让球胜平负。

纯计算——不读全局配置、不发请求、不看时钟。官方赔率权重与胜平负选号的三个
门槛都是参数（判据 10）。默认值取自迁移当时 `src/football/config.py` 的真实
取值，是公开契约的一部分，要有不传参数的用例守着（判据 29）。
"""

import re

# 迁移当时 config 的真实取值
LOTTERY_OFFICIAL_ODDS_WEIGHT = 0.80

# 原先内联在 `_spf_selection_profile` 里的三个门槛
DRAW_COVER_MIN_PROBABILITY = 0.24   # 平局概率够高才值得加保
DRAW_COVER_MAX_GAP = 0.12           # 首选非平时，与平局的差距上限
DRAW_PRIMARY_MAX_GAP = 0.08         # 首选是平时，与次选的差距上限

# **前两个门槛是耦合的**（判据 28）：概率和为 1 时，
# `平 < 0.24` 且 `首选 - 平 <= 0.12` 推出 `首选 <= 0.36`，
# 于是第三项 `= 1 - 首选 - 平 >= 0.40 > 首选`，与「首选是最大值」矛盾。
# 所以**归一化输入下 0.24 那道门槛永远不会单独决定结果**——差距那道更严。
# （已用 0.005 步长穷举整个单纯形验证：零个反例。）
# 把 `DRAW_COVER_MIN_PROBABILITY` 调低到 0.10 不会改变任何归一化输入的输出。
# 它只在调用方传未归一化的概率时才有作用。

MAX_LOTTERY_HANDICAP = 5
SETTLEMENT_RULE = '中国体彩：主队进球 + 让球数，与客队进球比较'
LINKED_RULE = '先取胜平负最高概率，再在同一赛果兼容的让球结果中分析'


def parse_lottery_handicap(value):
    """Parse the integer home-team handicap used by China Sports Lottery.

    This is deliberately separate from the Asian handicap.  For example, a
    lottery handicap of ``-1`` means the settlement score is home goals - 1
    versus away goals; quarter-ball Asian lines are never accepted here.
    """
    if value is None or value == '':
        return None
    match = re.search(r'[-+]?\d+(?:\.\d+)?', str(value).replace('（', '(').replace('）', ')'))
    if not match:
        return None
    try:
        number = float(match.group(0))
    except ValueError:
        return None
    if not number.is_integer() or abs(number) > MAX_LOTTERY_HANDICAP:
        return None
    return int(number)


def lottery_odds_probabilities(odds, keys):
    """Return normalized, overround-removed probabilities for one lottery market."""
    implied = {}
    for key in keys:
        try:
            value = float((odds or {}).get(key))
        except (TypeError, ValueError):
            return None
        if value <= 1.0:
            return None
        implied[key] = 1.0 / value
    total = sum(implied.values())
    return {key: value / total for key, value in implied.items()} if total > 0 else None


def blend_lottery_probabilities(model_probs, market_probs,
                                market_weight=LOTTERY_OFFICIAL_ODDS_WEIGHT):
    """模型概率与官方赔率概率的线性融合；无市场概率时原样返回模型概率"""
    if not market_probs:
        return dict(model_probs)
    weight = max(0.0, min(1.0, float(market_weight)))
    blended = {
        key: (1.0 - weight) * float(model_probs.get(key, 0.0))
        + weight * float(market_probs.get(key, 0.0))
        for key in model_probs
    }
    total = sum(blended.values())
    return {key: value / total for key, value in blended.items()} if total > 0 else blended


def spf_selection_profile(probabilities, *,
                          draw_cover_min=DRAW_COVER_MIN_PROBABILITY,
                          draw_cover_max_gap=DRAW_COVER_MAX_GAP,
                          draw_primary_max_gap=DRAW_PRIMARY_MAX_GAP):
    """Keep top-1 honest while exposing a close draw as explicit cover."""
    probs = {key: max(0.0, float((probabilities or {}).get(key, 0.0)))
             for key in ('胜', '平', '负')}
    ranked = sorted(probs, key=lambda key: (-probs[key], key))
    primary = ranked[0] if ranked and sum(probs.values()) > 0 else None
    if not primary:
        return {'primary': None, 'selections': [], 'mode': 'unavailable'}

    runner_up = ranked[1]
    gap = probs[primary] - probs[runner_up]
    selections = [primary]
    reason = 'top_probability'
    mode = 'single'

    if primary != '平':
        draw_gap = probs[primary] - probs['平']
        if probs['平'] >= draw_cover_min and draw_gap <= draw_cover_max_gap:
            selections.append('平')
            mode = 'draw_cover'
            reason = 'draw_probability_close_to_primary'
    elif probs[runner_up] >= draw_cover_min and gap <= draw_primary_max_gap:
        selections.append(runner_up)
        mode = 'draw_primary_cover'
        reason = 'draw_primary_but_margin_is_small'

    return {
        'primary': primary,
        'selections': selections,
        'mode': mode,
        'reason': reason,
        'primary_probability': probs[primary],
        'draw_probability': probs['平'],
        'top_gap': gap,
        'is_single': len(selections) == 1,
    }


def _normalize(values):
    total = sum(values.values())
    return {key: value / total for key, value in values.items()} if total > 0 else values


def _accumulate_outcomes(candidates, handicap):
    """把比分候选摊到胜平负与让球胜平负两组，并记下二者的联合分布"""
    spf = {'胜': 0.0, '平': 0.0, '负': 0.0}
    rqspf = {'让胜': 0.0, '让平': 0.0, '让负': 0.0} if handicap is not None else None
    joint = {}

    for item in candidates or []:
        try:
            (home_goals, away_goals), probability = item
            home_goals, away_goals = int(home_goals), int(away_goals)
            probability = float(probability)
        except (TypeError, ValueError):
            continue
        margin = home_goals - away_goals
        standard_label = '胜' if margin > 0 else '负' if margin < 0 else '平'
        spf[standard_label] += probability
        if rqspf is not None:
            adjusted = margin + handicap
            label = '让胜' if adjusted > 0 else '让负' if adjusted < 0 else '让平'
            rqspf[label] += probability
            key = (standard_label, label)
            joint[key] = joint.get(key, 0.0) + probability

    return spf, rqspf, joint


def _joint_recommendation(joint_ranked):
    if not joint_ranked:
        return None
    (standard_pick, handicap_pick), joint_probability = joint_ranked[0]
    return {
        'standard_prediction': standard_pick,
        'handicap_prediction': handicap_pick,
        'probability': joint_probability,
        'label': f'{standard_pick} + {handicap_pick}',
        'distribution': [
            {'standard': standard_result, 'handicap': handicap_result,
             'probability': probability}
            for (standard_result, handicap_result), probability in joint_ranked
        ],
    }


def _linked_recommendation(spf, rqspf, model_rqspf, joint_probs):
    """在胜平负首选兼容的让球结果里，按市场因子重新加权后取最高

    市场因子 = 融合后概率 / 模型概率——把官方赔率对让球盘的看法带进条件分布。
    """
    standard_pick = max(spf, key=spf.get)
    compatible = {
        rq_result: probability
        for (standard_result, rq_result), probability in joint_probs.items()
        if standard_result == standard_pick and probability > 0
    }
    adjusted = {}
    for rq_result, probability in compatible.items():
        model_value = float((model_rqspf or {}).get(rq_result, 0.0))
        fused_value = float(rqspf.get(rq_result, 0.0))
        market_factor = fused_value / model_value if model_value > 0 else 1.0
        adjusted[rq_result] = probability * market_factor
    adjusted_total = sum(adjusted.values())
    conditional = ({key: value / adjusted_total for key, value in adjusted.items()}
                   if adjusted_total > 0 else {})
    if not conditional:
        return None
    handicap_pick = max(conditional, key=conditional.get)
    return {
        'standard_prediction': standard_pick,
        'handicap_prediction': handicap_pick,
        'compatible_handicap_predictions': list(conditional),
        'handicap_conditional_probabilities': conditional,
        'conditional_probability': conditional[handicap_pick],
        'label': f'{standard_pick} ⇒ {handicap_pick}',
        'rule': LINKED_RULE,
    }


def lottery_market_probabilities(candidates, lottery_handicap=None,
                                 spf_odds=None, rqspf_odds=None, *,
                                 market_weight=LOTTERY_OFFICIAL_ODDS_WEIGHT,
                                 **selection_kwargs):
    """Build JCZQ probabilities from scores and independently priced official markets."""
    handicap = parse_lottery_handicap(lottery_handicap)
    spf, rqspf, joint_probs = _accumulate_outcomes(candidates, handicap)

    spf = _normalize(spf)
    if rqspf is not None:
        rqspf = _normalize(rqspf)
        joint_total = sum(joint_probs.values())
        if joint_total > 0:
            joint_probs = {key: value / joint_total for key, value in joint_probs.items()}

    joint_ranked = sorted(joint_probs.items(), key=lambda item: -item[1])
    joint_recommendation = _joint_recommendation(joint_ranked)

    model_spf = dict(spf)
    model_rqspf = dict(rqspf) if rqspf is not None else None
    market_spf = lottery_odds_probabilities(spf_odds, ('胜', '平', '负'))
    market_rqspf = (lottery_odds_probabilities(rqspf_odds, ('让胜', '让平', '让负'))
                    if rqspf is not None else None)

    spf = blend_lottery_probabilities(model_spf, market_spf, market_weight)
    spf_selection = spf_selection_profile(spf, **selection_kwargs)
    if rqspf is not None:
        rqspf = blend_lottery_probabilities(model_rqspf, market_rqspf, market_weight)

    linked_recommendation = None
    if rqspf is not None and joint_probs:
        linked_recommendation = _linked_recommendation(spf, rqspf, model_rqspf, joint_probs)

    primary_type = 'rqspf' if handicap not in (None, 0) else 'spf'
    primary_probs = rqspf if primary_type == 'rqspf' else spf

    return {
        'standard': {
            'type': 'spf', 'name': '胜平负', 'probabilities': spf,
            'model_probabilities': model_spf,
            'market_probabilities': market_spf,
            'market_weight': market_weight if market_spf else 0.0,
            'prediction': max(spf, key=spf.get) if sum(spf.values()) > 0 else None,
            'selections': spf_selection['selections'],
            'selection_profile': spf_selection,
        },
        'handicap': ({
            'type': 'rqspf', 'name': '让球胜平负', 'handicap': handicap,
            'probabilities': rqspf,
            'model_probabilities': model_rqspf,
            'market_probabilities': market_rqspf,
            'market_weight': market_weight if market_rqspf else 0.0,
            'prediction': (max(rqspf, key=rqspf.get)
                           if rqspf and sum(rqspf.values()) > 0 else None),
        } if rqspf is not None else None),
        'primary_market': primary_type,
        'primary': {
            'type': primary_type,
            'probabilities': primary_probs,
            'prediction': (max(primary_probs, key=primary_probs.get)
                           if primary_probs and sum(primary_probs.values()) > 0 else None),
        },
        'joint_recommendation': joint_recommendation,
        'linked_recommendation': linked_recommendation,
        'settlement_rule': SETTLEMENT_RULE,
    }


def apply_lottery_market_availability(lottery):
    """对已匹配的竞彩场次，关闭官方未开售的胜平负输出。

    **就地改传入的 dict**——迁移前就是这个契约，调用方依赖它。
    """
    spf_prediction_enabled = (
        not lottery.get('offer_matched') or bool(lottery.get('spf_available'))
    )
    if not spf_prediction_enabled:
        # 模型内部仍可用比分分布分析让球玩法，对外不产生 SPF 推荐。
        lottery['standard'] = None
        lottery['joint_recommendation'] = None
        lottery['linked_recommendation'] = None
    return spf_prediction_enabled
