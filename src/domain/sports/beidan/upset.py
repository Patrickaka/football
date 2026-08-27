"""爆冷风险：热门到底稳不稳，不稳的话往哪边防。

**这一层不预测爆冷，它衡量的是「热门有多不像热门」。** 三项概率挤在一起、
非热门合计过半，说明市场自己也没把握——那种场次单押热门的风险被低估了。

三档判定，每档三个条件全部满足才算：
- `high`   热门弱、差距小、非热门质量高 → 爆冷高风险
- `medium` 同样三条但门槛松一档 → 爆冷预警
- `confident` 反向的一档：**未预警**且热门够强、差距够大 → 热门稳胆

`confident` 与前两档不是同一个维度：它只在没预警的场次里再细分一次，
不改变 `level` 与 `alert` 的取值，所以加它不会破坏既有调用方。

这一层不读配置——八个门槛全部由调用方传入。
"""
HOME_WIN, DRAW, AWAY_WIN = '胜', '平', '负'
HIGH, MEDIUM, LOW = 'high', 'medium', 'low'

# 热门是某个结果时，防守方向该往哪两边押。
# **平局是热门时防的是两侧**，而不是「另一个平局」
REVERSE_DIRECTIONS = {
    HOME_WIN: ((DRAW, '防冷平'), (AWAY_WIN, '客胜冷门')),
    AWAY_WIN: ((DRAW, '防冷平'), (HOME_WIN, '主胜冷门')),
    DRAW: ((HOME_WIN, '主胜反向'), (AWAY_WIN, '客胜反向')),
}

EMPTY_RISK = {'level': LOW, 'alert': False, 'favorite': None,
              'favorite_prob': 0.0, 'upset_prob': 0.0, 'gap': 0.0}


def result_from_score(key):
    """从比分推出赛果。`(1, 0)`、`'1-0'`、`'1:0'` 三种写法都认。

    **认错了不会报错**，只会让爆冷比分挑到相反的方向去——所以三种写法
    都要有语料。解析不了返回 `None`，由调用方跳过。
    """
    try:
        if isinstance(key, (tuple, list)):
            home, away = int(key[0]), int(key[1])
        else:
            home, away = (int(part) for part
                          in str(key).replace(':', '-').split('-')[:2])
    except (ValueError, TypeError, IndexError):
        return None
    return HOME_WIN if home > away else (AWAY_WIN if home < away else DRAW)


def format_score(key):
    if isinstance(key, (tuple, list)):
        return f"{int(key[0])}-{int(key[1])}"
    return str(key)


def assess_risk(probabilities,
                high_fav_max=0.45, high_gap_max=0.10, high_mass_min=0.58,
                medium_fav_max=0.52, medium_gap_max=0.16, medium_mass_min=0.52,
                confident_fav_min=0.58, confident_gap_min=0.20):
    """按 1X2 概率评估爆冷风险，并给出防守方向。

    `upset_prob` 是**非热门的合计概率**，不是「爆冷的概率」——它衡量的是
    赔付风险摊在多少个结果上，与真实爆冷率是两回事。
    """
    if not probabilities:
        return dict(EMPTY_RISK)
    probs = {key: float(value) for key, value in probabilities.items()
             if value is not None}
    if not probs:
        return dict(EMPTY_RISK)

    ranked = sorted(probs.items(), key=lambda item: -item[1])
    favorite, favorite_prob = ranked[0]
    second_prob = ranked[1][1] if len(ranked) > 1 else 0.0
    gap = favorite_prob - second_prob
    upset_mass = 1.0 - favorite_prob

    if (favorite_prob < high_fav_max and gap <= high_gap_max
            and upset_mass >= high_mass_min):
        level, label, alert = HIGH, '爆冷高风险', True
    elif (favorite_prob < medium_fav_max and gap <= medium_gap_max
          and upset_mass >= medium_mass_min):
        level, label, alert = MEDIUM, '爆冷预警', True
    else:
        level, label, alert = LOW, '热门稳健', False

    # 反向的一档：只在未预警的场次里再细分，**不动 level 与 alert**
    confident = (not alert and favorite_prob >= confident_fav_min
                 and gap >= confident_gap_min)
    if confident:
        label = '热门稳胆'

    defensive = []
    if alert:
        defensive = [{'result': result, 'type': kind,
                      'probability': round(probs.get(result, 0.0), 6)}
                     for result, kind in REVERSE_DIRECTIONS.get(favorite, ())]
        defensive.sort(key=lambda item: -item['probability'])

    signals = []
    if alert:
        signals.append('热门强度不足' if favorite_prob >= high_fav_max else '弱热门')
        if gap <= high_gap_max:
            signals.append('三项概率胶着')
        if upset_mass >= high_mass_min:
            signals.append('非热门合计概率偏高')

    return {
        'level': level,
        'label': label,
        'alert': alert,
        'confident': confident,
        'favorite': favorite,
        'favorite_prob': round(favorite_prob, 6),
        'upset_prob': round(upset_mass, 6),
        'gap': round(gap, 6),
        'signals': signals,
        'defensive_selections': defensive,
        'recommended_cover': ('/'.join(item['result'] for item in defensive)
                              if defensive else None),
    }


def pick_scores(score_matrix, favorite_result, top_n=2):
    """从比分分布里挑出**与热门赛果相反方向**上概率最高的几个比分。

    热门主胜就挑平/负的比分，热门平局就挑胜/负两侧。挑同方向的没有意义——
    那不是防守，是加倍。
    """
    if not score_matrix or not favorite_result:
        return []
    if favorite_result == HOME_WIN:
        allowed = {DRAW, AWAY_WIN}
    elif favorite_result == AWAY_WIN:
        allowed = {HOME_WIN, DRAW}
    else:
        allowed = {HOME_WIN, AWAY_WIN}

    picked = []
    for key, prob in sorted(score_matrix.items(), key=lambda item: -item[1]):
        if result_from_score(key) in allowed:
            picked.append({'score': format_score(key),
                           'result': result_from_score(key),
                           'probability': round(float(prob), 6)})
        if len(picked) >= top_n:
            break
    return picked


def _label_of(score):
    """从一条比分记录里取赛果。记录可能是 dict，也可能是 (比分, 概率) 元组。"""
    if isinstance(score, dict):
        home, away = score.get('home_goals'), score.get('away_goals')
        if home is not None and away is not None:
            try:
                return result_from_score((int(home), int(away)))
            except (TypeError, ValueError):
                pass
        text = str(score.get('score', ''))
    else:
        text = str(score[0]) if score else ''
    return result_from_score(text) if '-' in text else None


def score_consistency(scores, prediction, top_n=3, min_agreement=0.45,
                      fallback_weight=0.25, min_fallback=0.1):
    """比分分布指向的赛果，与 1X2 推荐是否一致。

    **不一致本身不算冲突**——概率最高的比分是 1-1 而推荐主胜，很正常。
    只有当「支持推荐的比分权重」也低于 `min_agreement` 时才算冲突：
    那说明比分与 1X2 两条路径给出的是两个不同的答案。

    没有概率时按名次退化加权（第一名 1.0、第二名 0.75…），
    下限 `min_fallback` 防止排得靠后的比分权重变成 0 或负数。
    """
    if not scores or not prediction:
        return {'available': False, 'conflict': False}

    weights = {HOME_WIN: 0.0, DRAW: 0.0, AWAY_WIN: 0.0}
    top_result = None
    total = 0.0
    for index, score in enumerate(scores[:top_n]):
        result = _label_of(score)
        if not result:
            continue
        if top_result is None:
            top_result = result
        probability = (score.get('probability') if isinstance(score, dict)
                       else (score[1] if len(score) > 1 else None))
        weight = (float(probability) if probability is not None
                  else max(min_fallback, 1.0 - index * fallback_weight))
        weights[result] += weight
        total += weight

    if total <= 0:
        return {'available': False, 'conflict': False}

    agreement = weights.get(prediction, 0.0) / total
    return {
        'available': True,
        'conflict': top_result != prediction and agreement < min_agreement,
        'top_score_result': top_result,
        'agreement': round(agreement, 6),
        'result_weights': {key: round(value / total, 6)
                           for key, value in weights.items()},
    }
