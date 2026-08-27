"""推荐质量分档：这一注够不够格单选。

**默认是不够格。** 只有概率够高、领先够多、且与盘口方向不冲突，才允许单选；
其余一律建议双选或跳过。理由是北单一场只押一次，**把「两个选项差 3 个百分点」
当成强推荐，等于把随机波动包装成信心**。

分档结果里同时带上门槛值（`thresholds`）：过不了的时候，人需要看到差多少
才知道是接近了还是差得远。

这一层不读配置——五个门槛全部由调用方传入。
"""
HOME_WIN, DRAW, AWAY_WIN = '胜', '平', '负'

STRONG, MEDIUM, SPLIT, LOW, UNKNOWN = 'strong', 'medium', 'split', 'low', 'unknown'

# 亚盘方向与推荐结果冲突的两种组合。盘口在挺主队而模型推客胜（或反之），
# 说明两个信息源在打架——**这时候不该单选**，哪边对都可能
CONFLICTING = {
    ('home_backing', AWAY_WIN), ('away_laying', AWAY_WIN),
    ('away_backing', HOME_WIN), ('home_laying', HOME_WIN),
}

NO_DATA = {
    'level': UNKNOWN,
    'label': '无数据',
    'confidence': 0,
    'lead': 0,
    'top2': [],
    'advice': '跳过',
    'avoid_single': True,
}


def assess(probabilities, prediction=None, context=None,
           strong_probability=0.65, strong_lead=0.10,
           medium_probability=0.60, medium_lead=0.10,
           high_precision_probability=0.70, split_lead=0.035):
    """把一组结果概率分成 strong / medium / split / low / unknown。

    `split_lead` 是「分歧较大」那一档的门槛——**迁移前它是个写死在判断里的
    裸数字 0.035**，没有名字也没有出处。领先不到 3.5 个百分点时，谁排第一
    基本是噪声决定的，所以宁可报「分歧」也不报「低置信」：后者听起来
    像是有个答案只是没把握，前者才是实情。
    """
    if not probabilities:
        return dict(NO_DATA)

    ranked = sorted(((str(key), float(value))
                     for key, value in probabilities.items() if value is not None),
                    key=lambda item: -item[1])
    if not ranked:
        return dict(NO_DATA)

    top_key, top_prob = ranked[0]
    # 调用方指定了推荐项时以它为准——**评估的是「这一注」而不是「最高的那注」**
    if prediction and prediction != top_key:
        for key, prob in ranked:
            if key == prediction:
                top_key, top_prob = key, prob
                break

    second_prob = ranked[1][1] if len(ranked) > 1 else 0.0
    lead = top_prob - second_prob
    top2 = [{'option': key, 'probability': prob} for key, prob in ranked[:2]]

    context = context or {}
    score_consistency = context.get('score_consistency') or {}
    conflict = ((context.get('asian_direction'), top_key) in CONFLICTING
                or bool(score_consistency.get('conflict')))

    both = '/'.join(item['option'] for item in top2)
    pair_advice = f'建议双选 {both}' if len(top2) >= 2 else f'谨慎 {top_key}'
    if top_prob >= strong_probability and lead >= strong_lead and not conflict:
        level, label, advice = STRONG, '强推荐', f'单选 {top_key}'
    elif top_prob >= medium_probability and lead >= medium_lead and not conflict:
        level, label, advice = MEDIUM, '可参考', pair_advice
    elif lead < split_lead or conflict:
        level, label, advice = SPLIT, '分歧较大', pair_advice
    else:
        level, label, advice = LOW, '低置信', f'谨慎 {top_key}'

    return {
        'level': level,
        'label': label,
        'prediction': top_key,
        'confidence': round(top_prob, 6),
        'lead': round(lead, 6),
        'top2': top2,
        'advice': advice,
        'avoid_single': level != STRONG,
        'single_allowed': level == STRONG,
        'high_precision': level == STRONG and top_prob >= high_precision_probability,
        'thresholds': {
            'strong_probability': strong_probability,
            'strong_lead': strong_lead,
            'medium_probability': medium_probability,
            'medium_lead': medium_lead,
            'high_precision_probability': high_precision_probability,
        },
        'conflict': conflict,
        'score_consistency': score_consistency,
    }
