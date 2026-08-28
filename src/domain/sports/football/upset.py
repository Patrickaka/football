# -*- coding: utf-8 -*-
"""足球爆冷评估：由市场信号累加风险分，再分档给出预警与备选比分。

纯计算——不读全局配置、不碰存储与时钟。所有门槛都在模块级，可参数化覆盖。

**与北单的 upset 不是同一个形状**（判据 28 要求重新验算，不许照抄结论）：
北单那三个门槛是 `AND` 且互相耦合（`mass = 1 - fav` 让其中一条永远不单独
决定结果）；这里是**先累加十来个独立信号成 risk_score，再用 `OR` 分档**
（`fav_p < 0.45 或 risk_score >= 0.55` → high）。任一条单独成立即可，
不存在那种耦合不可达。
"""

import logging

log = logging.getLogger('domain.football.upset')

from .scoring_model import _result_label as result_label


def _evaluate_upset_profile(asian, euro, team=None, total=None,
                            anomaly=None, steam_result=None):
    """Build a direction-neutral, auditable upset-risk profile."""
    asian = asian or {}
    euro = euro or {}
    team = team or {}
    total = total or {}
    anomaly = anomaly or {}
    steam_result = steam_result or {}
    labels = {'home': '胜', 'draw': '平', 'away': '负'}
    close = {
        key: float((euro.get('close') or {}).get(key) or 0.0)
        for key in labels
    }
    open_probs = {
        key: float((euro.get('open') or {}).get(key) or 0.0)
        for key in labels
    }
    favorite_key = max(close, key=close.get) if any(close.values()) else None
    favorite = labels.get(favorite_key)
    ranked = sorted(close.values(), reverse=True)
    favorite_prob = ranked[0] if ranked else 0.0
    gap = favorite_prob - (ranked[1] if len(ranked) > 1 else 0.0)
    score = 0.0
    signals = []

    if favorite_prob and favorite_prob < 0.45:
        score += 0.35
        signals.append('热门概率不足45%，三项高度胶着')
    elif favorite_prob and favorite_prob < 0.52:
        score += 0.20
        signals.append('热门强度有限')
    if favorite_prob and gap < 0.08:
        score += 0.15
        signals.append('热门领先第二方向不足8%')
    elif favorite_prob and gap < 0.15:
        score += 0.08
        signals.append('热门领先优势偏小')

    if favorite_key and open_probs.get(favorite_key, 0) > 0:
        delta = close[favorite_key] - open_probs[favorite_key]
        if delta <= -0.03:
            score += 0.20 if delta <= -0.05 else 0.12
            signals.append(f"热门去水概率由{open_probs[favorite_key]:.1%}降至{close[favorite_key]:.1%}")
        strongest_reverse = max(
            (key for key in labels if key != favorite_key),
            key=lambda key: close[key] - open_probs.get(key, 0.0),
        )
        reverse_delta = close[strongest_reverse] - open_probs.get(strongest_reverse, 0.0)
        if reverse_delta >= 0.03:
            score += 0.15
            signals.append(f"反向{labels[strongest_reverse]}概率升高{reverse_delta:.1%}")

    kelly = euro.get('kelly') or {}
    if favorite_key and kelly.get('hardest') == favorite_key:
        score += 0.15
        signals.append('凯利指数将热门列为最难打出方向')
    favored_key = kelly.get('favored')
    if favorite_key and favored_key in labels and favored_key != favorite_key:
        score += 0.10
        signals.append(f"凯利指数相对支持反向{labels[favored_key]}")

    home_form = float((team.get('home_recent') or {}).get('form_pts') or 0.0) / 3.0
    away_form = float((team.get('away_recent') or {}).get('form_pts') or 0.0) / 3.0
    if favorite_key == 'home' and away_form - home_form >= 0.20:
        score += 0.15
        signals.append('客队近期状态明显优于主队热门')
    elif favorite_key == 'away' and home_form - away_form >= 0.20:
        score += 0.15
        signals.append('主队近期状态明显优于客队热门')

    open_handicap = asian.get('open_handicap')
    close_handicap = asian.get('handicap')
    if open_handicap is not None and close_handicap is not None:
        handicap_delta = float(close_handicap) - float(open_handicap)
        favor = asian.get('favor')
        favorite_weakened = (
            favorite_key == 'home' and favor == 'home' and handicap_delta <= -0.25
        ) or (
            favorite_key == 'away' and favor == 'away' and handicap_delta >= 0.25
        )
        if favorite_weakened:
            score += 0.18
            signals.append(f"热门方向降盘{float(open_handicap):+.2f}→{float(close_handicap):+.2f}")

    if favorite_key in {'home', 'away'}:
        open_water = asian.get('open_water') or {}
        close_water = asian.get('close_water') or {}
        if favorite_key in open_water and favorite_key in close_water:
            water_delta = float(close_water[favorite_key]) - float(open_water[favorite_key])
            if water_delta >= 0.08:
                score += 0.12
                signals.append(f"热门方向升水{water_delta:+.2f}")

    try:
        total_delta = float(total.get('close_line')) - float(total.get('open_line'))
    except (TypeError, ValueError):
        total_delta = 0.0
    if favorite_key in {'home', 'away'} and total_delta <= -0.25:
        score += 0.08
        signals.append('大小球降盘，热门穿透能力受压')

    deviation = (anomaly.get('euro_asian_deviation') or {}).get('abs_deviation')
    try:
        deviation = float(deviation or 0.0)
    except (TypeError, ValueError):
        deviation = 0.0
    if deviation >= 0.50:
        score += 0.18
        signals.append('欧赔与亚盘明显背离')
    elif deviation >= 0.35:
        score += 0.10
        signals.append('欧赔与亚盘存在背离')

    steam_summary = steam_result.get('summary') or {}
    steam_text = str(steam_summary.get('recommendation') or '')
    if steam_summary.get('dominant_signal') in {'steam_drop', 'trap'} or any(
        word in steam_text for word in ('冷门', '反向')
    ):
        score += 0.15
        signals.append('临场资金流向热门反方向')

    return {
        'risk_score': max(0.0, min(1.0, score)),
        'signals': signals,
        'favorite': favorite,
        'favorite_key': favorite_key,
        'favorite_prob': favorite_prob,
        'gap': gap,
    }


def _evaluate_upset_risk(asian, euro, team=None):
    """Compatibility wrapper for callers that only consume a numeric score."""
    return _evaluate_upset_profile(asian, euro, team)['risk_score']


def assess_football_upset(asian, euro, team, candidates, total=None,
                          anomaly=None, steam_result=None):
    """评估爆冷（热门被击败）风险，并挑出反向爆冷比分候选。

    对齐北单模块的爆冷识别能力：把足球模块内部已有的 ``_evaluate_upset_risk``
    （盘口走势 / 凯利 / 球队状态 / 欧赔变化）显式暴露为结构化的
    ``{level, alert, favorite, candidates}``，供前端展示「爆冷预警」。

    返回结构与北单 ``analyze_bifen`` 的 ``result['upset']`` 兼容。
    """
    profile = _evaluate_upset_profile(
        asian, euro, team, total=total, anomaly=anomaly,
        steam_result=steam_result,
    )
    risk_score = profile['risk_score']
    p_home = float(euro.get('close', {}).get('home', 0.0) or 0.0)
    p_draw = float(euro.get('close', {}).get('draw', 0.0) or 0.0)
    p_away = float(euro.get('close', {}).get('away', 0.0) or 0.0)
    probs = {'胜': p_home, '平': p_draw, '负': p_away}
    favorite = max(probs, key=probs.get)
    fav_p = probs[favorite]
    upset_p = 1.0 - fav_p
    ranked_p = sorted(probs.values(), reverse=True)
    gap = ranked_p[0] - (ranked_p[1] if len(ranked_p) > 1 else 0.0)

    # 两级阈值：以热门强度为主信号（参考北单回测：爆冷率随热门强度
    # 单调可分，fav<0.45 时爆冷率约 53%~62%），盘口/欧赔异常 risk_score 为辅。
    if fav_p < 0.45 or risk_score >= 0.55:
        level, alert = 'high', True
    elif fav_p < 0.52 or risk_score >= 0.3:
        level, alert = 'medium', True
    else:
        level, alert = 'low', False

    # 反向「稳胆」档：强热门 + 差距悬殊 → 405 场真实结算样本冷门率仅约 30%
    # (train 26.7% / test 33.9%，两半均稳)，区分真稳胆与约 44% 抛硬币的弱热门。
    # 与北单 assess_upset_risk 同阈值；risk_score 偏高（盘口异常）时不判稳胆。
    confident = (not alert and fav_p >= 0.58 and gap >= 0.20 and risk_score < 0.3)

    # 反向方向分成“逼平”和“真正逆转获胜”。旧逻辑把两者混在一起按
    # 单比分概率排序，结果所谓爆冷候选几乎永远又是 0-0 / 1-1。
    # 高风险时优先展示非热门方直接获胜；中风险才优先用平局防冷。
    if favorite == '胜':
        outright = lambda h, a: h < a
        cover_draw = lambda h, a: h == a
    elif favorite == '负':
        outright = lambda h, a: h > a
        cover_draw = lambda h, a: h == a
    else:
        outright = lambda h, a: h != a
        cover_draw = lambda h, a: False

    outright_picked = []
    draw_picked = []
    for (h, a), prob in (candidates or []):
        target = outright_picked if outright(h, a) else draw_picked if cover_draw(h, a) else None
        if target is not None:
            target.append({
                'score': f"{h}-{a}",
                'result': result_label(h, a),
                'probability': float(prob),
                'scenario': 'outright_upset' if target is outright_picked else 'draw_cover',
            })
    outright_picked.sort(key=lambda x: -x['probability'])
    draw_picked.sort(key=lambda x: -x['probability'])
    if level == 'high':
        picked = (outright_picked[:1] + draw_picked[:1])[:2]
    else:
        picked = (draw_picked[:1] + outright_picked[:1])[:2]

    label = {'high': '🔴高风险爆冷', 'medium': '🟠需警惕爆冷', 'low': '稳健'}.get(level, '稳健')
    if confident:
        label = '✅热门稳胆'
    reverse_labels = {
        '胜': [('平', '防冷平'), ('负', '客胜冷门')],
        '负': [('平', '防冷平'), ('胜', '主胜冷门')],
        '平': [('胜', '主胜反向'), ('负', '客胜反向')],
    }.get(favorite, [])
    defensive_selections = [
        {
            'result': result_label,
            'type': selection_type,
            'probability': probs.get(result_label, 0.0),
        }
        for result_label, selection_type in reverse_labels
    ] if alert else []
    defensive_selections.sort(key=lambda item: -item['probability'])
    return {
        'level': level,
        'label': label,
        'alert': alert,
        'confident': confident,
        'favorite': favorite,
        'favorite_prob': fav_p,
        'upset_prob': upset_p,
        'gap': gap,
        'risk_score': risk_score,
        'signals': profile.get('signals') or [],
        'defensive_selections': defensive_selections,
        'recommended_cover': '/'.join(
            item['result'] for item in defensive_selections
        ) if defensive_selections else None,
        'candidates': picked,
        'outright_candidates': outright_picked[:2],
        'draw_candidates': draw_picked[:2],
    }
