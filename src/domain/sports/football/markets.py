# -*- coding: utf-8 -*-
"""足球市场口径：亚盘 / 欧赔 / 大小球 / 凯利 / 离散度 / 联合异常。

纯计算——不读全局配置、不发请求、不看时钟。**每一个阈值都是参数**
（判据 10：迁移时的处置是「改成参数传入」，不是「把常量抄对」；
领域层内联配置本身就是错的，抄对了下次还会错）。

默认值取自迁移当时 `src/football/config.py` 的真实取值，它们是**公开契约的
一部分**，所以每个都要有一条不传参数的用例守着（判据 29）。

适配层 `src/football/markets.py` 负责把 config 里的值显式喂进来。
"""

import math

# 迁移当时 config 的真实取值（已实测，不是按命名推测——判据 10）
HANDICAP_TREND_EPS = 0.02
WATER_TREND_EPS = 0.05
EURO_PROB_TREND_EPS = 0.02
KELLY_BIAS_EPS = 2.0
TOTAL_LEAN_THRESHOLD = 0.55

# 原先内联在函数体里的常量，一并提到模块级并参数化
TREND_STRENGTH_SCALE = 0.5          # 让球/盘口变化归一到 [0,1] 的分母
SIGNAL_STRONG_DELTA = 0.5
SIGNAL_MEDIUM_DELTA = 0.25
ASIAN_LAMBDA_HOME_SCALE = 0.15
ASIAN_LAMBDA_AWAY_SCALE = -0.05
TOTAL_LAMBDA_SCALE = 0.6
TOTAL_LINE_TREND_EPS = 0.125        # 注意：与亚盘的 0.02 不是同一个数量级
KELLY_NEUTRAL_SPREAD = 1.0
KELLY_DIVERGED_SPREAD = 4.0
KELLY_CHANGE_EPS = 1.0
KELLY_SLOPE_EPS = 0.2
KELLY_TREND_RECENT_N = 5
EURO_HANDICAP_K = 1.8
MOMENTUM_SHIFT_SCALE = 1.8
MOMENTUM_SHIFT_CLAMP = 0.45
POISSON_TAIL_MAX_GOALS = 30
IMPLIED_TOTAL_BOUNDS = (0.3, 6.5)
IMPLIED_TOTAL_ITERS = 48
IMPLIED_TOTAL_PROB_CLAMP = (0.02, 0.98)
DEFAULT_RETURN_RATE = 92.0


def remove_vig(o1, o2, o3=None):
    """去水率，返回真实概率"""
    if o3 is None:
        p1, p2 = 1 / o1, 1 / o2
        total = p1 + p2
        return p1 / total, p2 / total
    p1, p2, p3 = 1 / o1, 1 / o2, 1 / o3
    total = p1 + p2 + p3
    return p1 / total, p2 / total, p3 / total


def handicap_trend_text(open_hcap, close_hcap, eps=HANDICAP_TREND_EPS):
    """让球走势的文字描述"""
    dh = close_hcap - open_hcap
    if dh > eps:
        return f"让球升高 {open_hcap:+.2f} → {close_hcap:+.2f}（主队被看好）"
    if dh < -eps:
        return f"让球降低 {open_hcap:+.2f} → {close_hcap:+.2f}（客队被看好）"
    return f"让球不变 {close_hcap:+.2f}（盘口稳定）"


def nudge_total_by_water(line, over_odds, under_odds, nudge=0.1):
    """按水位对盘口线做一次微调，作为隐含总进球的粗略基础

    低水的一方更被看好：大球低水上调，小球低水下调。
    """
    if over_odds < under_odds:
        return line + nudge
    if under_odds < over_odds:
        return line - nudge
    return line


def _signal_strength(delta, strong=SIGNAL_STRONG_DELTA, medium=SIGNAL_MEDIUM_DELTA):
    if abs(delta) >= strong:
        return 'strong'
    if abs(delta) >= medium:
        return 'medium'
    return 'weak'


def _handicap_expectation(hcap):
    """由让球值给出预期球差区间与描述"""
    a = abs(hcap)
    if a <= 0.25:
        return [0, 0.5], "势均力敌"
    if a <= 0.75:
        return [0.5, 1.5], "预期1球差"
    if a <= 1.25:
        return [1, 2], "预期1-2球差"
    if a <= 1.75:
        return [1.5, 2.5], "预期2球差"
    return [a - 0.25, a + 0.25], f"预期{a:.1f}球差以上"


def analyze_asian(data, *,
                  handicap_eps=HANDICAP_TREND_EPS,
                  water_eps=WATER_TREND_EPS,
                  trend_scale=TREND_STRENGTH_SCALE,
                  signal_strong=SIGNAL_STRONG_DELTA,
                  signal_medium=SIGNAL_MEDIUM_DELTA,
                  lambda_home_scale=ASIAN_LAMBDA_HOME_SCALE,
                  lambda_away_scale=ASIAN_LAMBDA_AWAY_SCALE):
    """解析亚盘：让球走势、水位走势、去水概率与强弱判断"""
    if not isinstance(data, dict):
        raise ValueError(f"亚盘数据格式错误，期望字典但得到: {type(data)}")
    if 'open' not in data:
        raise ValueError(f"亚盘数据缺少 'open' 键，可用键: {list(data.keys())}")
    if 'close' not in data:
        raise ValueError(f"亚盘数据缺少 'close' 键，可用键: {list(data.keys())}")

    op, cl = data['open'], data['close']
    hcap, open_hcap = cl['handicap'], op['handicap']
    dh = hcap - open_hcap

    if dh > handicap_eps:
        handicap_trend = f"让球升高 {open_hcap:+.2f} → {hcap:+.2f}（主队被看好）"
        trend_direction, trend_strength = 'up', min(dh / trend_scale, 1.0)
    elif dh < -handicap_eps:
        handicap_trend = f"让球降低 {open_hcap:+.2f} → {hcap:+.2f}（客队被看好）"
        trend_direction, trend_strength = 'down', min(-dh / trend_scale, 1.0)
    else:
        handicap_trend = f"让球不变 {hcap:+.2f}（盘口稳定）"
        trend_direction, trend_strength = 'stable', 0.0

    dw = cl['home_odds'] - op['home_odds']
    if dw > water_eps:
        water_trend, water_direction = "主队水位上升 → 资金偏向客队", 'up'
    elif dw < -water_eps:
        water_trend, water_direction = "主队水位下降 → 资金偏向主队", 'down'
    else:
        water_trend, water_direction = "水位基本稳定", 'stable'

    hp_o, ap_o = remove_vig(op['home_odds'], op['away_odds'])
    hp_c, ap_c = remove_vig(cl['home_odds'], cl['away_odds'])

    diff_range, diff_desc = _handicap_expectation(hcap)

    if hcap > 0:
        favor, favor_desc = 'home', f"主队让 {hcap} 球（主强客弱）"
        open_prob_label = {'home_give': hp_o, 'away_recv': ap_o}
        close_prob_label = {'home_give': hp_c, 'away_recv': ap_c}
    elif hcap < 0:
        favor, favor_desc = 'away', f"客队让 {abs(hcap)} 球（客强主弱）"
        open_prob_label = {'home_recv': hp_o, 'away_give': ap_o}
        close_prob_label = {'home_recv': hp_c, 'away_give': ap_c}
    else:
        favor, favor_desc = 'even', "平手盘（势均力敌）"
        open_prob_label = {'home': hp_o, 'away': ap_o}
        close_prob_label = {'home': hp_c, 'away': ap_c}

    return {
        'handicap': hcap,
        'open_handicap': open_hcap,
        'handicap_change': dh,
        'favor': favor, 'favor_desc': favor_desc, 'diff_desc': diff_desc,
        'diff_range': diff_range,
        'handicap_trend': handicap_trend, 'water_trend': water_trend,
        'trend_direction': trend_direction,
        'trend_strength': trend_strength,
        'signal_strength': _signal_strength(dh, signal_strong, signal_medium),
        'open_prob': open_prob_label,
        'close_prob': close_prob_label,
        'prob_change': {'home': hp_c - hp_o, 'away': ap_c - ap_o},
        'open_water': {'home': op['home_odds'], 'away': op['away_odds']},
        'close_water': {'home': cl['home_odds'], 'away': cl['away_odds']},
        'lambda_adjust': {'home': dh * lambda_home_scale,
                          'away': dh * lambda_away_scale},
    }


def return_rate_from_odds(home, draw, away, fallback=DEFAULT_RETURN_RATE):
    """由欧赔估算理论返还率（%），JSON 无返还率字段时兜底"""
    total = 1.0 / home + 1.0 / draw + 1.0 / away
    return 100.0 / total if total > 0 else fallback


def kelly_index_triple(home_odds, draw_odds, away_odds, p_home, p_draw, p_away):
    """三项凯利指数（%）= 赔率 × 去水概率 × 100，与 500.com 口径一致"""
    return {
        'home': home_odds * p_home * 100,
        'draw': draw_odds * p_draw * 100,
        'away': away_odds * p_away * 100,
    }


def kelly_outcome_label(key):
    return {'home': '主胜', 'draw': '平局', 'away': '客胜'}[key]


def linear_regression_slope(x_vals, y_vals):
    """线性回归斜率；样本不足或 x 无方差时返回 0.0"""
    n = len(x_vals)
    if n < 2:
        return 0.0
    mean_x = sum(x_vals) / n
    mean_y = sum(y_vals) / n
    numerator = sum((x_vals[i] - mean_x) * (y_vals[i] - mean_y) for i in range(n))
    denominator = sum((x_vals[i] - mean_x) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def analyze_kelly(ouzhi_data, probs_open, probs_close, *,
                  bias_eps=KELLY_BIAS_EPS,
                  neutral_spread=KELLY_NEUTRAL_SPREAD,
                  diverged_spread=KELLY_DIVERGED_SPREAD,
                  change_eps=KELLY_CHANGE_EPS,
                  default_return_rate=DEFAULT_RETURN_RATE):
    """欧赔凯利：初/终盘凯利、返还率对比、离散度与打出难度提示

    probs 取与计算凯利的赔率对应的那一组去水概率。
    """
    op, cl = ouzhi_data['open'], ouzhi_data['close']
    ph_o, pd_o, pa_o = probs_open
    ph_c, pd_c, pa_c = probs_close

    rr_o = op.get('return_rate') or return_rate_from_odds(
        op['home'], op['draw'], op['away'], default_return_rate)
    rr_c = cl.get('return_rate') or return_rate_from_odds(
        cl['home'], cl['draw'], cl['away'], default_return_rate)

    k_open = kelly_index_triple(op['home'], op['draw'], op['away'], ph_o, pd_o, pa_o)
    k_close = kelly_index_triple(cl['home'], cl['draw'], cl['away'], ph_c, pd_c, pa_c)
    delta = {k: k_close[k] - k_open[k] for k in k_close}

    labels = ('home', 'draw', 'away')
    spread = max(k_close.values()) - min(k_close.values())

    if spread < neutral_spread:
        hardest = favored = 'neutral'
    else:
        hardest = max(labels, key=lambda k: k_close[k] - rr_c)
        favored = min(labels, key=lambda k: k_close[k] - rr_c)

    risks, favors, kelly_changes = [], [], []
    for k in labels:
        name = kelly_outcome_label(k)
        diff = k_close[k] - rr_c
        if diff > bias_eps:
            risks.append(f"{name}凯利{k_close[k]:.1f}高于返还率{rr_c:.1f}（+{diff:.1f}）→ 打出偏难")
        elif diff < -bias_eps:
            favors.append(f"{name}凯利{k_close[k]:.1f}低于返还率（{diff:.1f}）→ 相对看好")
        if abs(delta[k]) >= change_eps:
            arrow = '↑' if delta[k] > 0 else '↓'
            kelly_changes.append(f"{name}凯利{arrow}{abs(delta[k]):.1f}")

    summary_parts = []
    if spread >= diverged_spread:
        summary_parts.append(f"凯利离散度{spread:.1f}，庄家态度分化明显")
    elif spread < neutral_spread:
        summary_parts.append(f"凯利离散度{spread:.1f}，三项较为均衡")
        summary_parts.append("暂无明显最难项")
    else:
        summary_parts.append(f"凯利离散度{spread:.1f}，三项较为均衡")
        summary_parts.append(f"最难项倾向{kelly_outcome_label(hardest)}")
    if favors:
        summary_parts.append(favors[0])

    return {
        'return_rate': {'open': rr_o, 'close': rr_c, 'delta': rr_c - rr_o},
        'open': k_open,
        'close': k_close,
        'delta': delta,
        'spread': spread,
        'hardest': hardest,
        'favored': favored,
        'risks': risks,
        'favors': favors,
        'kelly_changes': kelly_changes,
        'summary': '；'.join(summary_parts),
    }


_INSUFFICIENT_KELLY_TREND = {'slopes': {}, 'crossing_events': [], 'summary': '数据不足'}


def analyze_kelly_trend(series, recent_n=KELLY_TREND_RECENT_N, *,
                        bias_eps=KELLY_BIAS_EPS,
                        slope_eps=KELLY_SLOPE_EPS,
                        default_return_rate=DEFAULT_RETURN_RATE):
    """凯利时序：最近 N 条的斜率，以及穿越返还率的事件（诱盘检测）

    `series` 按**倒序**给出（最新在前），内部翻正后取前 N 条。
    """
    if not series or len(series) < 2:
        return dict(_INSUFFICIENT_KELLY_TREND)

    chrono = list(reversed(series))
    recent = chrono[:min(recent_n, len(chrono))]

    kelly_history, rr_history = [], []
    for rec in recent:
        if len(rec) >= 3:
            p_home, p_draw, p_away = remove_vig(rec[0], rec[1], rec[2])
            rr = rec[3] if len(rec) > 3 else return_rate_from_odds(
                rec[0], rec[1], rec[2], default_return_rate)
            kelly_history.append(
                kelly_index_triple(rec[0], rec[1], rec[2], p_home, p_draw, p_away))
            rr_history.append(rr)

    if len(kelly_history) < 2:
        return dict(_INSUFFICIENT_KELLY_TREND)

    labels = ['home', 'draw', 'away']
    x_vals = list(range(len(kelly_history)))
    slopes = {label: round(linear_regression_slope(x_vals, [kh[label] for kh in kelly_history]), 4)
              for label in labels}

    crossing_events = []
    for i in range(1, len(kelly_history)):
        for label in labels:
            prev_above = kelly_history[i - 1][label] > rr_history[i - 1] + bias_eps
            curr_above = kelly_history[i][label] > rr_history[i] + bias_eps
            if prev_above and not curr_above:
                crossing_events.append({
                    'type': 'cross_down', 'label': label,
                    'desc': f"{kelly_outcome_label(label)}凯利从高于返还率降至正常区间",
                })
            elif not prev_above and curr_above:
                crossing_events.append({
                    'type': 'cross_up', 'label': label,
                    'desc': f"{kelly_outcome_label(label)}凯利从正常区间升至高于返还率（可能诱盘）",
                })

    summary_parts = []
    for label in labels:
        slope = slopes[label]
        if abs(slope) > slope_eps:
            arrow = '↑' if slope > 0 else '↓'
            summary_parts.append(f"{kelly_outcome_label(label)}凯利{arrow}{abs(slope):.2f}/步")
    summary_parts.extend(event['desc'] for event in crossing_events)

    return {
        'slopes': slopes,
        'crossing_events': crossing_events,
        'summary': '；'.join(summary_parts) if summary_parts else '凯利走势平稳',
    }


def analyze_euro_momentum(series, *,
                          prob_eps=EURO_PROB_TREND_EPS,
                          shift_scale=MOMENTUM_SHIFT_SCALE,
                          shift_clamp=MOMENTUM_SHIFT_CLAMP):
    """由欧赔时序提取主/客胜概率累积走势，用于修正净胜球"""
    if not series or len(series) < 2:
        return {'shift_supremacy': 0.0, 'summary': '欧赔走势数据不足'}

    chrono = list(reversed(series))
    first = remove_vig(chrono[0][0], chrono[0][1], chrono[0][2])
    last = remove_vig(chrono[-1][0], chrono[-1][1], chrono[-1][2])
    d_home, d_away = last[0] - first[0], last[2] - first[2]
    shift = max(-shift_clamp, min(shift_clamp, (d_home - d_away) * shift_scale))

    parts = []
    if d_home > prob_eps:
        parts.append(f"主胜概率累积↑{d_home * 100:.1f}%")
    elif d_home < -prob_eps:
        parts.append(f"主胜概率累积↓{-d_home * 100:.1f}%")
    if d_away > prob_eps:
        parts.append(f"客胜概率累积↑{d_away * 100:.1f}%")
    elif d_away < -prob_eps:
        parts.append(f"客胜概率累积↓{-d_away * 100:.1f}%")

    return {
        'shift_supremacy': shift,
        'delta_home': d_home,
        'delta_away': d_away,
        'summary': '，'.join(parts) if parts else '欧赔走势平稳',
    }


def compute_dispersion(series):
    """离散度：初盘与终盘三项赔率差的方差"""
    if not series or len(series) < 2:
        return 0.0
    close, open_ = series[0], series[-1]
    diffs = [abs(close[i] - open_[i]) for i in range(3)
             if len(open_) > i and len(close) > i]
    if not diffs:
        return 0.0
    mean = sum(diffs) / len(diffs)
    return sum((d - mean) ** 2 for d in diffs) / len(diffs)


def compute_joint_anomaly(asian_data, total_data, *, water_eps=WATER_TREND_EPS):
    """联合异常：让球水位变化 × 大小球水位变化，以及「暗示大胜」的判定"""
    asian_op, asian_cl = asian_data['open'], asian_data['close']
    total_op, total_cl = total_data['open'], total_data['close']

    asian_water_change = asian_cl['home_odds'] - asian_op['home_odds']
    total_water_change = total_cl['over_odds'] - total_op['over_odds']

    hint_big_win = (asian_water_change < -water_eps
                    and total_water_change < -water_eps)

    return {
        'asian_water_change': round(asian_water_change, 4),
        'total_water_change': round(total_water_change, 4),
        'joint_water_feature': round(asian_water_change * total_water_change, 6),
        'hint_big_win': hint_big_win,
        'hint_desc': '主队水位下降+大球水位下降，暗示主队可能大胜' if hint_big_win else None,
    }


def euro_to_handicap_implied(p_home, p_away, k=EURO_HANDICAP_K):
    """由欧赔转换出理论让球值：(p_home - p_away) × k，k 通常在 1.5~2.0"""
    return (p_home - p_away) * k


def compute_euro_asian_deviation(euro_probs, asian_handicap, k=EURO_HANDICAP_K):
    """理论让球值（由欧赔转换）与实际亚盘让球值的偏差"""
    implied_handicap = euro_to_handicap_implied(
        euro_probs.get('home', 0.5), euro_probs.get('away', 0.5), k)
    deviation = implied_handicap - asian_handicap
    return {
        'implied_handicap': round(implied_handicap, 4),
        'actual_handicap': asian_handicap,
        'deviation': round(deviation, 4),
        'abs_deviation': round(abs(deviation), 4),
    }


def analyze_euro(data, *, prob_eps=EURO_PROB_TREND_EPS, **kelly_kwargs):
    """解析欧赔：初终盘 1X2 去水概率、凯利、走势与变化描述

    任何异常都归一成 `ValueError('欧赔分析失败: ...')`——调用方按这个契约接。
    """
    try:
        op, cl = data.get('open'), data.get('close')
        if not op or not cl:
            raise ValueError("欧赔数据缺少 open 或 close 字段")

        for field in ('home', 'draw', 'away'):
            if field not in op:
                raise ValueError(f"初盘数据缺少 {field} 字段")
            if field not in cl:
                raise ValueError(f"终盘数据缺少 {field} 字段")
        for field in ('home', 'draw', 'away'):
            if not isinstance(op[field], (int, float)) or op[field] <= 0:
                raise ValueError(f"初盘{field}赔率无效: {op[field]}")
            if not isinstance(cl[field], (int, float)) or cl[field] <= 0:
                raise ValueError(f"终盘{field}赔率无效: {cl[field]}")

        ph_o, pd_o, pa_o = remove_vig(op['home'], op['draw'], op['away'])
        ph_c, pd_c, pa_c = remove_vig(cl['home'], cl['draw'], cl['away'])

        for p, name in ((ph_o, '主胜初盘概率'), (pd_o, '平局初盘概率'), (pa_o, '客胜初盘概率'),
                        (ph_c, '主胜终盘概率'), (pd_c, '平局终盘概率'), (pa_c, '客胜终盘概率')):
            if not 0 <= p <= 1:
                raise ValueError(f"{name}超出范围: {p}")

        changes = []
        for delta, name in ((ph_c - ph_o, '主胜'), (pa_c - pa_o, '客胜'), (pd_c - pd_o, '平局')):
            if delta > prob_eps:
                changes.append(f"{name}概率↑{delta * 100:.1f}%")
            elif delta < -prob_eps:
                changes.append(f"{name}概率↓{-delta * 100:.1f}%")

        return {
            'open': {'home': ph_o, 'draw': pd_o, 'away': pa_o},
            'close': {'home': ph_c, 'draw': pd_c, 'away': pa_c},
            'raw_odds': {'open': dict(op), 'close': dict(cl)},
            'kelly': analyze_kelly(data, (ph_o, pd_o, pa_o), (ph_c, pd_c, pa_c),
                                   **kelly_kwargs),
            'momentum': analyze_euro_momentum(data.get('series', []), prob_eps=prob_eps),
            'changes': changes,
        }
    except Exception as e:
        raise ValueError(f"欧赔分析失败: {e}")


def _expected_goals_range(line, over_lean):
    """由盘口线与大小球倾向给出预期总进球区间"""
    if line <= 1.0:
        return [1, 3]
    if line <= 2.0:
        return [1, 4]
    if line <= 2.5:
        return [2, 4] if over_lean else [1, 3]
    if line <= 3.0:
        return [2, 5] if over_lean else [1, 3]
    if line <= 3.5:
        return [3, 6] if over_lean else [2, 4]
    lo = max(0, int(line))
    return [lo, lo + 2]


def analyze_total(data, *,
                  lean_threshold=TOTAL_LEAN_THRESHOLD,
                  line_eps=TOTAL_LINE_TREND_EPS,
                  trend_scale=TREND_STRENGTH_SCALE,
                  signal_strong=SIGNAL_STRONG_DELTA,
                  signal_medium=SIGNAL_MEDIUM_DELTA,
                  lambda_scale=TOTAL_LAMBDA_SCALE,
                  **implied_kwargs):
    """解析大小球：盘口线、去水概率、倾向与预期进球区间"""
    op, cl = data['open'], data['close']
    line, open_line = cl['line'], op['line']

    po_o, pu_o = remove_vig(op['over_odds'], op['under_odds'])
    po_c, pu_c = remove_vig(cl['over_odds'], cl['under_odds'])

    dl = line - open_line
    if dl > line_eps:
        line_trend = f"盘口升高 {open_line:.2f} → {line:.2f}（大球被看好）"
        trend_direction, trend_strength = 'up', min(dl / trend_scale, 1.0)
    elif dl < -line_eps:
        line_trend = f"盘口降低 {open_line:.2f} → {line:.2f}（小球被看好）"
        trend_direction, trend_strength = 'down', min(-dl / trend_scale, 1.0)
    else:
        line_trend = f"盘口稳定 {line:.2f}"
        trend_direction, trend_strength = 'stable', 0.0

    if po_c >= lean_threshold:
        lean, lean_desc = 'over', f"大球倾向（大球概率{po_c*100:.1f}%）"
    elif pu_c >= lean_threshold:
        lean, lean_desc = 'under', f"小球倾向（小球概率{pu_c*100:.1f}%）"
    else:
        lean, lean_desc = None, f"大小球均衡（线{line}，各约50%）"

    implied_total = implied_total_goals(line, po_c, **implied_kwargs)
    open_implied = implied_total_goals(op['line'], po_o, **implied_kwargs)

    return {
        'open_line': open_line, 'close_line': line,
        'line_change': dl,
        'line_trend': line_trend,
        'trend_direction': trend_direction,
        'trend_strength': trend_strength,
        'signal_strength': _signal_strength(dl, signal_strong, signal_medium),
        'implied_total': implied_total,
        'open_implied_total': open_implied,
        'implied_change': implied_total - open_implied,
        'lean': lean, 'lean_desc': lean_desc,
        'open_prob': {'over': po_o, 'under': pu_o},
        'close_prob': {'over': po_c, 'under': pu_c},
        'prob_change': {'over': po_c - po_o, 'under': pu_c - pu_o},
        'open_water': {'over': op['over_odds'], 'under': op['under_odds']},
        'close_water': {'over': cl['over_odds'], 'under': cl['under_odds']},
        'expected_goals': _expected_goals_range(line, lean == 'over'),
        'lambda_adjust': {'total': dl * lambda_scale},
    }


# ---- 泊松：F-4 会把仓库里四份泊松/DC 收编成一份，这里先原样搬过来 ----

def poisson_pmf(k, lam):
    """泊松概率质量函数 P(X=k)"""
    return math.exp(-lam) * lam ** k / math.factorial(k)


def poisson_tail_over(lam_total, line, max_goals=POISSON_TAIL_MAX_GOALS):
    """P(总进球 > line)；四分盘按相邻两个半球盘各半权重"""
    frac = round((line * 4) % 4)
    if frac in (1, 3):
        return 0.5 * poisson_tail_over(lam_total, line - 0.25, max_goals) \
             + 0.5 * poisson_tail_over(lam_total, line + 0.25, max_goals)
    k_min = math.floor(line + 0.501)
    return min(1.0, sum(poisson_pmf(k, lam_total) for k in range(k_min, max_goals)))


def implied_total_goals(line, p_over, tol=1e-4, *,
                        bounds=IMPLIED_TOTAL_BOUNDS,
                        iters=IMPLIED_TOTAL_ITERS,
                        prob_clamp=IMPLIED_TOTAL_PROB_CLAMP,
                        max_goals=POISSON_TAIL_MAX_GOALS):
    """由盘口线与去水大球概率二分反推期望总进球 λ_total

    `tol` 是历史遗留的形参，二分固定跑 `iters` 轮，不看它——保留是因为
    适配层与既有调用点按位置传过它。
    """
    p_over = max(prob_clamp[0], min(prob_clamp[1], p_over))
    lo, hi = bounds
    for _ in range(iters):
        mid = (lo + hi) / 2
        if poisson_tail_over(mid, line, max_goals) < p_over:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
