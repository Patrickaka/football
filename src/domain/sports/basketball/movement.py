"""赔率走势（Line Movement / Sharp Money）的纯计算。

把竞彩篮球的赔率变化转成可量化的交易信号：哪一侧被资金追捧、强度多大、
是否 steam（急单）、是否 stale（长时间无变化），以及在信号足够强且盘口与
水位不冲突时，允许走势改变弱模型的原始方向。

**本模块只做计算，不碰 IO。** 快照的采集与持久化（`track_basketball_odds`
那一路）留在旧模块，随赛前赔率追踪一起改成周期调度。这样切开是因为计算
部分是分析函数的直接依赖，若不先迁，领域层就得反向 import `src.basketball`。

约定：
- 让分用「主队加分值」：-3.5 -> -5.5 是主队让深，指向主队；
  大小分升盘指向大分、降盘指向小分。
- 只有至少两个有效快照、未过期、强度达标且盘口/水位不冲突，才允许反推。
- 任何玩法缺数据一律返回 None，由调用方回退到无走势逻辑。
"""
import logging
from datetime import datetime

log = logging.getLogger('domain.basketball.movement')

# steam：短时间内出现明显且单向的赔率位移（资金急涌）
STEAM_STRENGTH = 0.45
# 视为 stale 的静止时长（分钟）：盘口长期不动，信号价值低
STALE_MINUTES = 240
# 微调概率的最大幅度（防止走势喧宾夺主）
MAX_ADJUST_FACTOR = 0.10
INFERENCE_MIN_STRENGTH = 0.25

_LINE_MOVE_THRESHOLD = 0.5
_STEAM_FRESH_MINUTES = 90


def parse_ts(ts):
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(str(ts))
    except Exception:
        return None


def _sides_for(kind):
    """两路的命名随玩法而变：大小分是 over/under，其余是 home/away。"""
    return ('over', 'under') if kind == 'ou' else ('home', 'away')


def _water_side(first_move, second_move, kind):
    """从两侧水位位移判断资金指向。赔率下降 = 该侧被买入。

    四个阈值组合分两类：一侧明显降且另一侧明显升（对向确认），
    或一侧大幅降而另一侧基本没动（单向确认）。
    """
    first, second = _sides_for(kind)
    if first_move < -0.02 and second_move > 0.01:
        return first
    if second_move < -0.02 and first_move > 0.01:
        return second
    if first_move < -0.04 and second_move >= -0.005:
        return first
    if second_move < -0.04 and first_move >= -0.005:
        return second
    return 'flat'


def _line_side(line_move, kind):
    """从盘口位移判断指向。胜负玩法没有盘口，恒为 flat。"""
    if abs(line_move) < _LINE_MOVE_THRESHOLD:
        return 'flat'
    if kind == 'ou':
        return 'over' if line_move > 0 else 'under'
    if kind == 'ah':
        # handicap 是加在主队得分上的有符号值。数值下降（-3.5 -> -5.5，
        # 或 +5.5 -> +3.5）都代表主队变强。
        return 'home' if line_move < 0 else 'away'
    return 'flat'


def _combine_sides(water_side, line_side):
    """返回 (side, agreement, conflict)。

    冲突时保留较直接的水位方向用于展示，但由 conflict 标记禁止后续反推。
    """
    both_present = water_side != 'flat' and line_side != 'flat'
    conflict = both_present and water_side != line_side
    agreement = both_present and water_side == line_side
    side = water_side if water_side != 'flat' else line_side
    return side, agreement, conflict


def movement_from_snapshots(snaps, first_key, second_key, line_key=None,
                            kind='ml', now_fn=None):
    """从快照序列计算两路 movement。样本不足返回 None。

    `now_fn` 是可注入的时钟：window_min / last_move_age_min / steam / stale
    四个输出都依赖「现在」，不注入就无法写出可复现的测试。
    """
    now = (now_fn or datetime.now)()
    valid = [s for s in snaps if s.get(first_key) and s.get(second_key)]
    if len(valid) < 2:
        return None

    first, last = valid[0], valid[-1]
    # 赔率下降 = 该侧被买入 = 资金追捧
    first_move = float(last[first_key]) - float(first[first_key])
    second_move = float(last[second_key]) - float(first[second_key])

    opening_line = first.get(line_key) if line_key else None
    current_line = last.get(line_key) if line_key else None
    try:
        line_move = float(current_line) - float(opening_line)
    except (TypeError, ValueError):
        line_move = 0.0

    line_weight = 0.06 if kind in ('ah', 'ou') else 0.0
    raw_strength = (abs(first_move) + abs(second_move)
                    + min(abs(line_move), 5.0) * line_weight)

    water_side = _water_side(first_move, second_move, kind)
    line_side = _line_side(line_move, kind)
    side, agreement, conflict = _combine_sides(water_side, line_side)
    if agreement:
        raw_strength += 0.05
    strength = min(1.0, raw_strength / 0.4)

    window_start = parse_ts(first.get('ts'))
    window_min = (now - window_start).total_seconds() / 60.0 if window_start else 0.0
    last_move_age = _last_move_age(valid, first_key, second_key, line_key, now)

    return {
        'available': True,
        'side': side,
        'strength': round(strength, 4),
        'raw_strength': round(raw_strength, 4),
        'steam': (strength >= STEAM_STRENGTH
                  and (last_move_age is None or last_move_age <= _STEAM_FRESH_MINUTES)),
        'stale': last_move_age is not None and last_move_age >= STALE_MINUTES,
        'samples': len(valid),
        'window_min': round(max(0.0, window_min), 1),
        'last_move_age_min': round(last_move_age, 1) if last_move_age is not None else None,
        'home_move': round(first_move, 4),
        'away_move': round(second_move, 4),
        'line_move': round(line_move, 4),
        'opening_line': opening_line,
        'current_line': current_line,
        'kind': kind,
        'water_side': water_side,
        'line_side': line_side,
        'signal_agreement': agreement,
        'signal_conflict': conflict,
    }


def _last_move_age(valid, first_key, second_key, line_key, now):
    """距最后一次实际变盘过去了多少分钟；全程未动返回 None。"""
    for i in range(len(valid) - 1, 0, -1):
        changed = (valid[i].get(first_key) != valid[i - 1].get(first_key)
                   or valid[i].get(second_key) != valid[i - 1].get(second_key)
                   or (line_key and valid[i].get(line_key) != valid[i - 1].get(line_key)))
        if changed:
            ts = parse_ts(valid[i].get('ts'))
            return (now - ts).total_seconds() / 60.0 if ts else None
    return None


_DIRECTION_SIDES = {
    'home_backing': {'ou': 'over', 'ah': 'home', 'ml': 'home'},
    'over_backing': {'ou': 'over', 'ah': 'home', 'ml': 'home'},
    'away_backing': {'ou': 'under', 'ah': 'away', 'ml': 'away'},
    'under_backing': {'ou': 'under', 'ah': 'away', 'ml': 'away'},
    'line_up': {'ou': 'over', 'ah': 'away', 'ml': 'home'},
    'line_down': {'ou': 'under', 'ah': 'home', 'ml': 'away'},
}


def normalize_source_trend(trend, kind):
    """把源站 trend 字典规范化成与快照序列同构的 movement。

    两个来源的输出必须同构，否则下游每个消费点都要分别判断来源。
    """
    if not trend or not isinstance(trend, dict):
        return None

    side = _DIRECTION_SIDES.get(trend.get('direction'), {}).get(kind, 'flat')
    first_move = float(trend.get('home_move', 0) or 0)
    second_move = float(trend.get('away_move', 0) or 0)
    line_move = float(trend.get('line_move', 0) or 0)

    water_side = _water_side(first_move, second_move, kind)
    line_side = _line_side(line_move, kind)
    _, agreement, conflict = _combine_sides(water_side, line_side)

    raw_strength = float(trend.get('strength', 0) or 0)
    strength = min(1.0, raw_strength / 0.2)
    return {
        'available': True,
        'side': side,
        'strength': round(strength, 4),
        'raw_strength': round(raw_strength, 4),
        'steam': strength >= STEAM_STRENGTH,
        # 源站 trend 不带时间戳，无从判断静止时长。
        'stale': False,
        'samples': int(trend.get('samples', 0) or 0),
        'window_min': None,
        'last_move_age_min': None,
        'home_move': round(first_move, 4),
        'away_move': round(second_move, 4),
        'line_move': round(line_move, 4),
        'opening_line': trend.get('opening_line'),
        'current_line': trend.get('current_line'),
        'kind': kind,
        'water_side': water_side,
        'line_side': line_side,
        'signal_agreement': agreement,
        'signal_conflict': conflict,
    }


_MARKET_SIDE_NAMES = {
    'rqspf': {'home': '让胜', 'away': '让负'},
    'dx': {'over': '大分', 'under': '小分'},
}
_ALL_SIDE_NAMES = {'home': '让胜', 'away': '让负', 'over': '大分', 'under': '小分'}


def _inference_reason(movement, side_known, strength, samples):
    """反推为什么不可用——每个否决理由都要能单独说清，便于线上审计。"""
    if not (movement and movement.get('available') and side_known):
        return 'movement_unavailable'
    if samples < 2:
        return 'movement_samples_insufficient'
    if movement.get('stale'):
        return 'movement_stale'
    if movement.get('signal_conflict'):
        return 'water_line_conflict'
    if strength < INFERENCE_MIN_STRENGTH:
        return 'movement_signal_weak'
    return 'water_line_inference'


def infer_market_from_movement(movement, market):
    """把盘口/水位变化转成可审计的独立反推结论。"""
    movement = movement or {}
    mapping = _MARKET_SIDE_NAMES.get(market, {})
    side = movement.get('side', 'flat')
    strength = float(movement.get('strength', 0) or 0)
    samples = int(movement.get('samples', 0) or 0)
    available = bool(movement.get('available') and side in mapping)

    reason = _inference_reason(movement, available, strength, samples)
    actionable = available and reason == 'water_line_inference'
    return {
        'available': available,
        'actionable': actionable,
        'side': side,
        'recommendation': mapping.get(side),
        'strength': round(strength, 4),
        'confidence': _inference_confidence(actionable, strength, movement),
        'reason': reason,
        'samples': samples,
        'stale': bool(movement.get('stale')),
        'steam': bool(movement.get('steam')),
        'water_side': movement.get('water_side', side),
        'line_side': movement.get('line_side', 'flat'),
        'signal_agreement': bool(movement.get('signal_agreement')),
        'signal_conflict': bool(movement.get('signal_conflict')),
        'opening_line': movement.get('opening_line'),
        'current_line': movement.get('current_line'),
        'line_move': float(movement.get('line_move', 0) or 0),
        'first_water_move': float(movement.get('home_move', 0) or 0),
        'second_water_move': float(movement.get('away_move', 0) or 0),
    }


def _inference_confidence(actionable, strength, movement):
    if not actionable:
        return 'low'
    return 'high' if (strength >= 0.65 or movement.get('steam')) else 'medium'


def apply_market_inference(p_first, p_second, movement, market):
    """融合模型与水位反推；强走势可翻转弱模型，但不能碾压强模型。"""
    inference = infer_market_from_movement(movement, market)
    first_side, second_side = _sides_for('ou' if market == 'dx' else 'ah')
    model_side = first_side if p_first >= p_second else second_side
    inference['model_side_before'] = model_side
    inference['model_recommendation_before'] = _ALL_SIDE_NAMES.get(model_side)

    if not inference['actionable']:
        inference['reversed_model'] = False
        inference['probability_shift'] = 0.0
        return p_first, p_second, inference

    movement_edge = 0.025 + 0.075 * inference['strength']
    if inference['signal_agreement']:
        movement_edge += 0.015
    if inference['steam']:
        movement_edge += 0.01
    if inference['side'] != first_side:
        movement_edge = -movement_edge

    original_first = float(p_first)
    # 保留 55% 的原模型边际，让高确定性基本盘不被一次异常跳水硬翻。
    final_edge = (original_first - 0.5) * 0.55 + movement_edge
    final_first = max(0.34, min(0.66, 0.5 + final_edge))
    final_second = 1.0 - final_first

    final_side = first_side if final_first >= final_second else second_side
    inference['reversed_model'] = final_side != model_side
    inference['final_side'] = final_side
    inference['probability_shift'] = round(final_first - original_first, 4)
    return final_first, final_second, inference


def build_movement_for_match(match, history=None, source_bundle=None, now_fn=None):
    """为单场比赛构建 {spf, rqspf, dx} 三个玩法的 movement。

    优先级：源站赛程自带的 rf_trend / dx_trend > 详情 bundle >
    500 源累积的快照序列。任意玩法无数据则为 None，调用方回退原逻辑。
    """
    out = {
        'rqspf': _from_source(match, source_bundle, 'rf_trend', 'ah'),
        'dx': _from_source(match, source_bundle, 'dx_trend', 'ou'),
        'spf': _from_bundle(source_bundle, 'ml', 'ml'),
    }
    if history is None:
        return _ordered(out)

    snaps = history.get(match.get('id'))
    if snaps:
        for key, args in _SNAPSHOT_FIELDS.items():
            if out[key] is None:
                out[key] = movement_from_snapshots(snaps, *args, now_fn=now_fn)
    return _ordered(out)


_SNAPSHOT_FIELDS = {
    'spf': ('spf_home', 'spf_away', None, 'ml'),
    'rqspf': ('rqspf_home', 'rqspf_away', 'handicap', 'ah'),
    'dx': ('dx_over', 'dx_under', 'total_line', 'ou'),
}
_BUNDLE_KEYS = {'ah': 'ah', 'ou': 'ou', 'ml': 'ml'}


def _ordered(out):
    return {'spf': out['spf'], 'rqspf': out['rqspf'], 'dx': out['dx']}


def _from_source(match, bundle, trend_key, kind):
    if match.get('source') == 'zgzcw' and match.get(trend_key):
        return normalize_source_trend(match.get(trend_key), kind)
    return _from_bundle(bundle, _BUNDLE_KEYS[kind], kind)


def _from_bundle(bundle, bundle_key, kind):
    section = (bundle or {}).get(bundle_key) or {}
    if not section.get('available'):
        return None
    return normalize_source_trend(section.get('trend'), kind)


_TREND_DIRECTIONS = {
    'home': 'home_backing', 'away': 'away_backing',
    'over': 'over_backing', 'under': 'under_backing', 'flat': 'stable',
}


def movement_to_trend(movement):
    """把统一 movement 映射回源站 trend 的形状，供 adjust_two_way_by_trend 复用。"""
    if not movement or not movement.get('available'):
        return None
    return {
        'direction': _TREND_DIRECTIONS.get(movement.get('side'), 'stable'),
        'strength': movement.get('strength', 0.0),
        'home_move': movement.get('home_move', 0.0),
        'away_move': movement.get('away_move', 0.0),
        'line_move': movement.get('line_move', 0.0),
        'kind': movement.get('kind', 'ml'),
        'samples': movement.get('samples', 0),
    }


def adjust_two_way_by_trend(p_home, p_away, trend, factor=0.12):
    """按资金流向微调双边概率并归一化。"""
    if not trend or trend.get('direction') in (None, 'stable'):
        return p_home, p_away

    direction = trend['direction']
    strength = min(1.0, float(trend.get('strength') or 0) / 0.2)
    adj = factor * (0.5 + 0.5 * strength)

    if direction in ('home_backing', 'over_backing'):
        p_home *= (1 + adj)
        p_away *= (1 - adj * 0.6)
    elif direction in ('away_backing', 'under_backing'):
        p_away *= (1 + adj)
        p_home *= (1 - adj * 0.6)
    elif direction == 'line_up' and trend.get('kind') == 'ah':
        # handicap 加在主队一侧；数值上升代表客队方向增强。
        p_away *= (1 + adj * 0.5)
        p_home *= (1 - adj * 0.3)
    elif direction == 'line_down' and trend.get('kind') == 'ah':
        # 例如 -3.5 -> -5.5：主队让深，代表主队方向增强。
        p_home *= (1 + adj * 0.5)
        p_away *= (1 - adj * 0.3)

    total = p_home + p_away + 1e-9
    return p_home / total, p_away / total


def apply_movement(p_home, p_away, movement, factor=None):
    """用赔率走势微调双边概率。无走势或走势为 flat 时原样返回。"""
    if not movement or not movement.get('available') or movement.get('side') == 'flat':
        return p_home, p_away
    capped = max(0.0, min(MAX_ADJUST_FACTOR,
                          MAX_ADJUST_FACTOR if factor is None else factor))
    return adjust_two_way_by_trend(p_home, p_away, movement_to_trend(movement),
                                   factor=capped)


_RECOMMENDATION_SIDES = {
    '主胜': 'home', '让胜': 'home', '客胜': 'away', '让负': 'away',
    '大分': 'over', '小分': 'under',
}


def sharp_confirmation(movement, recommendation):
    """判断「聪明钱」是否确认本方的推荐方向。

    boost 是置信度提升档位（-0.5/0/0.5/1），供调用方调高或调低 confidence。
    资金逆向只降权、不反转——反转需要模型侧的新证据。
    """
    if not movement or not movement.get('available') or movement.get('side') == 'flat':
        return {'confirmed': False, 'reason': 'no_movement', 'boost': 0.0}

    rec_side = _RECOMMENDATION_SIDES.get(recommendation)
    if rec_side is None:
        return {'confirmed': False, 'reason': 'unknown_pick', 'boost': 0.0}
    if movement['side'] != rec_side:
        return {'confirmed': False, 'reason': 'contrary_flow', 'boost': -0.5}

    if movement.get('steam'):
        return {'confirmed': True, 'reason': 'steam_confirm', 'boost': 1.0}
    if movement.get('strength', 0.0) >= 0.3:
        return {'confirmed': True, 'reason': 'strong_confirm', 'boost': 0.5}
    return {'confirmed': True, 'reason': 'mild_confirm', 'boost': 0.0}


_SIDE_LABELS = {'home': '主队', 'away': '客队', 'over': '大分',
                'under': '小分', 'flat': '稳定'}
_MARKET_LABELS = {'spf': '胜负', 'rqspf': '让分', 'dx': '大小分'}


def describe_market_movement(movements, bets):
    """把三个玩法的走势合成一段面向用户的联合解读。"""
    movements = movements or {}
    bets = bets or {}
    signals = []
    aligned = contrary = steam = 0

    for key in ('spf', 'rqspf', 'dx'):
        movement = movements.get(key) or {}
        if not movement.get('available'):
            continue
        confirmed = ((bets.get(key) or {}).get('line_movement') or {}).get('confirmed')
        side = movement.get('side', 'flat')
        if side != 'flat':
            aligned += int(bool(confirmed))
            contrary += int(confirmed is False)
        steam += int(bool(movement.get('steam')))
        signals.append(_signal(key, movement, side, confirmed))

    if not signals:
        return {'available': False, 'verdict': '暂无有效盘口变化样本', 'signals': []}

    level, verdict = _verdict(steam, aligned, contrary)
    return {
        'available': True, 'level': level, 'verdict': verdict, 'signals': signals,
        'aligned_count': aligned, 'contrary_count': contrary, 'steam_count': steam,
    }


def _signal(key, movement, side, confirmed):
    line_move = float(movement.get('line_move', 0) or 0)
    line_text = ''
    if abs(line_move) >= 0.01:
        line_text = f"，盘口{'升' if line_move > 0 else '降'}{abs(line_move):g}分"
    side_label = _SIDE_LABELS.get(side, side)
    return {
        'market': key,
        'label': _MARKET_LABELS[key],
        'side': side,
        'side_label': side_label,
        'strength': round(float(movement.get('strength', 0) or 0), 3),
        'steam': bool(movement.get('steam')),
        'stale': bool(movement.get('stale')),
        'samples': int(movement.get('samples', 0) or 0),
        'line_move': line_move,
        'summary': f'{_MARKET_LABELS[key]}资金偏{side_label}{line_text}',
        'aligned_with_model': confirmed,
    }


def _verdict(steam, aligned, contrary):
    if steam and aligned:
        return 'strong', '急速资金与模型方向一致，属于强确认信号'
    if aligned > contrary:
        return 'positive', '盘口变化总体确认模型方向'
    if contrary > aligned:
        return 'warning', '盘口资金与模型方向冲突，建议降低置信度'
    return 'neutral', '各盘口信号分化，维持谨慎判断'
