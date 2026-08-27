"""斜连：同一位（或跨位）连续几期呈 ±1 等差，据此提示下期关注码。

**这是走势辅助信号，回测命中率接近随机。** 保留它是因为它给出的是人能看懂
的理由（「百位 3→4→5，下期关注 6」），不是因为它有预测优势——权重也因此
压得很低。任何把斜连当主信号的改动都会让推荐追着一段巧合走。

三种斜连各自独立：
- **同位斜连**：某一位在连续几期上等差（最强，长度越长强度越高）
- **跨期斜连**：近三期沿百→十→个的对角等差
- **位内斜连**：上一期自己三位就是等差，下期各位顺势延伸（最弱）

`±1` 一律**不含 9↔0 绕回**——绕回属于 `draw.neighbor` 那个概念，混进来会让
「等差」失去意义。
"""
from src.domain.numeric.lottery3d.space import DIGIT_SPACE, POSITION_NAMES, POSITIONS

# 位内斜连的强度。定得比同位斜连的起步值（1.0）低，因为它只用了一期数据。
IN_DRAW_STRENGTH = 0.6
# 同位斜连每多一期，强度加这么多。
LENGTH_STRENGTH_STEP = 0.25
CROSS_PERIOD_LENGTH = 3

NOTE = '斜连为走势辅助信号；历史回测命中率接近随机，请与和值/共现等一并参考。'


def step_between(previous, current):
    """等差步长，只认 ±1；其余（含 9↔0）返回 None。"""
    diff = current - previous
    return diff if diff in (-1, 1) else None


def _in_space(digit):
    return DIGIT_SPACE.contains(digit)


def detect_chain(digits_at_position, min_len, max_len):
    """同一位上结尾最长的等差链。找不到返回 None。

    **从长到短找，第一个成立的就是答案**——链越长信号越强，找到短的就停会
    系统性低估强度。
    """
    if len(digits_at_position) < min_len:
        return None

    upper = min(max_len, len(digits_at_position))
    for length in range(upper, min_len - 1, -1):
        sequence = digits_at_position[-length:]
        step = _uniform_step(sequence)
        if step is None:
            continue
        predicted = sequence[-1] + step
        if _in_space(predicted):
            return {'chain': sequence, 'step': step,
                    'predict_digit': predicted, 'length': length}
    return None


def _uniform_step(sequence):
    """整段是同一个 ±1 步长时返回它，否则 None。"""
    step = None
    for index in range(1, len(sequence)):
        current = step_between(sequence[index - 1], sequence[index])
        if current is None or (step is not None and current != step):
            return None
        step = current
    return step


def _sign(step):
    return '+' if step > 0 else ''


def position_signals(numbers, min_len, max_len):
    """每一位各自的同位斜连。"""
    signals = []
    for position in range(POSITIONS):
        series = [current[position] for current in numbers]
        found = detect_chain(series, min_len, max_len)
        if not found:
            continue
        strength = 1.0 + (found['length'] - min_len) * LENGTH_STRENGTH_STEP
        chain_text = '→'.join(map(str, found['chain']))
        signals.append({
            'type': 'position_slope',
            'position': position,
            'position_name': POSITION_NAMES[position],
            'chain': found['chain'],
            'step': found['step'],
            'predict_digit': found['predict_digit'],
            'length': found['length'],
            'label': (f"同位斜连 {POSITION_NAMES[position]}位 "
                      f"{_sign(found['step'])}{found['step']} "
                      f"({chain_text}) → 关注 {found['predict_digit']}"),
            'strength': round(strength, 2),
        })
    return signals


def cross_period_signals(numbers):
    """近三期沿对角线的等差。三种起始位轮换各查一次。"""
    if len(numbers) < CROSS_PERIOD_LENGTH:
        return []

    recent = numbers[-CROSS_PERIOD_LENGTH:]
    signals = []
    for offset in range(POSITIONS):
        values = [recent[k][(offset + k) % POSITIONS] for k in range(CROSS_PERIOD_LENGTH)]
        step = _uniform_step(values)
        if step is None:
            continue
        predicted = values[-1] + step
        if not _in_space(predicted):
            continue
        route = '→'.join(POSITION_NAMES[(offset + k) % POSITIONS]
                         for k in range(CROSS_PERIOD_LENGTH))
        signals.append({
            'type': 'cross_period_slope',
            'position': offset,
            'position_name': POSITION_NAMES[offset],
            'chain': values,
            'route': route,
            'step': step,
            'predict_digit': predicted,
            'length': CROSS_PERIOD_LENGTH,
            'label': (f"跨期斜连 {route} {_sign(step)}{step} "
                      f"({'→'.join(map(str, values))}) → "
                      f"下期{POSITION_NAMES[offset]}位关注 {predicted}"),
            'strength': 1.0,
        })
    return signals


def in_draw_signal(last_draw):
    """上一期自己三位成等差时，下期各位顺势延伸。没有则返回 None。"""
    step = _uniform_step(list(last_draw))
    if step is None:
        return None
    return {
        'type': 'in_draw_slope',
        'chain': list(last_draw),
        'step': step,
        'label': (f"上期位内斜连 {_sign(step)}{step} "
                  f"({'→'.join(map(str, last_draw))})，下期各位可顺势延伸"),
        'position_hints': [
            {'position_name': POSITION_NAMES[i], 'digit': last_draw[i] + step}
            for i in range(POSITIONS) if _in_space(last_draw[i] + step)
        ],
    }


def analyze(numbers, min_len, max_len):
    """把三种斜连汇总成信号列表与按位关注码。"""
    hints = {position: [] for position in range(POSITIONS)}
    signals = []

    for signal in position_signals(numbers, min_len, max_len):
        signals.append(signal)
        hints[signal['position']].append(_hint(signal))

    for signal in cross_period_signals(numbers):
        signals.append(signal)
        hints[signal['position']].append(_hint(signal))

    if numbers:
        in_draw = in_draw_signal(numbers[-1])
        if in_draw:
            for position in range(POSITIONS):
                predicted = numbers[-1][position] + in_draw['step']
                if _in_space(predicted):
                    hints[position].append({'digit': predicted,
                                            'strength': IN_DRAW_STRENGTH,
                                            'type': 'in_draw_slope'})
            signals.append(in_draw)

    return {
        'active': bool(signals),
        'signal_count': len(signals),
        'signals': signals,
        'position_hints': {POSITION_NAMES[i]: hints[i] for i in range(POSITIONS)},
        'note': NOTE,
    }


def _hint(signal):
    return {'digit': signal['predict_digit'],
            'strength': signal['strength'],
            'type': signal['type']}


def triplet_bonus(triple, analysis, weight):
    """一注与斜连关注码吻合时的加分，按关注码强度加权。"""
    hints = (analysis or {}).get('position_hints') or {}
    return weight * sum(
        float(hint.get('strength', 1.0))
        for position, name in enumerate(POSITION_NAMES)
        for hint in hints.get(name, [])
        if hint.get('digit') == triple[position])
