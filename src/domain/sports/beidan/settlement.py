"""北单赛果判定与历史可靠性校准。

两件事：从已结算快照里**读出实际发生了什么**（胜平负、让球胜平负、总进球、
比分），以及拿「历史上预测了多少 / 实际发生了多少」去修正当前这一注的概率。

这一层只做判定与算术。历史记录从哪来、盘口字符串怎么解析、门槛取多少，
都由调用方传进来——**让球值是解析后的浮点数，不是 `'(-1)'` 这种页面文本**。
"""

SPF_WIN, SPF_DRAW, SPF_LOSE = '胜', '平', '负'
RQSPF_WIN, RQSPF_PUSH, RQSPF_LOSE = '让胜', '让平', '让负'

# 校准因子的钳制区间。校准是「保守修正」而不是「重新预测」：
# 单次修正最多把一个选项抬高 16%、压低 14%，历史再极端也翻不了盘。
FACTOR_MIN, FACTOR_MAX = 0.86, 1.16
# 因子的先验样本量。平摊到每个选项上，样本少时把比值往 1 拉。
# 三档玩法每档摊到 2.0，八档的总进球每档只摊到 0.75——**档位越多先验越弱**，
# 这正是想要的：档位多意味着每档的实际样本更稀，但先验本身也不该更强势。
FACTOR_PRIOR = 6.0
# 同联赛的样本算 1.25 条。不是 2.0——联赛只是「更像」，不是「更对」。
SAME_LEAGUE_WEIGHT = 1.25


def implied_probability(odds):
    """欧赔 → 去水后的隐含概率。赔率非正或缺失的选项直接不参与。

    分母与分子用的是**同一道过滤**（`o and o > 0`），所以剩下几个选项就在
    几个选项之间归一化——含 0 赔率的那一档不会被算成无穷大概率。
    """
    if not odds:
        return {}

    total = sum(1 / value for value in odds.values() if value and value > 0)
    if total == 0:
        return {}

    return {key: (1 / value) / total
            for key, value in odds.items() if value and value > 0}


# 实际比分的取值候选。同一件事在快照里有五个可能的落点，因为写快照的代码
# 前后改过三轮，而历史记录不会跟着迁移——**旧记录只在旧的那个字段上有值**。
#
# **两条链的顺序不同，这是旧实现留下的**：胜平负/总进球/比分把顶层
# `actual_score` 排最前，让球胜平负把它排最后。两处同时有值且不一致时结论
# 相反。原样保留——没有任何证据表明哪一个是有意的，而统一它是一次行为改动。
_SCORE_PATHS = (
    (None, 'actual_score'),
    ('actual', 'score'),
    ('actual', 'actual_score'),
    ('settlement', 'score'),
    ('settlement', 'actual_score'),
)
_RQSPF_SCORE_PATHS = _SCORE_PATHS[1:] + _SCORE_PATHS[:1]


def _section(record, name):
    value = record.get(name)
    return value if isinstance(value, dict) else {}


def _first_present(record, paths):
    """按候选顺序取第一个真值。

    用真值而不是 `is not None`：这是旧实现的 `or` 链语义，空字符串与 0
    都会被跳过。对比分无害（`'0-0'` 是非空字符串），对总进球的直取字段
    **有害**——见 `actual_zjq`。
    """
    for container, key in paths:
        source = record if container is None else _section(record, container)
        value = source.get(key)
        if value:
            return value
    return None


def _parse_score(value):
    """`'2-1'` → `(2, 1)`。解析不出来返回 `None`。"""
    if not value or '-' not in str(value):
        return None
    try:
        home, away = map(int, str(value).split('-', 1))
    except (ValueError, TypeError):
        return None
    return home, away


def actual_spf(record):
    """已结算快照 → 实际胜平负。"""
    direct = _first_present(record, (
        (None, 'actual_spf'), ('actual', 'spf'),
        ('settlement', 'spf'), ('settlement', 'actual_spf'),
    ))
    if direct in (SPF_WIN, SPF_DRAW, SPF_LOSE):
        return direct

    score = _parse_score(_first_present(record, _SCORE_PATHS))
    if score is None:
        return None
    home, away = score
    if home > away:
        return SPF_WIN
    return SPF_LOSE if home < away else SPF_DRAW


def actual_zjq(record, max_goals=7):
    """已结算快照 → 实际总进球档位。`max_goals` 及以上并入 `'7+'` 那一档。

    **直取字段上的零球会被当成缺失**：候选链是 `or` 串起来的，`0` 是假值，
    于是 `actual_zjq=0` 的记录会继续往下找，最后落到比分上算。比分也没有时
    返回 `None` 而不是 `'0'`。原样保留（这条路径在线上从未被走到——
    没有任何记录被标记为已结算），但零球在真实赛果里并不罕见，
    真要启用结算回填时这里要先修。
    """
    ceiling = f'{max_goals}+'
    valid = {str(count) for count in range(max_goals)} | {ceiling}

    direct = _first_present(record, (
        (None, 'actual_zjq'), ('actual', 'zjq'),
        ('settlement', 'zjq'), ('settlement', 'actual_zjq'),
    ))
    if direct is not None and str(direct) in valid:
        return str(direct)

    score = _parse_score(_first_present(record, _SCORE_PATHS))
    if score is None:
        return None
    total = score[0] + score[1]
    return ceiling if total >= max_goals else str(total)


def actual_bifen(record):
    """已结算快照 → 规范化的比分字符串 `'h-a'`。

    没有直取捷径：比分只有一个来源，规范化只是把 `'2-1 '` 这类脏值收干净。
    """
    score = _parse_score(_first_present(record, _SCORE_PATHS))
    if score is None:
        return None
    return f'{score[0]}-{score[1]}'


def actual_rqspf(record, handicap):
    """已结算快照 + **解析后的**让球值 → 实际让球胜平负。

    `handicap` 是浮点数或 `None`，不是 `'(-1)'` 这种页面文本。解析属于
    适配层——领域层拿到 `None` 时按平手盘算，而这与「盘口就是 0」同解，
    所以适配层解析失败与没有盘口在这里不可分辨。
    """
    line = 0.0 if handicap is None else float(handicap)
    score = _parse_score(_first_present(record, _RQSPF_SCORE_PATHS))
    if score is None:
        return None
    margin = score[0] + line - score[1]
    if margin > 0:
        return RQSPF_WIN
    return RQSPF_LOSE if margin < 0 else RQSPF_PUSH


def _tally(records, options, actual_fn, section_key, league,
           limit, league_weight):
    """把历史记录归并成「每个选项预测了多少 / 实际发生了多少」。

    只统计**已结算**且在这个玩法上留了概率的记录。实际结果落在 `options`
    之外的（解析失败、玩法档位变过）整条跳过——只跳过这一条，不影响其它。
    """
    expected = {option: 0.0 for option in options}
    actuals = {option: 0.0 for option in options}
    samples = 0.0

    for record in records[:limit]:
        if not record.get('settled'):
            continue
        past_probs = _section(_section(record, section_key), 'probabilities')
        if not past_probs:
            continue
        actual = actual_fn(record)
        if actual not in expected:
            continue

        weight = league_weight if league and record.get('league') == league else 1.0
        for option in expected:
            expected[option] += float(past_probs.get(option, 0.0) or 0.0) * weight
        actuals[actual] += weight
        samples += weight

    return expected, actuals, samples


def calibration_factors(expected, actuals, prior=FACTOR_PRIOR,
                        factor_min=FACTOR_MIN, factor_max=FACTOR_MAX):
    """每个选项的修正因子：实际次数比预测期望高就抬，低就压。

    先验平摊到分子分母两侧（而不是只加在分子上），所以样本为零时因子恰好
    是 1.0——**没有历史时校准必须是恒等变换**，加在单侧会让它凭空偏向。
    """
    prior_each = prior / max(len(expected), 1)
    return {
        option: max(factor_min, min(factor_max,
                    (actuals[option] + prior_each)
                    / max(expected[option] + prior_each, 1e-9)))
        for option in expected
    }


def apply_history_calibration(probabilities, records, actual_fn, section_key,
                              league=None, min_samples=8, limit=200,
                              prior=FACTOR_PRIOR, league_weight=SAME_LEAGUE_WEIGHT,
                              factor_min=FACTOR_MIN, factor_max=FACTOR_MAX):
    """拿已结算历史对一注的概率做保守修正，返回 `(概率, 说明)`。

    不满足条件时**原样返回传进来的那个对象**，并在说明里写清是哪一道拦下的
    ——`applied=False` 有四种互不相同的原因，混成一个「没生效」就查不下去了。

    **因子按 `str(选项)` 建，而回写时用的是原始键**：调用方传元组键或整数键
    的话，`factors.get` 一个也对不上，每个选项静默回落成 1.0，校准等于没做。
    `recommending.py` 的比分那一路传的正是元组键的矩阵——它从投产起就没有
    校准过。原样保留（改掉是一次行为改动），但这是判据 26 那类
    「跨表示边界丢掉类型」的同一个形状，只是这次丢在内存里而不是存储上。
    """
    if not probabilities:
        return probabilities, {'applied': False, 'reason': 'empty_probabilities'}
    if not records:
        return probabilities, {'applied': False, 'reason': 'no_history'}

    options = [str(key) for key in probabilities]
    expected, actuals, samples = _tally(
        records, options, actual_fn, section_key, league, limit, league_weight)

    if samples < min_samples:
        return probabilities, {
            'applied': False,
            'reason': 'insufficient_settled_samples',
            'sample_count': round(samples, 3),
            'min_samples': min_samples,
        }

    factors = calibration_factors(expected, actuals, prior=prior,
                                  factor_min=factor_min, factor_max=factor_max)
    adjusted = {
        option: float(probabilities.get(option, 0.0) or 0.0) * factors.get(option, 1.0)
        for option in probabilities
    }
    total = sum(adjusted.values())
    if total <= 0:
        return probabilities, {'applied': False, 'reason': 'zero_adjusted_total'}

    return {option: value / total for option, value in adjusted.items()}, {
        'applied': True,
        'sample_count': round(samples, 3),
        'factors': {key: round(value, 6) for key, value in factors.items()},
        'actuals': {key: round(value, 3) for key, value in actuals.items()},
        'expected': {key: round(value, 3) for key, value in expected.items()},
    }


def record_key(match, fields=('date', 'num', 'home', 'away')):
    """历史记录的主键：四个字段用 `|` 串起来。

    缺字段落成空串而不是跳过——跳过会让 `2026-08-28||A|B` 与
    `2026-08-28|1|A|B` 之外的两条不同记录并成同一个键。
    """
    return '|'.join(str(match.get(field, '')) for field in fields)
