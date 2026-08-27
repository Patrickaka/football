"""数据与缓存的可用性判断：这份历史能不能用、这份 ML 缓存还算不算数。

**两个判断的默认都是「不可用」。** 数据有重复期号或断期就不让 ML 融合，
缓存缺时间戳、时间戳解析不了、版本对不上都算过期——用一份来路不明的
缓存去出号，比多训练一次贵得多。
"""

WARN_SHORT_HISTORY = 'history_too_short_for_ml_fusion'
WARN_DUPLICATE = 'duplicate_periods'
WARN_GAPS = 'period_gaps'

# 期号的年份占前四位，其余是当年的序号。3D 每天开一期，跨年时序号从 1 重来。
YEAR_DIGITS = 4
FIRST_SEQ_OF_YEAR = 1


def assess_history(periods, dates, min_periods_for_ml):
    """历史数据的质量摘要。

    **重复期号与断期都只计数、不修**：这一层的职责是把问题摆出来，
    补数据是抓取层的事。悄悄补上会让「数据是完整的」这个结论凭空成立。
    """
    duplicates = len(periods) - len(set(periods))
    gaps = _count_gaps(periods)

    warnings = []
    if len(periods) < min_periods_for_ml:
        warnings.append(WARN_SHORT_HISTORY)
    if duplicates:
        warnings.append(WARN_DUPLICATE)
    if gaps:
        warnings.append(WARN_GAPS)

    return {
        'periods': len(periods),
        'first_period': periods[0] if periods else None,
        'last_period': periods[-1] if periods else None,
        'first_date': dates[0] if dates else None,
        'last_date': dates[-1] if dates else None,
        'duplicate_periods': duplicates,
        'period_gaps': gaps,
        # 三个条件全满足才放行。ML 对断期特别敏感——它把序列当连续的来学。
        'ml_fusion_allowed': (len(periods) >= min_periods_for_ml
                              and duplicates == 0 and gaps == 0),
        'warnings': warnings,
    }


def _count_gaps(periods):
    """相邻两期的期号不连续算一次断期。

    **解析不了的期号跳过而不是算断期**：格式异常是另一类问题，把它算进
    断期数会让两种毛病混在一个数字里，谁也说不清。
    """
    gaps = 0
    for previous, current in zip(periods, periods[1:]):
        parsed = _parse_pair(previous, current)
        if parsed is None:
            continue
        (previous_year, previous_seq), (current_year, current_seq) = parsed
        if current_year == previous_year and current_seq - previous_seq != 1:
            gaps += 1
        elif current_year == previous_year + 1 and current_seq != FIRST_SEQ_OF_YEAR:
            gaps += 1
    return gaps


def _parse_pair(previous, current):
    try:
        return ((int(previous[:YEAR_DIGITS]), int(previous[YEAR_DIGITS:])),
                (int(current[:YEAR_DIGITS]), int(current[YEAR_DIGITS:])))
    except (ValueError, TypeError):
        return None


def is_cache_valid(cache, current_period, model_version, max_age, now, parse_time):
    """ML 预测缓存还能不能用。

    **每一条不满足都返回 False，没有「大概还行」**：拿一份基于旧数据或旧模型
    的预测去出号，错得毫无痕迹——号码看上去完全正常。

    `now` 与 `parse_time` 由调用方给：读时钟是副作用，不该长在判断里。
    """
    if not cache or cache.get('base_period') != current_period:
        return False
    if cache.get('model_version') != model_version:
        return False

    created_at = cache.get('created_at')
    if not created_at:
        # 没有时间戳就无从判断新旧。**当作有效**——这是迁移前的行为：
        # 期号与版本都对上了，缺个时间戳不足以推翻它。
        return True
    try:
        return now - parse_time(created_at) <= max_age
    except Exception:
        # 时间戳解析不了说明这份缓存本身有问题，不再信任它
        return False
