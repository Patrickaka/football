"""预测记录：结算、统计、稳定度。

**这一层不碰存储、不读时钟。** 记录从哪儿来、存到哪儿去、时间戳怎么打，
都由外面决定；这里只回答三个问题：这条记录中没中、一批记录的命中率是多少、
最近几期的推荐重复得有多厉害。

分开的理由很实在：结算逻辑要能在测试里喂任意记录跑，而只要它自己去
`kv_store.load()`，就得先造一套存储才能测一行判断。
"""
from collections import Counter


# ─── 结算 ───

# 判定「至少覆盖两个开奖号」的门槛。这是给用户看的直观指标：三位全中太罕见
# （1/1000），只报全中的话这份记录几乎永远是「没中」，看不出模型有没有在动。
GE2_DIGITS = 2


def settle(record, actual, max_overlap):
    """把一条预测记录结算掉。**原地改**，与迁移前一致。

    `max_overlap` 由调用方给（算重合度是选号层的事），这里只做判断。
    """
    label = ''.join(map(str, actual))
    top3 = record['zhixuan_top3']
    top30 = record['zhixuan']
    record['actual'] = label
    record['settled'] = True
    record['hit_top3'] = label in top3
    record['hit_top30'] = label in top30
    record['ge2_digit'] = max_overlap(label, top30) >= GE2_DIGITS
    return record


def pending(records):
    """还没结算的那些。"""
    return [record for record in records if not record.get('settled')]


def settle_all(records, periods, numbers, max_overlap):
    """按开奖结果结算全部待结算记录，返回 (结算条数, 是否有改动)。

    **一条记录预测的是它自己期号的下一期**：`period` 是基准期，
    对应的开奖在 `numbers[idx + 1]`。差一位就会把每条记录都对错开奖号，
    而结果看上去仍然是一份正常的命中率统计。
    """
    index_of = {period: index for index, period in enumerate(periods)}
    settled = 0
    for record in records:
        if record.get('settled'):
            continue
        index = index_of.get(record.get('period'))
        if index is None or index + 1 >= len(numbers):
            continue
        settle(record, numbers[index + 1], max_overlap)
        record['draw_period'] = periods[index + 1]
        settled += 1
    return settled, settled > 0


# ─── 统计 ───

HIT_FIELDS = ('hit_top3', 'hit_top30', 'ge2_digit')


def online_stats(records):
    """一批记录的命中率统计。

    **未结算的记录不进分母**——它们还没有结果，算进去等于用「还没开奖」
    冲淡命中率。但它们的条数要报出来，否则看的人分不清是没中还是没结算。
    """
    settled = [record for record in records if record.get('settled')]
    base = {'total_records': len(records),
            'settled_count': len(settled),
            'unsettled_count': len(records) - len(settled)}
    if not settled:
        return {**base, 'hit_top3_rate': 0.0, 'hit_top30_rate': 0.0,
                'ge2_digit_rate': 0.0, 'by_version': {}}

    counts = {field: sum(1 for record in settled if record[field])
              for field in HIT_FIELDS}
    return {
        **base,
        'hit_top3_count': counts['hit_top3'],
        'hit_top3_rate': counts['hit_top3'] / len(settled),
        'hit_top30_count': counts['hit_top30'],
        'hit_top30_rate': counts['hit_top30'] / len(settled),
        'ge2_digit_count': counts['ge2_digit'],
        'ge2_digit_rate': counts['ge2_digit'] / len(settled),
        'by_version': _by_version(settled),
    }


def _by_version(settled):
    """按版本分组统计。**换了版本的记录不该混在一起比**——那是两个模型。"""
    grouped = {}
    for record in settled:
        stats = grouped.setdefault(record['version'],
                                   {'count': 0, 'hit_top3': 0, 'hit_top30': 0})
        stats['count'] += 1
        stats['hit_top3'] += int(record['hit_top3'])
        stats['hit_top30'] += int(record['hit_top30'])
    for stats in grouped.values():
        stats['hit_top3_rate'] = stats['hit_top3'] / stats['count']
        stats['hit_top30_rate'] = stats['hit_top30'] / stats['count']
    return grouped


# ─── 稳定度 ───

STABILITY_WINDOW = 7
# 稳定度的两道界。**两头都不好**：太高说明推荐几期不动，用户看到的是同一批
# 号；太低说明每期都换一批，看不出模型有主张。
HIGH_STABILITY = 0.8
LOW_STABILITY = 0.3
# 对应的探索率：太稳就多探索一点打散，太随机就少探索一点收敛。
HIGH_EXPLORATION = 0.25
LOW_EXPLORATION = 0.08


def entry_numbers(entry):
    """兼容两种历史格式。**线上两种都还在**：新的是带期号的字典，
    旧的直接是号码列表。只认一种会让一半历史被当成空。"""
    return entry.get('recommendations', []) if isinstance(entry, dict) else entry


def stability(current, history):
    """当前推荐与最近几期的平均重叠率。

    **当前推荐为空时返回 0**：迁移前这里直接除以 `len(current)` 而崩掉。
    空推荐意味着「没有可比的东西」，与「历史为空」是同一种情况，
    该走同一条兜底路径。
    """
    current_set = set(current)
    if not current_set:
        return 0.0

    scores = []
    for entry in history[-STABILITY_WINDOW:]:
        previous = set(entry_numbers(entry))
        if not previous:
            continue
        scores.append(len(current_set & previous) / len(current_set))
    return sum(scores) / len(scores) if scores else 0.0


def stability_level(value):
    if value > HIGH_STABILITY:
        return 'high'
    if value < LOW_STABILITY:
        return 'low'
    return 'normal'


def exploration_rate(value, default):
    """按稳定度调整探索率。两端各自反向拉一把，中间用配置的默认值。"""
    if value > HIGH_STABILITY:
        return HIGH_EXPLORATION
    if value < LOW_STABILITY:
        return LOW_EXPLORATION
    return default


# ─── 组六轮换 ───

# 降权上限。再高就不是「轮换」而是「禁用」了——3D 选哪些码没有优势，
# 任意四个互异码的组六覆盖率恒为 4×6/1000，所以轮换零成本；
# 但把某个码彻底压死同样零收益，只是让推荐变得难以解释。
MAX_RECENT_PENALTY = 3.0


def recent_zu6_penalty(score, history, window, base, decay):
    """对最近几期组六四码用过的数字降权，让号码轮换起来。

    最近一期罚得最重，越久远越轻；连着几期都出现的数字累计罚得最多，
    优先被换出去。**这不是因为它们更不容易开出**——是为了别让用户连着
    几期看到同一组码。
    """
    adjusted = list(score)
    if not history:
        return adjusted

    base = min(float(base), MAX_RECENT_PENALTY)
    for age, entry in enumerate(reversed(history[-window:])):
        digits = entry.get('digits', []) if isinstance(entry, dict) else entry
        penalty = base * (decay ** age)
        for digit in digits:
            if 0 <= digit < len(adjusted):
                adjusted[digit] -= penalty
    return adjusted


# ─── 按期号去重的追加 ───

def upsert_by_period(history, period, fields, window, stamp=None):
    """按期号写入：同一期只留最后一次，超出窗口的丢掉。

    **推荐历史必须以「期」为单位，不能以「页面被访问了几次」为单位**——
    后者会让同一期的推荐在历史里出现十几遍，稳定度立刻虚高。
    """
    history = list(history)
    if history and isinstance(history[-1], dict) and history[-1].get('period') == period:
        history[-1].update(fields)
        if stamp:
            history[-1].update(stamp)
    else:
        entry = {'period': period, **fields}
        if stamp:
            entry.update(stamp)
        history.append(entry)
    return history[-window:]
