"""开奖记录。

**号码是这个领域里唯一不可失真的东西。** 开奖结果是既成事实，一旦记错，
后面所有的命中统计、策略验证、回测结论全部作废，而且不会有任何报错。
所以校验放在构造处：不合规的记录一律返回 None，而**绝不悄悄修正**——
把 19 个号码补成 20 个不是容错，是伪造。
"""
import hashlib
import json
from dataclasses import dataclass

DRAW_SIZE = 20
NUMBER_RANGE = (1, 80)

# 不同数据源对同一个字段有两种叫法
_NUMBER_KEYS = ('numbers', 'draw_numbers')
_DATE_KEYS = ('date', 'draw_date')


def checksum_of(numbers):
    """号码集合的短校验码。

    排序后再算：号码是集合而不是序列，顺序不同不代表内容不同。线上 2048 条
    历史记录都带着按这个算法算出的值，改算法会让它们全部显示为「校验不符」。
    """
    payload = json.dumps(sorted(numbers), separators=(',', ':'))
    return hashlib.md5(payload.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Draw:
    """一期开奖。不可变——既成事实不该有 setter。"""

    issue: str
    numbers: tuple
    date: str = ''
    source: str = ''
    fetched_at: str = ''
    checksum: str = ''

    @classmethod
    def parse(cls, record):
        """从原始记录构造。任何不合规都返回 None，不抛异常。

        返回 None 而非抛异常，是因为导入历史时一条坏记录不该让整批失败；
        调用方按 None 跳过即可。
        """
        if not isinstance(record, dict):
            return None

        numbers = _parse_numbers(_first(record, _NUMBER_KEYS))
        if numbers is None:
            return None

        issue = str(record.get('issue', '')).strip()
        if not issue:
            return None

        return cls(
            issue=issue,
            numbers=numbers,
            date=_first(record, _DATE_KEYS) or '',
            source=record.get('source', '') or '',
            fetched_at=record.get('fetched_at', '') or '',
            checksum=record.get('checksum') or checksum_of(numbers),
        )

    def hits(self, candidate):
        """候选号码里中了几个。

        按集合求交：候选里重复选同一个号只算一次——重复不会让它更容易中。
        """
        return len(set(candidate) & set(self.numbers))

    def to_dict(self):
        """纯 JSON 类型。进缓存与落库都要求这一点。"""
        return {
            'issue': self.issue,
            'numbers': list(self.numbers),
            'date': self.date,
            'source': self.source,
            'fetched_at': self.fetched_at,
            'checksum': self.checksum,
        }


def _first(record, keys):
    for key in keys:
        value = record.get(key)
        if value:
            return value
    return None


def _parse_numbers(raw):
    """解析并校验号码集合。合规才返回，否则 None。"""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    if not raw or not isinstance(raw, (list, tuple, set)):
        return None

    try:
        numbers = sorted(int(x) for x in raw)
    except (ValueError, TypeError):
        return None

    low, high = NUMBER_RANGE
    if len(numbers) != DRAW_SIZE or len(set(numbers)) != DRAW_SIZE:
        return None
    if any(n < low or n > high for n in numbers):
        return None
    return tuple(numbers)


@dataclass(frozen=True)
class DrawConflict:
    """同一期出现了两组不同的号码。

    带上双方完整记录（含来源）：判断哪边对要看数据源和抓取时间，
    只报一个期号的话，人还得回去翻原始文件。
    """

    issue: str
    kept: Draw
    rejected: Draw


def merge_draws(existing, incoming):
    """合并新旧开奖记录，返回 (合并结果, 冲突列表)。

    **冲突时保留旧值。** 开奖结果不会变，同一期出现两组号码只能是某一边
    错了；自动覆盖等于用未经核实的数据抹掉已经存在的。冲突原样报出来，
    交人工确认。

    结果按期号倒序——线上文件就是新→旧，下游多处直接取第一条当最新一期。
    """
    merged = {draw.issue: draw for draw in existing}
    conflicts = []

    for draw in incoming:
        current = merged.get(draw.issue)
        if current is None:
            merged[draw.issue] = draw
        elif current.numbers != draw.numbers:
            conflicts.append(DrawConflict(issue=draw.issue, kept=current,
                                          rejected=draw))

    ordered = sorted(merged.values(), key=lambda d: d.issue, reverse=True)
    return ordered, conflicts
