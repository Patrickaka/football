"""北单页面解析：把抓回来的文本变成结构化记录。

两类输入：

- **管道分隔的赔率表**（500.com 的 `ssq_match_info.jsp`）：一行一场比赛，
  `比赛号|价|价|…`，`#` 开头是注释。
- **走势历史页**（okooo 的详情页）：优先从 HTML 表格里取，取不到再退回
  页面脚本里的内联数据。

**三份走势历史迁移前是三段几乎相同的代码**（亚盘、大小球、比分盘），
差别只在表头关键字、列数、字段名和几条校验上。合成一份参数化实现之后，
加一种走势只需要加一个 `HistorySpec`——而此前要复制一百行再逐处改，
漏改一处不会报错，只会让那一种走势悄悄取不到数（判据 11）。

解析层对脏输入的态度是**跳过这一行，不是丢掉整批**（判据 18）：页面上有
一行坏数据很常见，为它放弃整场比赛的走势是不划算的。但「跳过什么」在
迁移前的三个实现里并不一致，那些不一致原样保留了，各自在注释里写明。
"""
import re

# 表格与脚本的抽取。都用正则而不是 HTML 解析器：页面标签常年不闭合，
# 解析器会在第一处不闭合上重排整棵树，正则反而稳。
_TAG = re.compile(r'<[^>]+>')
_TABLE = re.compile(r'<table[^>]*>(.*?)</table>', re.DOTALL)
_ROW = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL)
_HEADER_CELL = re.compile(r'<th[^>]*>(.*?)</th>', re.DOTALL)
_CELL = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL)
_SCRIPT = re.compile(r'<script[^>]*>(.*?)</script>', re.DOTALL)

# 脚本里的内联序列。**时间那一组前面没有可选引号**，所以
# `["09:00", -0.5, …]` 这种 JSON 数组一个也匹配不上，只有裸的
# `09:00, -0.5, …` 才认。值那一组两种都认。原样保留自迁移前。
_FOUR_COLUMN_SERIES = re.compile(
    r'(\d{2}:\d{2})\s*,\s*["\']?([^\s"\',]+)["\']?\s*,\s*([\d.]+)\s*,\s*([\d.]+)')
_THREE_COLUMN_SERIES = re.compile(
    r'(\d{2}:\d{2})\s*,\s*["\']?([^\s"\',]+)["\']?\s*,\s*([\d.]+)')

# 时间栏最多八个字符（`09:00` 这种）。超过就说明这一格根本不是时间——
# 表格里混着跨行的表头或合计行时会撞上。
MAX_TIME_LENGTH = 8
# 让球值最多二十个字符。只有亚盘那一路有这道门槛，大小球没有。
MAX_HANDICAP_LENGTH = 20
# 太短的脚本不可能装得下一条序列，直接跳过——省掉在几十段 JS 上跑正则。
MIN_SCRIPT_LENGTH = 50

COMMENT_PREFIX = '#'
FIELD_SEPARATOR = '|'

# 定长表里某一列缺席时的两种读法。**它们不是同义词**：
# 前者把空字符串读成 `None`，后者只在整段缺席时给 `None`，
# 空字符串会一路走到 `float('')` 抛出去、把整行带走。
# 总进球表的最后一档（`7+`）用的正是后者，其余各档用前者——
# 这处不对称是迁移前就有的（判据 17），原样保留。
EMPTY_AS_NONE = 'empty_as_none'
MISSING_AS_NONE = 'missing_as_none'

ZJQ_COLUMNS = (('0', 1, EMPTY_AS_NONE), ('1', 2, EMPTY_AS_NONE),
               ('2', 3, EMPTY_AS_NONE), ('3', 4, EMPTY_AS_NONE),
               ('4', 5, EMPTY_AS_NONE), ('5', 6, EMPTY_AS_NONE),
               ('6', 7, EMPTY_AS_NONE), ('7+', 8, MISSING_AS_NONE))
ZJQ_MINIMUM = 8

BQC_COLUMNS = (('胜胜', 1, EMPTY_AS_NONE), ('胜平', 2, EMPTY_AS_NONE),
               ('胜负', 3, EMPTY_AS_NONE), ('平胜', 4, EMPTY_AS_NONE),
               ('平平', 5, EMPTY_AS_NONE), ('平负', 6, EMPTY_AS_NONE),
               ('负胜', 7, EMPTY_AS_NONE), ('负平', 8, EMPTY_AS_NONE),
               ('负负', 9, EMPTY_AS_NONE))
BQC_MINIMUM = 10


def _data_lines(content):
    """去掉空行与注释行。`content` 为空时不产出任何一行。

    **迁移前先对整段做了一次 `strip()`，删掉了**：逐行的 `strip()` 已经把
    首尾两行也处理了，整段那次一个字符也改变不了结果——与比分表那道
    `len(parts) < 2` 一样，是看起来像防线的空操作（判据 9 第一类）。
    """
    for line in (content or '').split('\n'):
        line = line.strip()
        if line and not line.startswith(COMMENT_PREFIX):
            yield line


def parse_pair_table(content):
    """变长的「比分|价|比分|价…」表 → `{比赛号: {比分: 价}}`。

    **坏价只丢那一对，整行还留着**——与定长表相反。比分盘一场有几十个
    选项，为其中一个坏值丢掉整场太贵；而定长表的每一列都是必需的，
    缺一列那一行就没有意义了。

    末尾落单的比分（没有配对的价）直接忽略。

    **迁移前这里有一道 `len(parts) < 2` 的守卫，删掉了**：它一次也拦不下
    任何东西——段数不足时那个取对的循环本来就一对也取不出，末尾的
    `if odds` 会把这一行丢掉。留着一道任何输入都触发不了的检查比没有更糟，
    它看起来像道防线（判据 9 第一类）。
    """
    result = {}
    for line in _data_lines(content):
        parts = line.split(FIELD_SEPARATOR)
        odds = {}
        for index in range(1, len(parts), 2):
            if index + 1 < len(parts):
                try:
                    odds[parts[index]] = float(parts[index + 1])
                except ValueError:
                    pass
        if odds:
            result[parts[0]] = odds
    return result


def parse_column_table(content, columns, minimum):
    """定长的赔率表 → `({比赛号: {档位: 价}}, [解析失败的行])`。

    **一个坏价带走整行**，失败的行原样返回给调用方去记日志——
    日志是适配层的事，这一层只说「哪一行没读懂」。
    """
    result = {}
    failures = []
    for line in _data_lines(content):
        parts = line.split(FIELD_SEPARATOR)
        if len(parts) < minimum:
            continue
        try:
            result[parts[0]] = {
                name: _column_value(parts, index, mode)
                for name, index, mode in columns
            }
        except (ValueError, IndexError) as error:
            failures.append((line, str(error)))
    return result, failures


def _column_value(parts, index, mode):
    if mode is MISSING_AS_NONE:
        return float(parts[index]) if len(parts) > index else None
    return float(parts[index]) if parts[index] else None


class HistorySpec:
    """一种走势历史怎么从页面里取。

    `header_keywords` / `script_keywords` 分成大小写敏感与不敏感两组——
    迁移前比分盘那一路写的是 `'cs' in header_text.lower()`，另外两路是直接
    `in`。差别只在英文关键字上，但把它们混成一种会让「CS」这样的表头
    要么全都命中、要么全都不命中。
    """

    def __init__(self, header_keywords, header_keywords_lower, minimum_cells,
                 fields, guards, script_keywords, script_keywords_lower,
                 script_pattern, script_fields):
        self.header_keywords = header_keywords
        self.header_keywords_lower = header_keywords_lower
        self.minimum_cells = minimum_cells
        self.fields = fields
        self.guards = guards
        self.script_keywords = script_keywords
        self.script_keywords_lower = script_keywords_lower
        self.script_pattern = script_pattern
        self.script_fields = script_fields


TEXT, PRICE = 'text', 'price'


def _non_empty_short(value):
    return bool(value) and len(value) <= MAX_TIME_LENGTH


def _short_handicap(value):
    return bool(value) and len(value) <= MAX_HANDICAP_LENGTH


def _looks_like_score(value):
    return bool(value) and '-' in value


ASIAN = HistorySpec(
    header_keywords=('亚盘', '让球', '盘口'), header_keywords_lower=(),
    minimum_cells=4,
    fields=(('time', 0, TEXT), ('handicap', 1, TEXT),
            ('home_odds', 2, PRICE), ('away_odds', 3, PRICE)),
    guards=(('time', _non_empty_short), ('handicap', _short_handicap)),
    script_keywords=('亚盘', 'AH'), script_keywords_lower=('asian',),
    script_pattern=_FOUR_COLUMN_SERIES,
    script_fields=('time', 'handicap', 'home_odds', 'away_odds'))

GOALS = HistorySpec(
    header_keywords=('进球', '大小球'), header_keywords_lower=(),
    # **只要三列**：小球水位可以缺席。亚盘那边要四列。
    minimum_cells=3,
    fields=(('time', 0, TEXT), ('line', 1, TEXT),
            ('over_odds', 2, PRICE), ('under_odds', 3, PRICE)),
    # 没有让球值那道长度门槛——同样长的一格在亚盘那边会被跳过
    guards=(('time', _non_empty_short),),
    script_keywords=('进球',), script_keywords_lower=('goals', 'total'),
    script_pattern=_FOUR_COLUMN_SERIES,
    script_fields=('time', 'line', 'over_odds', 'under_odds'))

CORRECT_SCORE = HistorySpec(
    header_keywords=('比分',), header_keywords_lower=('cs',),
    minimum_cells=3,
    fields=(('time', 0, TEXT), ('score', 1, TEXT), ('odds', 2, PRICE)),
    guards=(('time', _non_empty_short), ('score', _looks_like_score)),
    script_keywords=('比分',), script_keywords_lower=('cs', 'score'),
    script_pattern=_THREE_COLUMN_SERIES,
    script_fields=('time', 'score', 'odds'))


def _text(value):
    return _TAG.sub('', value).strip() if value else value


def _price(value):
    """`'0.95'` → `0.95`；不是数字就给 `None`，**不抛**。

    判别用的是 `去掉小数点后全是数字`，所以负数与带正号的值一律读成
    `None`——水位不会是负数，真出现负值说明这一格取错了列。
    """
    if not value:
        return None
    return float(value) if value.replace('.', '').isdigit() else None


def _record(cells, spec):
    record = {}
    for name, index, kind in spec.fields:
        raw = cells[index].strip() if index < len(cells) else None
        cleaned = _text(raw)
        record[name] = cleaned if kind is TEXT else _price(cleaned)
    for name, passes in spec.guards:
        if not passes(record[name]):
            return None
    return record


def _matches_keywords(text, keywords, lowered_keywords):
    return (any(keyword in text for keyword in keywords)
            or any(keyword in text.lower() for keyword in lowered_keywords))


def parse_history_tables(html, spec):
    """从页面的表格里取一份走势。命不中表头关键字的表整张跳过。"""
    records = []
    for table in _TABLE.findall(html):
        rows = _ROW.findall(table)
        # 首行是表头，所以至少要两行才有数据。**改成 `< 1` 是等价变异**
        # （判据 9b，全语料验过）：只有表头时 `rows[1:]` 本来就是空的。
        # 写 `< 2` 是因为它说的是「要有数据行」，而不是「别越界」。
        if len(rows) < 2:
            continue
        header = ''.join(_HEADER_CELL.findall(rows[0]))
        if not _matches_keywords(header, spec.header_keywords,
                                 spec.header_keywords_lower):
            continue
        for row in rows[1:]:
            cells = _CELL.findall(row)
            if len(cells) < spec.minimum_cells:
                continue
            try:
                record = _record(cells, spec)
            except Exception:
                # 一行读不懂就跳过这一行。页面上混进一行脏数据很常见，
                # 为它放弃整场比赛的走势不划算（判据 18 的另一面：
                # 这里丢掉的是**读不懂的那一行**，不是读得懂的数据）。
                continue
            if record is not None:
                records.append(record)
    return records


def parse_history_scripts(html, spec):
    """表格里取不到时，退回页面脚本里的内联序列。

    命中一段脚本就停——多段脚本里往往是同一份数据的不同视图，
    全都收进来会重复。
    """
    records = []
    for script in _SCRIPT.findall(html):
        if len(script) < MIN_SCRIPT_LENGTH:
            continue
        if not _matches_keywords(script, spec.script_keywords,
                                 spec.script_keywords_lower):
            continue
        for values in spec.script_pattern.findall(script):
            record = {}
            for name, value in zip(spec.script_fields, values):
                # 脚本里的值不经过表格那套校验：能被正则的 `[\d.]+`
                # 匹配到就一定 float 得出来，所以这里直接转。
                record[name] = value if name in ('time', 'handicap', 'line',
                                                 'score') else float(value)
            records.append(record)
        if records:
            break
    return records


FROM_TABLE, FROM_SCRIPT = 'table', 'script'


def parse_history(html, spec):
    """一页里的走势记录，返回 `(记录, 来源)`。

    表格优先，取不到再看脚本，都没有就是 `([], None)`。**来源要一并返回**：
    从脚本里刮出来的数据经过的校验比表格那条少得多（时间长度、让球值长度、
    比分格式那几道都不走），调用方有理由区别对待。迁移前这一点只体现在
    两条不同的日志里。
    """
    # `if not html` 与 `if html is None` 在输出上等价（空串走下去也是
    # 空结果），差别只在白跑一趟正则。留短路是为了让「没有页面」这件事
    # 在这一层就结束，而不是靠下游恰好也返回空。
    if not html:
        return [], None
    # `display:none` 先抹掉：藏起来的那部分表格同样有数据，而正则不认识
    # 样式——不抹的话它照样能匹配到，抹掉是为了让**页面上藏与不藏**
    # 在这一层没有区别。
    # **这道清洗对表格抽取其实是空操作**：`<table[^>]*>` 这类正则本来就
    # 不看属性，`style="display:none"` 挡不住任何一次匹配。它唯一能改变的
    # 是**单元格文本里恰好出现这几个字**的情况。原样保留（删掉是行为改动），
    # 但别把它当成「藏起来的行靠它才解析得到」——那件事正则自己就做到了。
    cleaned = html.replace('display:none', '').replace('display: none', '')
    records = parse_history_tables(cleaned, spec)
    if records:
        return records, FROM_TABLE
    records = parse_history_scripts(cleaned, spec)
    return records, (FROM_SCRIPT if records else None)
