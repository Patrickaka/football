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
from datetime import datetime as _datetime, timedelta as _timedelta
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


# ── 赛程页 ───────────────────────────────────────────────────────
# 两个来源的页面结构完全不同：okooo 是一张规整的表格，500.com 是一堆
# 带 `shuju-*.shtml` 链接的锚点加上散落各处的时间、场次号与联赛块。
# 后者要把这些**分别抓出来再按比赛号拼回去**——页面上它们并不在一起。

_SCHEDULE_LINK = re.compile(
    r'shuju-(\d+)\.shtml.*?title="([^"]+?)VS([^"]+?)'
    r'(?:数据|盘口|百家|欧赔|亚赔|亚盘|指数|对比|分析)[^"]*"', re.DOTALL)
# 时间有两种排布：跨行的时间格在链接**之前**，或时间跟在链接**之后**。
# 两条都试，先命中的那条算数。
_SCHEDULE_TIME_PATTERNS = (
    re.compile(r'<td[^>]*?rowspan="2"[^>]*?>(\d{2}-\d{2}\s+\d{2}:\d{2})</td>.*?'
               r'shuju-(\d+)\.shtml', re.DOTALL),
    re.compile(r'shuju-(\d+)\.shtml.*?(\d{2}-\d{2}\s+\d{2}:\d{2})', re.DOTALL),
)
_SCHEDULE_NUM = re.compile(r'value="(\d+)"\s*/>\s*(周[一二三四五六日]\d{3})')
_SCHEDULE_LEAGUE_SPLIT = re.compile(
    r'<a[^>]*href="//liansai\.500\.com/zuqiu-\d+/"[^>]*>([^<]+)</a>')
_SCHEDULE_ID = re.compile(r'shuju-(\d+)\.shtml')
_DAY_AND_TIME = re.compile(r'(\d{2}-\d{2})\s+(\d{2}:\d{2})')

# 队名后面常跟着入口的名字（同一场比赛在页面上有九个入口）。要逐个剥掉。
_NAME_SUFFIXES = ('百家', '欧赔', '亚赔', '亚盘', '数据', '盘口', '指数',
                  '对比', '分析', '百家欧赔', '百家亚盘')

# 开赛前一小时算「进行中」，赛后三小时算「已结束」。三小时是一场球加中场
# 加伤停补时的上限——比这更短会把还在踢的比赛判成已结束。
IN_PROGRESS_BEFORE_HOURS = 1
FINISHED_AFTER_HOURS = 3
NOT_STARTED, IN_PROGRESS, FINISHED = 'not_started', 'in_progress', 'finished'

_EMPTY_MATCH = {
    'num': '', 'time': '', 'league': '',
    'spf_sp': None, 'spf_s': None, 'spf_f': None,
    'rqspf_sp': None, 'rqspf_s': None, 'rqspf_f': None,
    'handicap': None, 'status': NOT_STARTED,
}


def _strip_entry_suffixes(name):
    for suffix in _NAME_SUFFIXES:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    return name


def _schedule_times(html):
    """比赛号 → 时间。两种排布都试，**先命中的先占位**。"""
    times = {}
    for pattern in _SCHEDULE_TIME_PATTERNS:
        for match in pattern.finditer(html):
            first, second = match.group(1), match.group(2)
            match_id, value = ((first, second) if first.isdigit()
                               else (second, first))
            times.setdefault(match_id, value)
    return times


def _schedule_leagues(html):
    """联赛名与它下面的比赛。页面用联赛链接分段，段内的比赛都归它。"""
    blocks = _SCHEDULE_LEAGUE_SPLIT.split(html)
    leagues = {}
    current = ''
    for index, block in enumerate(blocks):
        if index % 2 == 1:
            current = block.strip()
            continue
        for match_id in _SCHEDULE_ID.findall(block):
            leagues.setdefault(match_id, current)
    return leagues


def _match_status(match, now):
    """按开赛时刻判断状态。

    **这里有一处迁移前就存在的缺陷，原样保留了**：拿不到开赛时刻时会走到
    「只按日期判断」那条分支，而它比较的是 `datetime` 与 `date`，
    Python 直接抛 `TypeError`。外层只 `catch ValueError`，于是异常一路冒到
    最外面，**整份赛程变成空列表**——一场没解析到时间的比赛，
    足以让当天所有比赛都消失。

    这条路只在 okooo 挂掉、回退到 500.com 时才走（线上 7 天内一次都没走过），
    所以迁移期没有动它。修它会改变回退路径的返回值，应单独决策。
    """
    try:
        if match['date'] and match['time']:
            kickoff = _datetime.strptime(f"{match['date']} {match['time']}",
                                         '%Y-%m-%d %H:%M')
        else:
            kickoff = None
    except ValueError:
        kickoff = None

    if kickoff:
        if kickoff < now - _timedelta(hours=FINISHED_AFTER_HOURS):
            return FINISHED
        if kickoff < now + _timedelta(hours=IN_PROGRESS_BEFORE_HOURS):
            return IN_PROGRESS
        return NOT_STARTED

    try:
        day = _datetime.strptime(match['date'], '%Y-%m-%d')
    except ValueError:
        return match['status']
    # ↓ `datetime < date`，必抛 TypeError。见上面的说明。
    return FINISHED if day < now.date() else match['status']


def parse_500_schedule(html, date, now):
    """500.com 的即时赔率页 → 未完结的比赛列表。

    时间、场次号、联赛名在页面上分散在三处，各自抓出来再按比赛号拼回去。
    已结束的比赛在返回前滤掉——调用方要的是「今天还能买的」。
    """
    matches = []
    for found in _SCHEDULE_LINK.finditer(html):
        match_id = found.group(1).strip()
        home = _strip_entry_suffixes(found.group(2).strip())
        away = _strip_entry_suffixes(found.group(3).strip())
        if home and away and match_id:
            matches.append(dict(_EMPTY_MATCH, id=match_id, home=home,
                                away=away, date=date))

    times = _schedule_times(html)
    leagues = _schedule_leagues(html)
    numbers = dict(_SCHEDULE_NUM.findall(html))

    for match in matches:
        match_id = match['id']
        if match_id in times:
            # 能进 `times` 的字符串都是被 `\d{2}-\d{2}\s+\d{2}:\d{2}`
            # 捕获出来的，而这里用的是同一个形状——**所以它必然匹配成功**。
            # 迁移前这里有一条 `else: match['time'] = when` 的兜底，
            # 任何输入都走不到，已删（判据 9 第一类）。
            parsed = _DAY_AND_TIME.match(times[match_id])
            match['time'] = parsed.group(2)
            # 时间里的日期与请求的日期不同时，记录跟着挪过去
            # ——跨零点的比赛属于第二天
            if parsed.group(1) != date[5:]:
                match['date'] = f'{date[:4]}-{parsed.group(1)}'
        if match_id in leagues:
            match['league'] = leagues[match_id].strip()
        if match_id in numbers:
            match['num'] = numbers[match_id]
        match['status'] = _match_status(match, now)

    return [match for match in matches if match['status'] != FINISHED]


# ── okooo 的单场页 ───────────────────────────────────────────────
_OKOOO_NUM = re.compile(r'<span class="xh"><i>(\d+)</i></span>')
_OKOOO_LEAGUE = re.compile(
    r'href="//www\.okooo\.com/soccer/league/\d+/"[^>]*>([^<]+)</a>')
_OKOOO_MATCH_ID = re.compile(r'/soccer/match/(\d+)')
_OKOOO_MTIME = re.compile(r'mTime="([^"]+)"')
_OKOOO_HOME = re.compile(
    r'<span class="homenameobj[^>]*title="([^"]+)"[^>]*>([^<]+)</span>')
_OKOOO_AWAY = re.compile(
    r'<span class="awaynameobj[^>]*title="([^"]+)"[^>]*>([^<]+)</span>')
_OKOOO_HANDICAP = re.compile(r'<span class="handicapobj[^>]*>([^<]+)</span>')
_OKOOO_ODDS = re.compile(r'<em[^>]*>([\d.]+)</em>')
_OKOOO_DATE = re.compile(r'(\d{4}-\d{2}-\d{2})')

# 比赛表是页面上的第二张表，前面还有目录之类。少于两张说明页面不对——
# 多半是被 WAF 换成了别的东西，而不是「今天没有比赛」。
OKOOO_MINIMUM_TABLES = 2
OKOOO_SCHEDULE_CELLS = 6
# 胜平负三个价在前，让球三个价在后。**让球那三个要六个都在才算数**：
# 只报了前三个时后三个是别的东西，取来会得到一组假赔率。
OKOOO_FULL_ODDS = 6


def _okooo_price(odds, index, required):
    return float(odds[index]) if len(odds) >= required else None


def parse_okooo_schedule(html, date, minimum_tables=OKOOO_MINIMUM_TABLES):
    """okooo 的单场页 → `(未完结比赛, 表格数)`。

    **表格数不足时返回 `(None, 数量)`**：那是「页面不对」，与「今天没有
    未完结比赛」不是一回事，而两者都会让调用方去找备用数据源——
    分开返回是为了让日志能说清是哪一种（§十一·3 那类故障里，
    「200 加 0 场比赛」最难查的地方正是分不清这两者）。
    """
    tables = _TABLE.findall(html)
    if len(tables) < minimum_tables:
        return None, len(tables)

    matches = []
    current_date = date
    for row in _ROW.findall(tables[1]):
        cells = _CELL.findall(row)
        if len(cells) < OKOOO_SCHEDULE_CELLS:
            found = _OKOOO_DATE.search(row)
            if found:
                current_date = found.group(1)
            continue

        home = _OKOOO_HOME.search(cells[2])
        away = _OKOOO_AWAY.search(cells[2])
        if not home or not away:
            continue

        num_found = _OKOOO_NUM.search(cells[0])
        league_found = _OKOOO_LEAGUE.search(cells[0])
        id_found = _OKOOO_MATCH_ID.search(row)
        num = num_found.group(1) if num_found else ''
        handicap_found = _OKOOO_HANDICAP.search(cells[2])
        odds = _OKOOO_ODDS.findall(cells[2])

        spf = [_okooo_price(odds, index, index + 1) for index in range(3)]
        rqspf = [_okooo_price(odds, index, OKOOO_FULL_ODDS)
                 for index in range(3, OKOOO_FULL_ODDS)]

        match_date, match_time = _okooo_kickoff(cells[1], current_date)
        score = _TAG.sub('', cells[5]).strip()
        matches.append({
            'id': (id_found.group(1) if id_found
                   else f'{current_date.replace("-", "")}_{num}'),
            'home': home.group(2).strip(), 'away': away.group(2).strip(),
            'num': num,
            'date': match_date, 'time': match_time,
            'league': league_found.group(1) if league_found else '',
            'spf_sp': spf[0], 'spf_s': spf[1], 'spf_f': spf[2],
            'rqspf_sp': rqspf[0], 'rqspf_s': rqspf[1], 'rqspf_f': rqspf[2],
            'rqspf_odds': (
                {'让胜': rqspf[0], '让平': rqspf[1], '让负': rqspf[2]}
                if all(value and value > 1.0 for value in rqspf) else None),
            'handicap': (handicap_found.group(1).strip()
                         if handicap_found else None),
            # **状态不看时钟，看比分栏**：有比分就是踢完了。比 500 那边
            # 按时间推算可靠——页面自己说的比我们算的准。
            'status': FINISHED if score and score != '-' else NOT_STARTED,
            'source': 'okooo',
        })

    return [m for m in matches if m['status'] != FINISHED], len(tables)


def _okooo_kickoff(cell, current_date):
    """开赛时刻优先取 `mTime` 属性，没有再从单元格文本里认。

    认不出格式时**原样留着那段文本**（`'稍后'` 这种），日期退回当前段落的
    日期。两个来源在这一点上处置相同——迁移时我以为它们不一样，
    黄金比对当场把这个想当然抓了出来。
    """
    mtime = _OKOOO_MTIME.search(cell)
    raw = mtime.group(1) if mtime else _TAG.sub('', cell).strip()
    parsed = _DAY_AND_TIME.match(raw)
    if parsed:
        return f'{current_date[:4]}-{parsed.group(1)}', parsed.group(2)
    return current_date, raw
