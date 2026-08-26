"""澳客页面的解析：混合过关赛程页，以及单场的欧赔 / 让分 / 大小三张详情页。

赛程页每行自带盘口变化历史（`rflist` 属性），这是让分与大小分走势的来源，
也是线上唯一活着的走势来源——**详情页在生产环境的出口 IP 上被 WAF 稳定
拦截**（2026-08-26 实测：服务器不通、本机通），所以 ml 共识（胜负走势）
在线上拿不到。代码照样迁完整，因为封锁是 IP 维度的、随时可能变。

两类页面的产物形状刻意做成同构：赛程行给 `rf_trend` / `dx_trend`，详情页
给 `bundle[kind]['trend']`，都是 `analyze_line_trend` 的输出，下游因此
不必区分数据来自哪一张页面。
"""
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

log = logging.getLogger('domain.basketball.okooo')

OKOOO_BASE = 'https://www.okooo.com'
OKOOO_HUNHE_URL = f'{OKOOO_BASE}/jingcailanqiu/hunhe/'
OKOOO_MATCH_URL = f'{OKOOO_BASE}/basketball/match/'

# 详情页三种玩法对应的 URL 段。ml 那张页面的路径是 odds 而不是 ml。
DETAIL_PATHS = {'ml': 'odds', 'ah': 'ah', 'ou': 'ou'}

_TAG = re.compile(r'<[^>]+>')
_WS = re.compile(r'\s+')
_TR = re.compile(r'<tr[^>]*>(.*?)</tr>', re.S)
_TR_WITH_ATTRS = re.compile(r'<tr([^>]*)>(.*?)</tr>', re.S)
_TD = re.compile(r'<td[^>]*>(.*?)</td>', re.S)
_TABLE = re.compile(r'<table[^>]*>(.*?)</table>', re.S)
_ISO_DATE = re.compile(r'(\d{4}-\d{2}-\d{2})')
_ROW_DATETIME = re.compile(r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})(?::\d{2})?')
_CLOCK = re.compile(r'(\d{2}:\d{2})')
_SCORE = re.compile(r'(\d+)\s*[-:]\s*(\d+)')
_SCORE_LOOSE = re.compile(r'\d+\s*[-:]\s*\d+')
_BRACKETS = re.compile(r'\[.*?\]')

# 赛程行的 class 标记。日期分组表头行没有它，用来区分「这是一场比赛」还是
# 「这是一个日期分隔」。
_MATCH_ROW_CLASS = 'alltrObj'
_MIN_CELLS = 7

_RFLIST_ENTRY = re.compile(
    r'(\d{2}/\d{2})\s+(\d{2}:\d{2})\s+([\d.]+)\s+\(([+\-]?\d+(?:\.\d+)?)\)\s+([\d.]+)')

# 走势判定阈值：水位缩短 0.03 以上才算被买入；盘口挪动满 1 分才算走盘
_WATER_MOVE = 0.03
_WATER_COUNTER_MOVE = 0.01
_LINE_MOVE = 1.0


def strip_html(text):
    return _WS.sub(' ', _TAG.sub(' ', text or '')).strip()


def safe_float(value):
    try:
        if value is None or value == '':
            return None
        return float(str(value).replace(',', '').strip())
    except (TypeError, ValueError):
        return None


def parse_rflist(rflist):
    """解析盘口历史属性：`07/14 11:47 1.75 (1.5) 1.65,07/14 10:17 ...`。

    页面按新→旧排列，返回按时间升序——走势计算全部假定「第一条是开盘」。
    认不出的片段直接丢弃：这个属性偶尔混进说明文字，为它整行报错不值得。
    """
    if not rflist:
        return []
    entries = []
    for part in str(rflist).split(','):
        matched = _RFLIST_ENTRY.match(part.strip())
        if not matched:
            continue
        entries.append({
            'date': matched.group(1),
            'time': matched.group(2),
            'home_odds': float(matched.group(3)),
            'line': float(matched.group(4)),
            'away_odds': float(matched.group(5)),
        })
    entries.reverse()
    return entries


def analyze_line_trend(history, kind='ah'):
    """盘口走势：水位缩短 = 资金偏该侧。样本不足两条时判为无变化。"""
    if not history or len(history) < 2:
        return {'direction': 'stable', 'strength': 0.0, 'kind': kind}

    first, last = history[0], history[-1]
    home_move = (last.get('home_odds') or 0) - (first.get('home_odds') or 0)
    away_move = (last.get('away_odds') or 0) - (first.get('away_odds') or 0)
    line_move = (last.get('line') or 0) - (first.get('line') or 0)

    return {
        'direction': _trend_direction(home_move, away_move, line_move, kind),
        'strength': round(abs(home_move) + abs(away_move) + abs(line_move) * 0.05, 4),
        'home_move': round(home_move, 4),
        'away_move': round(away_move, 4),
        'line_move': round(line_move, 4),
        'kind': kind,
        'samples': len(history),
        'opening_line': first.get('line'),
        'current_line': last.get('line'),
    }


def _trend_direction(home_move, away_move, line_move, kind):
    """优先看水位的对向变化，其次看单边跳水，最后才看盘口整体挪动。

    次序有讲究：对向变化是最干净的资金信号，盘口挪动则可能只是庄家调整
    风险敞口，两者同时出现时以水位为准。
    """
    if home_move < -_WATER_MOVE and away_move > _WATER_COUNTER_MOVE:
        return 'home_backing'
    if away_move < -_WATER_MOVE and home_move > _WATER_COUNTER_MOVE:
        return 'away_backing'
    if kind == 'ou' and home_move < -_WATER_MOVE:
        return 'over_backing'
    if kind == 'ou' and away_move < -_WATER_MOVE:
        return 'under_backing'
    if abs(line_move) >= _LINE_MOVE:
        return 'line_up' if line_move > 0 else 'line_down'
    return 'stable'


# ==================== 赛程页 ====================

_MATCH_ID = re.compile(r'/basketball/match/(\d+)/')
_NUM = re.compile(r'<i>(\d+)</i>')
_LEAGUE_TITLE = re.compile(
    r'href="[^"]*basketball/league/\d+/[^"]*"[^>]*title="([^"]+)"')
_LEAGUE_TEXT = re.compile(
    r'href="[^"]*basketball/league/\d+/[^"]*"[^>]*>([^<]+)</a>')
_LEAGUE_FALLBACK = re.compile(r'>(WNBA|NBA|CBA|NCAAB|欧篮联|美职女篮)</a>')
_TEAM_TITLE = re.compile(r'class="[^"]*duinameh[^"]*"[^>]*title="([^"]+)"')
_VS = re.compile(r'(.+?)\s+VS\s+(.+)', re.I)
_TEAMS_WITH_SCORE = re.compile(r'(.+?)\s+(\d+\s*[-:]\s*\d+)\s+(.+)')

_SF_ODDS = re.compile(r'class="betObj[^"]*"[^>]*>\s*([\d.]+)\s*<', re.S)
_SF_ODDS_ALT = re.compile(r'mixsf[\s\S]*?>([\d.]+)<[\s\S]*?>([\d.]+)<')
_RF_ODDS = re.compile(r'class="rbetObj[^"]*"[^>]*>\s*([\d.]+)\s*<', re.S)
_DX_ODDS = re.compile(r'class="dbetObj[^"]*"[^>]*>\s*([\d.]+)\s*<', re.S)
_RFLIST_ATTR = re.compile(r'rflist="([^"]*)"')
_HANDICAP_SPAN = re.compile(
    r'<span class="[^"]*rfsfrfzObj[^"]*"([^>]*)>([^<]+)</span>', re.S)
_TOTAL_SPAN = re.compile(
    r'<span class="[^"]*dxfjxzObj[^"]*"([^>]*)>([^<]+)</span>', re.S)
# 属性顺序在页面里出现过两种排法，两条都要试
_HANDICAP_ORDERED = re.compile(
    r'class="[^"]*rfsfrfzObj[^"]*"[^>]*rflist="([^"]*)"[^>]*>'
    r'\s*([+\-]?\d+(?:\.\d+)?)\s*<', re.S)
_HANDICAP_REVERSED = re.compile(
    r'rflist="([^"]*)"[^>]*class="[^"]*rfsfrfzObj[^"]*"[^>]*>'
    r'\s*([+\-]?\d+(?:\.\d+)?)', re.S)


def parse_schedule(html, date):
    """把混合过关赛程页解析成 match 列表（未做开赛过滤）。

    页面按日期分组：分组表头行携带日期、比赛行带 alltrObj。表头有时会落后于
    实际行，所以行内若带完整日期时间，以行内的为准。
    """
    tables = _TABLE.findall(html or '')
    if len(tables) < 2:
        log.warning('澳客篮球页未找到主表, tables=%d', len(tables))
        return []

    main = max(tables, key=len)
    current_date = date
    matches = []
    for attrs, body in _TR_WITH_ATTRS.findall(main):
        is_match_row = _MATCH_ROW_CLASS in _class_of(attrs)
        group_date = _ISO_DATE.search(body)
        if group_date and not is_match_row:
            current_date = group_date.group(1)
            continue
        if not is_match_row:
            continue
        try:
            match = parse_match_row(body, current_date)
        except Exception as exc:
            log.warning('解析澳客篮球行失败: %s', exc)
            continue
        if match:
            matches.append(match)
    return matches


def _class_of(attrs):
    matched = re.search(r'class="([^"]*)"', attrs)
    return matched.group(1) if matched else ''


def parse_match_row(row, date):
    cells = _TD.findall(row)
    if len(cells) < _MIN_CELLS:
        return None

    match_id = _MATCH_ID.search(row)
    if not match_id:
        return None

    teams = _parse_teams(cells[2])
    if teams is None:
        return None
    home, away = teams

    date, match_time = _row_datetime(row, cells[1], date)
    score_text = strip_html(cells[6])
    finished = bool(score_text and score_text != '-'
                    and _SCORE_LOOSE.search(score_text))
    rf_history, dx_history, odds = _parse_markets(cells[4])

    return {
        'id': match_id.group(1),
        'okooo_id': match_id.group(1),
        'date': date,
        'time': match_time,
        'num': _first_group(_NUM, cells[0], ''),
        'league': _parse_league(cells[0]),
        'home': home.strip(),
        'away': away.strip(),
        'status': 'finished' if finished else 'not_started',
        'home_score': _score(score_text, 1) if finished else None,
        'away_score': _score(score_text, 2) if finished else None,
        'source': 'okooo',
        'rf_history': rf_history,
        'dx_history': dx_history,
        'rf_trend': analyze_line_trend(rf_history, 'ah') if rf_history else None,
        'dx_trend': analyze_line_trend(dx_history, 'ou') if dx_history else None,
        **odds,
    }


def _first_group(pattern, text, default=None):
    matched = pattern.search(text)
    return matched.group(1) if matched else default


def _parse_league(cell):
    for pattern in (_LEAGUE_TITLE, _LEAGUE_TEXT):
        matched = pattern.search(cell)
        if matched and matched.group(1).strip():
            return matched.group(1).strip()
    return _first_group(_LEAGUE_FALLBACK, cell, '')


def _row_datetime(row, time_cell, fallback_date):
    """行内的完整日期时间优先于分组表头——表头会落后，把次日的比赛
    错标成今日，进而被开赛过滤误杀。"""
    row_dt = _ROW_DATETIME.search(row)
    if row_dt:
        return row_dt.group(1), row_dt.group(2)
    # 时间格里是「开赛 截止」两个时刻，取前一个
    return fallback_date, _first_group(_CLOCK, strip_html(time_cell), '')


def _parse_teams(cell):
    """优先取 title 属性里的全称；没有就退回纯文本，再区分未开赛与已完场。"""
    titles = _TEAM_TITLE.findall(cell)
    if len(titles) >= 2:
        return titles[0], titles[1]

    plain = strip_html(cell)
    versus = _VS.search(plain)
    if versus:
        home, away = versus.group(1), versus.group(2)
    else:
        with_score = _TEAMS_WITH_SCORE.search(plain)
        if not with_score:
            return None
        home, away = with_score.group(1), with_score.group(3)
    return _BRACKETS.sub('', home).strip(), _BRACKETS.sub('', away).strip()


def _parse_markets(cell):
    """一格里塞了三个玩法的水位、盘口与各自的变化历史。"""
    sf_odds = _SF_ODDS.findall(cell)
    if len(sf_odds) < 2:
        alternate = _SF_ODDS_ALT.findall(cell)
        sf_odds = list(alternate[0]) if alternate else []

    rf_odds = _RF_ODDS.findall(cell)
    dx_odds = _DX_ODDS.findall(cell)
    rf_rflist, handicap = _handicap(cell)
    dx_rflist, total_line = _total_line(cell)

    odds = {
        'spf_home': safe_float(_nth(sf_odds, 0)),
        'spf_away': safe_float(_nth(sf_odds, 1)),
        'rqspf_home': safe_float(_nth(rf_odds, 0)),
        'rqspf_away': safe_float(_nth(rf_odds, 1)),
        'dx_over': safe_float(_nth(dx_odds, 0)),
        'dx_under': safe_float(_nth(dx_odds, 1)),
        'handicap': handicap,
        'total_line': total_line,
    }
    return parse_rflist(rf_rflist), parse_rflist(dx_rflist), odds


def _nth(values, index):
    return values[index] if len(values) > index else None


def _handicap(cell):
    """让分值与它的历史。属性顺序页面上有两种排法，都要认。"""
    for pattern in (_HANDICAP_ORDERED, _HANDICAP_REVERSED):
        matched = pattern.search(cell)
        if matched:
            return matched.group(1), matched.group(2).strip()

    span = _HANDICAP_SPAN.search(cell)
    if not span:
        return '', None
    return _first_group(_RFLIST_ATTR, span.group(1), ''), span.group(2).strip()


def _total_line(cell):
    span = _TOTAL_SPAN.search(cell)
    if not span:
        return '', None
    return (_first_group(_RFLIST_ATTR, span.group(1), ''),
            safe_float(span.group(2)))


def _score(score_text, group):
    matched = _SCORE.search(score_text)
    return int(matched.group(group)) if matched else None


def select_live(matches, date, now):
    """按页面比分过滤完场，再按开赛时刻标进行中，最后只留未开赛的。

    完场以**比分**为准而不是时间：WNBA 有清晨开球的场次，按时间判断会被
    当成「昨天的比赛」误杀。
    """
    active = [m for m in matches if m.get('status') != 'finished']
    live = _same_day_or_nearest(active, date)

    for match in live:
        kickoff = _kickoff(match)
        if kickoff is not None:
            match['status'] = 'in_progress' if kickoff <= now else 'not_started'

    live = [m for m in live if m['status'] == 'not_started']
    log.info('澳客篮球获取到 %d 场未开赛比赛 (原始%d)', len(live), len(matches))
    return live


def _same_day_or_nearest(active, date):
    """当天的场次已全部结束时，页面会直接显示下一个销售日——此时只返回
    最近的那一天，不要把好几天的场次混在一起。"""
    same_day = [m for m in active if m.get('date') == date]
    if same_day:
        return same_day
    if not active:
        return []
    nearest = min(m.get('date') or '9999-12-31' for m in active)
    return [m for m in active if m.get('date') == nearest]


def _kickoff(match):
    try:
        return datetime.strptime(f"{match['date']} {match['time']}", '%Y-%m-%d %H:%M')
    except (ValueError, KeyError, TypeError):
        return None


class OkoooScheduleFetcher:
    def __init__(self, transport, now_fn=None):
        self._transport = transport
        self._now = now_fn or datetime.now

    def fetch(self, date=None):
        now = self._now()
        date = date or now.strftime('%Y-%m-%d')
        log.info('抓取澳客篮球赛程: %s', date)
        try:
            html = self._transport(OKOOO_HUNHE_URL)
        except Exception as exc:
            log.warning('澳客篮球赛程抓取失败: %s', exc)
            return []
        if not html:
            log.warning('澳客篮球赛程为空(WAF/网络)')
            return []
        return select_live(parse_schedule(html, date), date, now)


# ==================== 详情页 ====================

_AVERAGE_ML = re.compile(
    r'平均值</td>\s*<td[^>]*>\s*([\d.]+)\s*</td>\s*<td[^>]*>\s*([\d.]+)\s*</td>\s*'
    r'<td[^>]*>\s*([\d.]+)\s*</td>\s*<td[^>]*>\s*([\d.]+)\s*</td>', re.S)
_PLAIN_NUMBER = re.compile(r'>([+\-]?\d+(?:\.\d+)?)<')
_BOOK_SPAN = re.compile(r'<span>([^<]+)</span>')
_BOOK_NAME = re.compile(r'<span title="([^"]+)"')

# 各家行里前面还有序号、公司 id 等数字，靠「赔率-盘口-赔率」这个形状定位
# 真正的数据段。赔率在 1.01~5.0，让分盘口不超过 40 分，总分线在 100~280。
_ODDS_RANGE = (1.01, 5.0)
_HANDICAP_RANGE = 40
_TOTAL_RANGE = (100, 280)
_BOOK_CHUNK = 6


def parse_average_row(html, kind):
    """页脚的平均值行。只有欧赔那张页面给了「初赔 + 即时赔」四个数。"""
    if kind != 'ml':
        return {}
    matched = _AVERAGE_ML.search(html)
    if not matched:
        return {}
    return {
        'home_init': float(matched.group(1)),
        'away_init': float(matched.group(2)),
        'home': float(matched.group(3)),
        'away': float(matched.group(4)),
    }


def parse_book_rows(html, kind):
    """从各家赔率表里逐行取出开盘值与即时值。"""
    path = DETAIL_PATHS[kind]
    company_id = re.compile(rf'/{path}/(?:change|handicap|line)/(\d+)/')
    books = []
    for row in _TR.findall(html or ''):
        if not _is_book_row(row, path):
            continue
        book = _parse_book_row(row, kind, company_id)
        if book:
            books.append(book)
    return books


def _is_book_row(row, path):
    if f'/{path}/change/' in row or f'/{path}/handicap/' in row:
        return True
    return f'/{path}/' in row and 'change/' in row


def _parse_book_row(row, kind, company_id):
    base = {
        'company_id': _first_group(company_id, row, ''),
        'name': _first_group(_BOOK_NAME, row, ''),
    }
    if kind == 'ml':
        numbers = [safe_float(x) for x in _BOOK_SPAN.findall(row)]
        numbers = [x for x in numbers if x is not None]
        if len(numbers) < 4:
            return None
        return {**base, 'home_init': numbers[0], 'away_init': numbers[1],
                'home': numbers[2], 'away': numbers[3]}

    chunk = _locate_chunk(row, kind)
    if chunk is None:
        return None
    if kind == 'ah':
        keys = ('home_init', 'line_init', 'away_init', 'home', 'line', 'away')
    else:
        keys = ('over_init', 'line_init', 'under_init', 'over', 'line', 'under')
    return {**base, **dict(zip(keys, chunk))}


def _locate_chunk(row, kind):
    """在一串数字里找出「赔率 盘口 赔率」开头的六元组。

    不能直接取前六个：行首往往还有序号、公司编号这类数字，取错一位整行
    数据就全串了位，而且不会有任何报错。
    """
    numbers = [safe_float(x) for x in _PLAIN_NUMBER.findall(row)]
    numbers = [x for x in numbers if x is not None]
    if len(numbers) < _BOOK_CHUNK:
        return None

    start = 0
    for i in range(len(numbers) - 5):
        if _looks_like_odds_pair(numbers[i], numbers[i + 1], numbers[i + 2], kind):
            start = i
            break
    chunk = numbers[start:start + _BOOK_CHUNK]
    return chunk if len(chunk) == _BOOK_CHUNK else None


def _looks_like_odds_pair(first, line, second, kind):
    low, high = _ODDS_RANGE
    if not (low <= first <= high and low <= second <= high):
        return False
    if kind == 'ah':
        return abs(line) <= _HANDICAP_RANGE
    return _TOTAL_RANGE[0] <= line <= _TOTAL_RANGE[1]


_CONSENSUS_SPECS = {
    'ml': {'sides': ('home', 'away'), 'line': False},
    'ah': {'sides': ('home', 'away'), 'line': True},
    'ou': {'sides': ('over', 'under'), 'line': True},
}


def consensus_from_books(books, kind):
    """把各家的开盘与即时值平均成一份共识，并算出它自身的走势。

    三个玩法原本各写了一份几乎相同的实现，差别只有两路的字段名和有没有
    盘口。合成一份后，四舍五入位数、概率归一这些细节只有一处。
    """
    if not books:
        return {'available': False, 'book_count': 0}

    spec = _CONSENSUS_SPECS[kind]
    first, second = spec['sides']
    average = _averager(books)

    now_first, now_second = average(first), average(second)
    init_first, init_second = average(f'{first}_init'), average(f'{second}_init')
    line = average('line') if spec['line'] else None
    line_init = average('line_init') if spec['line'] else None

    out = {
        'available': True,
        'book_count': len(books),
        first: _round(now_first, 3),
        second: _round(now_second, 3),
        f'{first}_move': _move(now_first, init_first, 4),
        f'{second}_move': _move(now_second, init_second, 4),
    }
    if kind == 'ml':
        out['home_init'] = _round(init_first, 3)
        out['away_init'] = _round(init_second, 3)
    if spec['line']:
        out['line'] = round(line, 2) if line is not None else None
        out['line_init'] = round(line_init, 2) if line_init is not None else None
        out['line_move'] = (round(line - line_init, 2)
                            if line is not None and line_init is not None else 0.0)

    _add_probabilities(out, first, second, now_first, now_second)
    out['trend'] = analyze_line_trend(
        _consensus_history(now_first, now_second, line,
                           init_first, init_second, line_init, spec['line']),
        kind)
    return out


def _averager(books):
    def average(key):
        values = [b[key] for b in books if b.get(key) is not None]
        return sum(values) / len(values) if values else None

    return average


def _round(value, digits):
    return round(value, digits) if value else None


def _move(now, init, digits):
    return round(now - init, digits) if now and init else 0.0


def _add_probabilities(out, first, second, now_first, now_second):
    if not (now_first and now_second and now_first > 0 and now_second > 0):
        return
    p_first, p_second = 1 / now_first, 1 / now_second
    total = p_first + p_second
    out[f'{first}_prob'] = round(p_first / total, 4)
    out[f'{second}_prob'] = round(p_second / total, 4)


def _consensus_history(now_first, now_second, line,
                       init_first, init_second, line_init, has_line):
    """把「开盘共识 → 即时共识」当成两个采样点喂给走势分析。"""
    history = []
    if init_first and init_second and (not has_line or line_init is not None):
        history.append({'home_odds': init_first, 'away_odds': init_second,
                        'line': line_init if has_line else 0})
    if now_first and now_second and (not has_line or line is not None):
        history.append({'home_odds': now_first, 'away_odds': now_second,
                        'line': line if has_line else 0})
    return history


def build_bundle(match_id, pages):
    """把一场比赛的三张详情页合成共识包。缺页的玩法留 unavailable。"""
    bundle = {
        'match_id': str(match_id),
        'ml': {'available': False},
        'ah': {'available': False},
        'ou': {'available': False},
    }
    for kind in DETAIL_PATHS:
        html = pages.get(kind)
        if not html:
            continue
        books = parse_book_rows(html, kind)
        consensus = consensus_from_books(books, kind)
        if kind == 'ml':
            consensus = _prefer_page_average(consensus, parse_average_row(html, 'ml'),
                                             len(books))
        if consensus.get('available'):
            bundle[kind] = {**consensus, 'books_sample': books[:5]}
    return bundle


def _prefer_page_average(consensus, average, book_count):
    """欧赔页脚自带平均值，比自己对各家做算术平均更权威——它是页面方
    按自己的口径算的，包含了我们解析不到的那些家。"""
    if not average or not average.get('home'):
        return consensus

    home, away = average['home'], average['away']
    home_init, away_init = average.get('home_init'), average.get('away_init')
    out = {**consensus, 'available': True, 'home': home, 'away': away,
           'home_init': home_init, 'away_init': away_init, 'source': 'page_avg'}
    _add_probabilities(out, 'home', 'away', home, away)
    if home_init and away_init:
        out['home_move'] = round(home - home_init, 4)
        out['away_move'] = round(away - away_init, 4)
        out['trend'] = analyze_line_trend([
            {'home_odds': home_init, 'away_odds': away_init, 'line': 0},
            {'home_odds': home, 'away_odds': away, 'line': 0},
        ], 'ml')
    if not out.get('book_count'):
        out['book_count'] = book_count
    return out


def detail_url(match_id, kind):
    return f'{OKOOO_MATCH_URL}{match_id}/{DETAIL_PATHS[kind]}/'


class MarketBundleFetcher:
    """按场并发抓三张详情页并合成共识包。

    单场失败只影响这一场——线上常态是三张页面全被 WAF 拦掉，那就整体
    unavailable，走势退回赛程页自带的 rf_trend / dx_trend。
    """

    def __init__(self, transport, max_workers=6):
        self._transport = transport
        self._max_workers = max_workers

    def fetch_many(self, match_ids):
        ids = [str(i) for i in match_ids if i]
        if not ids:
            return {}
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {executor.submit(self.fetch_one, mid): mid for mid in ids}
            return {futures[f]: _bundle_result(futures[f], f)
                    for f in as_completed(futures)}

    def fetch_one(self, match_id):
        pages = {}
        for kind in DETAIL_PATHS:
            try:
                html = self._transport(detail_url(match_id, kind))
            except Exception as exc:
                log.warning('各家赔率抓取失败 %s/%s: %s', match_id, kind, exc)
                continue
            if html:
                pages[kind] = html
        return build_bundle(match_id, pages)


def _bundle_result(match_id, future):
    try:
        return future.result()
    except Exception as exc:
        log.warning('各家赔率抓取失败 %s: %s', match_id, exc)
        return {'match_id': match_id, 'ml': {'available': False},
                'ah': {'available': False}, 'ou': {'available': False}}
