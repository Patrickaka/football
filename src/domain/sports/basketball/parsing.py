"""500 源竞彩篮球赛程页的解析。

页面是一张表，每场一行、七个单元格：期号 / 联赛 / 时间 / 对阵 / 胜负 /
让分胜负 / 大小分。赔率都嵌在 `<span>` 里，缺盘的场次 span 数量会变少，
所以每个字段都按下标存在与否取值，取不到就是 None——**缺赔率必须留 None
而不是补 0**，下游用 `is None` 判断这场有没有开这个玩法。

迁移前这个解析器有两份逐字相同的副本：主流程一份，「今日无赛回退次日」
里又抄了一份。两份共约 50 行，任何正则调整都得改两处。这里合成一份，
回退只是换一个基准日期再解析一次。
"""
import logging
import re
from datetime import datetime, timedelta

log = logging.getLogger('domain.basketball.parsing')

BASE_URL = 'https://trade.500.com'
SCHEDULE_URL = f'{BASE_URL}/jclq/'

_TR = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL)
_TD = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL)
_SPAN = re.compile(r'<span[^>]*>([^<]*)</span>')
_TAG = re.compile(r'<[^>]*>')
# 期号形如「周三301」，必须以中文或字母开头——表头与广告行以此排除
_NUM_START = re.compile(r'^[一-龥a-zA-Z]')
_HOME_PREFIX = re.compile(r'^\[\w+\d*\]')
_AWAY_SUFFIX = re.compile(r'\[\w+\d*\]$')
# 时间单元格是 `08-27 07:00`（带月日），日期部分已经并进了 match['date']，
# 判断开赛时刻只需要其中的时分。
_CLOCK = re.compile(r'(\d{1,2}:\d{2})')

_MIN_CELLS = 7
_VS_MARKERS = ('VS', 'vs', '对')
# 开赛 3 小时后视为完场；开赛前 1 小时内视为进行中（页面此时已停售）
_FINISHED_AFTER = timedelta(hours=3)
_IN_PROGRESS_BEFORE = timedelta(hours=1)


def schedule_url(date):
    return f'{BASE_URL}/jclq/?playid=313&g=2&date={date}'


def parse_schedule(html, date):
    """把赛程页解析成 match 列表。date 只用于补全年份，不做过滤。"""
    matches = []
    for row in _TR.findall(html or ''):
        cells = _TD.findall(row)
        if len(cells) < _MIN_CELLS:
            continue
        try:
            match = _parse_row(cells, date)
        except Exception as exc:
            log.warning('解析篮球比赛失败: %s', exc)
            continue
        if match:
            matches.append(match)
    return matches


def _parse_row(cells, date):
    num = _text(cells[0])
    if not num or not _NUM_START.match(num):
        return None

    teams = _parse_teams(_text(cells[3]))
    if teams is None:
        return None
    home, away = teams

    match_time = _text(cells[2])
    match_date = _date_from_time(match_time, date)
    spf_home, spf_away = _two_odds(cells[4])
    rqspf_home, handicap, rqspf_away = _odds_around_line(cells[5])
    dx_over, total_line, dx_under = _odds_around_line(cells[6])

    return {
        'id': f'{match_date}_{home}_{away}',
        'date': match_date,
        'time': match_time,
        'num': num,
        'league': _text(cells[1]),
        'home': home,
        'away': away,
        'handicap': handicap,
        'rqspf_home': rqspf_home,
        'rqspf_away': rqspf_away,
        'spf_home': spf_home,
        'spf_away': spf_away,
        'total_line': _as_float(total_line),
        'dx_over': dx_over,
        'dx_under': dx_under,
        'status': 'not_started',
    }


def _text(cell):
    return _TAG.sub('', cell).strip()


def _parse_teams(team_text):
    """对阵单元格形如「[主]主队VS客队[客]」，分隔符有三种写法。"""
    for marker in _VS_MARKERS:
        index = team_text.find(marker)
        if index != -1:
            break
    else:
        return None

    home = _HOME_PREFIX.sub('', team_text[:index]).strip()
    away = _AWAY_SUFFIX.sub('', team_text[index + 2:].strip()).strip()
    return (home, away) if home and away else None


def _two_odds(cell):
    spans = _SPAN.findall(cell)
    return _as_float(_at(spans, 0)), _as_float(_at(spans, 1))


def _odds_around_line(cell):
    """让分与大小分的单元格是「本方水位 / 盘口 / 对方水位」三段。"""
    spans = _SPAN.findall(cell)
    return _as_float(_at(spans, 0)), _at(spans, 1), _as_float(_at(spans, 2))


def _at(spans, index):
    return spans[index] if len(spans) > index else None


def _as_float(value):
    """只接受纯数字文本。盘口那格可能是 `+13.5` 这类带符号的，留给调用方按原样保存。"""
    if value is None or not value.replace('.', '').isdigit():
        return None
    return float(value)


def _date_from_time(match_time, date):
    """时间单元格带月日（`08-27 07:00`），年份从请求日期取。"""
    if match_time and len(match_time) >= 5:
        return f'{date[:4]}-{match_time[:5]}'
    return date


def annotate_status(matches, now):
    """按开赛时间打状态。时间解析不了就保持原样——不猜。"""
    for match in matches:
        kickoff = _kickoff(match)
        if kickoff is None:
            continue
        if kickoff < now - _FINISHED_AFTER:
            match['status'] = 'finished'
        elif kickoff < now + _IN_PROGRESS_BEFORE:
            match['status'] = 'in_progress'
        else:
            match['status'] = 'not_started'


def select_upcoming(matches, now):
    """只留未开赛的。已开赛的既没有分析价值，也白白拖慢逐场计算。

    时间解析不了的一律保留：宁可多显示一场，也不静默吞掉。
    """
    return [m for m in matches
            if (_kickoff(m) is None) or (_kickoff(m) > now)]


def _kickoff(match):
    """从 date + 时间单元格算出开赛时刻。

    时间单元格的形状是 `08-27 07:00`，直接拿去拼 `%Y-%m-%d %H:%M` 会得到
    `2026-08-27 08-27 07:00`，必然 ValueError。迁移前正是这么写的，异常又被
    静默吞掉，于是 500 源的开赛过滤**从来没有生效过**：所有场次恒为
    not_started，已经打完的比赛照样出现在推荐列表里。只取其中的时分即可。
    """
    clock = _CLOCK.search(str(match.get('time') or ''))
    if not clock:
        return None
    try:
        return datetime.strptime(f"{match['date']} {clock.group(1)}",
                                 '%Y-%m-%d %H:%M')
    except (ValueError, KeyError, TypeError):
        return None


class ScheduleFetcher:
    """抓 + 解析 + 今日无赛时回退次日。

    transport 必须显式注入：真实实现会发网络请求，给默认值会让忘记注入的
    测试静默连上真实源站。
    """

    def __init__(self, transport, now_fn=None):
        self._transport = transport
        self._now = now_fn or datetime.now

    def fetch(self, date=None):
        now = self._now()
        date = date or now.strftime('%Y-%m-%d')
        try:
            matches = self._fetch_day(date)
            if not matches:
                tomorrow = (now + timedelta(days=1)).strftime('%Y-%m-%d')
                log.info('今日无比赛，尝试获取明日赛程: %s', tomorrow)
                matches = self._fetch_day(tomorrow)

            annotate_status(matches, now)
            upcoming = select_upcoming(matches, now)
            log.info('获取到 %d 场未开赛篮球比赛', len(upcoming))
            return upcoming
        except Exception as exc:
            log.error('抓取篮球赛程失败: %s', exc)
            return []

    def _fetch_day(self, date):
        log.info('抓取篮球赛程: %s', date)
        html = self._transport(schedule_url(date))
        if not html:
            log.warning('未获取到篮球赛程内容')
            return []
        return parse_schedule(html, date)
