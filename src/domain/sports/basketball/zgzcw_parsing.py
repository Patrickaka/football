"""中国足彩网竞彩篮球赛程解析。

胜负/让分和大小分位于两个公开页面；用同一 ``tr_<match_id>`` 合并，避免按
队名猜测。页面顺序是客队在前、主队在后，赔率也按“主负、主胜”展示，解析
时统一转换成领域层使用的主/客字段。
"""

import logging
import re
from datetime import datetime

log = logging.getLogger('domain.basketball.zgzcw')

ZGZCW_BASE = 'https://cp.zgzcw.com'
ZGZCW_SF_URL = f'{ZGZCW_BASE}/lottery/jchtplayvsForJsp.action?lotteryId=48&type=jcmini'
ZGZCW_DX_URL = f'{ZGZCW_BASE}/lottery/jcplayvsForJsp.action?lotteryId=29'

_ROW = re.compile(r'<tr\b([^>]*)>(.*?)</tr>', re.I | re.S)
_TAG = re.compile(r'<[^>]+>')
_ATTR_TEMPLATE = r'\b{name}=["\']([^"\']*)["\']'
_SCORE = re.compile(r'(\d+)\s*[:\-]\s*(\d+)')


def _attr(fragment, name, default=''):
    found = re.search(_ATTR_TEMPLATE.format(name=re.escape(name)), fragment or '', re.I)
    return found.group(1).strip() if found else default


def _text(fragment):
    return re.sub(r'\s+', ' ', _TAG.sub(' ', fragment or '')).strip()


def _cell(row, class_name):
    found = re.search(
        rf'<td\b([^>]*)class=["\'][^"\']*\b{re.escape(class_name)}\b[^"\']*["\'][^>]*>'
        rf'(.*?)</td>', row, re.I | re.S)
    return (found.group(1), found.group(2)) if found else ('', '')


def _team(row, class_name):
    _, body = _cell(row, class_name)
    link = re.search(r'<a\b[^>]*?(?:title=["\']([^"\']*)["\'])?[^>]*>(.*?)</a>',
                     body, re.I | re.S)
    if not link:
        return ''
    return _text(link.group(2)) or (link.group(1) or '').strip()


def _market(row, pid):
    found = re.search(
        rf'<div\b[^>]*\bpid=["\']{re.escape(str(pid))}["\'][^>]*>(.*?)</div>',
        row, re.I | re.S)
    if not found:
        return None, []
    block = found.group(1)
    line = re.search(r'<em\b[^>]*class=["\'][^"\']*(?:total|rq)[^"\']*["\'][^>]*>(.*?)</em>',
                     block, re.I | re.S)
    try:
        line_value = float(_text(line.group(1))) if line else None
    except ValueError:
        line_value = None
    odds = []
    for value in re.findall(r'<a\b[^>]*\bid=["\']td_[^"\']+["\'][^>]*>(.*?)</a>',
                            block, re.I | re.S):
        try:
            odds.append(float(_text(value)))
        except ValueError:
            pass
    return line_value, odds


def parse_page(html, kind):
    matches = {}
    for attrs, row in _ROW.findall(html or ''):
        row_id = _attr(attrs, 'id', '')
        if not row_id.startswith('tr_'):
            continue
        match_id = row_id[3:]
        away, home = _team(row, 'wh-4'), _team(row, 'wh-6')
        if not (match_id and home and away):
            continue
        kickoff = re.search(r'title=["\']比赛时间:([^"\']+)["\']', row, re.I)
        kickoff = kickoff.group(1).strip() if kickoff else ''
        date = kickoff[:10] if re.match(r'\d{4}-\d{2}-\d{2}', kickoff) else ''
        clock = re.search(r'(\d{2}:\d{2})', kickoff)
        num = _text(_cell(row, 'wh-1')[1])
        day = _text(re.search(r'<code\b[^>]*>(.*?)</code>', row,
                              re.I | re.S).group(1)) if re.search(
                                  r'<code\b[^>]*>(.*?)</code>', row, re.I | re.S) else ''
        order = re.search(r'(\d{3})', num)
        num = f'{day}{order.group(1)}' if order else num
        score = _SCORE.search(_text(_cell(row, 'wh-5')[1]))
        analysis = re.search(r'\bnewplayid=["\'](\d+)["\']', row, re.I)
        match = {
            'id': match_id, 'zgzcw_id': match_id,
            'analysis_id': analysis.group(1) if analysis else None,
            'date': date, 'time': clock.group(1) if clock else '', 'num': num,
            'league': _attr(attrs, 'm', '') or _text(_cell(row, 'wh-2')[1]),
            'home': home, 'away': away,
            'status': 'finished' if score else 'not_started',
            'home_score': int(score.group(2)) if score else None,
            'away_score': int(score.group(1)) if score else None,
            'source': 'zgzcw',
            'spf_home': None, 'spf_away': None,
            'rqspf_home': None, 'rqspf_away': None, 'handicap': None,
            'dx_over': None, 'dx_under': None, 'total_line': None,
        }
        if kind == 'sf':
            _, sf = _market(row, 26)
            handicap, rf = _market(row, 27)
            if len(sf) >= 2:
                match['spf_away'], match['spf_home'] = sf[0], sf[1]
            if len(rf) >= 2:
                match['rqspf_away'], match['rqspf_home'] = rf[0], rf[1]
                match['handicap'] = handicap
        else:
            total, dx = _market(row, 29)
            if len(dx) >= 2:
                match['dx_over'], match['dx_under'] = dx[0], dx[1]
                match['total_line'] = total
        matches[match_id] = match
    return matches


def merge_schedule_pages(sf_html, dx_html):
    sf, dx = parse_page(sf_html, 'sf'), parse_page(dx_html, 'dx')
    merged = []
    for match_id in sorted(set(sf) | set(dx), key=lambda value: int(value) if value.isdigit() else value):
        base = dict(sf.get(match_id) or dx.get(match_id))
        extra = dx.get(match_id)
        if extra:
            for key in ('dx_over', 'dx_under', 'total_line'):
                base[key] = extra.get(key)
        merged.append(base)
    return merged


def select_upcoming(matches, date, now):
    same_day = [match for match in matches if match.get('date') == date]
    candidates = same_day or matches
    upcoming = []
    for match in candidates:
        if match.get('status') == 'finished':
            continue
        try:
            kickoff = datetime.strptime(f"{match['date']} {match['time']}",
                                        '%Y-%m-%d %H:%M')
        except (KeyError, TypeError, ValueError):
            kickoff = None
        if kickoff is None or kickoff > now:
            upcoming.append(match)
    return upcoming


class ZgzcwScheduleFetcher:
    def __init__(self, transport, now_fn=None):
        self._transport = transport
        self._now = now_fn or datetime.now

    def fetch(self, date=None):
        now = self._now()
        date = date or now.strftime('%Y-%m-%d')
        suffix = '&issue=' + date
        try:
            sf_html = self._transport(ZGZCW_SF_URL + suffix)
            dx_html = self._transport(ZGZCW_DX_URL + suffix)
        except Exception as exc:
            log.warning('中国足彩网篮球赛程抓取失败: %s', exc)
            return []
        matches = merge_schedule_pages(sf_html or '', dx_html or '')
        live = select_upcoming(matches, date, now)
        log.info('中国足彩网篮球获取到 %d 场未开赛比赛', len(live))
        return live


class MarketBundleFetcher:
    """中国足彩网篮球页不公开逐公司历史，明确返回 unavailable。

    走势由本项目自己的低频快照补齐，不能把当前赔率伪造成历史序列。
    """

    def __init__(self, transport=None, max_workers=1):
        self._transport = transport

    def fetch_many(self, match_ids):
        return {str(match_id): self.fetch_one(match_id)
                for match_id in match_ids if match_id}

    @staticmethod
    def fetch_one(match_id):
        return {'match_id': str(match_id), 'ml': {'available': False},
                'ah': {'available': False}, 'ou': {'available': False}}
