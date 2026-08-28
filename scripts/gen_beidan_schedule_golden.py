"""从 beidan 的抓取模块生成「赛程解析」这半批的黄金快照。

两个解析器：okooo 的单场页（表格式）与 500.com 的即时赔率页（链接式）。
后者**读时钟**——比赛是否已结束由 `datetime.now()` 决定，所以语料要把
时钟钉死在几个不同的时刻上，而不是让它跟着今天跑（判据 24：写死一个
「未来的日期」的用例是一颗定时炸弹）。

语料来源：

1. **真实页面**：`tests/fixtures/index_jczq.html` 是 500.com 的真实快照
   （含 4 场比赛、16 处联赛块），正是 `fetch_beidan_schedule` 要解析的东西。
2. **按解析代码构造**：okooo 那张表按 `fetch_okooo_schedule` 里实际用的
   选择器铺出来——`<span class="xh"><i>`、`homenameobj`、`handicapobj`、
   `<em>` 里的赔率、`mTime=` 属性。线上抓不到真页面（WAF），这是缺口。

用法：
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 \\
        scripts/gen_beidan_schedule_golden.py /tmp/beidan_schedule_old.json
"""
import datetime as _datetime
import json
import pathlib
import sys
from unittest import mock

sys.path.insert(0, '.')

from tests.domain.golden import as_comparable

import src.beidan.fetching as fetching_mod

REAL_500_PAGE = pathlib.Path('tests/fixtures/index_jczq.html').read_text(
    encoding='utf-8')

# 时钟钉死在四个时刻上：远早于全部比赛、当天、赛后三小时内、远晚于全部。
# 三小时是「已结束」的门槛，一小时是「进行中」的门槛。
CLOCKS = {
    'long_before': _datetime.datetime(2020, 1, 1, 0, 0),
    'match_day_morning': _datetime.datetime(2026, 8, 28, 8, 0),
    'match_day_evening': _datetime.datetime(2026, 8, 28, 19, 0),
    'long_after': _datetime.datetime(2030, 1, 1, 0, 0),
}


def _frozen(moment):
    class Frozen(_datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return moment
    return Frozen


# ── 500.com 的即时赔率页 ────────────────────────────────────────
def _link(match_id, home, away, suffix='数据'):
    return f'<a href="shuju-{match_id}.shtml" title="{home}VS{away}{suffix}"></a>'


def _time_cell(match_id, when):
    return f'<td rowspan="2">{when}</td><a href="shuju-{match_id}.shtml"></a>'


def _league_block(name, *links):
    return (f'<a href="//liansai.500.com/zuqiu-100/">{name}</a>'
            + ''.join(links))


FIVE_HUNDRED_PAGES = {
    'real_page': REAL_500_PAGE,
    'empty': '',
    'no_links': '<html><body>什么也没有</body></html>',
    # 一场比赛、有时间 → 状态由时钟决定
    'one_match': _link('999001', '甲队', '乙队')
                 + _time_cell('999001', '08-28 19:30'),
    # **没有任何时间模式**：`match['time']` 保持空字符串，于是后面那条
    # 「按日期判断是否已结束」的分支会拿 datetime 与 date 比较 —— TypeError。
    # 外层只 catch 了 ValueError，异常冒到最外面，**整份赛程变成空列表**。
    'match_without_time': _link('999001', '甲队', '乙队'),
    # 标题后缀要剥干净：页面上同一场比赛有好几个入口
    'name_suffixes': ''.join(
        _link(f'99900{i}', f'甲队{suffix}', f'乙队{suffix}', suffix)
        for i, suffix in enumerate(('百家', '欧赔', '亚赔', '亚盘', '数据',
                                    '盘口', '指数', '对比', '分析'), start=1))
        + ''.join(_time_cell(f'99900{i}', '08-28 19:30') for i in range(1, 10)),
    # 时间里的日期与请求的日期不同 → 记录跟着挪到那一天
    'date_rollover': _link('999001', '甲队', '乙队')
                     + _time_cell('999001', '08-29 02:00'),
    # 第二种时间模式：时间跟在链接后面而不是前面
    'time_after_link': _link('999001', '甲队', '乙队')
                       + '<a href="shuju-999001.shtml"></a>08-28 19:30',
    # 时间格式认不出来 → 原样塞进 `time`，随后 strptime 失败
    'unparsable_time': _link('999001', '甲队', '乙队')
                       + '<td rowspan="2">稍后</td><a href="shuju-999001.shtml"></a>',
    # 联赛块：块与块之间的比赛各归各的联赛
    'leagues': _league_block('英超', _link('999001', '甲队', '乙队'))
               + _league_block('西甲', _link('999002', '丙队', '丁队'))
               + _time_cell('999001', '08-28 19:30')
               + _time_cell('999002', '08-28 21:30'),
    # 场次号
    'with_num': _link('999001', '甲队', '乙队')
                + '<input value="999001" /> 周五001'
                + _time_cell('999001', '08-28 19:30'),
    # 队名为空 → 整条跳过
    'empty_team_name': '<a href="shuju-999001.shtml" title="VS乙队数据"></a>',
}

# ── okooo 的单场页 ──────────────────────────────────────────────
def _okooo_row(num='001', league='英超', match_id='1320957',
               time_cell='08-28 19:30', mtime=None, score='-',
               home='安山小绿人', away='大邱FC', handicap='(-1)',
               odds=('1.80', '3.60', '4.20', '2.20', '3.40', '3.10')):
    first = (f'<span class="xh"><i>{num}</i></span>'
             f'<a href="//www.okooo.com/soccer/league/100/">{league}</a>')
    time_attr = f' mTime="{mtime}"' if mtime else ''
    second = f'<span{time_attr}>{time_cell}</span>'
    teams = (f'<span class="homenameobj" title="{home}">{home}</span>'
             f'<span class="awaynameobj" title="{away}">{away}</span>')
    if handicap is not None:
        teams += f'<span class="handicapobj">{handicap}</span>'
    teams += ''.join(f'<em>{value}</em>' for value in odds)
    cells = [first, second, teams, '', '', score]
    return ('<tr>' + f'<a href="/soccer/match/{match_id}/"></a>'
            + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')


def _okooo_page(*rows, tables=2):
    filler = '<table><tr><td>目录</td></tr></table>'
    main = '<table>' + ''.join(rows) + '</table>'
    return '<html><body>' + filler * (tables - 1) + main + '</body></html>'


OKOOO_PAGES = {
    'empty': '',
    # 表格不足两个 → 走备用数据源
    'single_table': '<html><table><tr><td>只有一个</td></tr></table></html>',
    'normal': _okooo_page(_okooo_row()),
    'two_matches': _okooo_page(_okooo_row(),
                               _okooo_row(num='002', match_id='1320958',
                                          home='丙队', away='丁队')),
    # 比分不是 `-` → 已结束 → 被过滤掉 → 触发「没有未完结比赛」的回退
    'finished': _okooo_page(_okooo_row(score='2:1')),
    'mixed_status': _okooo_page(_okooo_row(score='2:1'),
                                _okooo_row(num='002', match_id='1320958')),
    # 少于六个 td 的行：找日期，不找比赛
    'date_row': _okooo_page('<tr><td>2026-08-29</td></tr>', _okooo_row()),
    # `mTime` 属性优先于单元格文本
    'mtime_wins': _okooo_page(_okooo_row(time_cell='稍后', mtime='08-29 02:00')),
    'mtime_unparsable': _okooo_page(_okooo_row(time_cell='x', mtime='稍后')),
    'cell_time_unparsable': _okooo_page(_okooo_row(time_cell='稍后')),
    # 赔率个数：让球那三个要**六个都在**才算数
    'three_odds': _okooo_page(_okooo_row(odds=('1.80', '3.60', '4.20'))),
    'five_odds': _okooo_page(_okooo_row(
        odds=('1.80', '3.60', '4.20', '2.20', '3.40'))),
    'no_odds': _okooo_page(_okooo_row(odds=())),
    # 让球赔率里有一个不到 1.0 → `rqspf_odds` 整组不给
    'rqspf_below_one': _okooo_page(_okooo_row(
        odds=('1.80', '3.60', '4.20', '2.20', '0.95', '3.10'))),
    'no_handicap': _okooo_page(_okooo_row(handicap=None)),
    'no_league': _okooo_page(_okooo_row(league=None)) if False else _okooo_page(
        _okooo_row().replace('<a href="//www.okooo.com/soccer/league/100/">英超</a>', '')),
    'no_num': _okooo_page(_okooo_row().replace(
        '<span class="xh"><i>001</i></span>', '')),
    # 没有 match_id 链接 → 用日期加场次号拼一个
    'no_match_id': _okooo_page(_okooo_row().replace(
        '<a href="/soccer/match/1320957/"></a>', '')),
    # 缺主队或客队 → 整行跳过
    'missing_home': _okooo_page(_okooo_row().replace('homenameobj', 'xx')),
}


def entries():
    for clock_name, moment in CLOCKS.items():
        with mock.patch.object(fetching_mod, 'datetime', _frozen(moment)):
            for page_name, html in FIVE_HUNDRED_PAGES.items():
                with mock.patch.object(fetching_mod, 'fetch', return_value=html):
                    yield (f'500:{page_name}:{clock_name}',
                           fetching_mod.fetch_beidan_schedule(
                               '2026-08-28', source='jczq'))

    # okooo 那条不读时钟，但取不到东西时会回退到 500——把回退也打桩掉，
    # 好让这一组只反映 okooo 自己的解析结果
    for page_name, html in OKOOO_PAGES.items():
        with mock.patch.object(fetching_mod, 'fetch_okooo', return_value=html):
            with mock.patch.object(fetching_mod, 'fetch_beidan_schedule',
                                   return_value=[{'id': 'FALLBACK'}]):
                yield (f'okooo:{page_name}',
                       fetching_mod.fetch_okooo_schedule('2026-08-28'))


def main(out_path):
    golden = {key: as_comparable(value) for key, value in entries()}
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(golden, fh, ensure_ascii=False, sort_keys=True, indent=1)
    print(f'共 {len(golden)} 条 → {out_path}')


if __name__ == '__main__':
    main(sys.argv[1])
