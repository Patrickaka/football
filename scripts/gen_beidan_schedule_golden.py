"""从 beidan 的抓取模块生成「赛程解析」这半批的黄金快照。

只覆盖 500.com 的即时赔率页（链接式）。它**读时钟**——比赛是否已结束由
`datetime.now()` 决定，所以语料要把时钟钉死在几个不同的时刻上，而不是让它
跟着今天跑（判据 24：写死一个「未来的日期」的用例是一颗定时炸弹）。

语料是 `tests/fixtures/index_jczq.html`，500.com 的真实快照（含 4 场比赛、
16 处联赛块），正是 `fetch_beidan_schedule` 要解析的东西。

**中国足彩网单场页那 19 条语料已经删掉**：它们编码的是旧解析器的边界，
换到 `tr_<id>` 行 / `wh-N` 列 / `tn` 属性的结构化解析后一律得到空列表。
新解析器的覆盖在
`tests/domain/sports/beidan/test_schedule_parsing.py::ZgzcwScheduleTests`。

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

# ── 单场页的语料已随旧解析器一起移除 ───────────────────────────────
# 原先这里有 19 条澳客单场页语料。zgzcw 的单场页换成了按 `tr_<id>` 行、
# `wh-N` 列、`tn` 属性取值的结构化解析，旧语料喂进去只会得到空列表。
# 新解析器的覆盖在
# `tests/domain/sports/beidan/test_schedule_parsing.py::ZgzcwScheduleTests`。


def entries():
    for clock_name, moment in CLOCKS.items():
        with mock.patch.object(fetching_mod, 'datetime', _frozen(moment)):
            for page_name, html in FIVE_HUNDRED_PAGES.items():
                with mock.patch.object(fetching_mod, 'fetch', return_value=html):
                    yield (f'500:{page_name}:{clock_name}',
                           fetching_mod.fetch_beidan_schedule(
                               '2026-08-28', source='jczq'))


def main(out_path):
    golden = {key: as_comparable(value) for key, value in entries()}
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(golden, fh, ensure_ascii=False, sort_keys=True, indent=1)
    print(f'共 {len(golden)} 条 → {out_path}')


if __name__ == '__main__':
    main(sys.argv[1])
