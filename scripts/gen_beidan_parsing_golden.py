"""从 beidan 的抓取模块生成「页面解析」这半批的黄金快照。

覆盖 4-T6 前半的六个解析器：三张管道分隔的赔率表（比分/总进球/半全场）
与三份走势历史（亚盘/大小球/比分盘）。

**网络用打桩替掉，打在被测路径之外**：`fetch` 与 `fetch_okooo` 换成直接返回
语料。抓取本身不是这一批要迁的东西。

语料的来源分三类，**看得见地分开**：

1. **真实页面**：`tests/fixtures/basketball/okooo_ah.html.gz` 是 2026-08-26
   抓的真页面。**它是篮球的详情页**，那张表头含「盘口」的表其实是博彩公司
   列表而不是时间序列，解析出来是 `time='序'`、`handicap='公司名'` 这种东西
   ——语料留着不是因为它有意义，而是因为它是**真的 HTML**：
   标签嵌套、空白、编码都是真实的，能照出「在真页面上会发生什么」。
2. **按解析代码构造**：其余 HTML 与管道表按 `fetching.py` / `schedules.py`
   里实际用的正则与下标铺出来，**不是按字段名猜的**。
   每一条都对着一个具体分支：表头关键字、列数不足、时间过长、
   赔率非数字、script 回退、多个 URL 依次尝试。
3. **抓不到的那两类**：`ssq_match_info.jsp` 线上实测 **HTTP 404**
   （2026-08-28），okooo 详情页对临时进程一律 WAF 拦截。所以这两类没有真实
   捕获，语料只能构造——**这是这一批已知的覆盖缺口**，写在这里免得被忘掉。

用法：
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 \\
        scripts/gen_beidan_parsing_golden.py /tmp/beidan_parsing_old.json
"""
import gzip
import json
import pathlib
import sys
from unittest import mock

sys.path.insert(0, '.')

from tests.domain.golden import as_comparable

import src.beidan.fetching as fetching_mod
import src.beidan.schedules as schedules_mod

FIXTURES = pathlib.Path('tests/fixtures')

# ── 管道分隔的赔率表 ─────────────────────────────────────────────
# 形如 `比赛号|价|价|...`，`#` 开头是注释行。
# 三个解析器的**严格度不同**，语料要能把差别照出来：
#   比分   —— 变长的「比分|价」对，坏价只跳过那一对，整行还留着
#   总进球 —— 定长 8 列（第 8 列可缺），坏价**整行丢弃**
#   半全场 —— 定长 9 列，坏价**整行丢弃**
BIFEN_TABLES = {
    'normal': '\n'.join([
        '# 注释行会被跳过',
        '1320957|1-0|8.00|1-1|7.50|2-1|9.00|0-1|11.00',
        '1320958|0-0|9.50|1-0|7.20',
        '',
        '1320959|2-2|15.00',
    ]),
    # 坏价只丢那一对：这一行仍然进结果，只是少一个比分
    'one_bad_price': '1320957|1-0|8.00|1-1|x|2-1|9.00',
    # 全是坏价 → `odds` 为空 → **整行不进结果**
    'all_bad_prices': '1320957|1-0|x|1-1|y',
    # 奇数段：最后一个比分没有配对的价，直接被忽略
    'dangling_score': '1320957|1-0|8.00|1-1',
    'only_id': '1320957',
    'empty': '',
    'comments_only': '# a\n# b',
    'blank_lines': '\n\n   \n',
}

ZJQ_TABLES = {
    'normal': '\n'.join([
        '1320957|11.00|5.60|3.90|4.30|6.50|11.00|21.00|26.00',
        '1320958|12.00|6.00|4.00|4.50|7.00|12.00|22.00|27.00',
    ]),
    # 恰好 8 段 → 有 '0'~'6'，`7+` 为 None
    'eight_parts': '1320957|11.00|5.60|3.90|4.30|6.50|11.00|21.00',
    # 差一段 → 整行跳过（门槛是 `len(parts) < 8`）
    'seven_parts': '1320957|11.00|5.60|3.90|4.30|6.50|11.00',
    # 空字段读成 None，不是 0
    'empty_fields': '1320957|11.00||3.90|4.30|6.50|11.00|21.00|26.00',
    # 一个坏价 → **整行丢弃**（与比分那张表不同）
    'one_bad_price': '1320957|11.00|x|3.90|4.30|6.50|11.00|21.00|26.00',
    'multi_line_one_bad': '\n'.join([
        '1320957|11.00|5.60|3.90|4.30|6.50|11.00|21.00|26.00',
        '1320958|11.00|x|3.90|4.30|6.50|11.00|21.00|26.00',
    ]),
    'empty': '',
}

BQC_TABLES = {
    'normal': '1320957|3.20|15.00|41.00|7.50|6.10|12.00|51.00|17.00|4.30',
    # 恰好 10 段是门槛，9 段跳过
    'nine_parts': '1320957|3.20|15.00|41.00|7.50|6.10|12.00|51.00|17.00',
    'empty_fields': '1320957|3.20||41.00|7.50|6.10|12.00|51.00|17.00|4.30',
    'one_bad_price': '1320957|3.20|x|41.00|7.50|6.10|12.00|51.00|17.00|4.30',
    'extra_parts': '1320957|3.20|15.00|41.00|7.50|6.10|12.00|51.00|17.00|4.30|9.99',
    'empty': '',
}


# ── 走势历史的 HTML ──────────────────────────────────────────────
def _table(header_cells, rows):
    head = '<tr>' + ''.join(f'<th>{c}</th>' for c in header_cells) + '</tr>'
    body = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>'
                   for row in rows)
    return f'<table>{head}{body}</table>'


def _page(*parts):
    return '<html><body>' + ''.join(parts) + '</body></html>'


def _script(text):
    return f'<script type="text/javascript">{text}</script>'


# 表头必须命中关键字，否则整张表跳过——三个解析器的关键字不同
ASIAN_ROWS = [('09:00', '-0.5', '0.95', '0.90'),
              ('10:00', '-0.75', '0.88', '0.98'),
              ('11:00', '-0.75', '0.92', '0.94')]
GOALS_ROWS = [('09:00', '2.5', '0.95', '0.90'),
              ('10:00', '2.5', '0.80', '1.05')]
CS_ROWS = [('09:00', '1-0', '8.00'),
           ('10:00', '1-1', '7.50')]

ASIAN_PAGES = {
    'table_asian': _page(_table(['时间', '亚盘', '主', '客'], ASIAN_ROWS)),
    'table_handicap_keyword': _page(_table(['时间', '让球', '主', '客'], ASIAN_ROWS)),
    'table_panluo_keyword': _page(_table(['时间', '盘口', '主', '客'], ASIAN_ROWS)),
    # 表头不含关键字 → 整张表跳过
    'table_wrong_header': _page(_table(['时间', '欧赔', '主', '客'], ASIAN_ROWS)),
    # 只有表头没有数据行 → `len(rows) < 2` 跳过
    'header_only': _page(_table(['时间', '亚盘', '主', '客'], [])),
    # 列数不足 4
    'three_cells': _page(_table(['时间', '亚盘', '主'],
                                [('09:00', '-0.5', '0.95')])),
    # 时间为空 / 超过 8 个字符 → 跳过该行
    'bad_time': _page(_table(['时间', '亚盘', '主', '客'],
                             [('', '-0.5', '0.95', '0.90'),
                              ('2026-08-28 09:00:00', '-0.5', '0.95', '0.90'),
                              ('09:00', '-0.5', '0.95', '0.90')])),
    # 让球值超过 20 个字符 → 跳过该行。**第一版写了 12 个字，够不到门槛**，
    # 两行都留下了，这条用例看起来在测长度、实际什么也没测（判据 28）
    'long_handicap': _page(_table(['时间', '亚盘', '主', '客'],
                                  [('09:00', '主队让半球再加两个球外加一个角球和一张黄牌', '0.95', '0.90'),
                                   ('10:00', '-0.5', '0.95', '0.90')])),
    'handicap_exactly_twenty': _page(_table(['时间', '亚盘', '主', '客'],
                                            [('09:00', '主' * 20, '0.95', '0.90')])),
    # 非数字的水位读成 None，而不是丢掉整行
    'non_numeric_odds': _page(_table(['时间', '亚盘', '主', '客'],
                                     [('09:00', '-0.5', '封', '-'),
                                      ('10:00', '-0.5', '0.95', '0.90')])),
    # 单元格里嵌着标签，要先剥干净
    'nested_tags': _page(_table(['时间', '亚盘', '主', '客'],
                                [('<b>09:00</b>', '<i>-0.5</i>',
                                  '<span>0.95</span>', '0.90')])),
    # `display:none` 会被先替换掉——藏起来的行同样要解析
    'hidden_rows': _page(
        '<table style="display:none">'
        + _table(['时间', '亚盘', '主', '客'], ASIAN_ROWS)[len('<table>'):]),
    # 表格路走不通时回退到 script。**取值形式必须是「裸的 HH:MM 打头」**：
    # 那条正则的第一组前面没有可选引号，所以 `["09:00", ...]` 这种 JSON 数组
    # 一个也匹配不上——第一版语料写的正是 JSON 形式，于是这条分支从没走到。
    # 下面一条是能匹配的形式，再下面一条把匹配不上的形式也钉住。
    'script_only': _page(_script(
        '亚盘走势 var data = ' + 'x' * 60
        + ' 09:00, -0.5, 0.95, 0.90 ; 10:00, -0.75, 0.88, 0.98 ;')),
    'script_json_style': _page(_script(
        '亚盘走势 var data = [' + 'x' * 60
        + '["09:00", -0.5, 0.95, 0.90],["10:00", -0.75, 0.88, 0.98]];')),
    # 值带引号是认的——不认的只有时间那一组带引号
    'script_quoted_value': _page(_script(
        '亚盘 ' + 'x' * 60 + ' 09:00,"-0.5",0.95,0.90')),
    'script_ah_keyword': _page(_script(
        'AH series ' + 'x' * 60 + ' 09:00, -0.5, 0.95, 0.90')),
    'script_too_short': _page(_script('09:00, -0.5, 0.95, 0.90')),
    'script_wrong_keyword': _page(_script(
        '欧赔 ' + 'x' * 60 + ' 09:00, -0.5, 0.95, 0.90')),
    # 表格有数据时 script 不该再跑
    'table_wins_over_script': _page(
        _table(['时间', '亚盘', '主', '客'], ASIAN_ROWS),
        _script('亚盘 ' + 'x' * 60 + ' 23:59, -9.5, 1.11, 2.22')),
    'no_tables': _page('<div>什么也没有</div>'),
    'empty': '',
}

GOALS_PAGES = {
    'table_goals': _page(_table(['时间', '进球', '大', '小'], GOALS_ROWS)),
    'table_daxiao_keyword': _page(_table(['时间', '大小球', '大', '小'], GOALS_ROWS)),
    'table_wrong_header': _page(_table(['时间', '亚盘', '大', '小'], GOALS_ROWS)),
    # 只有三列 —— 大小球允许，小球水位读成 None（比亚盘松一列）
    'three_cells': _page(_table(['时间', '进球', '大'],
                                [('09:00', '2.5', '0.95')])),
    'bad_time': _page(_table(['时间', '进球', '大', '小'],
                             [('', '2.5', '0.95', '0.90'),
                              ('09:00', '2.5', '0.95', '0.90')])),
    # **没有让球值那道长度门槛**——同样的输入在亚盘那边会被跳过
    'long_line': _page(_table(['时间', '进球', '大', '小'],
                              [('09:00', '大小球二点五球球球球球球球球球', '0.95', '0.90')])),
    'non_numeric_odds': _page(_table(['时间', '进球', '大', '小'],
                                     [('09:00', '2.5', '封', '-')])),
    'script_only': _page(_script(
        '进球走势 ' + 'x' * 60 + ' 09:00, 2.5, 0.95, 0.90 ; 10:00, 3, 0.88, 0.98')),
    'script_json_style': _page(_script(
        '进球走势 ' + 'x' * 60 + ' ["09:00", 2.5, 0.95, 0.90]')),
    'script_total_keyword': _page(_script(
        'total goals ' + 'x' * 60 + ' 09:00, 2.5, 0.95, 0.90')),
    'no_tables': _page('<div>空</div>'),
    'empty': '',
}

CS_PAGES = {
    'table_cs': _page(_table(['时间', '比分', '赔率'], CS_ROWS)),
    'table_cs_english_header': _page(_table(['time', 'CS', 'odds'], CS_ROWS)),
    'table_wrong_header': _page(_table(['时间', '亚盘', '赔率'], CS_ROWS)),
    # 比分里没有 `-` → 跳过该行
    'bad_score': _page(_table(['时间', '比分', '赔率'],
                              [('09:00', '10', '8.00'),
                               ('10:00', '', '8.00'),
                               ('11:00', '1-0', '8.00')])),
    'bad_time': _page(_table(['时间', '比分', '赔率'],
                             [('', '1-0', '8.00'), ('09:00', '1-0', '8.00')])),
    'non_numeric_odds': _page(_table(['时间', '比分', '赔率'],
                                     [('09:00', '1-0', '封')])),
    'script_only': _page(_script(
        '比分走势 ' + 'x' * 60 + ' 09:00, 1-0, 8.00 ; 10:00, 1-1, 7.50')),
    'script_json_style': _page(_script(
        '比分走势 ' + 'x' * 60 + ' ["09:00", "1-0", 8.00]')),
    'script_score_keyword': _page(_script(
        'score series ' + 'x' * 60 + ' 09:00, 1-0, 8.00')),
    'no_tables': _page('<div>空</div>'),
    'empty': '',
}

# 真实页面：2026-08-26 抓的 okooo 篮球亚盘页，表头含「盘口」、两行可解析。
# **这是这一批唯一一份真实捕获的走势页**（足球详情页对临时进程一律被 WAF 拦）。
REAL_OKOOO_AH = gzip.open(FIXTURES / 'basketball/okooo_ah.html.gz',
                          'rt', encoding='utf-8').read()

# 四个候选 URL 依次尝试：前几个空、某一个有数据
URL_SEQUENCES = {
    'first_hits': [ASIAN_PAGES['table_asian'], '', '', ''],
    'third_hits': ['', '', ASIAN_PAGES['table_asian'], ''],
    'all_empty': ['', '', '', ''],
    'all_none': [None, None, None, None],
    # 前一个页面能解析出东西就不再往后走
    'first_wins': [ASIAN_PAGES['table_asian'], ASIAN_PAGES['script_only'], '', ''],
    # 页面有内容但解析不出记录 → 继续试下一个
    'unparsable_then_hit': [ASIAN_PAGES['no_tables'],
                            ASIAN_PAGES['table_asian'], '', ''],
}


def _fetch_table(content):
    return mock.patch.object(schedules_mod, 'fetch', return_value=content)


def _fetch_pages(pages):
    return mock.patch.object(fetching_mod, 'fetch_okooo', side_effect=list(pages))


def entries():
    for name, content in BIFEN_TABLES.items():
        with _fetch_table(content):
            yield f'bifen_table:{name}', schedules_mod.fetch_beidan_bifen('2026-08-28')
    for name, content in ZJQ_TABLES.items():
        with _fetch_table(content):
            yield f'zjq_table:{name}', schedules_mod.fetch_beidan_zjq('2026-08-28')
    for name, content in BQC_TABLES.items():
        with _fetch_table(content):
            yield f'bqc_table:{name}', schedules_mod.fetch_beidan_bqc('2026-08-28')
    # 抓取返回空 → 三个都给空字典
    for label, fn in (('bifen', schedules_mod.fetch_beidan_bifen),
                      ('zjq', schedules_mod.fetch_beidan_zjq),
                      ('bqc', schedules_mod.fetch_beidan_bqc)):
        for empty in (None, ''):
            with _fetch_table(empty):
                yield f'{label}_table:fetch_{empty!r}', fn('2026-08-28')

    for name, page in ASIAN_PAGES.items():
        with _fetch_pages([page] * 4):
            yield (f'asian:{name}',
                   fetching_mod.fetch_okooo_asian_history('1320957'))
    for name, page in GOALS_PAGES.items():
        with _fetch_pages([page] * 4):
            yield (f'goals:{name}',
                   fetching_mod.fetch_okooo_goals_history('1320957'))
    for name, page in CS_PAGES.items():
        with _fetch_pages([page] * 4):
            yield f'cs:{name}', fetching_mod.fetch_okooo_cs_history('1320957')

    with _fetch_pages([REAL_OKOOO_AH] * 4):
        yield 'asian:real_okooo_page', fetching_mod.fetch_okooo_asian_history('1320957')
    with _fetch_pages([REAL_OKOOO_AH] * 4):
        yield 'goals:real_okooo_page', fetching_mod.fetch_okooo_goals_history('1320957')
    with _fetch_pages([REAL_OKOOO_AH] * 4):
        yield 'cs:real_okooo_page', fetching_mod.fetch_okooo_cs_history('1320957')

    for name, pages in URL_SEQUENCES.items():
        with _fetch_pages(pages):
            yield (f'asian_urls:{name}',
                   fetching_mod.fetch_okooo_asian_history('1320957'))


def main(out_path):
    golden = {key: as_comparable(value) for key, value in entries()}
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(golden, fh, ensure_ascii=False, sort_keys=True, indent=1)
    print(f'共 {len(golden)} 条 → {out_path}')


if __name__ == '__main__':
    main(sys.argv[1])
