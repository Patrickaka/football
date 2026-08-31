"""从 beidan 的抓取模块生成「页面解析」这半批的黄金快照。

覆盖三张管道分隔的赔率表：比分 / 总进球 / 半全场。

**网络用打桩替掉，打在被测路径之外**：`fetch` 换成直接返回语料。
抓取本身不是这一批要迁的东西。

语料按解析代码构造——按 `schedules.py` 里实际用的正则与下标铺出来，
**不是按字段名猜的**。每一条都对着一个具体分支：列数不足、赔率非数字、
注释行、空表。三个解析器的严格度不同，语料要能把差别照出来。

**走势历史（亚盘/大小球/比分盘）已经不在这一批里**：那 53 条语料编码的是
旧解析器的边界，换到 zgzcw 的结构化历史页后一律解析为空。详见下方注释。

用法：
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 \\
        scripts/gen_beidan_parsing_golden.py /tmp/beidan_parsing_old.json
"""
import json
import sys
from unittest import mock

sys.path.insert(0, '.')

from tests.domain.golden import as_comparable

import src.beidan.fetching as fetching_mod
import src.beidan.schedules as schedules_mod

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


# ── 走势历史的语料已随旧解析器一起移除 ─────────────────────────────
# 原先这里有 53 条 asian / goals / cs 语料，编码的是**旧解析器**的边界：
# 表头关键字、列数不足、时间格式、script 回退……zgzcw 的历史页换成了按
# `chupan-w-0` / `cid` / `firsttime` 取值的结构化解析，上面每一条喂进去
# 都只会得到空历史——留着不是覆盖，是 53 条恒真断言。
# 新解析器的覆盖在 `tests/domain/test_zgzcw_sources.py` 与
# `tests/domain/sports/beidan/test_parsing.py::AdapterTests`。


def _fetch_table(content):
    return mock.patch.object(schedules_mod, 'fetch', return_value=content)


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


def main(out_path):
    golden = {key: as_comparable(value) for key, value in entries()}
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(golden, fh, ensure_ascii=False, sort_keys=True, indent=1)
    print(f'共 {len(golden)} 条 → {out_path}')


if __name__ == '__main__':
    main(sys.argv[1])
