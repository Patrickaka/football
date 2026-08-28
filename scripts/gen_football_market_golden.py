# -*- coding: utf-8 -*-
"""生成盘口四件套（market_db / similar_market / market_clustering / steam_move）的黄金语料。

覆盖清单**对着领域层导出逐个核过**。唯一一个恒空的是
`half_full_probs_from_records`——它的 docstring 明写「已废弃，不再生成伪半场
数据」，空字典就是正确行为。
"""
import datetime
import itertools

from src.domain.sports.football import market_matching as mm, steam

HANDICAPS = [-3.0, -1.75, -1.0, -0.75, -0.27, -0.25, 0.0, 0.13, 0.25, 0.5, 0.75, 1.0, 2.5, 3.5]
TOTALS = [0.5, 1.25, 1.75, 2.0, 2.25, 2.5, 2.63, 2.75, 3.0, 3.25, 4.5]
RAW_VALUES = ['0', '-0.5', '1.5', '受让半球', '平手/半球', '', 'abc', None, '2.5']
ODD_PAIRS = [(0.9, 0.9), (1.05, 0.75), (0.75, 1.05), (1.9, 1.9), (0.5, 1.5)]
DATES = ['2026-08-28', '28/08/2026', '2026/08/28', '', 'bad', None, '08-28']
LEAGUES = ('英超', '西甲', '中超', '某某丙级', '')

ASIAN = {'open_handicap': 0.5, 'handicap': 0.25,
         'open_time': '2026-08-28 10:00:00', 'close_time': '2026-08-28 19:30:00',
         'open_water': {'home': 0.9, 'away': 0.9},
         'close_water': {'home': 1.05, 'away': 0.75}}
TOTAL = {'open_line': 2.5, 'close_line': 2.75,
         'open_time': '2026-08-28 10:00:00', 'close_time': '2026-08-28 19:30:00',
         'open_water': {'over': 0.9, 'under': 0.9},
         'close_water': {'over': 1.02, 'under': 0.78}}
MATCH_TIME = '2026-08-28 20:00:00'
RECORD_DATA = {'asian': 0.5, 'asian_odds_home': 0.0, 'asian_odds_away': 0.0,
               'total': 2.5, 'total_over': 0.0, 'total_under': 0.0,
               'euro_home': 2.0, 'euro_draw': 3.4, 'euro_away': 3.8,
               'result': 'H', 'goals_home': 2, 'goals_away': 1,
               'date': '2026-08-28', 'league': '英超',
               'home_team': 'A', 'away_team': 'B', 'season': '2026-27'}
# **键名要用 MatchRecord 真正读的那一套**：`euro_home` 不是 `home_odds`、
# `result` 不是 `ftr`、赛季是 `'2026-27'` 不是 `'2526'`。第一版全喂错了，
# 于是 `filter_record` 对任何输入都返回 False，三道过滤一条都没走到（判据 23）。
NOW = datetime.datetime(2026, 8, 28, 12, 0, 0)


def entries():
    for h in HANDICAPS:
        yield f'norm_asian:{h}', mm.normalize_asian(h)
        yield f'internal_asian:{h}', mm.to_internal_asian(h)
        yield f'round_hcap:{h}', mm.round_to_standard(h, False)
    for t in TOTALS:
        yield f'norm_ou:{t}', mm.normalize_ou(t)
        yield f'round_total:{t}', mm.round_to_standard(t, True)
    for v in RAW_VALUES:
        yield f'parse_hcap:{v!r}', mm.parse_handicap_value(v)
        yield f'parse_odds:{v!r}', mm.parse_odds_value(v)
        yield f'parse_float:{v!r}', mm.parse_float(v)
    for home, away in ODD_PAIRS:
        yield f'hcap_from_odds:{home}/{away}', mm.normalize_handicap_from_odds(home, away)
        yield f'infer_hcap:{home}/{away}', mm.infer_handicap(home, away)
    for home, away in itertools.product((1.5, 2.0, 3.5), (0.8, 1.2, 2.5)):
        yield f'estimate_total:{home}/{away}', mm.estimate_total(home, away)
    for date in DATES:
        # `_parse_record_date` 返回 datetime，JSON 存不下——存 ISO 串
        parsed = mm._parse_record_date(date)
        yield f'record_date:{date!r}', parsed.isoformat() if parsed else parsed
    from src.football.similar_market import MatchRecord as _MR
    for days in (0, 10, 90, 400, 1200):
        # **收的是 MatchRecord 不是日期**——第一版喂了 date，每条都落到默认 0.7
        stamp = (datetime.date(2026, 8, 28) - datetime.timedelta(days=days)).isoformat()
        record = _MR(dict(RECORD_DATA, date=stamp))
        yield f'recency:{days}', mm._recency_weight(record, NOW)
    for asian in (-2.5, -0.5, 0.0, 0.25, 1.0, 2.5):
        yield f'dynamic_k:{asian}', mm.dynamic_k(asian)
    for league in LEAGUES:
        yield f'league_tier:{league!r}', mm.league_tier(league)
    for hcap, ou in itertools.product((-0.5, 0.0, 0.75), (2.25, 2.5, 3.0)):
        yield f'score_key:{hcap}/{ou}', mm.market_score_key(hcap, ou)
        yield f'half_full:{hcap}/{ou}', mm.half_full_probs_from_records(hcap, ou)
    for odds in ({'over': 1.9, 'under': 1.9}, {'over': 1.6, 'under': 2.3}, {}):
        yield f'implied_total:{sorted(odds.items())}', mm.implied_total_from_odds(2.5, odds)

    from src.football.similar_market import MatchRecord
    for i, kw in enumerate(({}, {'league': '友谊赛'}, {'season': '2009-10'},
                            {'euro_home': 1.001}, {'euro_home': 500.0},
                            {'result': ''}, {'asian': -1.5},
                            {'asian_odds_home': 0.9, 'asian_odds_away': 0.9})):
        record = MatchRecord(dict(RECORD_DATA, **kw))
        yield f'features:{i}', mm.extract_features(record)
        for flag in (True, False):
            yield f'filter:{i}/{flag}', mm.filter_record(record, '', flag, flag, flag)

    for name, data in (('full', ASIAN), ('empty', {}),
                       ('no_time', {k: v for k, v in ASIAN.items() if 'time' not in k}),
                       ('reverse', dict(ASIAN, close_water={'home': 0.75, 'away': 1.05}))):
        yield f'asian_steam:{name}', steam._analyze_asian_steam(data, MATCH_TIME)
        yield f'trap:{name}', steam._analyze_trap_pattern(data)
    for name, data in (('full', TOTAL), ('empty', {}), ('reverse', dict(TOTAL, close_line=2.25))):
        yield f'total_steam:{name}', steam._analyze_total_steam(data, MATCH_TIME)
        yield f'total_trap:{name}', steam._analyze_total_trap(data)
    for start, end in (('2026-08-28 10:00:00', '2026-08-28 18:00:00'),
                       ('2026-08-28 18:00:00', '2026-08-28 10:00:00'),
                       ('bad', '2026-08-28 10:00:00'), ('', ''), (None, None)):
        yield f'time_diff:{start}', steam._calculate_time_diff(start, end)
        yield f'time_left:{start}', steam._calculate_time_remaining(start, end)
    for match_time in ('08-28 20:00', '2026-08-28 20:00', '2026-08-28 20:00:00',
                       '', 'bad', None):
        # **注入固定时钟**：不带年的时间串会补"当前年"，不注入的话黄金跨年就红
        yield f'norm_time:{match_time!r}', steam._normalize_match_time(match_time, now=NOW)
    signals = steam._analyze_asian_steam(ASIAN, MATCH_TIME).get('signals') or []
    yield 'summary:real', steam._summarize_signals(signals)
    yield 'summary:empty', steam._summarize_signals([])
