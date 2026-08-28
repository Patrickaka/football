# -*- coding: utf-8 -*-
"""生成 football parsing / lottery 的黄金语料条目。

被 `tests/domain/sports/football/test_parsing.py` 与 `scripts/regen_golden.py`
共用——生成与比对必须走同一个 `entries()`。

语料三部分：
- **真实 HTML**：`tests/fixtures/football_odds_pages.json.gz` 是 2026-08-28 从
  500.com 抓的六页（两场 × 亚盘/大小球/数据分析），只抓一次、存进版本控制
  （判据 20：测试依赖的东西必须在版本控制里）。
- **真实竞彩赔率**：`tests/fixtures/football_lottery_corpus.json`，从线上 114 场
  `match_analysis` 缓存里去重出的 45 组官方赔率与让球值。
- **合成**：真实语料没碰到的分支——让球 0（平手）、各档让球文本、畸形输入。

**比分候选是合成的**：缓存里没存 `candidates`，用泊松矩阵按五组 λ 构造。
这是「我们自己算的」，不是库算的，所以进黄金是安全的（判据 20b）。
"""
import gzip
import json
import math
import pathlib

from src.domain.sports.football import lottery as lot
from src.domain.sports.football import parsing as p

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGES = json.loads(gzip.decompress((ROOT / 'tests/fixtures/football_odds_pages.json.gz').read_bytes()))
LOTTERY = json.loads((ROOT / 'tests/fixtures/football_lottery_corpus.json').read_text(encoding='utf-8'))

# 迁移当时 config 的真实取值（写死不 import——判据 4）
LEAGUE_PROFILES = {
    'default': {'avg_goal': 1.42, 'home_boost': 1.06, 'low_score': 0.92, 'draw_mult': 1.0},
    '英超': {'avg_goal': 1.52, 'home_boost': 1.08, 'low_score': 0.88, 'draw_mult': 0.95},
    '西甲': {'avg_goal': 1.38, 'home_boost': 1.10, 'low_score': 0.95, 'draw_mult': 1.02},
    '意甲': {'avg_goal': 1.35, 'home_boost': 1.05, 'low_score': 0.98, 'draw_mult': 1.05},
}

HANDICAP_TEXTS = [
    '平手', '半球', '一球', '球半', '两球', '两球半', '三球', '三球半',
    '平手/半球', '半球/一球', '一球/球半', '球半/两球', '两球/两球半',
    '两球半/三球', '三球/三球半', '平半', '半一',
    '受平手', '受半球', '受一球', '受球半', '受两球',
    '受让平手', '受让半球', '受让一球', '受让球半', '受让两球', '受让两球半',
    '受平半', '受半一', '0.5', '-0.75', '1.25', 'abc', '', ' 半球 ',
]

TOTAL_TEXTS = ['0.5/1', '1/1.5', '1.5/2', '2/2.5', '2.5/3', '3/3.5', '3.5/4',
               '2.5', '3', 'x', '']

LOTTERY_HANDICAP_INPUTS = [None, '', '0', '-1', '1', '2.5', '-2', '5', '6', '-6',
                           '1.5', '(-1)', '（-1）', '让1', 'abc', 0, 1, -1, 5, 6, 2.0]

TOTAL_SHAPES = [{}, {'close_line': 2.5}, {'line': 3.0}, {'close': {'line': 1.5}},
                {'close_line': 0}, {'close_line': None, 'line': 2.0}, {'close': {}},
                {'close_line': 0, 'line': 0, 'close': {'line': 0}}]

SPF_PROBS = [
    {'胜': 0.5, '平': 0.3, '负': 0.2}, {'胜': 0.4, '平': 0.3, '负': 0.3},
    {'胜': 0.34, '平': 0.33, '负': 0.33}, {'胜': 0.3, '平': 0.4, '负': 0.3},
    {'胜': 0.28, '平': 0.36, '负': 0.36}, {'胜': 0, '平': 0, '负': 0},
    {}, None, {'胜': 0.6, '平': 0.24, '负': 0.16}, {'胜': 0.36, '平': 0.25, '负': 0.39},
    {'胜': 0.45, '平': 0.24, '负': 0.31}, {'胜': 0.45, '平': 0.23, '负': 0.32},
    {'胜': 0.40, '平': 0.29, '负': 0.31}, {'胜': 0.34, '平': 0.35, '负': 0.31},
]

LAMBDAS = [(1.5, 1.1), (2.4, 0.8), (0.7, 2.1), (1.2, 1.2), (3.0, 0.4)]


def poisson_candidates(lam_home, lam_away, max_goals=6):
    """按独立泊松铺一张比分候选表——**我们自己算的**，与第三方库无关"""
    out = []
    for h in range(max_goals + 1):
        ph = math.exp(-lam_home) * lam_home ** h / math.factorial(h)
        for a in range(max_goals + 1):
            pa = math.exp(-lam_away) * lam_away ** a / math.factorial(a)
            out.append(((h, a), ph * pa))
    return out


CANDIDATES = {f'{lh}-{la}': poisson_candidates(lh, la) for lh, la in LAMBDAS}


def _asian_handicap(h):
    return {'asian': {'close': {'handicap': h}}}


def entries():
    # --- 文本解析 ---
    for t in HANDICAP_TEXTS:
        yield f'parse_handicap:{t!r}', p.parse_handicap(t)
        yield f'handicap_text:{t!r}', p.handicap_text_to_num(t)
    yield 'handicap_text:None', p.handicap_text_to_num(None)
    for t in TOTAL_TEXTS:
        yield f'parse_total_line:{t!r}', p.parse_total_line(t)
    for v in LOTTERY_HANDICAP_INPUTS:
        yield f'lottery_handicap:{v!r}', lot.parse_lottery_handicap(v)
    for shape in TOTAL_SHAPES:
        yield f'close_total:{shape}', p.get_close_total_line(shape)
        yield f'close_total_d:{shape}', p.get_close_total_line(shape, 3.5)

    # --- 真实页面 ---
    for key, html in sorted(PAGES.items()):
        yield f'html_text_len:{key}', len(p.html_to_text(html))
        yield f'html_text_head:{key}', p.html_to_text(html)[:400]
        try:
            yield f'avg:{key}', p.extract_avg_numbers(html)[:20]
        except ValueError as e:
            yield f'avg:{key}', f'ValueError: {e}'
        for company in ('Bet365', 'Pinnacle'):
            for is_total in (False, True):
                yield (f'company:{key}:{company}:{is_total}',
                       p.extract_company_odds(html, company, is_total))
        page = key.split(':')[0]
        if page in ('yazhi', 'daxiao'):
            nums = p.extract_avg_numbers(html)
            if page == 'yazhi':
                yield f'yazhi_page:{key}', p.yazhi_from_page(html, nums)
            else:
                yield f'daxiao_nums:{key}', p.daxiao_from_avg_numbers(nums)
        else:
            for home, away in (('阿森纳', '切尔西'), ('曼联', '利物浦'), ('', '')):
                yield (f'team_strength:{key}:{home}/{away}',
                       p.parse_team_strength(html, home, away))
            yield f'form_matches:{key}', len(p.RECENT_FORM_PAT.findall(html))

    # --- 队名匹配 ---
    for ctx in ('阿森纳 近10场战绩', '切尔西队', '曼彻斯特联', '', '联队'):
        for name in ('阿森纳', '切尔西', '曼彻斯特联', '曼联', '', None):
            yield f'team_in_ctx:{ctx!r}:{name!r}', p.team_in_context(ctx, name)

    # --- 让球片段 ---
    segments = ['平均值 0.945 平手 0.884 0.872 半球 0.933',
                '平均值 0.923 -0.688 0.894 0.893 -0.938 0.926',
                '平均值 0.9 受让半球 0.9 0.9 受让一球 0.9',
                '什么也没有']
    for i, seg in enumerate(segments):
        for before, after in ((0.945, 0.884), (0.923, 0.894), (0.9, 0.9), (1.0, 1.0)):
            yield (f'hcap_seg:{i}:{before}:{after}',
                   p.extract_handicap_from_segment(seg, before, after))

    # --- 分歧指数 ---
    for bi, bet in enumerate((None, _asian_handicap(None), _asian_handicap(0.0),
                              _asian_handicap(0.5), _asian_handicap(-0.5))):
        for pi, pin in enumerate((None, _asian_handicap(None), _asian_handicap(0.0),
                                  _asian_handicap(0.75), _asian_handicap(-0.75),
                                  _asian_handicap(1.5))):
            for avg in (0.0, 0.25, -0.25, 0.5):
                yield f'consensus:{bi}:{pi}:{avg}', p.bookmaker_consensus(bet, pin, avg)

    # --- 竞彩 ---
    for cname, cands in sorted(CANDIDATES.items()):
        for i, rec in enumerate(LOTTERY):
            yield (f'lottery:{cname}:{i}',
                   lot.lottery_market_probabilities(cands, rec['lottery_handicap'],
                                                    rec['spf_odds'], rec['rqspf_odds']))
        for hc in (None, 0, 1, -1, 2, -2, 3, 6, '让1'):
            yield f'lottery_syn:{cname}:{hc!r}', lot.lottery_market_probabilities(cands, hc)
        yield f'lottery_empty:{cname}', lot.lottery_market_probabilities([], 1)
    yield 'lottery_none', lot.lottery_market_probabilities(None, 1)
    yield ('lottery_bad',
           lot.lottery_market_probabilities([('x', 'y'), ((1,), 0.5), ((1, 0), 'z')], 1))

    for i, probs in enumerate(SPF_PROBS):
        yield f'spf_selection:{i}', lot.spf_selection_profile(probs)
    for i, odds in enumerate(({'胜': 2.0, '平': 3.0, '负': 4.0},
                              {'胜': 1.0, '平': 3.0, '负': 4.0},
                              {'胜': 'x', '平': 3.0, '负': 4.0}, None, {})):
        yield f'lottery_odds:{i}', lot.lottery_odds_probabilities(odds, ('胜', '平', '负'))
    for i, (model, market) in enumerate((
            ({'胜': 0.5, '平': 0.3, '负': 0.2}, {'胜': 0.4, '平': 0.35, '负': 0.25}),
            ({'胜': 0.5, '平': 0.3, '负': 0.2}, None),
            ({'胜': 0.0, '平': 0.0, '负': 0.0}, None))):
        for w in (0.0, 0.5, 0.8, 1.0, 1.5, -0.5):
            yield f'blend_lottery:{i}:{w}', lot.blend_lottery_probabilities(model, market, w)

    # --- 融合与画像 ---
    for close in (1.0, 2.0, 0.0):
        for open_ in (None, 1.0, 3.0):
            for w in (0.72, 0.0, 1.0):
                yield f'blend_close_open:{close}:{open_}:{w}', p.blend_close_open(close, open_, w)
            yield f'blend_close_open_default:{close}:{open_}', p.blend_close_open(close, open_)
    for name in ('英超', '英格兰超级联赛', '西甲', '意甲', '', '未知联赛', None):
        static = p.resolve_static_league_profile(name, LEAGUE_PROFILES)
        yield f'static_profile:{name}', static
        for live in (None,
                     {'avg_goal': 1.8, 'draw_rate': 0.30, 'sample_size': 200},
                     {'avg_goal': 1.8, 'draw_rate': 0.30, 'sample_size': 49},
                     {'avg_goal': 1.8, 'draw_rate': 0.30, 'sample_size': 50}):
            sample = (live or {}).get('sample_size')
            yield f'blend_profile:{name}:{sample}', p.blend_league_profiles(static, live, name)
    for i, matches in enumerate((
            [], None,
            [{'score': '2-1'}, {'score': '0-0'}, {'score': '1-3'}],
            [{'score': 'bad'}, {'score': None}, {'score': '1-1'}],
            [{'score': '3-2'}] * 60)):
        yield f'league_from_matches:{i}', p.league_profile_from_matches(matches)

    # --- 赔率值与欧赔序列 ---
    for v in (None, '', '0', '-1', '1.5', 1.5, 0, -1, 'abc'):
        try:
            yield f'odds_value:{v!r}', p.parse_odds_value(v, 'f', 'm1')
        except ValueError as e:
            yield f'odds_value:{v!r}', f'ValueError: {e}'
    series = [[2.0, 3.4, 3.8, 93.1], [2.1, 3.3, 3.7, 93.0], [2.2, 3.2, 3.6, 92.9]]
    for name, s in (('ok', series), ('empty', []), ('notlist', {'a': 1}),
                    ('short_close', [[2.0, 3.0]] + series),
                    ('short_open', series + [[2.0, 3.0]]),
                    ('zero', [[0, 3.0, 4.0, 93.0]] + series),
                    ('str', [['x', 3.0, 4.0, 93.0]] + series),
                    ('no_rr', [[2.0, 3.0, 4.0], [2.1, 3.1, 4.1]])):
        try:
            yield f'ouzhi:{name}', p.ouzhi_from_series(s, 'm1')
        except ValueError as e:
            yield f'ouzhi:{name}', f'ValueError: {e}'

    # --- 单家公司行 → 市场结构 ---
    asian_row = [0.95, 0.5, 0.85, 0.90, 0.75, 0.90, '08-28 10:00', '08-27 10:00']
    total_row = [0.90, 2.5, 0.90, 0.85, 2.75, 0.95, '08-28 10:00', '08-27 10:00']
    yield 'company_markets:both', p.company_odds_to_markets(asian_row, total_row)
    yield 'company_markets:asian_only', p.company_odds_to_markets(asian_row, None)
    yield 'company_markets:none', p.company_odds_to_markets(None, total_row)
    yield 'company_markets:short', p.company_odds_to_markets(asian_row[:6], total_row[:6])
