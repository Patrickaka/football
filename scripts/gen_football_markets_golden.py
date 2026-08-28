# -*- coding: utf-8 -*-
"""生成 football markets 的黄金语料条目。

被 `tests/domain/sports/football/test_markets.py` 与 `scripts/regen_golden.py`
共用——**生成与比对必须走同一个 entries()**，两边各写一套，比的就不是同一个
东西了。

语料两部分：
- **真实**：从线上 114 个 `match_analysis` 缓存里反推出的 56 组亚盘/欧赔/大小球
  （缓存存的是输出，但输出里带 `raw_odds` 与 `open_water`/`close_water`，够反推）。
- **合成**：真实语料没碰到的分支（判据 8「黄金覆盖了输出不等于行为被测到」）。
  三处盲区：大小球 signal 全是 weak、预期进球有三档没走到、
  **凯利的离散度恒等于 0**（见下）。

**凯利那条为什么必须合成**：`kelly_i = o_i × p_i × 100`，而 `analyze_euro`
传的 `p_i` 是同一组赔率的去水概率 `(1/o_i)/Σ(1/o_j)`，于是
`o_i × p_i ≡ 100/Σ(1/o_j)`——**三项数学上恒等，spread 恒为 0**（实测 ≤1.4e-14）。
所以线上 `hardest`/`favored` 永远是 `neutral`、`risks`/`favors` 恒空。
`analyze_kelly` 是导出的公开函数，传别的概率就走得到，属判据 9 第三行
「当前调用方碰巧不触发」——补语料，不删代码。
"""
import json
import pathlib

from src.domain.sports.football import markets as m

CORPUS = json.load(open(
    pathlib.Path(__file__).resolve().parents[1] / 'tests/fixtures/football_markets_corpus.json',
    encoding='utf-8'))


def _devig_probs(side):
    return m.remove_vig(side['home'], side['draw'], side['away'])


def _kelly_prob_variants(euro):
    """每组欧赔配几套概率，覆盖 spread 的三档

    - `devig`：与线上一致，spread ≡ 0 → neutral
    - `devig_tiny`：±0.0005 微扰 → spread 落在 (0, 1) 内，**仍是 neutral**。
      这一档专为 `KELLY_NEUTRAL_SPREAD = 1.0` 而设：不喂它的话，把门槛从
      1.0 改成 0.1 是零反应的（devig 那档 spread≈1e-14 在两个门槛下都算中性）。
    - `devig_nudge`：±0.004 微扰 → spread 落在 1~4 的中间档
    - 五组手写三元组 → spread ≥ 4 的分化档
    """
    devig = _devig_probs(euro['close'])
    yield 'devig', devig
    yield 'devig_tiny', (devig[0] + 0.0005, devig[1], devig[2] - 0.0005)
    nudged = (devig[0] + 0.004, devig[1], devig[2] - 0.004)
    yield 'devig_nudge', nudged
    for i, probs in enumerate(CORPUS['kelly_probs']):
        yield f'explicit{i}', tuple(probs)


def entries():
    for i, d in enumerate(CORPUS['asian']):
        yield f'asian:{i}', m.analyze_asian(d)
        yield f'asian_default:{i}', m.analyze_asian(d)
        yield f'handicap_trend:{i}', m.handicap_trend_text(
            d['open']['handicap'], d['close']['handicap'])

    for i, d in enumerate(CORPUS['total']):
        yield f'total:{i}', m.analyze_total(d)
        yield f'nudge_total:{i}', m.nudge_total_by_water(
            d['close']['line'], d['close']['over_odds'], d['close']['under_odds'])
        yield f'implied_total:{i}', m.implied_total_goals(
            d['close']['line'], m.remove_vig(d['close']['over_odds'], d['close']['under_odds'])[0])

    for i, d in enumerate(CORPUS['euro']):
        yield f'euro:{i}', m.analyze_euro(d)
        for name, probs in _kelly_prob_variants(d):
            yield f'kelly:{i}:{name}', m.analyze_kelly(d, probs, probs)
        yield f'momentum:{i}', m.analyze_euro_momentum(_euro_series(i))
        yield f'dispersion:{i}', m.compute_dispersion(_euro_series(i))
        for n in (2, 3, 5, 8):
            yield f'kelly_trend:{i}:{n}', m.analyze_kelly_trend(_euro_series(i), n)

    for i in range(min(len(CORPUS['asian']), len(CORPUS['total']))):
        yield f'joint:{i}', m.compute_joint_anomaly(CORPUS['asian'][i], CORPUS['total'][i])

    for i, d in enumerate(CORPUS['euro']):
        ph, _, pa = _devig_probs(d['close'])
        for hcap in (-1.5, -0.25, 0.0, 0.75, 2.0):
            yield f'euro_asian_dev:{i}:{hcap}', m.compute_euro_asian_deviation(
                {'home': ph, 'away': pa}, hcap)
            for k in (1.5, 1.8, 2.0):
                yield f'euro_to_hcap:{i}:{hcap}:{k}', m.euro_to_handicap_implied(ph, pa, k)

    for lam in (0.8, 1.7, 2.6, 3.4):
        for line in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5):
            yield f'poisson_tail:{lam}:{line}', m.poisson_tail_over(lam, line)
        for k in range(0, 8):
            yield f'poisson_pmf:{lam}:{k}', m.poisson_pmf(k, lam)

    for xs, ys in (([0, 1, 2], [1.0, 2.0, 3.0]), ([0, 1, 2], [3.0, 2.0, 1.0]),
                   ([0, 0, 0], [1.0, 2.0, 3.0]), ([0], [1.0]), ([], [])):
        yield f'slope:{xs}:{ys}', m.linear_regression_slope(xs, ys)


def _euro_series(i):
    """用相邻五组欧赔拼一段时序（倒序，最新在前——与线上 JSON 的顺序一致）"""
    euro = CORPUS['euro']
    return [[euro[(i + j) % len(euro)]['close']['home'],
             euro[(i + j) % len(euro)]['close']['draw'],
             euro[(i + j) % len(euro)]['close']['away'],
             euro[(i + j) % len(euro)]['close'].get('return_rate', 93.0)]
            for j in range(5)]
