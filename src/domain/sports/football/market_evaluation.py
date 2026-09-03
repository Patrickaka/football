# -*- coding: utf-8 -*-
"""市场评估底座：把任何概率来源当成"一个预测者"，用同一把尺子量。

**没有 I/O**：入口收的是 football-data 格式的行（dict），读 CSV 留在适配层。

为什么先做这个：现在的 1X2 输出锚定到市场（强度 1.0），页面上的 ✅/❌ 量的
是庄家而不是模型。要判断任何"提高准确性"的改动值不值，得先有一把不依赖
生产管线、能对比多个来源的尺子——Pinnacle 收盘是最接近有效的参照，
Bet365/市场均价是竞彩这类软盘的代理。

指标只用 log loss / Brier / ECE 加 EV 策略回报。Top-1 命中率被热门主导，
只作辅助展示。
"""

import math
from typing import Dict, Iterable, List, Optional, Sequence

from .markets import remove_vig

OUTCOMES = ('H', 'D', 'A')

# football-data 列名。C = closing，无 C 的是开盘。
MARKET_SOURCES: Dict[str, Dict[str, str]] = {
    'pinnacle_open': {'H': 'PSH', 'D': 'PSD', 'A': 'PSA'},
    'pinnacle_close': {'H': 'PSCH', 'D': 'PSCD', 'A': 'PSCA'},
    'b365_open': {'H': 'B365H', 'D': 'B365D', 'A': 'B365A'},
    'b365_close': {'H': 'B365CH', 'D': 'B365CD', 'A': 'B365CA'},
    'avg_open': {'H': 'AvgH', 'D': 'AvgD', 'A': 'AvgA'},
    'avg_close': {'H': 'AvgCH', 'D': 'AvgCD', 'A': 'AvgCA'},
    'max_close': {'H': 'MaxCH', 'D': 'MaxCD', 'A': 'MaxCA'},
}

_LOG_FLOOR = 1e-12


def _odds(row: Dict, column: str) -> Optional[float]:
    value = row.get(column)
    if value in (None, ''):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 1.0 else None


DEVIG_METHODS = ('proportional', 'power')


def _power_devig(inverse: Sequence[float]) -> List[float]:
    """求 k 使 Σ (1/o)^k = 1。

    比例去水把水按相同比例从每项扣掉，等于默认庄家在每个选项上抽同样的
    利润率；实际上庄家在冷门上抽得多得多（热门-冷门偏差）。幂法让冷门被压
    得更狠，是文献里最常用的修正之一。k=1 时退化为无水的原始值。
    """
    total = sum(inverse)
    if abs(total - 1.0) < 1e-12:
        return list(inverse)
    lo, hi = (1.0, 8.0) if total > 1.0 else (0.05, 1.0)
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if sum(value ** mid for value in inverse) > 1.0:
            lo = mid
        else:
            hi = mid
    exponent = (lo + hi) / 2.0
    powered = [value ** exponent for value in inverse]
    normaliser = sum(powered)
    return [value / normaliser for value in powered]


def implied_probabilities(home_odds, draw_odds, away_odds,
                          method: str = 'proportional') -> Optional[Dict[str, float]]:
    """三项赔率去水成概率；任一项缺失或不大于 1 视为不可用。"""
    if method not in DEVIG_METHODS:
        raise ValueError(f'未知去水方法: {method!r}，可选 {DEVIG_METHODS}')
    odds = [home_odds, draw_odds, away_odds]
    try:
        odds = [float(value) for value in odds]
    except (TypeError, ValueError):
        return None
    if any(value <= 1.0 for value in odds):
        return None
    if method == 'power':
        home, draw, away = _power_devig([1.0 / value for value in odds])
    else:
        home, draw, away = remove_vig(*odds)
    return {'H': home, 'D': draw, 'A': away}


def source_odds(row: Dict, source: str) -> Optional[Dict[str, float]]:
    columns = MARKET_SOURCES[source]
    odds = {outcome: _odds(row, columns[outcome]) for outcome in OUTCOMES}
    return odds if all(odds.values()) else None


def source_probabilities(row: Dict, source: str,
                         devig: str = 'proportional') -> Optional[Dict[str, float]]:
    odds = source_odds(row, source)
    if odds is None:
        return None
    return implied_probabilities(odds['H'], odds['D'], odds['A'], method=devig)


def score_rows(rows: Iterable[Dict], source: str,
               devig: str = 'proportional') -> List[Dict]:
    """把每行变成 {probs, result, row}；缺来源列或缺赛果的行跳过。

    未知来源名直接 KeyError——静默返回空列表会让"来源拼错"看起来像
    "这个赛季没数据"。
    """
    MARKET_SOURCES[source]
    scored = []
    for row in rows:
        result = row.get('FTR')
        if result not in OUTCOMES:
            continue
        probs = source_probabilities(row, source, devig)
        if probs is None:
            continue
        scored.append({'probs': probs, 'result': result, 'row': row})
    return scored


def probability_metrics(scored: Sequence[Dict], ece_bins: int = 10) -> Dict:
    """log loss / Brier / Top-1 / ECE。

    ECE 按"报出的最高概率"分桶，桶内 |平均置信度 − 实际命中率| 加权平均——
    衡量的是"说 70% 时是不是真有 70%"，与 log loss 互补。
    """
    n = len(scored)
    if n == 0:
        return {'n': 0, 'log_loss': None, 'brier': None,
                'top1_hit_rate': None, 'ece': None}
    log_loss = brier = hits = 0.0
    buckets = [[0.0, 0.0, 0] for _ in range(ece_bins)]
    for item in scored:
        probs, result = item['probs'], item['result']
        log_loss -= math.log(max(probs.get(result, 0.0), _LOG_FLOOR))
        brier += sum((probs.get(outcome, 0.0) - (1.0 if outcome == result else 0.0)) ** 2
                     for outcome in OUTCOMES)
        top = max(OUTCOMES, key=lambda outcome: probs.get(outcome, 0.0))
        confidence = probs.get(top, 0.0)
        hit = 1.0 if top == result else 0.0
        hits += hit
        index = min(ece_bins - 1, int(confidence * ece_bins))
        buckets[index][0] += confidence
        buckets[index][1] += hit
        buckets[index][2] += 1
    ece = sum(abs(conf_sum / count - hit_sum / count) * count / n
              for conf_sum, hit_sum, count in buckets if count)
    return {
        'n': n,
        'log_loss': log_loss / n,
        'brier': brier / n,
        'top1_hit_rate': hits / n,
        'ece': ece,
    }


def expected_value_picks(rows: Iterable[Dict], sharp_source: str, soft_source: str,
                         threshold: float = 0.0, devig: str = 'proportional') -> List[Dict]:
    """尖锐价当真概率，软盘赔率当可下注价，挑 EV 超过阈值的选项。

    EV = p_sharp × odds_soft − 1。同一场三项各自判断，可能一场出多个。
    """
    picks = []
    for row in rows:
        result = row.get('FTR')
        if result not in OUTCOMES:
            continue
        sharp = source_probabilities(row, sharp_source, devig)
        soft = source_odds(row, soft_source)
        if sharp is None or soft is None:
            continue
        for outcome in OUTCOMES:
            ev = sharp[outcome] * soft[outcome] - 1.0
            if ev > threshold:
                picks.append({
                    'selection': outcome,
                    'ev': ev,
                    'odds': soft[outcome],
                    'sharp_prob': sharp[outcome],
                    'won': outcome == result,
                    'row': row,
                })
    return picks


def roi_summary(picks: Sequence[Dict]) -> Dict:
    """每注一单位本金的回报率，附正态近似的 95% 区间。

    区间不是装饰：几十注的 ROI 正负全靠运气，没有区间就会把噪声当成优势。
    """
    n = len(picks)
    if n == 0:
        return {'n': 0, 'roi': None, 'hit_rate': None, 'roi_ci95': None,
                'avg_odds': None, 'avg_ev': None}
    profits = [(pick['odds'] - 1.0) if pick['won'] else -1.0 for pick in picks]
    roi = sum(profits) / n
    variance = sum((profit - roi) ** 2 for profit in profits) / max(1, n - 1)
    half_width = 1.96 * math.sqrt(variance / n)
    return {
        'n': n,
        'roi': roi,
        'hit_rate': sum(1 for pick in picks if pick['won']) / n,
        'roi_ci95': (roi - half_width, roi + half_width),
        'avg_odds': sum(pick['odds'] for pick in picks) / n,
        'avg_ev': sum(pick.get('ev', 0.0) for pick in picks) / n,
    }


def evaluate_sources(rows: Sequence[Dict], sources: Sequence[str],
                     devig: str = 'proportional') -> Dict[str, Dict]:
    """多个来源在同一批行上的概率指标，键为来源名。"""
    return {source: probability_metrics(score_rows(rows, source, devig)) for source in sources}


def evaluate_ev_strategy(rows: Sequence[Dict], sharp_source: str, soft_sources: Sequence[str],
                         thresholds: Sequence[float],
                         devig: str = 'proportional') -> Dict[str, Dict[float, Dict]]:
    """尖锐 vs 各软盘、各阈值下的 EV 策略回报，键为软盘名 → 阈值。"""
    return {
        soft: {
            threshold: roi_summary(
                expected_value_picks(rows, sharp_source, soft, threshold, devig))
            for threshold in thresholds
        }
        for soft in soft_sources
    }
