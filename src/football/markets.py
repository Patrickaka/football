# -*- coding: utf-8 -*-
"""足球市场分析的适配层：喂配置、发请求，算法在 `domain/sports/football/markets`

领域层不读全局配置，所以本层的职责就是把 `config.py` 里那几个阈值显式传进去。
`fetch_ouzhi_company` 是网络本体，留在这里。

导出名与迁移前逐个一致（存量测试按名字导入），只是实现搬走了。
下划线开头的那几个在领域层去掉了下划线——它们本来就被 `__init__.py` 导出，
"私有"只是个名字上的错觉。
"""

from ..common.logger import setup_logger
from ..domain.sports.football import markets as _m

log = setup_logger('football')
from . import fetching as _fetching_mod

from .config import (
    BASE, EURO_PROB_TREND_EPS, HANDICAP_TREND_EPS, KELLY_BIAS_EPS, OUZHI_JSON_URL,
    TOTAL_LEAN_THRESHOLD, WATER_TREND_EPS,
)

remove_vig = _m.remove_vig
kelly_index_triple = _m.kelly_index_triple
compute_dispersion = _m.compute_dispersion
euro_to_handicap_implied = _m.euro_to_handicap_implied
compute_euro_asian_deviation = _m.compute_euro_asian_deviation
implied_total_goals = _m.implied_total_goals

_kelly_outcome_label = _m.kelly_outcome_label
_linear_regression_slope = _m.linear_regression_slope
_return_rate_from_odds = _m.return_rate_from_odds
_poisson_pmf = _m.poisson_pmf
_poisson_tail_over = _m.poisson_tail_over


def _analyze_handicap_trend(open_hcap, close_hcap):
    """分析让球走势"""
    return _m.handicap_trend_text(open_hcap, close_hcap, eps=HANDICAP_TREND_EPS)


def calculate_implied_total(line, over_odds, under_odds):
    """根据大小球盘口和水位计算隐含总进球数"""
    return _m.nudge_total_by_water(line, over_odds, under_odds)


def analyze_asian(data):
    """解析亚盘，返回让球走势、水位走势、真实概率与强弱判断"""
    return _m.analyze_asian(data,
                            handicap_eps=HANDICAP_TREND_EPS,
                            water_eps=WATER_TREND_EPS)


def analyze_kelly(ouzhi_data, probs_open, probs_close):
    """欧赔凯利指数分析：初/终盘凯利、返还率对比、离散度与打出难度提示"""
    return _m.analyze_kelly(ouzhi_data, probs_open, probs_close,
                            bias_eps=KELLY_BIAS_EPS)


def analyze_kelly_trend(series, recent_n=5):
    """凯利指数时序分析：最近 N 条的斜率与穿越返还率的事件"""
    return _m.analyze_kelly_trend(series, recent_n, bias_eps=KELLY_BIAS_EPS)


def analyze_euro_momentum(series):
    """由欧赔时间序列提取主/客胜概率走势，用于修正净胜球"""
    return _m.analyze_euro_momentum(series, prob_eps=EURO_PROB_TREND_EPS)


def compute_joint_anomaly(asian_data, total_data):
    """计算联合异常特征：让球水位变化 × 大小球水位变化"""
    return _m.compute_joint_anomaly(asian_data, total_data, water_eps=WATER_TREND_EPS)


def analyze_euro(data):
    """解析欧赔，返回初终盘 1X2 真实概率、凯利、走势与变化趋势"""
    return _m.analyze_euro(data,
                           prob_eps=EURO_PROB_TREND_EPS,
                           bias_eps=KELLY_BIAS_EPS)


def analyze_total(data):
    """解析大小球，返回盘口线、大小球真实概率、倾向与期望进球区间"""
    return _m.analyze_total(data, lean_threshold=TOTAL_LEAN_THRESHOLD)


def fetch_ouzhi_company(match_id, cid=1):
    """抓取指定公司的欧赔时间序列（cid=1 为威廉希尔等）"""
    url = f'{OUZHI_JSON_URL}?fid={match_id}&cid={cid}&type=europe&r=1'
    referer = f'{BASE}/fenxi/ouzhi-{match_id}.shtml'
    try:
        series = _fetching_mod.fetch_json(url, referer=referer)
        if isinstance(series, list) and len(series) >= 2:
            return series
    except Exception:
        pass
    return None
