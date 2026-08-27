# -*- coding: utf-8 -*-
"""北单概率建模：泊松/DC矩阵/λ推导/盘口解析/比分锚定"""

import sys
import math
import re
from collections import defaultdict
import time
import json
import urllib.request
import urllib.error
import random
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from ..common.logger import setup_logger
from ..common.paths import data_path
from ..common import kv_store

log = setup_logger('beidan')

from .config import (
    LEAGUE_PROFILES, MAX_GOALS, SCORE_SPLIT,
)









DC_RHO = 0.0


OU_TOTAL_BLEND = 0.6


TARGET_TOTAL_MIN = 1.8


TARGET_TOTAL_MAX = 3.6


FACTOR_MIN = 0.85


FACTOR_MAX = 1.15


STRENGTH_SPLIT = SCORE_SPLIT  # = 0.45


BEIDAN_STRONG_MIN_PROBABILITY = 0.65


BEIDAN_STRONG_MIN_LEAD = 0.10


BEIDAN_MEDIUM_MIN_PROBABILITY = 0.60


BEIDAN_MEDIUM_MIN_LEAD = 0.10


BEIDAN_HIGH_PRECISION_MIN_PROBABILITY = 0.70


SCORE_OUTCOME_ANCHOR_STRENGTH = 0.75

# ─── 领域层适配 ───
#
# 算法在 `src/domain/sports/beidan/` 的 scoring_model（泊松/DC/锚定）、
# handicap（让球解析与结算线）、totals（大小球与目标总进球）三个模块里。
# 这里只做两件事：**把上面那些配置常量喂进去**，以及保住旧名字——
# `__init__.py` 导出了它们，`recommending`/`markets` 按名字导入。

from src.domain.sports.beidan import handicap as _handicap
from src.domain.sports.beidan import scoring_model as _model
from src.domain.sports.beidan import totals as _totals

poisson_pmf = _model.poisson_pmf
aggregate_goals_from_scores = _model.aggregate_goals
parse_beidan_handicap = _handicap.parse
rqspf_probs_from_score_probs = _handicap.rqspf_from_scores
_asian_line_parts = _handicap.line_parts
_asian_over_profit = _handicap.over_profit
_to_euro_odds = _totals.to_euro_odds
_parse_total_line_value = _totals.parse_line_value
implied_total_from_ou = _totals.implied_total


def anchor_score_outcomes(score_dist, target_probabilities,
                          strength=SCORE_OUTCOME_ANCHOR_STRENGTH):
    """按市场概率部分锚定比分矩阵的胜平负质量。

    领域层的 `anchor_outcomes` 要求 `strength` 必传——**它是配置，
    不该藏在算法的默认值里**。默认值留在这一层。
    """
    return _model.anchor_outcomes(score_dist, target_probabilities, strength)


def _profile(league):
    """联赛档案。**查表是配置层的事**——领域层只接收 avg_goals 这个数。"""
    return LEAGUE_PROFILES.get(league, {'avg_goals': 2.6, 'draw_rate': 0.27})


def euro_implied_lambdas(p_home, p_draw, p_away, target_total):
    return _model.lambdas_from_probs(p_home, p_draw, p_away, target_total, SCORE_SPLIT)


def match_lambdas(home_prob, draw_prob, away_prob, target_total):
    """由 1X2 概率与目标总进球计算主客 λ。

    **签名比迁移前少了 `split`**：它在函数体里从没出现过——这个函数只是
    转手调 `euro_implied_lambdas`，而那边用的是模块级的 `SCORE_SPLIT`。
    三个不同的 split 值算出完全一样的结果，而三处调用方一个都没传过它。
    领域层的 `lambdas_from_probs` 让这个参数真正生效，默认仍是 `SCORE_SPLIT`，
    所以现有行为不变。
    """
    return _model.lambdas_from_probs(home_prob, draw_prob, away_prob,
                                     target_total, SCORE_SPLIT)


def calibrate_draw_probability(p_home, p_draw, p_away, handicap,
                               home_draw_rate=0.25, away_draw_rate=0.25,
                               league_draw_rate=0.25):
    return _model.calibrate_draw(p_home, p_draw, p_away, handicap,
                                 home_draw_rate, away_draw_rate, league_draw_rate)


def build_dixon_coles_matrix(lam_home, lam_away, rho=DC_RHO, max_goals=MAX_GOALS):
    return _model.dixon_coles_matrix(lam_home, lam_away, rho, max_goals)


def match_target_total(league='', total_over_odds=None, total_under_odds=None,
                       asian_factor=1.0, goals_factor=1.0, total_line=2.5):
    return _totals.target_total(
        _profile(league)['avg_goals'], total_over_odds, total_under_odds,
        asian_factor, goals_factor, total_line,
        blend=OU_TOTAL_BLEND, factor_low=FACTOR_MIN, factor_high=FACTOR_MAX,
        total_low=TARGET_TOTAL_MIN, total_high=TARGET_TOTAL_MAX)


def predict_scores_by_poisson(home_prob, draw_prob, away_prob, league='', handicap=0,
                              total_over_odds=None, total_under_odds=None, use_dc=True,
                              total_line=2.5):
    profile = _profile(league)
    p_home, p_draw, p_away = _model.calibrate_draw(
        home_prob, draw_prob, away_prob, handicap,
        league_draw_rate=profile['draw_rate'])

    target_total = match_target_total(
        league=league, total_over_odds=total_over_odds,
        total_under_odds=total_under_odds, total_line=total_line)
    lam_home, lam_away = match_lambdas(p_home, p_draw, p_away, target_total)

    if use_dc:
        score_probs = _model.dixon_coles_matrix(lam_home, lam_away, DC_RHO, MAX_GOALS)
    else:
        score_probs = _model.independent_poisson_matrix(lam_home, lam_away, MAX_GOALS)

    score_probs, outcome_anchor = _model.anchor_outcomes(
        score_probs, {'胜': p_home, '平': p_draw, '负': p_away},
        SCORE_OUTCOME_ANCHOR_STRENGTH)

    return {
        'top3': _model.top_scores(score_probs, 3),
        'score_probs': score_probs,
        'lambda_home': lam_home,
        'lambda_away': lam_away,
        'target_total': target_total,
        '1x2_prob': {'H': p_home, 'D': p_draw, 'A': p_away},
        'outcome_anchor': outcome_anchor,
    }
