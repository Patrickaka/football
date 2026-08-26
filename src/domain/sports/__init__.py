"""赛事盘口领域基座：比赛、盘口、赔率、结算。

供 football / beidan / basketball 三个实现共用。
"""
from .match import Match
from .odds import Odds, odds_to_prob

__all__ = ['Match', 'Odds', 'odds_to_prob']
