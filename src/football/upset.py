# -*- coding: utf-8 -*-
"""【适配层】足球爆冷评估：算法在 `domain/sports/football/upset`"""

from ..common.logger import setup_logger
from ..domain.sports.football import upset as _u

log = setup_logger('football')

_evaluate_upset_profile = _u._evaluate_upset_profile
_evaluate_upset_risk = _u._evaluate_upset_risk
assess_football_upset = _u.assess_football_upset
