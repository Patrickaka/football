"""High-precision selection gates for football 1X2 markets.

The gate deliberately abstains when the available evidence is not strong
enough.  A target accuracy is not presented as achieved until an independent
out-of-sample audit supports it.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..domain.sports.football.accuracy_gate import (  # noqa: F401
    MIN_MARKET_MARGIN,
    MIN_PREDICTION_RELIABILITY,
    MIN_TOP2_MARGIN,
    RQSPF_MIN_PROBABILITY,
    SPF_HISTORICAL_PROXY,
    SPF_LEAGUE_POLICIES,
    SPF_MIN_PROBABILITY,
    TARGET_ACCURACY,
    TOTAL_GOALS_LEAGUE_POLICIES,
    _league_policy,
    _market_agrees,
    _spf_policy,
    _static_spf_policy,
    _top_pick,
    build_accuracy_gate,
    build_total_goals_gate,
    has_static_spf_policy,
    prediction_reliability,
)
