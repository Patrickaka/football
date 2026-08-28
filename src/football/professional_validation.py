"""Leakage-resistant evaluation utilities for football betting models.

The module intentionally separates probability quality from betting returns.
Records must be in chronological order and thresholds are selected using only
the training side of each walk-forward fold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..domain.sports.football.validation import (  # noqa: F401
    LABELS,
    RQSPF_LABELS,
    _max_drawdown,
    _normalize_labeled,
    _rqspf_actual,
    _rqspf_odds,
    blend_record_with_market,
    evaluate_rqspf_records,
    evaluate_strategy,
    multiclass_metrics,
    normalize_probabilities,
    probabilities_from_odds,
    select_market_residual_weight,
    select_threshold,
    walk_forward_evaluate,
)
