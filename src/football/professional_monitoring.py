"""Production monitoring for calibrated football probabilities and market timing."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Sequence

from ..domain.sports.football.monitoring import (  # noqa: F401
    _actual_rqspf,
    _normalise,
    _window_metrics,
    build_professional_monitoring,
    calibration_report,
    upset_alert_report,
)
from ..domain.sports.football.stats import wilson_interval  # noqa: F401
