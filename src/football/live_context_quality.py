"""Quality gate for volatile pre-match context.

Missing context does not invent a prediction adjustment. The gate produces an
auditable confidence multiplier and can block official bets when lineups were
required but are unavailable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from ..domain.sports.football.context import (  # noqa: F401
    _parse_timestamp,
    assess_live_context,
)
