"""Bundled audited football validation baseline.

This snapshot travels with the application so production status never depends
on a mutable report file being present.  A newer report may override it, but
must use the same schema.
"""

from copy import deepcopy

from ..domain.sports.football.monitoring import (  # noqa: F401
    AUDITED_PROFESSIONAL_BASELINE,
    BASELINE_GENERATED_AT,
    BASELINE_VERSION,
    bundled_professional_baseline,
)
