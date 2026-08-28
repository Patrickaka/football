"""Auditable per-match evidence coverage and system capability reporting."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from ..domain.sports.football.readiness import (  # noqa: F401
    _live_item_verified,
    _present,
    _probability_divergence,
    build_match_evidence_profile,
    build_professional_decision_gate,
    build_system_gap_assessment,
)
