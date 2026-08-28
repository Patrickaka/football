"""Read-only production validation for league-specific SPF admission.

The threshold is selected on an earlier chronological segment and must hold on
the later segment before it can be used by live recommendations.  No database
state is changed here; MySQL-backed settled prediction records are only read.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Iterable, Mapping



from ..domain.sports.football.league_gate import (  # noqa: F401
    MIN_HOLDOUT_CI_LOW,
    MIN_HOLDOUT_SELECTIONS,
    MIN_TOTAL_ROWS,
    MIN_TRAIN_SELECTIONS,
    TARGET_ACCURACY,
    THRESHOLDS,
    _candidate_label,
    _gate_row,
    _metrics,
    _normalise_league,
    build_production_league_spf_policies,
    validate_league_spf_policy,
)

_CACHE_TTL_SECONDS = 600
_POLICY_CACHE: tuple[float, dict[str, dict[str, Any]]] | None = None














def load_production_league_spf_policy(league: Any) -> dict[str, Any]:
    """Load cached policies from MySQL-first storage and return one league.

    The complete production history is read at most once per TTL window.  Live
    match batches commonly contain several previously unseen leagues, so a
    per-league database scan would multiply the same expensive read.
    """
    global _POLICY_CACHE
    key = _normalise_league(league)
    if not key:
        return {"supported": False, "reason": "缺少联赛标识"}
    now = time.monotonic()
    if _POLICY_CACHE is None or now - _POLICY_CACHE[0] >= _CACHE_TTL_SECONDS:
        try:
            from ..common import repositories

            built = build_production_league_spf_policies(
                repositories.football_prediction_load()
            )
            policies = {
                _normalise_league(name): policy
                for name, policy in built.items()
            }
        except Exception as exc:
            policies = {
                "__load_error__": {
                    "supported": False,
                    "reason": f"生产预测历史读取失败: {exc}",
                }
            }
        _POLICY_CACHE = (now, policies)

    policies = _POLICY_CACHE[1]
    if "__load_error__" in policies:
        return {
            **policies["__load_error__"],
            "league": str(league or ""),
        }
    return policies.get(key, {
        "supported": False,
        "league": str(league or ""),
        "sample_count": 0,
        "reason": f"生产可审计样本不足{MIN_TOTAL_ROWS}场",
    })

