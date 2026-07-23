"""Bundled audited football validation baseline.

This snapshot travels with the application so production status never depends
on a mutable report file being present.  A newer report may override it, but
must use the same schema.
"""

from copy import deepcopy


BASELINE_VERSION = "football-oos-2026-07-23-v1"
BASELINE_GENERATED_AT = "2026-07-23T14:13:38+08:00"

AUDITED_PROFESSIONAL_BASELINE = {
    "method": "expanding-window-walk-forward",
    "out_of_sample_n": 1804,
    "model_metrics": {
        "n": 1804,
        "accuracy": 0.5238359201773836,
        "logloss": 0.9965126754063698,
        "brier": 0.593481284345111,
    },
    "market_baseline_metrics": {
        "n": 1804,
        "accuracy": 0.5360310421286031,
        "logloss": 0.9773157271504305,
        "brier": 0.5816143229560724,
    },
    "strategy": {
        "bets": 293,
        "wins": 175,
        "hit_rate": 0.5972696245733788,
        "coverage": 0.16241685144124168,
        "profit": -5.620000000000001,
        "roi": -0.019180887372013657,
        "max_drawdown_units": 14.170000000000002,
        "mean_clv": -0.006273251680351913,
        "clv_samples": 293,
    },
    "audit": {
        "model_split": "expanding chronological window",
        "threshold_split": "earlier out-of-sample predictions only",
        "raw_samples": 3504,
        "model_oos_samples": 2304,
    },
}


def bundled_professional_baseline():
    return deepcopy(AUDITED_PROFESSIONAL_BASELINE)
