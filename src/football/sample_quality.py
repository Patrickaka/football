#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Historical sample quality helpers for football prediction calibration/backtests.

评估逻辑已迁至 `src.domain.sports.football.quality`，这里只做转发，
保持既有 import 路径不变。
"""

from ..domain.sports.football.quality import (  # noqa: F401
    FRIENDLY_KEYWORDS,
    GRADE_RANK,
    _has_any,
    _is_friendly,
    assess_record_quality,
    filter_quality_records,
)
