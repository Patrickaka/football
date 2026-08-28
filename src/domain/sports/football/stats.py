# -*- coding: utf-8 -*-
"""比例的区间估计。

`wilson_interval` 单独放在这里是为了**断环**：迁移前
`production_league_gate` 顶层 import 它、`professional_monitoring`
又延迟 import `build_production_league_spf_policies`，两个模块互相咬。
提到共用模块后依赖变成单向。
"""

import math


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = hits / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)
