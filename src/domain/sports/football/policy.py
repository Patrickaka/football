# -*- coding: utf-8 -*-
"""预测策略的口径：联赛画像、分桶键、参数取值范围、比分矩阵的混合与选点。

**没有存储**：调参配置的读写（kv_store 与磁盘档案）、
`get_prediction_policy` 与 `apply_score_distribution_policy`
都留在适配层 `src/football/prediction_policy.py`。

`PARAM_RANGES` 是**领域契约**（判据 29）：调参搜索的合法区间由它定，
不是某个调用方的默认值。
"""

from typing import Dict, Tuple


LOW_GOAL_LEAGUE_HINTS = ('意甲', '葡超', '希腊', '阿甲', '巴乙', '日乙')

HIGH_VARIANCE_HINTS = ('荷甲', '挪超', '瑞典', '巴甲', '美职')

CUP_HINTS = ('杯', '欧冠', '欧联', '世俱', '世界杯', '欧洲杯')

POLICY_PARAM_KEYS = {
    'static_market_cap',
    'change_market_cap',
    'half_full_real_weight',
    'half_full_market_cap',
    'draw_bias',
    'low_score_bias',
    'high_score_bias',
    'late_market_weight_bias',
}

PARAM_ALIASES = {
    'market_db_weight': 'static_market_cap',
    'market_change_weight': 'change_market_cap',
    'half_full_history_weight': 'half_full_real_weight',
    'score_draw_bias': 'draw_bias',
    'late_market_weight': 'late_market_weight_bias',
    'time_layer_market_weight': 'late_market_weight_bias',
}

PARAM_RANGES = {
    'static_market_cap': (0.0, 0.30),
    'change_market_cap': (0.0, 0.30),
    'half_full_real_weight': (0.0, 0.45),
    'half_full_market_cap': (0.0, 0.20),
    'draw_bias': (0.75, 1.25),
    'low_score_bias': (0.75, 1.30),
    'high_score_bias': (0.75, 1.30),
    'late_market_weight_bias': (-0.08, 0.08),
}

def _contains_any(text: str, hints: Tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(hint.lower() in lowered for hint in hints)

def _league_text(league=None, league_profile=None) -> str:
    if league:
        return str(league)
    if isinstance(league_profile, dict):
        return str(league_profile.get('name') or league_profile.get('league') or '')
    return ''

def _empty_tuning_config() -> Dict:
    return {
        'version': 1,
        'updated_at': None,
        'global': {},
        'leagues': {},
        'buckets': {},
        'history': [],
    }

def _canonical_params(params: Dict) -> Dict:
    cleaned = {}
    for key, value in (params or {}).items():
        canonical = PARAM_ALIASES.get(key, key)
        if canonical not in POLICY_PARAM_KEYS:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        low, high = PARAM_RANGES[canonical]
        cleaned[canonical] = max(low, min(high, number))
    return cleaned

def get_total_bucket(total_line) -> str:
    if total_line is None:
        return 'unknown'
    try:
        line = float(total_line)
    except (TypeError, ValueError):
        return 'unknown'
    if line <= 2.25:
        return 'low'
    if line >= 3.0:
        return 'high'
    return 'normal'

def get_handicap_bucket(handicap) -> str:
    if handicap is None:
        return 'unknown'
    try:
        depth = abs(float(handicap))
    except (TypeError, ValueError):
        return 'unknown'
    if depth <= 0.25:
        return 'level'
    if depth < 1.0:
        return 'mid'
    return 'deep'

def policy_bucket_key(league=None, total_line=None, handicap=None, league_profile=None) -> str:
    league_name = _league_text(league, league_profile) or '*'
    return f"{league_name}|{get_total_bucket(total_line)}|{get_handicap_bucket(handicap)}"

def normalize_score_matrix(matrix: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    total = sum(matrix.values())
    if total <= 0:
        return matrix
    return {score: prob / total for score, prob in matrix.items()}

def blend_score_matrices(primary, secondary, secondary_weight: float = 0.50):
    """Convex blend of two score distributions with auditable fixed weight."""
    weight = max(0.0, min(1.0, float(secondary_weight)))
    primary = dict(primary or {})
    secondary = dict(secondary or {})
    keys = set(primary) | set(secondary)
    blended = {
        score: (1.0 - weight) * float(primary.get(score, 0.0))
        + weight * float(secondary.get(score, 0.0))
        for score in keys
    }
    return normalize_score_matrix(blended)

def select_diverse_score_scenarios(candidates, limit: int = 5):
    """Rank display scenarios without changing the underlying probabilities.

    The primary score follows the aggregate 1X2 direction. Remaining slots
    cover distinct total-goal scenarios before filling by raw probability.
    This avoids presenting the global modal score (often 1-1) as if it were
    the model's only match script.
    """
    candidates = list(candidates or [])
    if not candidates:
        return []

    def outcome(score):
        home, away = score
        return "H" if home > away else ("D" if home == away else "A")

    masses = {"H": 0.0, "D": 0.0, "A": 0.0}
    for score, probability in candidates:
        masses[outcome(score)] += float(probability)
    primary_outcome = max(masses, key=masses.get)
    primary = next(
        (item for item in candidates if outcome(item[0]) == primary_outcome),
        candidates[0],
    )

    selected = [primary]
    selected_scores = {primary[0]}
    used_totals = {sum(primary[0])}

    global_mode = candidates[0]
    if global_mode[0] not in selected_scores:
        selected.append(global_mode)
        selected_scores.add(global_mode[0])
        used_totals.add(sum(global_mode[0]))

    for item in candidates:
        score = item[0]
        if score in selected_scores or sum(score) in used_totals:
            continue
        selected.append(item)
        selected_scores.add(score)
        used_totals.add(sum(score))
        if len(selected) >= limit:
            return selected

    for item in candidates:
        if item[0] in selected_scores:
            continue
        selected.append(item)
        selected_scores.add(item[0])
        if len(selected) >= limit:
            break
    return selected
