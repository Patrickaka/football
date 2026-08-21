# -*- coding: utf-8 -*-
"""快乐8候选池生成：整形/多样化/多票"""

import math
import json
import time
import hashlib
import uuid
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple
from itertools import combinations
from pathlib import Path

from src.common.paths import data_path
from src.common.repositories import doc_store
from src.common.logger import setup_logger

log = setup_logger('kl8')
from . import strategies as _strategies_mod

from .config import (
    KL8_DEFAULT_HISTORY, KL8_DRAW_COUNT, KL8_NUM_RANGE,
)


def _clean_pick_numbers(numbers, expected_len: int) -> List[int]:
    """Return a validated pick list, or [] when the pick is malformed."""
    if not isinstance(numbers, (list, tuple, set)):
        return []
    try:
        nums = [int(n) for n in numbers]
    except (TypeError, ValueError):
        return []
    if len(nums) != expected_len or len(set(nums)) != expected_len:
        return []
    if any(n < 1 or n > KL8_NUM_RANGE for n in nums):
        return []
    return nums


def _default_repeat_cap(target_size: int) -> int:
    """Default cap for numbers repeated from the previous draw.

    KL8 draws 20 out of 80, so each previous number has about a 25% chance to
    repeat. The cap allows normal overlap without letting repeats dominate.
    """
    if target_size <= 0:
        return 0
    return max(1, min(target_size, math.ceil(target_size * 0.40)))


def _adaptive_repeat_cap(history_data: List[Dict], target_size: int, lookback: int = 20) -> int:
    """Adapt the final repeat cap to recent draw-to-draw overlap.

    The static cap is intentionally conservative. When recent draws have a
    higher-than-normal overlap, small plays such as select 5/6 should not be
    forced to throw away otherwise strong candidates from the latest draw.
    """
    base_cap = _default_repeat_cap(target_size)
    if target_size <= 0 or len(history_data or []) < 2:
        return base_cap

    recent = history_data[:lookback + 1]
    overlaps = []
    for idx in range(len(recent) - 1):
        newer = set(recent[idx].get('numbers', []))
        older = set(recent[idx + 1].get('numbers', []))
        if newer and older:
            overlaps.append(len(newer & older))

    if not overlaps:
        return base_cap

    mean_overlap = sum(overlaps) / len(overlaps)
    if mean_overlap >= 6.5:
        adjustment = 0.15
    elif mean_overlap >= 5.5:
        adjustment = 0.10
    elif mean_overlap <= 3.5:
        adjustment = -0.10
    else:
        adjustment = 0.0

    ratio = max(0.25, min(0.55, 0.40 + adjustment))
    return max(1, min(target_size, math.ceil(target_size * ratio)))


def _adaptive_repeat_target(history_data: List[Dict], target_size: int, minimum: int = 0,
                            lookback: int = 20) -> Dict:
    """Map recent draw overlap to a suitable overlap inside a small pick.

    If adjacent draws recently overlap by r out of 20 numbers, a shape-matched
    pick of size n contains about n*r/20 previous-draw numbers. This only controls
    structure; it is not treated as evidence of predictive advantage.
    """
    overlaps = []
    recent = (history_data or [])[:lookback + 1]
    for idx in range(len(recent) - 1):
        newer = set(recent[idx].get('numbers', []))
        older = set(recent[idx + 1].get('numbers', []))
        if newer and older:
            overlaps.append(len(newer & older))
    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 5.0
    target = round(target_size * mean_overlap / KL8_DRAW_COUNT)
    cap = _adaptive_repeat_cap(history_data, target_size, lookback)
    target = max(int(minimum or 0), min(cap, target_size, target))
    return {
        'target': target,
        'cap': cap,
        'mean_draw_overlap': round(mean_overlap, 2),
        'sample_size': len(overlaps),
    }


def _enforce_minimum_repeats(
    selected: List[Tuple[int, float]],
    candidates: List[Tuple[int, float]],
    last_numbers: Optional[set],
    minimum: int,
) -> List[Tuple[int, float]]:
    """Ensure a final pick contains a small, explicit previous-draw overlap.

    This is a shape constraint, not a predictive edge. Replacements preserve the
    candidate ranking as much as possible: add the best missing repeat and remove
    the lowest-ranked non-repeat until the requested floor is met.
    """
    if not selected or minimum <= 0:
        return selected
    last_numbers = set(last_numbers or ())
    minimum = min(len(selected), int(minimum))
    result = list(selected)
    present = {num for num, _ in result}
    repeat_count = sum(1 for num, _ in result if num in last_numbers)
    if repeat_count >= minimum:
        return result

    replacements = [item for item in candidates if item[0] in last_numbers and item[0] not in present]
    while repeat_count < minimum and replacements:
        replacement = replacements.pop(0)
        removable_idx = next(
            (idx for idx in range(len(result) - 1, -1, -1) if result[idx][0] not in last_numbers),
            None,
        )
        if removable_idx is None:
            break
        present.discard(result[removable_idx][0])
        result[removable_idx] = replacement
        present.add(replacement[0])
        repeat_count += 1
    score_order = {num: idx for idx, (num, _) in enumerate(candidates)}
    return sorted(result, key=lambda item: score_order.get(item[0], len(score_order)))


def _diversify_candidate_pool(
    candidates: List[Tuple[int, float]],
    target_size: int,
    last_numbers: Optional[set] = None,
    max_last_numbers: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """Lightly diversify candidates when scores are close.

    KL8 naturally repeats numbers from the previous draw. This keeps score
    order as the first priority, then limits zone, 012-road and repeat
    concentration without trying to avoid repeats outright.
    """
    if target_size <= 0 or not candidates:
        return []

    last_numbers = last_numbers or set()
    max_zone = max(2, math.ceil(target_size / 16) + 1)
    max_road = max(3, math.ceil(target_size / 3) + 1)
    default_repeat_cap = _default_repeat_cap(target_size)
    max_repeat = default_repeat_cap if max_last_numbers is None else max(0, min(target_size, int(max_last_numbers)))

    selected = []
    protected = []
    deferred = []
    zone_counts = Counter()
    road_counts = Counter()
    repeat_count = 0
    best_score = float(candidates[0][1]) if candidates else 0.0
    score_floor = best_score * 0.90 if best_score > 0 else best_score - 0.10

    for num, score in candidates:
        zone = (num - 1) // 5 + 1
        road = num % 3
        is_repeat = num in last_numbers
        violates = (
            zone_counts[zone] >= max_zone
            or road_counts[road] >= max_road
            or (is_repeat and repeat_count >= max_repeat)
        )

        if not violates:
            selected.append((num, score))
            zone_counts[zone] += 1
            road_counts[road] += 1
            if is_repeat:
                repeat_count += 1
            if len(selected) >= target_size:
                return selected
        elif score >= score_floor:
            protected.append((num, score))
        else:
            deferred.append((num, score))

    seen = {num for num, _ in selected}
    fallback = protected + deferred
    fallback.sort(key=lambda item: (-item[1], item[0]))
    for num, score in fallback:
        if num not in seen:
            selected.append((num, score))
            seen.add(num)
        if len(selected) >= target_size:
            break

    return selected[:target_size]


def _zone_spread_candidate_pool(
    candidates: List[Tuple[int, float]],
    target_size: int,
) -> List[Tuple[int, float]]:
    if target_size <= 0 or not candidates:
        return []

    selected_nums = []
    zone_counts = Counter()
    max_zone = max(1, math.ceil(target_size / 16))
    score_lookup = {num: score for num, score in candidates}

    for num, _ in candidates[:max(target_size * 4, 20)]:
        zone = (num - 1) // 5 + 1
        if zone_counts[zone] >= max_zone:
            continue
        selected_nums.append(num)
        zone_counts[zone] += 1
        if len(selected_nums) >= target_size:
            break

    for num, _ in candidates:
        if len(selected_nums) >= target_size:
            break
        if num not in selected_nums:
            selected_nums.append(num)

    return [(num, score_lookup.get(num, 0.0)) for num in selected_nums[:target_size]]


def _prize_floor_candidate_pool(
    candidates: List[Tuple[int, float]],
    target_size: int,
    last_numbers: Optional[set] = None,
    max_last_numbers: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """Build a higher-variance pool for plays whose first prize tier is 3 hits.

    For select-6, taking the top six often just extends select-5 by one number.
    This keeps a strong core, then adds two later candidates with zone/road
    separation so the recommendation is not only a near-duplicate of select-5.
    """
    if target_size <= 0 or not candidates:
        return []
    last_numbers = last_numbers or set()
    repeat_cap = (
        _default_repeat_cap(target_size)
        if max_last_numbers is None
        else max(0, min(target_size, int(max_last_numbers)))
    )
    score_lookup = {num: score for num, score in candidates}

    if target_size <= 4:
        wing_count = 1
    elif target_size <= 7:
        wing_count = 2
    else:
        wing_count = max(2, math.ceil(target_size * 0.30))
    core_size = max(1, target_size - wing_count)
    selected = list(candidates[:core_size])
    selected_nums = {num for num, _ in selected}
    selected_zones = {(num - 1) // 5 for num, _ in selected}
    selected_roads = {num % 3 for num, _ in selected}
    repeat_count = sum(1 for num, _ in selected if num in last_numbers)

    wing_start = min(len(candidates), target_size)
    wing_candidates = candidates[wing_start:] + candidates[core_size:wing_start]
    scored_wings = []
    for idx, (num, score) in enumerate(wing_candidates):
        if num in selected_nums:
            continue
        zone = (num - 1) // 5
        road = num % 3
        repeat_penalty = 0.18 if num in last_numbers and repeat_count >= repeat_cap else 0.0
        spread_bonus = (0.08 if zone not in selected_zones else 0.0) + (0.04 if road not in selected_roads else 0.0)
        tail_bonus = 0.03 if idx < max(target_size, 8) else 0.0
        scored_wings.append((score + spread_bonus + tail_bonus - repeat_penalty, num, score))

    scored_wings.sort(key=lambda item: (-item[0], item[1]))
    for _, num, score in scored_wings:
        selected.append((num, score))
        selected_nums.add(num)
        selected_zones.add((num - 1) // 5)
        selected_roads.add(num % 3)
        if num in last_numbers:
            repeat_count += 1
        if len(selected) >= target_size:
            break

    if len(selected) < target_size:
        for num, score in candidates:
            if num not in selected_nums:
                selected.append((num, score))
                selected_nums.add(num)
            if len(selected) >= target_size:
                break

    return [(num, score_lookup.get(num, score)) for num, score in selected[:target_size]]


def _high_tier_chase_candidate_pool(
    candidates: List[Tuple[int, float]],
    target_size: int,
    last_numbers: Optional[set] = None,
    max_last_numbers: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """Concentrated high-tier pool for select 5/6 chase targets.

    To get close to 4/5 hits, the pick must behave like one strong cluster.
    This intentionally preserves top-ranked candidates and only uses the repeat
    cap as a soft fallback, avoiding the spread penalties used by balanced
    modes.
    """
    if target_size <= 0 or not candidates:
        return []

    last_numbers = last_numbers or set()
    repeat_cap = target_size if max_last_numbers is None else max(0, min(target_size, int(max_last_numbers)))
    selected = []
    deferred_repeats = []

    for num, score in candidates:
        if num in last_numbers and sum(1 for n, _ in selected if n in last_numbers) >= repeat_cap:
            deferred_repeats.append((num, score))
            continue
        selected.append((num, score))
        if len(selected) >= target_size:
            return selected

    seen = {num for num, _ in selected}
    for num, score in deferred_repeats + candidates:
        if num not in seen:
            selected.append((num, score))
            seen.add(num)
        if len(selected) >= target_size:
            break

    return selected[:target_size]


def _shape_targets(target_size: int) -> Dict:
    """Return neutral KL8 draw-shape targets scaled to the pick size."""
    zone_base, zone_rem = divmod(max(target_size, 0), 4)
    zone_targets = [zone_base + (1 if idx < zone_rem else 0) for idx in range(4)]
    half_low = target_size // 2
    half_high = target_size - half_low
    return {
        'zone20_targets': zone_targets,
        'odd_range': (half_low, half_high),
        'small_range': (half_low, half_high),
    }


def _shape_profile(numbers: List[int], last_numbers: Optional[set] = None) -> Dict:
    nums = sorted(int(n) for n in numbers)
    last_numbers = last_numbers or set()
    zone20 = [
        sum(1 for n in nums if 1 <= n <= 20),
        sum(1 for n in nums if 21 <= n <= 40),
        sum(1 for n in nums if 41 <= n <= 60),
        sum(1 for n in nums if 61 <= n <= 80),
    ]
    zone10 = [
        sum(1 for n in nums if start <= n <= start + 9)
        for start in range(1, 80, 10)
    ]
    odd = sum(1 for n in nums if n % 2 == 1)
    even = len(nums) - odd
    small = sum(1 for n in nums if n <= 40)
    big = len(nums) - small
    return {
        'zone20': zone20,
        'zone10': zone10,
        'odd_even': {'odd': odd, 'even': even},
        'big_small': {'small': small, 'big': big},
        'sum': sum(nums),
        'repeat_from_last': sum(1 for n in nums if n in last_numbers),
    }


def _shape_penalty(numbers: List[int], target_size: int, last_numbers: Optional[set], repeat_cap: int) -> float:
    """Lower is better. Penalize shapes that drift from common KL8 balance."""
    if len(numbers) != target_size:
        return 999.0

    profile = _shape_profile(numbers, last_numbers)
    targets = _shape_targets(target_size)
    penalty = 0.0

    penalty += sum(
        abs(actual - target)
        for actual, target in zip(profile['zone20'], targets['zone20_targets'])
    ) * 1.15

    odd = profile['odd_even']['odd']
    odd_low, odd_high = targets['odd_range']
    if odd < odd_low:
        penalty += (odd_low - odd) * 1.0
    elif odd > odd_high:
        penalty += (odd - odd_high) * 1.0

    small = profile['big_small']['small']
    small_low, small_high = targets['small_range']
    if small < small_low:
        penalty += (small_low - small) * 0.85
    elif small > small_high:
        penalty += (small - small_high) * 0.85

    repeat_count = profile['repeat_from_last']
    if repeat_count > repeat_cap:
        penalty += (repeat_count - repeat_cap) * 0.95

    return penalty


def _shape_balanced_candidate_pool(
    candidates: List[Tuple[int, float]],
    target_size: int,
    last_numbers: Optional[set] = None,
    max_last_numbers: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """Select a pool that keeps score order but favors normal KL8 shapes.

    The shape constraints are intentionally soft: four 20-number zones should
    stay near an even split, odd/even and small/big should stay near half, and
    repeats from the previous draw should stay close to the adaptive cap.
    """
    if target_size <= 0 or not candidates:
        return []

    last_numbers = last_numbers or set()
    repeat_cap = (
        _default_repeat_cap(target_size)
        if max_last_numbers is None
        else max(0, min(target_size, int(max_last_numbers)))
    )
    score_lookup = {num: score for num, score in candidates}
    selected: List[Tuple[int, float]] = []
    selected_nums = set()
    targets = _shape_targets(target_size)
    zone_caps = [max(1, target + 1) for target in targets['zone20_targets']]

    for num, score in candidates:
        if len(selected) >= target_size:
            break
        trial_nums = [n for n, _ in selected] + [num]
        profile = _shape_profile(trial_nums, last_numbers)
        zone_idx = (num - 1) // 20
        odd = profile['odd_even']['odd']
        small = profile['big_small']['small']
        repeat_count = profile['repeat_from_last']

        if profile['zone20'][zone_idx] > zone_caps[zone_idx]:
            continue
        if odd > targets['odd_range'][1] + 1:
            continue
        if small > targets['small_range'][1] + 1:
            continue
        if repeat_count > repeat_cap + 1:
            continue

        selected.append((num, score))
        selected_nums.add(num)

    if len(selected) < target_size:
        for num, score in candidates:
            if num not in selected_nums:
                selected.append((num, score))
                selected_nums.add(num)
            if len(selected) >= target_size:
                break

    # One deterministic improvement pass: replace weaker shape outliers when a
    # nearby-scored candidate materially improves the final structure.
    candidate_window = candidates[:max(target_size * 5, 30)]
    improved = True
    while improved:
        improved = False
        current_nums = [num for num, _ in selected]
        current_penalty = _shape_penalty(current_nums, target_size, last_numbers, repeat_cap)
        current_score = sum(score for _, score in selected)
        selected_set = {num for num, _ in selected}

        best_swap = None
        for out_num, out_score in list(selected):
            for in_num, in_score in candidate_window:
                if in_num in selected_set:
                    continue
                trial = [(n, s) for n, s in selected if n != out_num] + [(in_num, in_score)]
                trial_nums = [n for n, _ in trial]
                trial_penalty = _shape_penalty(trial_nums, target_size, last_numbers, repeat_cap)
                score_loss = max(0.0, current_score - sum(s for _, s in trial))
                gain = current_penalty - trial_penalty - score_loss * 0.10
                if gain > 0.35 and (best_swap is None or gain > best_swap[0]):
                    best_swap = (gain, out_num, in_num, in_score)

        if best_swap:
            _, out_num, in_num, in_score = best_swap
            selected = [(n, s) for n, s in selected if n != out_num]
            selected.append((in_num, score_lookup.get(in_num, in_score)))
            selected.sort(key=lambda item: (-item[1], item[0]))
            improved = True

    return [(num, score_lookup.get(num, score)) for num, score in selected[:target_size]]


def _score_candidate_selection(
    pool: List[Tuple[int, float]],
    candidates: List[Tuple[int, float]],
    target_size: int,
    last_numbers: Optional[set],
    repeat_cap: int,
) -> float:
    if not pool or target_size <= 0:
        return -1.0

    top_score = float(candidates[0][1]) if candidates else 1.0
    if top_score == 0:
        top_score = 1.0
    avg_score = sum(float(score) for _, score in pool) / max(len(pool), 1)
    score_ratio = avg_score / max(abs(top_score), 1e-9)

    nums = [num for num, _ in pool]
    zone_count = len({(num - 1) // 5 for num in nums})
    road_count = len({num % 3 for num in nums})
    repeat_count = sum(1 for num in nums if num in (last_numbers or set()))
    repeat_penalty = max(0, repeat_count - repeat_cap) / max(target_size, 1)
    shape_penalty = _shape_penalty(nums, target_size, last_numbers, repeat_cap) / max(target_size, 1)

    zone_score = zone_count / min(target_size, 16)
    road_score = road_count / min(target_size, 3)
    shape_score = max(0.0, 1.0 - shape_penalty / 2.0)
    return (
        score_ratio * 0.60
        + zone_score * 0.10
        + road_score * 0.08
        + shape_score * 0.18
        - repeat_penalty * 0.16
    )


def _select_final_candidate_pool(
    candidates: List[Tuple[int, float]],
    target_size: int,
    last_numbers: Optional[set] = None,
    max_last_numbers: Optional[int] = None,
    selection_mode: str = 'balanced',
) -> Tuple[List[Tuple[int, float]], str]:
    if target_size <= 0 or not candidates:
        return [], selection_mode

    last_numbers = last_numbers or set()
    repeat_cap = (
        _default_repeat_cap(target_size)
        if max_last_numbers is None
        else max(0, min(target_size, int(max_last_numbers)))
    )

    modes = {
        'top_ranked': candidates[:target_size],
        'concentrated': candidates[:target_size],
        'high_tier_chase': _high_tier_chase_candidate_pool(
            candidates,
            target_size,
            last_numbers,
            max_last_numbers=repeat_cap,
        ),
        'balanced': _diversify_candidate_pool(
            candidates,
            target_size,
            last_numbers,
            max_last_numbers=repeat_cap,
        ),
        'diversified': _diversify_candidate_pool(
            candidates,
            target_size,
            last_numbers,
            max_last_numbers=repeat_cap,
        ),
        'low_repeat': _diversify_candidate_pool(
            candidates,
            target_size,
            last_numbers,
            max_last_numbers=max(0, repeat_cap - 1),
        ),
        'repeat_follow': _diversify_candidate_pool(
            candidates,
            target_size,
            last_numbers,
            max_last_numbers=min(target_size, repeat_cap + 1),
        ),
        'zone_spread': _zone_spread_candidate_pool(candidates, target_size),
        'prize_floor': _prize_floor_candidate_pool(
            candidates,
            target_size,
            last_numbers,
            max_last_numbers=repeat_cap,
        ),
        'shape_balanced': _shape_balanced_candidate_pool(
            candidates,
            target_size,
            last_numbers,
            max_last_numbers=repeat_cap,
        ),
    }

    if selection_mode in modes and selection_mode != 'best_variant':
        return modes[selection_mode], selection_mode

    scored = []
    for mode, pool in modes.items():
        if selection_mode == 'best_variant' and mode == 'low_repeat':
            continue
        scored.append((
            _score_candidate_selection(pool, candidates, target_size, last_numbers, repeat_cap),
            mode,
            pool,
        ))
    scored.sort(key=lambda item: (-item[0], item[1]))
    _, mode, pool = scored[0]
    return pool, mode


def generate_multi_slips(analyzer: 'KL8Analyzer', select_n: int, n_slips: int = 8,
                          pick_size: int = None) -> List[List[int]]:
    """生成 n_slips 组差异化选号集合（v9.6: 覆盖最大化结构）。

    背景：对公平摇奖，单号没有预测 edge，因此本函数不再通过“特征扰动”假装产生
    不同视角，而是直接构造覆盖最大化的号码组合：
    (1) 取完整 80 码排名（当前为公平确定性随机，未来若验证出有效策略也可复用）；
    (2) 第 0 组为排名最前的 pick_size 个号码，与主推号码保持一致；
    (3) 后续组按排名顺序依次取不重叠的号码块，在 12x6 配置下可覆盖 72 个号码；
    (4) 总槽位超过 80 时循环复用排名靠前号码。

    数学效果（公平摇奖假设）：12 组 x 6 码 disjoint 结构的最佳组命中 ≥3 约 95%、
    ≥4 约 35%，显著优于当前 12 组都挤在同一 top40 排名里的相关结构（≥3 约 82%、
    ≥4 约 25%）。这是组合层面的覆盖率杠杆，不改变任何单号的理论开出概率。

    参数：
      select_n : 玩法选号数（决定用哪套策略，如 6=选6）。
      n_slips  : 生成的组数。
      pick_size: 每组输出的号码个数，默认=select_n。设成 >select_n（如7）可让
                 每组覆盖更多号码，适配“选5复式”等玩法。

    返回 list（长度=n_slips），每个元素为排序后的号码列表。
    """
    if pick_size is None:
        pick_size = select_n
    s_key = f'select_{select_n}'
    strategy = _strategies_mod.resolve_play_strategy(s_key, allow_reference=True)
    if strategy is None:
        return []
    base_weights = {k: float(v) for k, v in (strategy.get('feature_weights') or {}).items() if v}
    if not base_weights:
        return []

    window = strategy.get('window_size', KL8_DEFAULT_HISTORY)
    predictor = analyzer._build_window_analyzer(window)
    repeat_direction = strategy.get('repeat_direction', 'neutral')
    frequency_mode = strategy.get('frequency_mode', 'mean_reversion')

    # 取完整 80 码排名，按排名顺序切分为连续的 disjoint 块 → 覆盖最大化
    ranking = predictor.get_ensemble_ranking(
        top_n=KL8_NUM_RANGE, feature_weights=base_weights,
        repeat_direction=repeat_direction, frequency_mode=frequency_mode,
    )
    if len(ranking) < pick_size:
        return []

    ordered = [it['num'] for it in ranking]
    total_slots = n_slips * pick_size

    if total_slots <= len(ordered):
        coverage = ordered[:total_slots]
    else:
        # 全 80 码仍不够时循环复用排名靠前的号码
        coverage = ordered[:]
        idx = 0
        while len(coverage) < total_slots:
            coverage.append(ordered[idx % len(ordered)])
            idx += 1

    slips = [sorted(coverage[i * pick_size:(i + 1) * pick_size]) for i in range(n_slips)]
    return slips


def _simulate_multi_slip_coverage(
    slips: List[List[int]],
    simulations: int = 12000,
    seed_key: str = '',
) -> Dict:
    """Estimate portfolio hit rates using the tickets' actual overlap.

    Unlike ``1-(1-p)^n``, this simulation does not assume tickets are
    independent.  It samples fair 20-of-80 draws with a deterministic seed so
    the same prediction always renders the same stable figures.
    """
    import random as _rng

    clean = [set(int(n) for n in slip) for slip in (slips or []) if slip]
    if not clean or simulations <= 0:
        return {}
    rng = _rng.Random(f'kl8_coverage_v1_{seed_key}')
    ge3 = ge4 = ge5 = ge6 = total_best = 0
    for _ in range(simulations):
        draw = set(rng.sample(range(1, KL8_NUM_RANGE + 1), KL8_DRAW_COUNT))
        best = max(len(ticket & draw) for ticket in clean)
        total_best += best
        ge3 += int(best >= 3)
        ge4 += int(best >= 4)
        ge5 += int(best >= 5)
        ge6 += int(best >= 6)

    overlaps = [
        len(clean[i] & clean[j])
        for i in range(len(clean))
        for j in range(i + 1, len(clean))
    ]
    return {
        'method': 'deterministic_monte_carlo_actual_overlap',
        'simulations': simulations,
        'at_least_one_ge3': round(ge3 / simulations, 6),
        'at_least_one_ge4': round(ge4 / simulations, 6),
        'at_least_one_ge5': round(ge5 / simulations, 6),
        'at_least_one_ge6': round(ge6 / simulations, 6),
        'average_best_hits': round(total_best / simulations, 4),
        'unique_number_count': len(set().union(*clean)),
        'max_pair_overlap': max(overlaps, default=0),
        'average_pair_overlap': round(sum(overlaps) / len(overlaps), 3) if overlaps else 0.0,
    }


