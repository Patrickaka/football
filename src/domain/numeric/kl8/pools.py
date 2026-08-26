"""从候选序列里挑出最终推荐的那几个号。

输入是投票排好序的 `[(号码, 票数), ...]`，输出是同样形状的一个子集。中间隔着
七种挑法，差别只在「除了票数还看什么」：`concentrated` 什么都不看，直接取前
若干；`zone_spread` 只管区间铺开；`shape_balanced` 逐个试算形态。**它们都不
改变任何号码的中奖概率**，只决定推荐看起来是集中还是分散。

**重号（与上一期重复的号）是这里反复出现的那条轴。** 80 选 20 意味着每个号
有 25% 的概率连续开出，所以重号既不该被当成信号追，也不该一律回避——几个
上限函数做的都是「允许正常比例的重号，但别让它主导」。
"""
import math
from collections import Counter

from src.domain.numeric.kl8 import shape
from src.domain.numeric.kl8.space import DRAW_COUNT, SPACE

ZONE_SIZE = 5        # 16 个 5 码区，与评分里的 position_residual 同粒度
ROAD_MODULUS = 3     # 012 路

# 重号占比的基准与浮动区间。0.40 比理论的 0.25 宽，是留给「最近确实重得多」
# 的余地；再宽就等于按上期抄作业了。
BASE_REPEAT_RATIO = 0.40
MIN_REPEAT_RATIO = 0.25
MAX_REPEAT_RATIO = 0.55
REPEAT_LOOKBACK = 20


def clean_pick_numbers(numbers, expected_len):
    """校验一组号码，不合格返回空列表。

    个数不对、有重复、越界、非整数——任何一条不满足都判空，**不做部分修补**。
    修补出来的那几个号会被当成一次真实推荐记进命中率统计。
    """
    if not isinstance(numbers, (list, tuple, set)):
        return []
    try:
        nums = [int(n) for n in numbers]
    except (TypeError, ValueError):
        return []
    if len(nums) != expected_len or len(set(nums)) != expected_len:
        return []
    if any(not SPACE.contains(n) for n in nums):
        return []
    return nums


# ─── 重号上限 ───

def default_repeat_cap(target_size):
    """静态上限：不超过选号数的 40%，且至少留一个位置给重号。"""
    if target_size <= 0:
        return 0
    return max(1, min(target_size, math.ceil(target_size * BASE_REPEAT_RATIO)))


def adaptive_repeat_cap(history_data, target_size, lookback=REPEAT_LOOKBACK):
    """按最近相邻两期的实际重合度调整上限。

    静态上限是刻意保守的。最近确实重得多的时候，选 5、选 6 这种小盘不该被迫
    把上期开出的强号全扔掉。
    """
    base_cap = default_repeat_cap(target_size)
    if target_size <= 0 or len(history_data or []) < 2:
        return base_cap

    overlaps = _adjacent_overlaps(history_data, lookback)
    if not overlaps:
        return base_cap

    mean_overlap = sum(overlaps) / len(overlaps)
    ratio = max(MIN_REPEAT_RATIO,
                min(MAX_REPEAT_RATIO, BASE_REPEAT_RATIO + _ratio_adjustment(mean_overlap)))
    return max(1, min(target_size, math.ceil(target_size * ratio)))


def _ratio_adjustment(mean_overlap):
    """重合度分档加减。20 个号里平均重 5 个是常态，档位围绕它设。"""
    if mean_overlap >= 6.5:
        return 0.15
    if mean_overlap >= 5.5:
        return 0.10
    if mean_overlap <= 3.5:
        return -0.10
    return 0.0


def adaptive_repeat_target(history_data, target_size, minimum=0, lookback=REPEAT_LOOKBACK):
    """一注里「应该」含几个重号。

    相邻两期若重合 r 个（共开 20 个），那么 n 个号的一注按同样比例含
    n*r/20 个重号。**这只用来定结构，不当成预测优势。**
    """
    overlaps = _adjacent_overlaps(history_data, lookback)
    # 没有样本时退回理论值：80 选 20，期望重合 20*0.25 = 5
    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else DRAW_COUNT * 0.25
    cap = adaptive_repeat_cap(history_data, target_size, lookback)
    target = round(target_size * mean_overlap / DRAW_COUNT)
    return {
        'target': max(int(minimum or 0), min(cap, target_size, target)),
        'cap': cap,
        'mean_draw_overlap': round(mean_overlap, 2),
        'sample_size': len(overlaps),
    }


def _adjacent_overlaps(history_data, lookback):
    """相邻两期重合了几个号。任一期为空就跳过，别把缺数据算成「重合 0 个」。"""
    recent = (history_data or [])[:lookback + 1]
    overlaps = []
    for idx in range(len(recent) - 1):
        newer = set(recent[idx].get('numbers', []))
        older = set(recent[idx + 1].get('numbers', []))
        if newer and older:
            overlaps.append(len(newer & older))
    return overlaps


def _resolved_cap(target_size, max_last_numbers):
    """没指定就用静态上限；指定了就夹到 [0, 选号数] 之间。"""
    if max_last_numbers is None:
        return default_repeat_cap(target_size)
    return max(0, min(target_size, int(max_last_numbers)))


def enforce_minimum_repeats(selected, candidates, last_numbers, minimum):
    """保证一注里含够指定个数的重号。

    这是形态约束，不是预测优势。换号时尽量不动排名：补进最靠前的缺失重号，
    换掉最靠后的非重号。
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

    replacements = [item for item in candidates
                    if item[0] in last_numbers and item[0] not in present]
    while repeat_count < minimum and replacements:
        replacement = replacements.pop(0)
        removable_idx = next((idx for idx in range(len(result) - 1, -1, -1)
                              if result[idx][0] not in last_numbers), None)
        if removable_idx is None:
            break
        present.discard(result[removable_idx][0])
        result[removable_idx] = replacement
        present.add(replacement[0])
        repeat_count += 1
    score_order = {num: idx for idx, (num, _) in enumerate(candidates)}
    return sorted(result, key=lambda item: score_order.get(item[0], len(score_order)))


# ─── 七种挑法 ───

def concentrated(candidates, target_size, last_numbers=None, max_last_numbers=None):
    """直接取票数最高的前若干个。什么结构都不管。"""
    if target_size <= 0:
        return []
    return list(candidates[:target_size])


def diversify(candidates, target_size, last_numbers=None, max_last_numbers=None):
    """票数优先，再限制区间、012 路与重号的扎堆程度。

    kl8 本来就常带出上期的号，所以这里不追求回避重号，只是不让任何一种集中
    走到极端。**票数仍是第一优先**——被上限挡下的高分号进「保护区」，
    补位时优先回来，不会因为一次撞限就永远出局。
    """
    if target_size <= 0 or not candidates:
        return []

    last_numbers = last_numbers or set()
    max_zone = max(2, math.ceil(target_size / 16) + 1)
    max_road = max(3, math.ceil(target_size / 3) + 1)
    max_repeat = _resolved_cap(target_size, max_last_numbers)

    selected, protected, deferred = [], [], []
    zone_counts, road_counts = Counter(), Counter()
    repeat_count = 0
    best_score = float(candidates[0][1])
    # 与最高分差一成以内的算「保护区」：它们只是撞了结构上限，不是分低
    score_floor = best_score * 0.90 if best_score > 0 else best_score - 0.10

    for num, score in candidates:
        zone, road = _zone_of(num), _road_of(num)
        is_repeat = num in last_numbers
        violates = (zone_counts[zone] >= max_zone
                    or road_counts[road] >= max_road
                    or (is_repeat and repeat_count >= max_repeat))
        if not violates:
            selected.append((num, score))
            zone_counts[zone] += 1
            road_counts[road] += 1
            repeat_count += int(is_repeat)
            if len(selected) >= target_size:
                return selected
        elif score >= score_floor:
            protected.append((num, score))
        else:
            deferred.append((num, score))

    # 补位前重排：保护区与放弃区是分别攒的，直接拼起来顺序就乱了，
    # 高分号会排在低分号后面。
    fallback = sorted(protected + deferred, key=lambda item: (-item[1], item[0]))
    return _fill_up(selected, fallback, target_size)


def zone_spread(candidates, target_size, last_numbers=None, max_last_numbers=None):
    """只管一件事：别让号码挤在同几个 5 码区里。"""
    if target_size <= 0 or not candidates:
        return []

    score_lookup = dict(candidates)
    max_zone = max(1, math.ceil(target_size / 16))
    zone_counts = Counter()
    picked = []
    # 只在靠前的一段里挑。翻遍整条候选就等于把区间限制作废了
    for num, _ in candidates[:max(target_size * 4, 20)]:
        if zone_counts[_zone_of(num)] >= max_zone:
            continue
        picked.append(num)
        zone_counts[_zone_of(num)] += 1
        if len(picked) >= target_size:
            break

    for num, _ in candidates:
        if len(picked) >= target_size:
            break
        if num not in picked:
            picked.append(num)

    return [(num, score_lookup.get(num, 0.0)) for num in picked[:target_size]]


def prize_floor(candidates, target_size, last_numbers=None, max_last_numbers=None):
    """给「中三个就有奖」的玩法留一点方差。

    选 6 直接取前六，往往只是选 5 多带一个号，两注实质是同一注。这里保住一个
    强核心，再挑两个区间与路数都岔开的靠后号码，让推荐不至于互为副本。
    """
    if target_size <= 0 or not candidates:
        return []
    last_numbers = last_numbers or set()
    repeat_cap = _resolved_cap(target_size, max_last_numbers)
    score_lookup = dict(candidates)

    wing_count = _wing_count(target_size)
    core_size = max(1, target_size - wing_count)
    selected = list(candidates[:core_size])
    selected_nums = {num for num, _ in selected}
    selected_zones = {_zone_of(num) for num, _ in selected}
    selected_roads = {_road_of(num) for num, _ in selected}
    repeat_count = sum(1 for num, _ in selected if num in last_numbers)

    for _, num, score in _scored_wings(candidates, target_size, core_size, selected_nums,
                                       selected_zones, selected_roads, last_numbers,
                                       repeat_count, repeat_cap):
        selected.append((num, score))
        selected_nums.add(num)
        selected_zones.add(_zone_of(num))
        selected_roads.add(_road_of(num))
        repeat_count += int(num in last_numbers)
        if len(selected) >= target_size:
            break

    selected = _fill_up(selected, candidates, target_size)
    return [(num, score_lookup.get(num, score)) for num, score in selected[:target_size]]


def _wing_count(target_size):
    """核心之外留几个「岔开」的位置。小盘留不起太多，大盘按三成留。"""
    if target_size <= 4:
        return 1
    if target_size <= 7:
        return 2
    return max(2, math.ceil(target_size * 0.30))


def _scored_wings(candidates, target_size, core_size, selected_nums,
                  selected_zones, selected_roads, last_numbers,
                  repeat_count, repeat_cap):
    """给候选里的「侧翼」重新打分：岔开有加分，超额重号有减分。"""
    wing_start = min(len(candidates), target_size)
    # 先看名次在选号数之外的，再回头看核心与它之间那一段
    wing_candidates = candidates[wing_start:] + candidates[core_size:wing_start]
    scored = []
    for idx, (num, score) in enumerate(wing_candidates):
        if num in selected_nums:
            continue
        repeat_penalty = 0.18 if num in last_numbers and repeat_count >= repeat_cap else 0.0
        spread_bonus = ((0.08 if _zone_of(num) not in selected_zones else 0.0)
                        + (0.04 if _road_of(num) not in selected_roads else 0.0))
        tail_bonus = 0.03 if idx < max(target_size, 8) else 0.0
        scored.append((score + spread_bonus + tail_bonus - repeat_penalty, num, score))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


def high_tier_chase(candidates, target_size, last_numbers=None, max_last_numbers=None):
    """冲高奖档时用的集中池。

    要接近中 4、中 5，一注得表现得像一个强簇，所以这里**不加任何岔开惩罚**，
    重号上限也只当软兜底：撞限的号先放一边，位置不够时照样回来。
    """
    if target_size <= 0 or not candidates:
        return []

    last_numbers = last_numbers or set()
    # 与别处不同：没指定上限就等于不限。集中是这个模式的全部目的。
    repeat_cap = (target_size if max_last_numbers is None
                  else max(0, min(target_size, int(max_last_numbers))))
    selected, deferred = [], []

    for num, score in candidates:
        over_cap = sum(1 for n, _ in selected if n in last_numbers) >= repeat_cap
        if num in last_numbers and over_cap:
            deferred.append((num, score))
            continue
        selected.append((num, score))
        if len(selected) >= target_size:
            return selected

    return _fill_up(selected, deferred + list(candidates), target_size)


def shape_balanced(candidates, target_size, last_numbers=None, max_last_numbers=None):
    """票数优先，但让最终形态贴近一期正常开奖的样子。

    约束是软的：四个大区别太偏、奇偶与大小别一边倒、重号别超过上限太多。
    先按票数逐个试着放，再做一轮**确定性**的交换改良——交换必须同时满足
    「形态改善够多」与「分数损失够小」，否则不换。
    """
    if target_size <= 0 or not candidates:
        return []

    last_numbers = last_numbers or set()
    repeat_cap = _resolved_cap(target_size, max_last_numbers)
    score_lookup = dict(candidates)
    selected = _shape_first_pass(candidates, target_size, last_numbers, repeat_cap)
    selected = _fill_up(selected, candidates, target_size)
    selected = _shape_improve(selected, candidates, target_size, last_numbers,
                              repeat_cap, score_lookup)
    return [(num, score_lookup.get(num, score)) for num, score in selected[:target_size]]


def _shape_first_pass(candidates, target_size, last_numbers, repeat_cap):
    """按票数顺序放号，放下去会让形态越界的就跳过。

    界限都留了 1 的余量：卡得刚好会让高分号被形态挤掉太多，得不偿失。
    """
    target = targets_with_caps(target_size)
    selected = []
    for num, score in candidates:
        if len(selected) >= target_size:
            break
        trial = [n for n, _ in selected] + [num]
        actual = shape.profile(trial, last_numbers)
        if actual['zone20'][_zone20_of(num)] > target['zone_caps'][_zone20_of(num)]:
            continue
        if actual['odd_even']['odd'] > target['odd_range'][1] + 1:
            continue
        if actual['big_small']['small'] > target['small_range'][1] + 1:
            continue
        if actual['repeat_from_last'] > repeat_cap + 1:
            continue
        selected.append((num, score))
    return selected


def targets_with_caps(target_size):
    """中性形态目标，外加每个大区的硬上限（目标 + 1）。"""
    target = dict(shape.targets(target_size))
    target['zone_caps'] = [max(1, want + 1) for want in target['zone20_targets']]
    return target


def _shape_improve(selected, candidates, target_size, last_numbers, repeat_cap, score_lookup):
    """反复找最划算的一次交换，直到没有划算的为止。

    只在靠前的一段候选里找替补——翻遍整条会让低分号仅凭形态就挤进来。
    """
    window = candidates[:max(target_size * 5, 30)]
    while True:
        swap = _best_swap(selected, window, target_size, last_numbers, repeat_cap)
        if swap is None:
            return selected
        _, out_num, in_num, in_score = swap
        selected = [(n, s) for n, s in selected if n != out_num]
        selected.append((in_num, score_lookup.get(in_num, in_score)))
        selected.sort(key=lambda item: (-item[1], item[0]))


# 形态收益要盖过这个门槛才换。太低会让选号跟着形态反复抖动。
SWAP_GAIN_THRESHOLD = 0.35
# 分数损失折算成形态收益的比例。形态是次要目标，所以折得很狠。
SCORE_LOSS_WEIGHT = 0.10


def _best_swap(selected, window, target_size, last_numbers, repeat_cap):
    """在所有「换出一个、换进一个」里挑收益最大的那次，都不划算就返回 None。"""
    current_nums = [num for num, _ in selected]
    current_penalty = shape.penalty(current_nums, target_size, last_numbers, repeat_cap)
    current_score = sum(score for _, score in selected)
    selected_set = set(current_nums)

    best = None
    for out_num, _ in list(selected):
        for in_num, in_score in window:
            if in_num in selected_set:
                continue
            trial = [(n, s) for n, s in selected if n != out_num] + [(in_num, in_score)]
            trial_penalty = shape.penalty([n for n, _ in trial], target_size,
                                          last_numbers, repeat_cap)
            score_loss = max(0.0, current_score - sum(s for _, s in trial))
            gain = current_penalty - trial_penalty - score_loss * SCORE_LOSS_WEIGHT
            if gain > SWAP_GAIN_THRESHOLD and (best is None or gain > best[0]):
                best = (gain, out_num, in_num, in_score)
    return best


def _fill_up(selected, extras, target_size):
    """位置没填满时按给定顺序补，已在池里的跳过。"""
    if len(selected) >= target_size:
        return selected[:target_size]
    result = list(selected)
    seen = {num for num, _ in result}
    for num, score in extras:
        if num not in seen:
            result.append((num, score))
            seen.add(num)
        if len(result) >= target_size:
            break
    return result[:target_size]


def _zone_of(number):
    return (number - 1) // ZONE_SIZE + 1


def _zone20_of(number):
    return (number - 1) // shape.ZONE20_SIZE


def _road_of(number):
    return number % ROAD_MODULUS


# ─── 最终选池 ───

# 每种模式怎么建。值是 (建池函数, 重号上限怎么改) —— 三种 diversify 变体的
# 差别**只有重号上限**，写成同一个函数加一个偏移，接反的可能性就只剩一处。
MODE_BUILDERS = {
    'top_ranked': (concentrated, None),
    'concentrated': (concentrated, None),
    'high_tier_chase': (high_tier_chase, 0),
    'balanced': (diversify, 0),
    'diversified': (diversify, 0),
    'low_repeat': (diversify, -1),
    'repeat_follow': (diversify, +1),
    'zone_spread': (zone_spread, None),
    'prize_floor': (prize_floor, 0),
    'shape_balanced': (shape_balanced, 0),
}

BEST_VARIANT = 'best_variant'
# 挑最优变体时不考虑 low_repeat：它是刻意压重号的偏好档，不该在「哪个更好」
# 的比较里胜出——那样等于把一个偏好悄悄变成默认。
BEST_VARIANT_EXCLUDES = ('low_repeat',)


def build_pool(mode, candidates, target_size, last_numbers, repeat_cap):
    """按名字建一个候选池。名字不认识时抛错——调用方应当先查 `MODE_BUILDERS`。"""
    builder, cap_offset = MODE_BUILDERS[mode]
    cap = repeat_cap if cap_offset is None else _shifted_cap(repeat_cap, cap_offset, target_size)
    return builder(candidates, target_size, last_numbers, max_last_numbers=cap)


def _shifted_cap(repeat_cap, offset, target_size):
    if offset == 0:
        return repeat_cap
    if offset < 0:
        return max(0, repeat_cap + offset)
    return min(target_size, repeat_cap + offset)


def select_final(candidates, target_size, last_numbers=None,
                 max_last_numbers=None, selection_mode='balanced'):
    """定下最终这一注，返回 (号码池, 实际用的模式)。

    **只建被点名的那一种池。** 迁移前是十种全建好再挑一种用，其中
    `shape_balanced` 一家就占一次预测的 15%，而线上十一个玩法请求的都是
    `concentrated`——最便宜的那个。

    模式名不认识（包括传 `None`）时**不报错，而是走「挑最优变体」那条路**。
    这是迁移前就有的行为，且线上真实可达：策略试验表里 2433 条记录的
    `final_selection_mode` 就是 `None`。返回值第二项报的是实际用的模式，
    所以它不是无声的——但它确实让一个拼错的模式名看起来像是生效了。
    """
    if target_size <= 0 or not candidates:
        return [], selection_mode

    last_numbers = last_numbers or set()
    repeat_cap = _resolved_cap(target_size, max_last_numbers)

    if selection_mode in MODE_BUILDERS:
        return build_pool(selection_mode, candidates, target_size,
                          last_numbers, repeat_cap), selection_mode

    excludes = BEST_VARIANT_EXCLUDES if selection_mode == BEST_VARIANT else ()
    return _best_scoring_pool(candidates, target_size, last_numbers, repeat_cap, excludes)


def _best_scoring_pool(candidates, target_size, last_numbers, repeat_cap, excludes):
    """把每种池都建出来打分，取最高的。并列按模式名排序，保证可复现。"""
    scored = []
    for mode in MODE_BUILDERS:
        if mode in excludes:
            continue
        pool = build_pool(mode, candidates, target_size, last_numbers, repeat_cap)
        scored.append((score_selection(pool, candidates, target_size, last_numbers, repeat_cap),
                       mode, pool))
    scored.sort(key=lambda item: (-item[0], item[1]))
    _, mode, pool = scored[0]
    return pool, mode


# 打分的五项权重。票数占六成——结构再好也不能盖过排名本身。
SCORE_WEIGHTS = {'score': 0.60, 'zone': 0.10, 'road': 0.08, 'shape': 0.18, 'repeat': 0.16}


def score_selection(pool, candidates, target_size, last_numbers, repeat_cap):
    """给一个候选池打分，越高越好。空池给 -1，让它永远不会被选中。"""
    if not pool or target_size <= 0:
        return -1.0

    top_score = float(candidates[0][1]) if candidates else 1.0
    if top_score == 0:
        top_score = 1.0
    avg_score = sum(float(score) for _, score in pool) / max(len(pool), 1)
    score_ratio = avg_score / max(abs(top_score), 1e-9)

    nums = [num for num, _ in pool]
    zone_score = len({_zone_of(n) for n in nums}) / min(target_size, 16)
    road_score = len({_road_of(n) for n in nums}) / min(target_size, ROAD_MODULUS)
    repeat_count = sum(1 for n in nums if n in (last_numbers or set()))
    repeat_ratio = max(0, repeat_count - repeat_cap) / max(target_size, 1)
    shape_ratio = shape.penalty(nums, target_size, last_numbers, repeat_cap) / max(target_size, 1)
    shape_score = max(0.0, 1.0 - shape_ratio / 2.0)

    return (score_ratio * SCORE_WEIGHTS['score']
            + zone_score * SCORE_WEIGHTS['zone']
            + road_score * SCORE_WEIGHTS['road']
            + shape_score * SCORE_WEIGHTS['shape']
            - repeat_ratio * SCORE_WEIGHTS['repeat'])
