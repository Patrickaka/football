"""给全部 1000 注打分排序，选出推荐的那几注。

**直选的中奖概率恒定 1/1000，这里没有任何一项能改变它。** 打分决定的是
推荐哪几注、以及它们看起来像不像一次真实开奖（和值、跨度、形态、冷热分布）。
`rank_triplets` 的几个开关（探索、扰动、冷热平衡、去相关）都是「让推荐更好看
或更轮换」的手段，不是提高命中的手段。

**打分与分解是同一份实现。** 迁移前它们是两个函数各写一遍全部特征——
结果 `triplet_weight_detail` 漏了形态先验那一项，用户看到的「为什么推荐这注」
加起来对不上旁边的分数（组六差 4.32 分）。合成一处之后，加一个特征只能
同时进两边。
"""
import random
from collections import Counter
from dataclasses import dataclass, field
from itertools import product

from src.domain.numeric.lottery3d import draw as draw_props
from src.domain.numeric.lottery3d import history, slope, weights as W
from src.domain.numeric.lottery3d.space import DIGIT_SPACE, POSITIONS


@dataclass(frozen=True)
class TripletContext:
    """打一整轮分所需的、**与具体候选无关**的东西。

    建一次、用一千次。迁移前 `form_switch` 与 `sum_interval` 是在逐注循环里
    各算一遍的——它们只依赖历史，一千注算出来的是同一个值。
    """

    weights: W.TripletWeights
    flags: dict
    danma: frozenset = frozenset()
    kill: frozenset = frozenset()
    form_switch: dict = field(default_factory=dict)
    sum_interval: dict = field(default_factory=dict)


def build_context(meta, weights, flags, danma=(), kill=(),
                  form_switch=None, sum_interval=None):
    return TripletContext(
        weights=weights, flags=flags,
        danma=frozenset(danma or ()), kill=frozenset(kill or ()),
        form_switch=form_switch or {}, sum_interval=sum_interval or {})


def _term_base(triple, score, meta, context):
    """三位基础分：分位评分为主、全局评分为辅。没有分位分时退回纯全局分。"""
    position_scores = meta.get('pos_scores')
    if position_scores and len(position_scores) == POSITIONS:
        return (context.weights.triplet_position
                * sum(position_scores[i][triple[i]] for i in range(POSITIONS))
                + context.weights.triplet_global
                * sum(score[digit] for digit in triple))
    return sum(score[digit] for digit in triple)


def _term_danma(triple, context):
    return context.weights.danma_hit * sum(1 for d in triple if d in context.danma)


def _term_kill(triple, context):
    return -context.weights.kill_penalty * sum(1 for d in triple if d in context.kill)


def _term_sum_span(triple, meta, context):
    """和值与跨度：离中心越近分越高，落在热区再加一份固定分。"""
    if not context.flags.get('sum_span', True):
        return 0.0
    total = sum(triple)
    span = draw_props.span(triple)
    value = (W.SUM_GAUSSIAN_WEIGHT
             * history.gaussian_score(total, meta['sum_center'],
                                      context.weights.sum_sigma)
             + W.SPAN_GAUSSIAN_WEIGHT
             * history.gaussian_score(span, meta['span_center'],
                                      context.weights.span_sigma))
    if total in meta['hot_sum_set']:
        value += W.HOT_SUM_BONUS
    if span in meta['hot_span_set']:
        value += W.HOT_SPAN_BONUS
    if (total % 10) in meta['sum_tail_top']:
        value += W.HOT_SUM_TAIL_BONUS
    return value


def _term_pattern(triple, meta, context):
    if not (context.flags.get('consecutive', True)
            and draw_props.has_consecutive_digits(*triple)):
        return 0.0
    dynamic = meta.get('dynamic') or {}
    return dynamic.get('w_consecutive', context.weights.consecutive)


def _term_last_repeat(triple, meta, context):
    """与上期的关系：同位相同加分，整注照抄或同一组数字重罚。

    加分与惩罚方向相反是有意的：**带出一两个上期的号很正常，整注照抄不是**。
    """
    if not context.flags.get('lag1_repeat', True):
        return 0.0
    last = meta.get('last_draw')
    if not last:
        return 0.0

    dynamic = meta.get('dynamic') or {}
    per_position = dynamic.get('w_pos_repeat', context.weights.position_repeat)
    multiplier = dynamic.get('pos_mult', [1.0] * POSITIONS)
    value = sum(per_position * multiplier[i]
                for i in range(POSITIONS) if triple[i] == last[i])

    if tuple(triple) == tuple(last):
        value -= dynamic.get('w_full_repeat_penalty', 0.0)
    elif set(triple) == set(last):
        value -= dynamic.get('w_same_set_penalty', 0.0)
    return value


def _term_ratio(triple, meta, context):
    """奇偶比与大小比和近期分布的吻合度。按占比给分，不是按次数。"""
    if not context.flags.get('ratio', True):
        return 0.0
    value = 0.0
    for key_fn, freq_key, total_key in (
            (draw_props.odd_even_key, 'oe_freq', 'oe_total'),
            (draw_props.big_small_key, 'bs_freq', 'bs_total')):
        frequency = meta.get(freq_key)
        if frequency:
            value += (context.weights.ratio_match * frequency.get(key_fn(triple), 0)
                      / meta.get(total_key, 1))
    return value


def _term_pair(triple, meta, context):
    high_pairs = meta.get('high_pairs') or set()
    if not (context.flags.get('pair', True) and high_pairs):
        return 0.0
    return history.pair_bonus(triple, high_pairs, context.weights.pair_bonus)


def _term_slope(triple, meta, context):
    if not context.flags.get('slope', True):
        return 0.0
    return slope.triplet_bonus(triple, meta.get('slope'), context.weights.slope_match)


def _term_form_switch(triple, context):
    """同一形态连开多期后，给另一种形态加分。值已在 context 里算好。"""
    if not (context.flags.get('form_switch', True) and context.form_switch):
        return 0.0
    form = draw_props.classify_form(triple)
    key = draw_props.ZU6 if form == draw_props.ZU6 else draw_props.ZU3
    return context.form_switch.get(key, 0.0)


def _term_sum_interval(triple, context):
    if not (context.flags.get('sum_span', True) and context.sum_interval):
        return 0.0
    return context.sum_interval.get('bonus', {}).get(sum(triple), 0.0)


def _term_form_prior(triple, context):
    """按形态的**理论**占比加分，让推荐池的形态构成贴近真实开奖。

    选哪几注没有优势（直选恒 1/1000），这一项只调整「组六占多少、组三占
    多少」。它是三项里最大的一份（组六 4.32 分），迁移前 `detail` 把它整个
    漏掉了，所以分解加起来一直对不上总分。
    """
    return (context.weights.form_prior
            * draw_props.THEORY_FORM_P[draw_props.classify_form(triple)])


def score_terms(triple, score, meta, context):
    """一注的全部得分项，按 `detail` 的分组返回。

    **打分与分解共用它**：`weight()` 求和，`detail()` 原样呈现。
    """
    return {
        'base_digit': _term_base(triple, score, meta, context),
        'danma': _term_danma(triple, context),
        'kill': _term_kill(triple, context),
        'sum_span': _term_sum_span(triple, meta, context),
        'pattern': _term_pattern(triple, meta, context),
        'last_repeat': _term_last_repeat(triple, meta, context),
        'ratio_match': _term_ratio(triple, meta, context),
        'pair': _term_pair(triple, meta, context),
        'slope': _term_slope(triple, meta, context),
        'form_switch': _term_form_switch(triple, context),
        'sum_interval': _term_sum_interval(triple, context),
        'form_prior': _term_form_prior(triple, context),
    }


def weight(triple, score, meta, context):
    return sum(score_terms(triple, score, meta, context).values())


def detail(triple, score, meta, context):
    terms = score_terms(triple, score, meta, context)
    return {**terms, 'total': sum(terms.values())}


# ─── 选池 ───

def blend_dan_score(score, meta):
    """胆码/杀码用的融合分：分位分占大头，与直选的分位排序保持一致。"""
    if not meta.get('pos_scores'):
        return list(score)
    return [score[digit] * DAN_GLOBAL_SHARE
            + sum(meta['pos_scores'][p][digit] for p in range(POSITIONS)) * DAN_POSITION_SHARE
            for digit in DIGIT_SPACE.numbers()]


DAN_GLOBAL_SHARE = 0.45
DAN_POSITION_SHARE = 0.55


def select_danma(score_rank, top_pool, random_rate, enable_random=True, rng=random):
    """选两个胆码：多数时候取前两名，小概率从前若干名里随机挑。

    随机不是为了更准，是为了**别让同两个数字长期霸榜**——分数接近时
    第二名和第三名的差别没有意义，却会让推荐连着几期长一个样。
    """
    pool = [digit for digit, _ in score_rank[:top_pool]]
    if enable_random and rng.random() < random_rate:
        return rng.sample(pool, 2)
    return pool[:2]


def select_diverse_pool(pool, top_n, candidate_size, diversity_weight,
                        correlation_penalty, correlation_threshold,
                        use_diversity=True, use_correlation=True):
    """贪心选池：在原始分之外，兼顾数字覆盖面与彼此的重合度。

    每次从剩余候选里挑「原始分 + 新覆盖的数字数 × 奖励 − 重合惩罚」最高的
    那注。**贪心而不是全局最优**：全局最优要枚举组合，而这里的目标本身
    就是启发式的，多算的那点精度换不来任何命中率。
    """
    candidates = sorted(pool, key=lambda item: -item[0])[:candidate_size]
    selected, selected_digits = [], []

    while candidates and len(selected) < top_n:
        covered = set().union(*selected_digits) if selected_digits else set()
        best, best_score = None, -float('inf')
        for item_weight, number in candidates:
            digits = set(number)
            overlap = (sum(correlation_penalty for previous in selected_digits
                           if len(digits & previous) >= correlation_threshold)
                       if use_correlation else 0.0)
            # 只有已经选了东西之后，「新覆盖」才有意义
            new_cover = len(digits - covered) if use_diversity and selected_digits else 0.0
            candidate_score = item_weight + new_cover * diversity_weight - overlap
            if candidate_score > best_score:
                best_score, best = candidate_score, (item_weight, number)
        if best is None:
            break
        selected.append(best)
        selected_digits.append(set(best[1]))
        candidates.remove(best)
    return selected


def position_constrained_pool(score, meta, context, per_position):
    """百/十/个各取前若干码做笛卡尔积。

    用途是**补漏**：全量排序会漏掉「三位各自都很热、合起来分不算最高」的组合。
    """
    position_scores = meta.get('pos_scores')
    if not position_scores:
        return []
    tops = [sorted(DIGIT_SPACE.numbers(), key=lambda d: -position_scores[i][d])[:per_position]
            for i in range(POSITIONS)]
    pool = [(weight(triple, score, meta, context), _label(triple))
            for triple in product(*tops)]
    pool.sort(key=lambda item: -item[0])
    return pool


def merge_pools(*pools, top_n):
    """按号码去重后合并，取分最高的若干注。先出现的那份保留。"""
    seen, merged = set(), []
    for pool in pools:
        for item in pool:
            if item[1] not in seen:
                seen.add(item[1])
                merged.append(item)
    merged.sort(key=lambda item: -item[0])
    return merged[:top_n]


def _label(triple):
    return ''.join(str(digit) for digit in triple)


# ─── 排名主流程 ───

NOISE_POOL_SIZE = 50        # 只给前 50 注加扰动：后面的注本来就进不了推荐
EXPLORATION_POOL_SIZE = 50
BALANCE_MIN_KEEP = 100      # 冷热平衡后至少保留这么多注，别直接砍到 top_n
BALANCE_KEEP_FACTOR = 4
CANDIDATE_FACTOR = 5        # 贪心选池的候选集是目标注数的几倍
POSITION_MERGE_MAX_N = 5    # 只有 Top5 以内才值得合并分位候选池


def rank_triplets(score, meta, context, top_n, *,
                  hot_cold=None, recent_recommendations=None,
                  penalise_recent=None, diversity=None,
                  enable_exploration=False, apply_noise=False,
                  enable_diversity=False, enable_correlation=False,
                  position_top_k=None, rng=random):
    """给全部 1000 注打分，按几道处理后返回前 top_n 注。

    处理顺序是有讲究的：扰动与去重惩罚要在**排序之前**生效，冷热平衡在
    候选池还足够大的时候做，多样性选池最后做。顺序换了，后面那道就只能在
    前面那道砍剩的池子里挑。
    """
    pool = [(weight(triple, score, meta, context), _label(triple))
            for triple in product(DIGIT_SPACE.numbers(), repeat=POSITIONS)]
    pool.sort(key=lambda item: -item[0])

    if apply_noise:
        pool = _add_noise(pool, context.weights.noise, rng)
    if recent_recommendations and penalise_recent:
        pool = sorted(penalise_recent(pool, recent_recommendations),
                      key=lambda item: -item[0])
    if hot_cold:
        pool = _balance_hot_cold(pool, top_n, hot_cold)

    if enable_exploration and rng.random() < context.weights.exploration_rate:
        return _explore(pool, top_n, rng)

    if enable_diversity or enable_correlation:
        result = select_diverse_pool(
            pool, top_n=top_n,
            candidate_size=max(top_n * CANDIDATE_FACTOR, diversity['candidate_size']),
            diversity_weight=context.weights.diversity,
            correlation_penalty=context.weights.correlation_penalty,
            correlation_threshold=context.weights.correlation_threshold,
            use_diversity=enable_diversity, use_correlation=enable_correlation)
    else:
        result = pool[:top_n]

    if (top_n <= POSITION_MERGE_MAX_N and meta.get('pos_scores')
            and position_top_k and not enable_exploration
            and not enable_diversity and not enable_correlation):
        position_pool = position_constrained_pool(score, meta, context, position_top_k)
        if position_pool:
            result = merge_pools(position_pool, pool, top_n=top_n)
    return result


def _add_noise(pool, amplitude, rng):
    """给头部加一点随机，免得同分的号长期霸榜。"""
    head = [(item_weight + rng.uniform(-amplitude, amplitude), number)
            for item_weight, number in pool[:NOISE_POOL_SIZE]]
    return sorted(head + pool[NOISE_POOL_SIZE:], key=lambda item: -item[0])


def _explore(pool, top_n, rng):
    """探索模式：从头部随机取，而不是永远取最高分的那几注。"""
    head = pool[:EXPLORATION_POOL_SIZE] if len(pool) >= EXPLORATION_POOL_SIZE else pool
    if len(head) < top_n:
        return head
    rng.shuffle(head)
    return head[:top_n]


def _balance_hot_cold(pool, top_n, hot_cold):
    """按注里冷热号的构成分三档，各取一定比例。

    分档标准不对称是有意的：含两个以上热号算热注，而只要**同时**含冷号和
    温号就算冷注——冷号本来就少，要求两个冷号的话这一档会几乎是空的。
    """
    hot, warm, cold = hot_cold['hot'], hot_cold['warm'], hot_cold['cold']
    keep = max(top_n * BALANCE_KEEP_FACTOR, BALANCE_MIN_KEEP)
    needed = {'hot': max(1, int(keep * hot_cold['hot_share'])),
              'warm': max(1, int(keep * hot_cold['warm_share'])),
              'cold': max(1, int(keep * hot_cold['cold_share']))}

    buckets = {'hot': [], 'warm': [], 'cold': []}
    for item in pool:
        digits = {int(c) for c in item[1]}
        if len(digits & set(hot)) >= 2:
            buckets['hot'].append(item)
        elif digits & set(cold) and digits & set(warm):
            buckets['cold'].append(item)
        else:
            buckets['warm'].append(item)

    balanced = []
    for name in ('hot', 'warm', 'cold'):
        balanced.extend(sorted(buckets[name], key=lambda item: -item[0])[:needed[name]])
    if len(balanced) < keep:
        chosen = set(balanced)
        balanced.extend([item for item in pool if item not in chosen][:keep - len(balanced)])
    return balanced[:keep]


def build_detail_list(items, score, meta, context, ndigits=1):
    """给推荐列表逐注附上得分拆解，供「为什么推荐这注」展示。"""
    result = []
    for item_weight, number in items:
        triple = tuple(int(c) for c in number)
        terms = score_terms(triple, score, meta, context)
        result.append({
            'num': number,
            'score': round(item_weight, ndigits),
            'detail': {name: round(value, ndigits) for name, value in terms.items()},
        })
    return result
