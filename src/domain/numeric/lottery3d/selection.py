"""选号：把数字评分变成一份能照着买的方案。

三种玩法各占一节，共同的前提只有一条——**选哪些码没有优势**。3D 是公平
摇奖，实测评分选码约等于随机选码。唯一真正的杠杆是**覆盖多少注**，所以
这里的每个档位都同时给出注数、成本与命中率，让人按预算挑，而不是让人以为
挑对了码。

- **组六**：N 码 → C(N,3) 注。持有 K 注互异组六，无条件命中 = K×6/1000
  （要开奖为组六、且三码全在池内）；给定开奖为组六时 = K/120。
- **组三**：一个对子 {a,b} → 2 注组选三，覆盖全部 6 种排列。任取 K 组的
  条件命中 = K/45，**与选哪些码无关**。
- **胆拖杀**：胆码是评分最高的两个，杀码是最低的一两个。杀码**降权而不是
  排除**——排除等于断言它开不出来，而它照样有 1/10 的机会。

这里没有一个函数能提高中奖概率。
"""
import math
from collections import Counter
from itertools import combinations

from src.domain.numeric.lottery3d import draw as draw_props
from src.domain.numeric.lottery3d.space import DIGIT_SPACE, POSITIONS

TICKET_PRICE = 2

# 组合数常量。写成名字而不是散在各处的字面量：它们是「命中率」那几个分母，
# 看到 120 得先想一下才知道是 C(10,3)。
ZU6_TOTAL_COMBOS = math.comb(DIGIT_SPACE.size, POSITIONS)          # C(10,3) = 120
ZU3_TOTAL_PAIRS = math.comb(DIGIT_SPACE.size, 2)                    # C(10,2) = 45
# 一注组六对应 6 种排列，全部直选组合共 1000 注
ZU6_PERMUTATIONS = 6
ALL_STRAIGHT = DIGIT_SPACE.size ** POSITIONS

# 组三样本不足时退回的更长窗口，以及退无可退时的无信息先验。
ZU3_FALLBACK_WINDOW = 60
ZU3_UNIFORM_PRESENCE = 0.2


def _digits():
    return DIGIT_SPACE.numbers()


# ─── 胆拖杀 ───

TUOMA_SLICE = slice(2, 6)   # 拖码取第 3~6 名
# 末位与倒数第二名分差超过它时，只杀最后一个——差距明显说明末位是真的孤立，
# 一并杀掉倒数第二个会误伤。
KILL_GAP = 3


def pick_dan_tuo_kill(score, select_danma):
    """胆码、拖码、杀码，以及完整排名。

    `select_danma` 由调用方给：胆码带随机成分，随机源不该长在这里。
    """
    rank = sorted(enumerate(score), key=lambda item: item[1], reverse=True)
    danma = select_danma(rank)
    tuoma = [digit for digit, _ in rank[TUOMA_SLICE]]
    if rank[-1][1] + KILL_GAP < rank[-2][1]:
        kill = [rank[-1][0]]
    else:
        kill = [digit for digit, _ in rank[-2:]]
    return danma, tuoma, kill, rank


def effective_digit_score(score, digit, kill, kill_penalty):
    """杀码**降权**后的单码分。

    降权而不是从候选里剔除：剔除等于断言这个号开不出来，而它和别的号一样
    有 1/10 的机会。降权只是表达「更不看好」。
    """
    return score[digit] - (kill_penalty if digit in set(kill or ()) else 0.0)


# ─── 组六 ───

def zu6_pool(score, pool_size, kill=None, kill_penalty=0.0):
    """组六复式选号：按有效分取前 N 码，升序返回。

    **只看单码分**。迁移前这里还挂着 `pair_freq` 与 `numbers` 两个参数，
    函数体一个都没用到，而调用方在认真地传——包括回测与两个分析脚本。
    它们本该由一个「组合得分」函数消费，而那个函数零调用方。一起删了。
    """
    rank = sorted(_digits(),
                  key=lambda digit: -effective_digit_score(score, digit, kill, kill_penalty))
    return sorted(rank[:pool_size])


def zu6_notes(digits):
    """N 码 → C(N,3) 注组六。返回 (组合元组, 号码字符串)。"""
    combos = [tuple(sorted(combo)) for combo in combinations(digits, POSITIONS)]
    return combos, [''.join(map(str, combo)) for combo in combos]


def _coverage(note_count):
    """一组注的命中率。两种口径都给，因为它们回答的是不同的问题。"""
    return {
        # 无条件：**含「开奖须为组六」这一步**。用户看到的实际命中频率是这个。
        'hit_rate': round(note_count * ZU6_PERMUTATIONS / ALL_STRAIGHT, 4),
        # 条件：已知开奖是组六时。它更高，但不是能指望的那个数。
        'conditional_hit_rate': round(note_count / ZU6_TOTAL_COMBOS, 4),
    }


def zu6_payload(digits, **extra):
    """一组组六号码的完整说明：注数、成本、两种命中率、全部组合。"""
    digits = sorted(int(digit) for digit in digits)
    combos, combo_strings = zu6_notes(digits)
    return {
        'digits': digits,
        'digits_str': ''.join(map(str, digits)),
        'notes': len(combos),
        'cost': len(combos) * TICKET_PRICE,
        **_coverage(len(combos)),
        'combos': combo_strings,
        **extra,
    }


def zu6_primary(score, size, kill=None, kill_penalty=0.0):
    """主推池。与同尺寸的档位取号一致（同一个 `zu6_pool`）。"""
    payload = zu6_payload(zu6_pool(score, size, kill, kill_penalty), is_primary=True)
    payload['size'] = size
    return payload


def zu6_coverage_tiers(score, sizes, primary_size, kill=None, kill_penalty=0.0):
    """把各档位摊开，供按预算挑。

    存在的意义是**让人按注数挑，而不是以为挑对了码**——选哪些码没有优势，
    注数是唯一真正变化的量。
    """
    tiers = []
    for size in sizes:
        payload = zu6_payload(zu6_pool(score, size, kill, kill_penalty))
        payload.pop('digits')
        payload['size'] = size
        payload['is_primary'] = size == primary_size
        tiers.append(payload)
    return tiers


# 四码均衡分的几个系数。目标是让四个码「看起来像一次正常开奖」：
# 奇偶各半、大小各半、跨度别太窄、别全是连号。都是结构约束，不是优势。
BALANCE_ODD_PENALTY = 1.0
BALANCE_BIG_PENALTY = 0.8
BALANCE_SPAN_BONUS = 0.15
BALANCE_SPAN_CAP = 8
BALANCE_ADJACENT_PENALTY = 0.35
BALANCE_KILL_PENALTY = 1.2
# 「扩散」变体额外奖励跨度，让它明显区别于「均衡」
SPREAD_SPAN_BONUS = 0.3
VARIANT_POOL_SIZE = 8    # 变体只在前 8 名里组合，再往下分数已经没有意义
ZU6_FOUR_SIZE = 4


def zu6_balance_score(combo, score, kill, kill_penalty):
    """四码组合的均衡分：单码分打底，形态偏离扣分。"""
    digits = tuple(sorted(combo))
    base = sum(effective_digit_score(score, digit, kill, kill_penalty) for digit in digits)
    odd = sum(1 for digit in digits if digit % 2)
    big = sum(1 for digit in digits if digit >= draw_props.BIG_SMALL_THRESHOLD)
    span = digits[-1] - digits[0]
    adjacent = sum(1 for a, b in zip(digits, digits[1:]) if b - a == 1)
    killed = sum(1 for digit in digits if digit in set(kill or ()))
    return (base
            - abs(odd - 2) * BALANCE_ODD_PENALTY
            - abs(big - 2) * BALANCE_BIG_PENALTY
            + min(span, BALANCE_SPAN_CAP) * BALANCE_SPAN_BONUS
            - adjacent * BALANCE_ADJACENT_PENALTY
            - killed * BALANCE_KILL_PENALTY)


def zu6_four_variants(score, limit, kill=None, kill_penalty=0.0, use_kill=False):
    """几组风格不同的四码，供对照着看覆盖差异。

    四个风格都是**确定性**的：主推按分、均衡按形态、避杀绕开杀码、扩散再加
    跨度奖励。不引入随机——同一份输入两次给出不同的四码，用户无从判断
    是模型变了还是掷了骰子。
    """
    kill_for_rank = kill if use_kill else None
    rank = sorted(_digits(),
                  key=lambda d: -effective_digit_score(score, d, kill_for_rank, kill_penalty))
    candidates = rank[:VARIANT_POOL_SIZE]

    variants, seen = [], set()

    def add(label, digits):
        key = tuple(sorted(digits))
        if key in seen or len(key) != ZU6_FOUR_SIZE:
            return
        seen.add(key)
        variants.append(zu6_payload(key, label=label))

    def by_balance(combo, extra=0.0):
        return zu6_balance_score(combo, score, kill, kill_penalty) + extra

    add('主推', zu6_pool(score, ZU6_FOUR_SIZE, kill_for_rank, kill_penalty))
    add('均衡', max(combinations(candidates, ZU6_FOUR_SIZE), key=by_balance))

    killed = set(kill or ())
    without_kill = [digit for digit in rank if digit not in killed][:6]
    if len(without_kill) >= ZU6_FOUR_SIZE:
        add('避杀', without_kill[:ZU6_FOUR_SIZE])

    add('扩散', max(combinations(candidates, ZU6_FOUR_SIZE),
                    key=lambda c: by_balance(c, (max(c) - min(c)) * SPREAD_SPAN_BONUS)))

    for combo in sorted(combinations(candidates, ZU6_FOUR_SIZE),
                        key=by_balance, reverse=True):
        add('备选', combo)
        if len(variants) >= limit:
            break
    return variants[:limit]


def evaluate_zu6_pool(numbers, sizes, trials, score_fn, min_train):
    """逐期样本外检验号码池：**每一期只用它之前的数据选码**。

    用当期数据选码再拿当期开奖去检验，命中率会好看得离谱而毫无意义——
    这个函数存在的全部理由就是把那条泄漏堵死。

    `ge2_rate` 回答用户最直观的问题：「至少覆盖两个开奖号」有多频繁。
    完整命中只在组六期统计，因为非组六期本来就不可能全中。
    """
    sizes = tuple(sorted({int(size) for size in sizes
                          if POSITIONS <= int(size) <= DIGIT_SPACE.size}))
    if len(numbers) <= min_train or not sizes:
        return {'trials': 0, 'zu6_draws': 0, 'tiers': {}}

    start = max(min_train, len(numbers) - max(1, int(trials)))
    stats = {size: {'full_hit': 0, 'ge2_hit': 0, 'overlap_sum': 0} for size in sizes}
    zu6_draws = evaluated = 0

    for index in range(start, len(numbers)):
        train = numbers[:index]
        actual = set(numbers[index])
        is_zu6 = len(actual) == POSITIONS
        zu6_draws += int(is_zu6)
        evaluated += 1
        scores = score_fn(train)
        for size in sizes:
            pool = set(zu6_pool(scores, size))
            overlap = len(actual & pool)
            stats[size]['overlap_sum'] += overlap
            stats[size]['ge2_hit'] += int(overlap >= 2)
            stats[size]['full_hit'] += int(is_zu6 and actual <= pool)

    return {'trials': evaluated, 'zu6_draws': zu6_draws,
            'tiers': {str(size): _tier_stats(size, item, evaluated, zu6_draws)
                      for size, item in stats.items()}}


def _tier_stats(size, item, evaluated, zu6_draws):
    notes = math.comb(size, POSITIONS)
    return {
        'size': size,
        'trials': evaluated,
        'zu6_draws': zu6_draws,
        'full_hit': item['full_hit'],
        'conditional_full_rate': _rate(item['full_hit'], zu6_draws),
        'unconditional_full_rate': _rate(item['full_hit'], evaluated),
        'ge2_rate': _rate(item['ge2_hit'], evaluated),
        'avg_unique_overlap': round(item['overlap_sum'] / evaluated, 3) if evaluated else 0.0,
        # 实测值旁边always放理论值：差得远说明选码有问题，差不多说明选码没用
        'theoretical_conditional_rate': round(notes / ZU6_TOTAL_COMBOS, 4),
        'theoretical_unconditional_rate': round(notes * ZU6_PERMUTATIONS / ALL_STRAIGHT, 4),
    }


def _rate(hit, total):
    return round(hit / total, 4) if total else 0.0


# ─── 组三 ───

def zu3_presence(numbers, window, min_samples):
    """组三条件下各数字的出现率：**只看历史上开出组三的那些期**。

    与组六同思路，只回答「这个数字会不会进入组三的号码集合」，不问位置、
    不问哪个号重复。样本不足时退到更长的窗口；仍然没有就返回均匀先验——
    **不是 0**，0 会被下游当成「这个号不会出现」。
    """
    recent = [set(draw) for draw in numbers[-window:]
              if draw_props.classify_form(draw) == draw_props.ZU3]
    if len(recent) < min_samples:
        recent = [set(draw) for draw in numbers[-ZU3_FALLBACK_WINDOW:]
                  if draw_props.classify_form(draw) == draw_props.ZU3]
    if not recent:
        return {digit: ZU3_UNIFORM_PRESENCE for digit in _digits()}

    counts = Counter()
    for draw in recent:
        counts.update(draw)
    return {digit: counts.get(digit, 0) / len(recent) for digit in _digits()}


def zu3_pair_scores(presence):
    """45 个无序数对的条件概率，按独立性假设 P({a,b}) ∝ r_a·r_b 后归一化。

    独立性显然不成立，但**这不影响结论**：下面的档位命中率是 K/45，与
    这里排出的顺序无关。这个排序只决定「先列哪几对」。
    """
    scored = [((a, b), presence[a] * presence[b])
              for a in _digits() for b in _digits() if a < b]
    total = sum(value for _, value in scored) or 1.0
    return [(pair, value / total) for pair, value in scored]


def zu3_straight_combos(pair):
    """对子 {a,b} 覆盖的全部 6 注单选（aab/aba/baa/abb/bab/bba）。"""
    a, b = sorted(pair)
    combos = set()
    for repeated, single in ((a, b), (b, a)):
        for position in range(POSITIONS):
            slots = [repeated] * POSITIONS
            slots[position] = single
            combos.add(''.join(map(str, slots)))
    return sorted(combos)


def zu3_group_notes(pair):
    """对子的**组选三**表达：2 注覆盖全部 6 种排列。

    组选三一注 = 三码含一个重复位，本身就代表 3 种排列。对子 {a,b} 有
    「双 a」与「双 b」两个方向，2 注即可全覆盖——与 6 注单选命中概率完全
    相同，成本只有三分之一。
    """
    a, b = sorted(pair)
    return sorted({f'{repeated}{repeated}{single}'
                   for repeated, single in ((a, b), (b, a))})


def zu3_pairs(presence, limit):
    """条件概率最高的若干个对子，每个带上注数与成本。"""
    scored = sorted(zu3_pair_scores(presence), key=lambda item: -item[1])[:limit]
    pairs = []
    for (a, b), probability in scored:
        group_notes = zu3_group_notes((a, b))
        straight = zu3_straight_combos((a, b))
        pairs.append({
            'digits': [a, b],
            'digits_str': f'{a}{b}',
            'prob': round(probability, 4),
            'notes': len(group_notes),
            'cost': len(group_notes) * TICKET_PRICE,
            'zu_notes': group_notes,
            'combos': straight,
            'direct_notes': len(straight),
            'direct_cost': len(straight) * TICKET_PRICE,
        })
    return pairs, sum(probability for _, probability in scored)


def zu3_coverage_tiers(presence, sizes):
    """K 组对子 → 2K 注组选三，条件命中率 K/45，**线性**。

    线性是这里唯一诚实的说法：多买多中，与买哪几组无关。
    """
    scored = sorted(zu3_pair_scores(presence), key=lambda item: -item[1])
    tiers = []
    for size in sizes:
        size = min(size, ZU3_TOTAL_PAIRS)
        top = scored[:size]
        tiers.append({
            'size': size,
            'pairs': [list(pair) for pair, _ in top],
            # 迁移前这里写的是 `for a, b in top`，而 top 的元素是
            # `((a, b), 概率)`——于是 a 拿到整个对子、b 拿到概率，渲染成
            # `(2, 5)0.06338028169014084`。改对了。
            'pairs_str': ' '.join(f'{a}{b}' for (a, b), _ in top),
            'notes': size * 2,
            'cost': size * 2 * TICKET_PRICE,
            'conditional_hit_rate': round(size / ZU3_TOTAL_PAIRS, 4),
            # 直选口径一并给出作对比：同样的 K 组覆盖，组选三成本只有三分之一
            'direct_notes': size * ZU6_PERMUTATIONS,
            'direct_cost': size * ZU6_PERMUTATIONS * TICKET_PRICE,
        })
    return tiers


# ─── 形态概率 ───

# 四个来源的融合权重。近窗最重、转移次之，历史与理论只做兜底——
# 但**融合出来的仍然是噪声**：形态没有短期可预测性，见 `form_bet` 的说明。
FORM_BLEND = {'recent': 0.40, 'markov': 0.35, 'historical': 0.15, 'theory': 0.10}
# 组三概率相对其基准抬升/回落多少才值得标出来
FORM_SIGNAL_MARGIN = 0.03


def form_probability(forms, recent_p, markov_p):
    """把四个来源融合成一份形态概率。

    `recent_p` 与 `markov_p` 由调用方算好传入（窗口与平滑系数是配置问题）。
    """
    counts = Counter(forms)
    historical = {form: counts.get(form, 0) / len(forms) for form in draw_props.THEORY_FORM_P}
    blended = {form: (FORM_BLEND['recent'] * recent_p[form]
                      + FORM_BLEND['markov'] * markov_p[form]
                      + FORM_BLEND['historical'] * historical[form]
                      + FORM_BLEND['theory'] * draw_props.THEORY_FORM_P[form])
               for form in draw_props.THEORY_FORM_P}
    total = sum(blended.values()) or 1.0
    return historical, {form: value / total for form, value in blended.items()}


def form_streak(forms):
    """结尾连续多少期是同一形态。"""
    if not forms:
        return 0
    last = forms[-1]
    streak = 0
    for form in reversed(forms):
        if form != last:
            break
        streak += 1
    return streak


def form_signal(zu3_elevation):
    if zu3_elevation > FORM_SIGNAL_MARGIN:
        return 'elevated'
    if zu3_elevation < -FORM_SIGNAL_MARGIN:
        return 'depressed'
    return 'normal'
