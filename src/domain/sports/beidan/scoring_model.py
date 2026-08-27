"""比分概率模型：泊松、Dixon-Coles 修正、λ 推导、按市场锚定 1X2。

**没有一个函数能提高中奖概率。** 它们做的是把「市场认为主队多强、这场几个球」
翻译成一张比分概率表，好让各玩法从同一份分布里取数——同一场比赛的胜平负、
让球、比分、总进球因此必然自洽。

这一层不读配置：`max_goals`、`rho`、`split`、锚定强度一律由调用方传入。
"""
import math
from collections import defaultdict

HOME_WIN, DRAW, AWAY_WIN = '胜', '平', '负'
# 1X2 的三套别名。页面、接口、历史记录各用一套，都得认
OUTCOME_ALIASES = {
    HOME_WIN: (HOME_WIN, 'H', 'home'),
    DRAW: (DRAW, 'D', 'draw'),
    AWAY_WIN: (AWAY_WIN, 'A', 'away'),
}
# 总进球分桶的上界，超过归入 '7+'。**键是字符串**——'7+' 不是数字，
# 整个桶只能统一用字符串，混着放下游 sorted 就会炸
GOALS_BUCKET_MAX = 7
OVERFLOW_BUCKET = '7+'
# λ 的下限。0 会让泊松退化成「必然 0 球」，而那不是任何真实球队
MIN_LAMBDA = 0.01
EPSILON = 1e-9


def poisson_pmf(k, mu):
    """泊松概率质量。均值非正时只有 0 球可能——这是退化情形，不是错误。"""
    if mu <= 0:
        return 0.0 if k > 0 else 1.0
    return (mu ** k) * math.exp(-mu) / math.factorial(k)


def lambdas_from_probs(home_prob, draw_prob, away_prob, target_total, split):
    """由 1X2 概率与目标总进球分出主客两队的 λ。

    `split` 决定强弱差距在总进球里占多大比例：0 表示平均分，
    越大越向强队倾斜。**迁移前这个参数是死的**——`match_lambdas` 收着它，
    转手调的函数却用模块级常量，三个不同的 split 值算出完全一样的结果。
    这里让它真正生效，默认值就是迁移前那个常量，所以现有调用行为不变。
    """
    strength = (home_prob - away_prob) / (home_prob + draw_prob + away_prob + EPSILON)
    lam_home = target_total * (0.5 + strength * split)
    lam_away = target_total * (0.5 - strength * split)
    return max(MIN_LAMBDA, lam_home), max(MIN_LAMBDA, lam_away)


def calibrate_draw(home_prob, draw_prob, away_prob, handicap_value,
                   home_draw_rate=0.25, away_draw_rate=0.25, league_draw_rate=0.25,
                   big_handicap=1.0, small_handicap=0.5,
                   big_discount=0.8, small_discount=0.95,
                   low_ratio=0.8, high_ratio=1.3,
                   lift=1.2, cut=0.9):
    """按参照平局率微调平局概率，然后重新归一化。

    **盘口越大，平局越少**：让一球以上的比赛，强弱差距本身就压低了平局，
    所以参照率要打折。这不是在预测平局，是在纠正 1X2 去水后系统性的平局偏差。
    """
    reference = (home_draw_rate + away_draw_rate + league_draw_rate) / 3
    if handicap_value is not None:
        try:
            value = handicap_value
            if isinstance(value, str):
                value = float(value.replace('(', '').replace(')', ''))
            if abs(value) >= big_handicap:
                reference *= big_discount
            elif abs(value) >= small_handicap:
                reference *= small_discount
        except (ValueError, TypeError):
            # 盘口解析不了就不调整，而不是让整场预测失败
            pass

    total = home_prob + draw_prob + away_prob + EPSILON
    current = draw_prob / total
    if current < reference * low_ratio:
        draw_prob *= lift
    elif current > reference * high_ratio:
        draw_prob *= cut

    total_new = home_prob + draw_prob + away_prob + EPSILON
    return home_prob / total_new, draw_prob / total_new, away_prob / total_new


def dixon_coles_matrix(lam_home, lam_away, rho, max_goals):
    """Dixon-Coles 修正后的联合比分分布 `{(主, 客): 概率}`。

    独立泊松低估了 0-0、1-1 这类低比分的相关性（两队互相压制时进球一起变少），
    DC 用一个 τ 因子修正**四格**：0-0、0-1、1-0、1-1，其余原样。

    **`rho = 0` 时它精确退化成独立泊松。** 线上现在正是 0（`694bdec` 有意设的），
    所以这段修正当前不生效——但配置随时会改回来，它不是死代码。
    """
    probs = {}
    raw_sum = 0.0
    for home in range(max_goals + 1):
        for away in range(max_goals + 1):
            base = poisson_pmf(home, lam_home) * poisson_pmf(away, lam_away)
            if home == 0 and away == 0:
                tau = 1 - lam_home * lam_away * rho
            elif home == 0 and away == 1:
                tau = 1 + lam_home * rho
            elif home == 1 and away == 0:
                tau = 1 + lam_away * rho
            elif home == 1 and away == 1:
                tau = 1 - rho
            else:
                tau = 1.0
            probs[(home, away)] = base * tau
            raw_sum += base * tau

    if raw_sum <= 0:
        # rho 取值极端时 τ 可能把整张表压成非正数。退回独立泊松而不是抛——
        # 出不了号比出错号更糟，但**悄悄给一张全 0 的表**是最糟的
        probs = {key: poisson_pmf(key[0], lam_home) * poisson_pmf(key[1], lam_away)
                 for key in probs}
        raw_sum = sum(probs.values()) + EPSILON
    return {key: value / raw_sum for key, value in probs.items()}


def independent_poisson_matrix(lam_home, lam_away, max_goals, floor=1e-6):
    """不带 DC 修正的比分分布，低于 `floor` 的格子直接丢掉。

    与 `dixon_coles_matrix(rho=0)` 的差别只在这个截断：那边保留全部格子，
    这边丢掉尾部再归一。**两者结果因此不完全相同**，不能互相替代。
    """
    probs = {}
    for home in range(max_goals + 1):
        for away in range(max_goals + 1):
            prob = poisson_pmf(home, lam_home) * poisson_pmf(away, lam_away)
            if prob > floor:
                probs[(home, away)] = prob
    total = sum(probs.values()) + EPSILON
    return {key: value / total for key, value in probs.items()}


def aggregate_goals(score_dist, bucket_max=GOALS_BUCKET_MAX,
                    overflow=OVERFLOW_BUCKET):
    """把比分分布聚合成总进球分布。

    **键是字符串**（`'0'`…`'6'`、`'7+'`）：溢出桶本来就不是数字，
    整个映射只能统一成字符串。混着放 int 和 str，下游一 `sorted()` 就炸——
    进球数校准那条链踩过一次（kv 往返把 int 键变成 str，两处失效都不报错）。
    """
    goals = defaultdict(float)
    for (home, away), prob in score_dist.items():
        total = home + away
        goals[overflow if total >= bucket_max else str(total)] += prob
    return dict(goals)


def _outcome_of(home, away):
    return HOME_WIN if home > away else (AWAY_WIN if home < away else DRAW)


def anchor_outcomes(score_dist, target_probabilities, strength):
    """把比分矩阵的胜平负质量部分地拉向市场概率。

    模型算出的 1X2 与市场报价总有差距。**全量对齐会丢掉模型的信息，
    完全不对齐又会让推荐和盘口明显打架**，所以按 `strength` 取幂做部分对齐：
    0 是完全不动，1 是完全对齐市场。

    返回 `(新分布, 说明)`。**任何一步走不通都返回原分布加一个 `applied: False`
    的理由**，而不是抛异常或悄悄返回半成品——调用方据此决定要不要展示。
    """
    if not score_dist or not target_probabilities:
        return score_dist, {'applied': False, 'reason': 'missing_distribution_or_target'}

    target = {}
    for label, aliases in OUTCOME_ALIASES.items():
        value = next((target_probabilities.get(key) for key in aliases
                      if target_probabilities.get(key) is not None), 0.0)
        try:
            target[label] = max(0.0, float(value))
        except (TypeError, ValueError):
            target[label] = 0.0
    target_total = sum(target.values())
    if target_total <= 0:
        return score_dist, {'applied': False, 'reason': 'invalid_target'}
    target = {key: value / target_total for key, value in target.items()}

    current = {HOME_WIN: 0.0, DRAW: 0.0, AWAY_WIN: 0.0}
    normalized = {}
    for score, probability in score_dist.items():
        try:
            home, away = int(score[0]), int(score[1])
            probability = max(0.0, float(probability))
        except (TypeError, ValueError, IndexError):
            continue
        normalized[(home, away)] = normalized.get((home, away), 0.0) + probability
        current[_outcome_of(home, away)] += probability

    raw_total = sum(current.values())
    # 三种结果里任何一种质量为 0，比例就没法算——这时候不对齐，
    # 而不是给一个除零后的巨大因子
    if raw_total <= 0 or any(current[key] <= 0 or target[key] <= 0 for key in current):
        return score_dist, {'applied': False, 'reason': 'incomplete_outcome_mass'}
    current = {key: value / raw_total for key, value in current.items()}

    weight = max(0.0, min(1.0, float(strength)))
    adjusted = {}
    for (home, away), probability in normalized.items():
        label = _outcome_of(home, away)
        adjusted[(home, away)] = probability * (target[label] / current[label]) ** weight

    adjusted_total = sum(adjusted.values())
    if adjusted_total <= 0:
        return score_dist, {'applied': False, 'reason': 'zero_adjusted_total'}
    adjusted = {score: value / adjusted_total for score, value in adjusted.items()}

    after = {label: sum(value for score, value in adjusted.items()
                        if _outcome_of(*score) == label)
             for label in (HOME_WIN, DRAW, AWAY_WIN)}
    return adjusted, {
        'applied': True,
        'strength': weight,
        'before': {key: round(value, 6) for key, value in current.items()},
        'target': {key: round(value, 6) for key, value in target.items()},
        'after': {key: round(value, 6) for key, value in after.items()},
    }


def top_scores(score_dist, count):
    """概率最高的几个比分，供展示用。"""
    ranked = sorted(score_dist.items(), key=lambda item: -item[1])
    return [{'score': f"{home}-{away}", 'probability': prob,
             'home_goals': home, 'away_goals': away}
            for (home, away), prob in ranked[:count]]
