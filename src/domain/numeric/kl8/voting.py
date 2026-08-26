"""kl8 的多模型投票：从排名到最终选号的那一段管道。

名字叫「多模型」，实际只剩一个模型。迁移前还挂着贝叶斯与马尔可夫两个，
但线上 23564 条策略试验、381 份预测快照里，这两个的权重**无一例外是 0**，
`MODEL_CONFIG` 里也标着 `enabled: false`。它们不是被少用，是从未被用过。

投票这一步做的事：把排名换算成票数（名次越靠前票越多），累加，排序，
交给候选池整形。只剩一个模型时它等价于「按排名取前若干」，保留投票这层
是因为权重仍然是策略的一部分——`rank: 0.5` 与 `rank: 1.0` 对**单模型**
确实没有区别，但「有没有信号」的判断要看它。

候选池整形（`PoolShaper`）是注入进来的。整形逻辑现在还在
`src/kl8/candidates.py`，阶段 3-10 迁进领域层，届时换掉实现即可，
这里一行不用改。
"""
from src.domain.numeric.kl8 import scoring

# 只剩这一个模型。权重字典里出现别的键时，为零可以容忍——线上 23564 条
# 历史试验记录的权重字典都带着 `bayesian`/`markov` 两个 0，改动它们的形状
# 会让新旧记录对不上。但**非零就必须报错**：静默忽略等于把用户指定的模型
# 悄悄换成另一个，而结果看上去完全正常（判据 3）。
KNOWN_MODELS = ('rank',)

NO_SIGNAL_STATUS = 'no_validated_signal'
NO_SIGNAL_MESSAGE = '暂无通过回测验证的有效特征，不输出号码推荐'

# 投票用的排名长度。比请求的池子宽，让整形有得挑；上限是号码空间本身。
MIN_MODEL_TOP_N = 40
# 候选池至少这么大。7 是复式玩法的最小池，比它还小就没有整形余地了。
MIN_POOL_SIZE = 7


class PoolShaper:
    """候选池整形的端口。

    两个函数总是成对出现、由同一份实现提供，所以合成一个对象传，
    而不是两个各自游离的参数。
    """

    def __init__(self, diversify, select_final):
        self.diversify = diversify
        self.select_final = select_final


def _check_model_weights(model_weights):
    """未知模型的权重非零时报错，而不是当它不存在。"""
    unknown = sorted(name for name, weight in model_weights.items()
                     if name not in KNOWN_MODELS and weight)
    if unknown:
        raise ValueError(
            f'未知的模型权重 {unknown}：这些模型已不存在，'
            f'继续按剩余模型出号会得到一个看不出问题的错结果。'
            f'当前可用的模型是 {list(KNOWN_MODELS)}。')


def _no_signal(version):
    return {
        'selected': [],
        'candidates': [],
        'votes': {},
        'status': NO_SIGNAL_STATUS,
        'message': NO_SIGNAL_MESSAGE,
        'version': version,
    }


def _tally(ranked_numbers, weight):
    """名次换票数：第一名满票，最后一名接近零票，线性递减。"""
    total = max(len(ranked_numbers), 1)
    return {num: (1.0 - (rank / total)) * weight
            for rank, num in enumerate(ranked_numbers)}


def vote(statistics, feature_weights, model_weights, shaper, *,
         version, based_on_issue='', pick_n=5, top_n=20,
         pool_diversify=True, pool_max_last_numbers=None,
         final_selection_mode='balanced', **ranking_options):
    """跑一遍投票管道，返回选号与全部中间结果。

    **没有有效信号时返回空选号，而不是随便给一组。** 判断要同时满足两头：
    模型有权重、且该模型的特征有权重。只有模型权重的话，排名会退化成
    「按号码大小排序」——那看起来像结果，实际什么也没说。
    """
    _check_model_weights(model_weights)

    rank_weight = model_weights.get('rank', 0.0)
    has_feature = any(w > 0 for w in feature_weights.values())
    if not (rank_weight > 0 and has_feature):
        return _no_signal(version)

    model_top_n = min(scoring.SPACE.size, max(top_n, MIN_MODEL_TOP_N))
    ranking = scoring.ensemble_ranking(
        statistics, feature_weights, top_n=model_top_n,
        based_on_issue=based_on_issue, **ranking_options)
    votes = _tally([item['num'] for item in ranking], rank_weight)

    candidates = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
    pool_size = max(top_n, MIN_POOL_SIZE)
    last_numbers = statistics.get('last_numbers', set())
    if pool_diversify:
        candidate_pool = shaper.diversify(
            candidates, pool_size, last_numbers,
            max_last_numbers=pool_max_last_numbers)
    else:
        candidate_pool = candidates[:pool_size]

    final_pool, selected_mode = shaper.select_final(
        candidate_pool, pick_n, last_numbers,
        max_last_numbers=pool_max_last_numbers,
        selection_mode=final_selection_mode)

    return {
        'selected': [num for num, _ in final_pool],
        'candidates': candidate_pool,
        'selected_pool': final_pool,
        'votes': dict(votes),
        'diversified': pool_diversify,
        'pool_max_last_numbers': pool_max_last_numbers,
        'final_selection_mode': selected_mode,
        'raw_candidate_count': len(candidates),
        'version': version,
    }
