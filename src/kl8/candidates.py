# -*- coding: utf-8 -*-
"""候选池：旧名字对领域层的适配层。

挑法本身（形态、重号上限、七种建池、最终选池、多注覆盖）都在
`src/domain/numeric/kl8/` 下的 `shape` / `pools` / `portfolio` 三个模块里。
这里只留两样东西：**旧的下划线名字**，因为 `analyzer.py`、`src/kl8/__init__.py`
和 `scripts/` 下若干脚本都按这些名字导入；以及 `generate_multi_slips`，
它要先从分析器拿到排名，才谈得上切注。
"""
from typing import Dict, List

from src.domain.numeric.kl8 import pools, portfolio, shape
from src.domain.numeric.kl8.space import SPACE

from . import strategies as _strategies_mod
from .config import KL8_DEFAULT_HISTORY

_clean_pick_numbers = pools.clean_pick_numbers
_default_repeat_cap = pools.default_repeat_cap
_adaptive_repeat_cap = pools.adaptive_repeat_cap
_adaptive_repeat_target = pools.adaptive_repeat_target
_enforce_minimum_repeats = pools.enforce_minimum_repeats
_diversify_candidate_pool = pools.diversify
_zone_spread_candidate_pool = pools.zone_spread
_prize_floor_candidate_pool = pools.prize_floor
_high_tier_chase_candidate_pool = pools.high_tier_chase
_shape_balanced_candidate_pool = pools.shape_balanced
_score_candidate_selection = pools.score_selection
_select_final_candidate_pool = pools.select_final
_shape_targets = shape.targets
_shape_profile = shape.profile
_shape_penalty = shape.penalty
_simulate_multi_slip_coverage = portfolio.simulate_coverage


def generate_multi_slips(analyzer, select_n: int, n_slips: int = 8,
                         pick_size: int = None) -> List[List[int]]:
    """生成 n_slips 组互不重叠的选号，用于提高组合层面的覆盖率。

    这里只做「拿到排名」这一段：解析玩法策略、按窗口造分析器、取满 80 码的
    排名。怎么切注是领域问题，在 `portfolio.coverage_slips` 里——切法不该
    知道分析器长什么样。

    参数：
      select_n : 玩法选号数（决定用哪套策略，如 6=选6）。
      n_slips  : 生成的组数。
      pick_size: 每组的号码个数，默认=select_n。设成 >select_n（如 7）可让
                 每组覆盖更多号码，适配「选5复式」等玩法。
    """
    ranked = _ranked_numbers_for_play(analyzer, select_n)
    if not ranked:
        return []
    return portfolio.coverage_slips(ranked, n_slips,
                                    select_n if pick_size is None else pick_size)


def _ranked_numbers_for_play(analyzer, select_n: int) -> List[int]:
    """按该玩法的策略取满号码空间的排名。没有可用策略或权重时返回空。"""
    strategy = _strategies_mod.resolve_play_strategy(f'select_{select_n}',
                                                     allow_reference=True)
    if strategy is None:
        return []
    weights = {k: float(v) for k, v in (strategy.get('feature_weights') or {}).items() if v}
    if not weights:
        return []

    predictor = analyzer._build_window_analyzer(
        strategy.get('window_size', KL8_DEFAULT_HISTORY))
    ranking = predictor.get_ensemble_ranking(
        top_n=SPACE.size, feature_weights=weights,
        repeat_direction=strategy.get('repeat_direction', 'neutral'),
        frequency_mode=strategy.get('frequency_mode', 'mean_reversion'),
    )
    return [item['num'] for item in ranking]
