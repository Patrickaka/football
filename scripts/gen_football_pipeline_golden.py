# -*- coding: utf-8 -*-
"""生成 analyze_match 两个抽出段（市场锚定 / 结果组装）的黄金语料。

语料与测试共用 `tests/domain/sports/football/_pipeline_corpus`，
两边看的是同一批输入。
"""
from copy import deepcopy

from src.domain.sports.football.analysis_result import build_analysis_result
from src.domain.sports.football.market_anchoring import anchor_candidates_to_market
from tests.domain.golden import describe_exception
from tests.domain.sports.football._pipeline_corpus import (
    ASIANS, BASE_PARTS, CANDIDATES, EUROS, PROFILES, TOTALS,
)

_SEEN = {}


def _key(label):
    """键必须唯一——同名 label 会互相覆盖，黄金里少掉的条目一声不响。"""
    _SEEN[label] = _SEEN.get(label, 0) + 1
    return label if _SEEN[label] == 1 else f'{label}#{_SEEN[label]}'


def _y(label, call):
    try:
        yield _key(label), call()
    except Exception as exc:
        yield _key(label), describe_exception(exc)


def entries():
    _SEEN.clear()
    for total in TOTALS:
        for euro in EUROS:
            for asian in ASIANS:
                for profile in PROFILES:
                    label = (f'anchor/{total and total.get("line")}'
                             f'/{euro and euro.get("H")}'
                             f'/{asian and asian.get("handicap")}'
                             f'/{profile and profile.get("applied")}')
                    yield from _y(label,
                                  lambda t=total, e=euro, a=asian, p=profile:
                                  anchor_candidates_to_market(
                                      list(CANDIDATES), t, e, a, p))

    # `build_analysis_result` 会**就地**往 lottery 里塞 accuracy_gate，
    # 浅拷贝挡不住——不深拷贝的话第二次调用看到的就是被污染的输入，
    # 黄金因此不可复现
    yield from _y('assemble/base',
                  lambda: build_analysis_result(**deepcopy(BASE_PARTS)))

    for key in sorted(BASE_PARTS):
        for blank in (None, {}, []):
            parts = deepcopy(BASE_PARTS)
            parts[key] = blank
            yield from _y(f'assemble/{key}={blank!r}',
                          lambda p=parts: build_analysis_result(**p))
