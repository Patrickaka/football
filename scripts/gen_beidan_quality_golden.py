"""从 beidan 的 quality/upset 实现生成黄金快照。

两个模块都是纯计算（AST 查过：零网络、零存储、零时钟），直接调用即可。

**语料按阈值的两侧铺开**：三档爆冷判定（high/medium/confident）各有三个条件，
只喂「明显超过」或「明显不足」的样本，把阈值改一点点是测不出来的（判据 5）。

用法：
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 \\
        scripts/gen_beidan_quality_golden.py /tmp/beidan_quality_old.json
"""
import json
import sys

sys.path.insert(0, '.')

from tests.domain.golden import as_comparable

import src.beidan.quality as quality_mod
import src.beidan.upset as upset_mod

# 1X2 概率组，围绕三档阈值的边界构造。
#   high     : fav < 0.45  且 gap <= 0.10 且 upset_mass >= 0.58
#   medium   : fav < 0.52  且 gap <= 0.16 且 upset_mass >= 0.52
#   confident: 未预警 且 fav >= 0.58 且 gap >= 0.20
PROB_SETS = {
    # 三档各取「刚好命中」与「差一点」两组
    'high_hit': {'胜': 0.42, '平': 0.33, '负': 0.25},
    'high_miss_fav': {'胜': 0.46, '平': 0.30, '负': 0.24},      # fav 刚超 0.45
    'high_miss_gap': {'胜': 0.44, '平': 0.28, '负': 0.28},      # gap 刚超 0.10
    # medium 的三个条件里 mass >= 0.52 最容易被忽略——**上一版这三组的名字
    # 都带 medium，实际却全落在 low**（fav=0.50 时 mass 只有 0.50）。
    # 语料的名字必须与它真正走到的分支相符，否则读的人会以为这一档测过了
    'medium_hit': {'胜': 0.48, '平': 0.34, '负': 0.18},          # 三条恰好都满足
    'medium_miss_fav': {'胜': 0.53, '平': 0.30, '负': 0.17},     # fav 刚超 0.52
    'medium_miss_gap': {'胜': 0.48, '平': 0.31, '负': 0.21},     # gap 刚超 0.16
    'medium_miss_mass': {'胜': 0.50, '平': 0.35, '负': 0.15},    # mass 差 0.02
    'confident_hit': {'胜': 0.62, '平': 0.22, '负': 0.16},
    'confident_miss_fav': {'胜': 0.57, '平': 0.25, '负': 0.18},  # fav 差一点
    'confident_miss_gap': {'胜': 0.60, '平': 0.45, '负': -0.05},  # gap 差一点
    'draw_favorite': {'胜': 0.30, '平': 0.40, '负': 0.30},
    'away_favorite': {'胜': 0.20, '平': 0.28, '负': 0.52},
    'split_hair': {'胜': 0.35, '平': 0.34, '负': 0.31},          # lead < 0.035
    # quality 的 strong 档：top >= 0.65 且 lead >= 0.10。
    # **上一版语料最高只有 0.62，strong 与 high_precision 两条分支一次都没走到**
    'strong_hit': {'胜': 0.68, '平': 0.20, '负': 0.12},
    'strong_miss_prob': {'胜': 0.64, '平': 0.22, '负': 0.14},    # top 差一点
    'strong_miss_lead': {'胜': 0.66, '平': 0.60, '负': -0.26},   # lead 差一点
    'high_precision_hit': {'胜': 0.75, '平': 0.15, '负': 0.10},  # 再过 0.70 那道
    'high_precision_miss': {'胜': 0.69, '平': 0.19, '负': 0.12}, # 是 strong 但不够精
    'empty': {},
    'all_none': {'胜': None, '平': None, '负': None},
}

# 亚盘方向 × 比分一致性，用来触发 quality 的 conflict 分支
CONTEXTS = {
    'none': None,
    'home_backing': {'asian_direction': 'home_backing'},
    'away_backing': {'asian_direction': 'away_backing'},
    'score_conflict': {'score_consistency': {'conflict': True}},
}

# 比分分布：元组键与字符串键两种形态都要覆盖——`_result_from_score`
# 两种都认，而**认错了不会报错，只会让爆冷比分挑反方向**
SCORE_MATRICES = {
    'tuple_keys': {(1, 0): 0.12, (1, 1): 0.11, (0, 1): 0.09,
                   (2, 1): 0.08, (0, 0): 0.07, (2, 0): 0.06},
    'string_keys': {'1-0': 0.12, '1-1': 0.11, '0-1': 0.09,
                    '2-1': 0.08, '0-0': 0.07, '2-0': 0.06},
    'colon_keys': {'1:0': 0.12, '1:1': 0.11, '0:1': 0.09},
    'malformed': {'abc': 0.5, (1, 0): 0.3, None: 0.2},
    'empty': {},
}

# `assess_score_consistency` 的入参有两种形态：dict 列表与 (比分, 概率) 元组列表
SCORE_LISTS = {
    'dicts_home': [{'score': '2-1', 'probability': 0.12, 'home_goals': 2, 'away_goals': 1},
                   {'score': '1-0', 'probability': 0.10, 'home_goals': 1, 'away_goals': 0},
                   {'score': '1-1', 'probability': 0.09, 'home_goals': 1, 'away_goals': 1}],
    'dicts_mixed': [{'score': '1-1', 'probability': 0.12},
                    {'score': '2-1', 'probability': 0.10},
                    {'score': '0-1', 'probability': 0.09}],
    'tuples': [('2-1', 0.12), ('1-1', 0.10), ('0-1', 0.09)],
    'no_probability': [{'score': '2-1'}, {'score': '1-0'}, {'score': '1-1'}],
    'unparsable': [{'score': 'x'}, {'score': '1-0', 'probability': 0.1}],
    'empty': [],
}


def entries():
    for name, probs in PROB_SETS.items():
        yield f'upset_risk:{name}', upset_mod.assess_upset_risk(probs)
        for ctx_name, context in CONTEXTS.items():
            yield (f'quality:{name}:{ctx_name}',
                   quality_mod.assess_recommendation_quality(probs, context=context))
        # 指定 prediction 与 top1 不同时，quality 会改用指定的那个
        for prediction in ('胜', '平', '负'):
            yield (f'quality_pred:{name}:{prediction}',
                   quality_mod.assess_recommendation_quality(probs, prediction=prediction))

    for matrix_name, matrix in SCORE_MATRICES.items():
        for favorite in ('胜', '平', '负', None):
            for top_n in (1, 2, 3):
                yield (f'upset_scores:{matrix_name}:{favorite}:{top_n}',
                       upset_mod.pick_upset_scores(matrix, favorite, top_n=top_n))

    for list_name, scores in SCORE_LISTS.items():
        for prediction in ('胜', '平', '负', None):
            yield (f'consistency:{list_name}:{prediction}',
                   upset_mod.assess_score_consistency(scores, prediction))

    for key in ((1, 0), (0, 1), (1, 1), '2-1', '1:1', 'bad', None):
        yield f'result_from_score:{key}', upset_mod._result_from_score(key)


def main(out_path):
    golden = {key: as_comparable(value) for key, value in entries()}
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(golden, fh, ensure_ascii=False, sort_keys=True, indent=1)
    print(f'共 {len(golden)} 条 → {out_path}')


if __name__ == '__main__':
    main(sys.argv[1])
