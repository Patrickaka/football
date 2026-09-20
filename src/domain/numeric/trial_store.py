"""策略试验记录的持久化。

迁移前这些记录躺在一个 18MB 的 JSON 文件里，而写法是：每新增**一条**记录，
就把内存里全部 23564 条去重、按 `indent=2` 序列化、整文件重写。一次策略
验证会产生成百上千条试验，18MB 就被反复重写成百上千次。

入库后逐条追加。去重键与旧实现保持一致
（strategy_id + play_type + tournament_round + tested_at），
因为线上历史数据正是按它去过重的，换一个键会让重跑迁移时凭空多出记录。
"""
import json
import logging

from src.domain.numeric.repository import StrategyTrialRepository

log = logging.getLogger('domain.numeric.trial_store')

KEY_FIELDS = ('strategy_id', 'play_type', 'tournament_round', 'tested_at')
_JSON_FIELDS = ('feature_weights', 'model_weights')
_PLAIN_FIELDS = ('window_size', 'repeat_direction', 'raw_p_value',
                 'fdr_adjusted_p', 'validation_lift', 'n_permutations')
# 后加的字段，老记录没有。「没有这个键」与「值为空」是两件事：线上有 15949
# 条显式把 pool_max_last_numbers 记为 null，也有 3641 条根本没有
# practical_score 这个键。前者是「测过，无此项」，后者是「那时还没这项指标」。
_OPTIONAL_FIELDS = ('pool_diversify', 'pool_max_last_numbers', 'frequency_mode',
                    'final_selection_mode', 'practical_score')
#: 有列可落的字段。其余一律进 `extra`，一个都不能丢。
_MODELED_FIELDS = frozenset(KEY_FIELDS + _JSON_FIELDS + _PLAIN_FIELDS
                            + _OPTIONAL_FIELDS)
#: 落在 Float 列上的字段，不接受 bool。
_NUMERIC_FIELDS = frozenset(('raw_p_value', 'fdr_adjusted_p', 'validation_lift'))


def _numeric_or_none(value):
    """bool 一律当作「没有值」。

    `True`/`False` 是 int 的子类，存进 Float 列会变成 1.0/0.0。对 p 值来说
    **0.0 看起来像极其显著**，而它其实是个非法输入——调用方的
    `_safe_fdr_p_value` 本来认得出 False 并按 1.0（不显著）处理，经库往返
    一圈就认不出了，于是一个无效策略被判成显著，而且一声不响。
    """
    return None if isinstance(value, bool) else value


class TrialStore:
    def __init__(self, db, game):
        self.db = db
        self.game = game
        self._repo = StrategyTrialRepository(db)

    def append(self, trial):
        """追加一条。键已存在时保持原样，不覆盖。"""
        return self.append_many([trial])

    def append_many(self, trials):
        """批量追加，返回实际写入条数。

        批内与库内都去重：同一批里出现重复键时保留先出现的那条，
        与旧实现遍历时「先到先得」的行为一致。
        """
        rows, seen = [], set()
        existing = self._existing_keys()
        for trial in trials:
            key = tuple(str(trial.get(f, '')) for f in KEY_FIELDS)
            if key in seen or key in existing:
                continue
            seen.add(key)
            rows.append(_trial_to_row(self.game, trial))
        self._repo.insert_many(rows)
        return len(rows)

    def load(self):
        return [_row_to_trial(row)
                for row in self._repo.find_by(game=self.game,
                                              order_by=['play_type', 'tested_at'])]

    def by_play_type(self, play_type):
        """按玩法取全部试验，按测试时间排序。

        FDR 校正取「刚添加的那条」是按位置索引拿的，顺序不稳就会取错，
        算出来的校正 p 值会安静地对应到另一条策略上。
        """
        rows = self._repo.find_by(game=self.game, play_type=play_type,
                                  order_by='tested_at')
        return [_row_to_trial(row) for row in rows]

    def p_values(self, play_type):
        return [t.get('raw_p_value') for t in self.by_play_type(play_type)]

    def family_trials(self, play_types, exclude_rounds=()):
        """取一个玩法族的全部试验，按测试时间排序。

        FDR 的族不是单个 play_type——`MAIN_PLAYS` 的几种玩法合成一族，
        校正要在整族上做。`find_by` 只支持等值过滤，所以逐个玩法查完再合并；
        族里只有两三种玩法，这点往返换掉一个通用 IN 查询是划算的。
        """
        excluded = set(exclude_rounds)
        found = [trial for play_type in play_types
                 for trial in self.by_play_type(play_type)
                 if trial.get('tournament_round') not in excluded]
        return sorted(found, key=lambda trial: str(trial.get('tested_at')))

    def update_fields(self, trial, **values):
        """按四元键定位一条并改其中几个字段，其余字段原样保留。

        返回是否命中。调用方给的是内存里的记录，而库里那条可能带着它没有的
        字段（`extra` 里的那些），所以先读回整行再改，不能拿内存副本覆盖。
        """
        key = {field: str(trial.get(field, '')) for field in KEY_FIELDS}
        rows = self._repo.find_by(game=self.game, **key)
        if not rows:
            return False
        row = {**rows[0], **values}
        self._repo.upsert(row, key_cols=['game', *KEY_FIELDS])
        return True

    def replace(self, trial):
        """按四元键整行覆盖已有记录；键不存在时什么都不做。

        与 `append` 相反：那个「键已存在就保持原样」，这个专门用来把库里
        的旧行刷成文件里的新内容。返回是否命中。
        """
        key = {field: str(trial.get(field, '')) for field in KEY_FIELDS}
        if not self._repo.find_by(game=self.game, **key):
            return False
        self._repo.upsert(_trial_to_row(self.game, trial),
                          key_cols=['game', *KEY_FIELDS])
        return True

    def count(self, play_type=None):
        filters = {'game': self.game}
        if play_type is not None:
            filters['play_type'] = play_type
        return len(self._repo.find_by(**filters))

    def _existing_keys(self):
        return {tuple(str(row.get(f, '')) for f in KEY_FIELDS)
                for row in self._repo.find_by(game=self.game)}


def _trial_to_row(game, trial):
    row = {'game': game}
    row.update({field: str(trial.get(field, '')) for field in KEY_FIELDS})
    for field in _JSON_FIELDS:
        value = trial.get(field)
        row[field] = json.dumps(value, ensure_ascii=False) if value is not None else None
    row.update({field: _numeric_or_none(trial.get(field)) if field in _NUMERIC_FIELDS
                else trial.get(field)
                for field in _PLAIN_FIELDS})
    row.update({field: trial.get(field) for field in _OPTIONAL_FIELDS})
    # 记下这条记录当时到底带了哪些可选字段，读回时才分得清
    # 「值为空」和「根本没有这个键」
    present = [f for f in _OPTIONAL_FIELDS if f in trial]
    row['optional_present'] = json.dumps(present, separators=(',', ':'))
    extra = {k: v for k, v in trial.items() if k not in _MODELED_FIELDS}
    row['extra'] = json.dumps(extra, ensure_ascii=False) if extra else None
    return row


def _row_to_trial(row):
    trial = {field: row.get(field) for field in KEY_FIELDS}
    for field in _JSON_FIELDS:
        raw = row.get(field)
        trial[field] = json.loads(raw) if raw else None
    trial.update({field: row.get(field) for field in _PLAIN_FIELDS})

    try:
        present = json.loads(row.get('optional_present') or '[]')
    except (TypeError, ValueError):
        present = []
    for field in _OPTIONAL_FIELDS:
        if field in present:
            trial[field] = row.get(field)

    try:
        trial.update(json.loads(row.get('extra') or '{}'))
    except (TypeError, ValueError):
        log.warning('试验记录 %s 的 extra 无法解析，该条将缺少未建模字段',
                    tuple(row.get(f) for f in KEY_FIELDS))
    return trial
