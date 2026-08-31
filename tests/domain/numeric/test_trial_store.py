"""策略试验记录的持久化。

**这是阶段 3 要解决的最大一处数据问题。** 线上
`data/kl8_strategy_trials.json` 是 18MB、23564 条，而写入方式是：每新增
**一条**试验记录，就把内存里全部记录去重、按 `indent=2` 序列化、整文件重写。
一次策略验证会产生成百上千条试验，于是 18MB 被反复重写成百上千次。

结构依据实读线上文件（判据 4）：四元键
(strategy_id, play_type, tournament_round, tested_at) 零重复，与旧实现的
去重键一致；12 个字段恒存在，5 个可选（pool_diversify 23190 条、
frequency_mode / final_selection_mode 21131 条、practical_score 19923 条、
pool_max_last_numbers 只有 7241 条有值，其余为 null）。
"""
import unittest

from src.domain.numeric.repository import create_all
from src.domain.numeric.trial_store import TrialStore
from src.foundation.store import Database, make_engine

# 取自线上真实记录
REAL = {
    'strategy_id': 'candidate_freq_50',
    'play_type': 'select_3',
    'feature_weights': {'frequency': 1.0, 'position_residual': 0.0,
                        'road_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0,
                        'big_small': 0.0},
    'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
    'window_size': 50,
    'repeat_direction': 'neutral',
    'raw_p_value': 0.05994,
    'validation_lift': -0.0222,
    'n_permutations': 1000,
    'tested_at': '2026-06-26T14:07:25',
    'tournament_round': 'per_play_validation',
    'fdr_adjusted_p': 0.05994,
    'pool_diversify': True,
    'pool_max_last_numbers': None,
    'frequency_mode': 'neutral',
    'final_selection_mode': 'balanced',
    'practical_score': 0.4231,
}


def _trial(**overrides):
    return {**REAL, **overrides}


class _Base(unittest.TestCase):
    def setUp(self):
        self.db = Database(make_engine('sqlite+pysqlite:///:memory:'))
        create_all(self.db)
        self.store = TrialStore(self.db, game='kl8')


class RoundTripTests(_Base):
    def test_empty_store(self):
        self.assertEqual(self.store.load(), [])

    def test_round_trip_preserves_every_field(self):
        self.store.append(_trial())
        self.assertEqual(self.store.load(), [REAL])

    def test_nested_weight_dicts_survive(self):
        """特征权重的键随版本增删——线上一共出现过 13 种特征名，
        而单条记录只带其中几个。拆成列的话每加一个特征就要改表。"""
        self.store.append(_trial())
        loaded = self.store.load()[0]
        self.assertEqual(loaded['feature_weights'], REAL['feature_weights'])
        self.assertEqual(loaded['model_weights'], REAL['model_weights'])

    def test_optional_fields_absent_stay_absent(self):
        """线上 23564 条里有 3641 条没有 practical_score。补一个默认值进去，
        等于凭空造出「这条策略的实用分是 0」这个结论。"""
        sparse = {k: v for k, v in REAL.items()
                  if k not in ('practical_score', 'frequency_mode')}
        self.store.append(sparse)
        loaded = self.store.load()[0]
        self.assertNotIn('practical_score', loaded)
        self.assertNotIn('frequency_mode', loaded)

    def test_none_valued_field_is_kept_as_none(self):
        """pool_max_last_numbers 有 15949 条显式为 null——「有这个键且值为空」
        与「没有这个键」是两件事。"""
        self.store.append(_trial())
        self.assertIn('pool_max_last_numbers', self.store.load()[0])
        self.assertIsNone(self.store.load()[0]['pool_max_last_numbers'])

    def test_boolean_survives_as_boolean(self):
        self.store.append(_trial(pool_diversify=False))
        self.assertIs(self.store.load()[0]['pool_diversify'], False)


class AppendTests(_Base):
    """**逐条追加**，而不是整表重写。这正是本批要解决的问题。"""

    def test_appending_does_not_rewrite_the_rest(self):
        self.store.append(_trial(tested_at='2026-06-26T14:07:25'))
        self.store.append(_trial(tested_at='2026-06-26T14:07:26'))
        self.assertEqual(self.store.count(), 2)

    def test_same_key_is_deduplicated(self):
        """去重键与旧实现一致：strategy_id + play_type + tournament_round
        + tested_at。"""
        self.store.append(_trial())
        self.store.append(_trial(raw_p_value=0.999))
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(self.store.load()[0]['raw_p_value'], 0.05994)

    def test_each_key_component_distinguishes(self):
        self.store.append(_trial())
        for field, value in (('strategy_id', 'other'), ('play_type', 'select_4'),
                             ('tournament_round', 'other_round'),
                             ('tested_at', '2026-06-26T14:07:26')):
            with self.subTest(field=field):
                self.store.append(_trial(**{field: value}))
        self.assertEqual(self.store.count(), 5)

    def test_append_many(self):
        self.store.append_many([_trial(tested_at=f'2026-06-26T14:07:{i:02d}')
                                for i in range(10)])
        self.assertEqual(self.store.count(), 10)

    def test_append_many_deduplicates_within_the_batch(self):
        self.store.append_many([_trial(), _trial(raw_p_value=0.5)])
        self.assertEqual(self.store.count(), 1)

    def test_append_nothing(self):
        self.assertEqual(self.store.append_many([]), 0)


class QueryTests(_Base):
    """FDR 校正要按玩法取出全部试验的 p 值——这是唯一的真实查询模式。"""

    def setUp(self):
        super().setUp()
        self.store.append_many([
            _trial(play_type='select_3', tested_at='2026-06-26T14:07:25',
                   raw_p_value=0.1),
            _trial(play_type='select_3', tested_at='2026-06-26T14:07:26',
                   raw_p_value=0.2),
            _trial(play_type='select_5', tested_at='2026-06-26T14:07:27',
                   raw_p_value=0.3),
        ])

    def test_by_play_type(self):
        self.assertEqual([t['raw_p_value'] for t in self.store.by_play_type('select_3')],
                         [0.1, 0.2])
        self.assertEqual(len(self.store.by_play_type('select_5')), 1)
        self.assertEqual(self.store.by_play_type('不存在'), [])

    def test_by_play_type_is_ordered_by_test_time(self):
        """FDR 校正取「刚添加的那条」是按位置索引拿的，顺序不稳就会取错，
        算出来的校正 p 值会安静地对应到另一条策略上。

        strategy_id 与 tested_at 刻意反向：主键顺序是
        (strategy_id, ..., tested_at)，不显式按时间排的话，返回的就是
        主键顺序——在这组数据上恰好是倒过来的。
        """
        store = TrialStore(self.db, game='ordering')
        store.append_many([
            _trial(strategy_id='zzz', tested_at='2026-06-26T14:07:01'),
            _trial(strategy_id='aaa', tested_at='2026-06-26T14:07:99'),
        ])
        self.assertEqual([t['tested_at'] for t in store.by_play_type('select_3')],
                         ['2026-06-26T14:07:01', '2026-06-26T14:07:99'])

    def test_p_values_shortcut(self):
        self.assertEqual(self.store.p_values('select_3'), [0.1, 0.2])

    def test_count_by_play_type(self):
        self.assertEqual(self.store.count(play_type='select_3'), 2)
        self.assertEqual(self.store.count(), 3)


class GameIsolationTests(_Base):
    def test_games_do_not_see_each_other(self):
        other = TrialStore(self.db, game='other-game')
        self.store.append(_trial())
        self.assertEqual(other.load(), [])
        self.assertEqual(other.count(), 0)


if __name__ == '__main__':
    unittest.main()
