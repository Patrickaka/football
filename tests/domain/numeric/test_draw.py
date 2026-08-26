"""开奖记录：数字彩票领域的核心值对象。

字段与约束依据 2026-08-26 实读线上 `data/kl8_history.json`（判据 4）：
2048 期，每期恰好 20 个号码，issue 唯一，另带 date / source / fetched_at /
checksum 四个溯源字段。

**号码是这个领域里唯一不可失真的东西**。开奖结果是既成事实，一旦记错，
后面所有的命中统计、策略验证、回测结论全部作废，而且不会有任何报错——
所以校验放在构造处，而不是等用到时再说。
"""
import unittest

from src.domain.numeric.draw import Draw, DrawConflict, checksum_of, merge_draws

VALID = [4, 9, 10, 12, 17, 18, 22, 28, 33, 38, 42, 44, 47, 48, 61, 63, 64, 67, 73, 74]
# 取自线上真实记录
REAL = {
    'issue': '2026227', 'numbers': list(VALID), 'date': '2026-08-25',
    'source': 'api_huiniao', 'fetched_at': '2026-08-26T14:54:41',
    'checksum': 'ec0a8edd6cbd',
}


class ChecksumTests(unittest.TestCase):
    def test_matches_the_checksum_stored_online(self):
        """线上 2048 条记录都带着这个校验码，算法必须一致——否则迁移进来的
        历史记录会全部显示为「校验不符」。"""
        self.assertEqual(checksum_of(VALID), 'ec0a8edd6cbd')

    def test_order_does_not_matter(self):
        self.assertEqual(checksum_of(list(reversed(VALID))), checksum_of(VALID))

    def test_different_numbers_give_different_checksums(self):
        other = VALID[:-1] + [75]
        self.assertNotEqual(checksum_of(other), checksum_of(VALID))


class DrawParsingTests(unittest.TestCase):
    def test_parses_a_real_record(self):
        draw = Draw.parse(REAL)
        self.assertEqual(draw.issue, '2026227')
        self.assertEqual(draw.numbers, tuple(VALID))
        self.assertEqual(draw.date, '2026-08-25')
        self.assertEqual(draw.source, 'api_huiniao')
        self.assertEqual(draw.checksum, 'ec0a8edd6cbd')

    def test_numbers_are_sorted(self):
        draw = Draw.parse({**REAL, 'numbers': list(reversed(VALID))})
        self.assertEqual(draw.numbers, tuple(sorted(VALID)))

    def test_accepts_a_json_encoded_number_list(self):
        """上游 API 偶尔把号码序列化成字符串再塞进 JSON。"""
        import json

        draw = Draw.parse({**REAL, 'numbers': json.dumps(VALID)})
        self.assertEqual(draw.numbers, tuple(VALID))

    def test_accepts_the_alternate_field_names(self):
        """不同数据源用 draw_numbers / draw_date，同一个东西两种叫法。"""
        draw = Draw.parse({'issue': '1', 'draw_numbers': VALID,
                           'draw_date': '2026-08-25'})
        self.assertEqual(draw.numbers, tuple(VALID))
        self.assertEqual(draw.date, '2026-08-25')

    def test_checksum_is_derived_when_missing(self):
        draw = Draw.parse({k: v for k, v in REAL.items() if k != 'checksum'})
        self.assertEqual(draw.checksum, 'ec0a8edd6cbd')

    def test_issue_is_normalised_to_text(self):
        """期号在不同源里有时是整数。它是主键，类型必须稳定。"""
        draw = Draw.parse({**REAL, 'issue': 2026227})
        self.assertEqual(draw.issue, '2026227')


class DrawRejectionTests(unittest.TestCase):
    """坏数据一律返回 None 而不是抛异常：一条坏记录不该让整批导入失败。

    但**不能悄悄修正**——比如把 19 个号码补成 20 个。开奖结果是既成事实，
    修正等于伪造。
    """

    def _rejects(self, record, why):
        with self.subTest(why=why):
            self.assertIsNone(Draw.parse(record))

    def test_rejects_malformed_records(self):
        cases = [
            ('不是字典', 'plain string'),
            ('号码缺失', {'issue': '1'}),
            ('号码为空', {'issue': '1', 'numbers': []}),
            ('号码不是列表', {'issue': '1', 'numbers': 42}),
            ('坏 JSON 字符串', {'issue': '1', 'numbers': '[1,2,'}),
            ('号码含非数字', {'issue': '1', 'numbers': ['a'] * 20}),
            ('只有 19 个', {'issue': '1', 'numbers': VALID[:19]}),
            ('有 21 个', {'issue': '1', 'numbers': VALID + [80]}),
            ('号码重复', {'issue': '1', 'numbers': VALID[:19] + [VALID[0]]}),
            ('号码为 0', {'issue': '1', 'numbers': [0] + VALID[1:]}),
            ('号码超过 80', {'issue': '1', 'numbers': VALID[:19] + [81]}),
            ('期号为空', {'issue': '', 'numbers': VALID}),
            ('期号只有空白', {'issue': '   ', 'numbers': VALID}),
        ]
        for why, record in cases:
            self._rejects(record, why)


class HitTests(unittest.TestCase):
    """命中数是这个领域所有回测与验证的基本单位。"""

    def setUp(self):
        self.draw = Draw.parse(REAL)

    def test_counts_intersection(self):
        self.assertEqual(self.draw.hits([4, 9, 10]), 3)
        self.assertEqual(self.draw.hits([1, 2, 3]), 0)
        self.assertEqual(self.draw.hits(VALID), 20)

    def test_ignores_duplicates_in_the_candidate(self):
        """候选里重复选同一个号，不该被算成命中两次。"""
        self.assertEqual(self.draw.hits([4, 4, 4]), 1)

    def test_empty_candidate_hits_nothing(self):
        self.assertEqual(self.draw.hits([]), 0)


class MergeTests(unittest.TestCase):
    """合并新旧开奖记录。

    **号码冲突时保留旧值**——开奖结果不会变，冲突只能是某一边错了，
    自动覆盖等于用未经核实的数据抹掉已有的。冲突要报出来等人工确认。
    """

    OLD = [Draw.parse(REAL),
           Draw.parse({**REAL, 'issue': '2026226', 'date': '2026-08-24'})]

    def test_new_issues_are_appended(self):
        fresh = Draw.parse({**REAL, 'issue': '2026228'})
        merged, conflicts = merge_draws(self.OLD, [fresh])
        self.assertEqual(len(merged), 3)
        self.assertEqual(conflicts, [])

    def test_identical_repeat_is_not_a_conflict(self):
        merged, conflicts = merge_draws(self.OLD, [Draw.parse(REAL)])
        self.assertEqual(len(merged), 2)
        self.assertEqual(conflicts, [])

    def test_conflicting_numbers_keep_the_old_value(self):
        changed = Draw.parse({**REAL, 'numbers': VALID[:19] + [80]})
        merged, conflicts = merge_draws(self.OLD, [changed])
        by_issue = {d.issue: d for d in merged}
        self.assertEqual(by_issue['2026227'].numbers, tuple(VALID),
                         '冲突时用新值覆盖了旧值')
        self.assertEqual(len(conflicts), 1)

    def test_conflict_report_carries_both_sides(self):
        """报告要能让人直接判断哪边对，不必再去翻原始文件。"""
        changed = Draw.parse({**REAL, 'numbers': VALID[:19] + [80],
                              'source': 'api_other'})
        _, conflicts = merge_draws(self.OLD, [changed])
        conflict = conflicts[0]
        self.assertIsInstance(conflict, DrawConflict)
        self.assertEqual(conflict.issue, '2026227')
        self.assertEqual(conflict.kept.numbers, tuple(VALID))
        self.assertEqual(conflict.rejected.numbers, tuple(VALID[:19] + [80]))
        self.assertEqual(conflict.kept.source, 'api_huiniao')
        self.assertEqual(conflict.rejected.source, 'api_other')

    def test_result_is_ordered_newest_first(self):
        """线上文件就是新→旧，下游多处直接取 [0] 当最新一期。"""
        older = Draw.parse({**REAL, 'issue': '2026100', 'date': '2026-04-01'})
        newer = Draw.parse({**REAL, 'issue': '2026228', 'date': '2026-08-26'})
        merged, _ = merge_draws(self.OLD, [older, newer])
        self.assertEqual([d.issue for d in merged],
                         ['2026228', '2026227', '2026226', '2026100'])

    def test_merging_into_an_empty_history(self):
        merged, conflicts = merge_draws([], [Draw.parse(REAL)])
        self.assertEqual(len(merged), 1)
        self.assertEqual(conflicts, [])


class SerialisationTests(unittest.TestCase):
    def test_round_trips_through_a_plain_dict(self):
        """进缓存与落库都要求纯 JSON 类型（裁决 C 的同一条约束）。"""
        import json

        draw = Draw.parse(REAL)
        payload = draw.to_dict()
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        self.assertEqual(Draw.parse(payload), draw)

    def test_equality_is_by_value(self):
        self.assertEqual(Draw.parse(REAL), Draw.parse(dict(REAL)))
        self.assertNotEqual(Draw.parse(REAL),
                            Draw.parse({**REAL, 'issue': '2026226'}))


if __name__ == '__main__':
    unittest.main()
