# -*- coding: utf-8 -*-
"""跨站队名差异大量出现在词中间，前后缀白名单接不住。

北单来源（zgzcw/okooo）与 500.com 对同一支队的写法常差一个字，且差异位置在
词中间：曼斯菲德/曼斯菲尔德、基马诺克/基尔马诺克。原规则只容忍整段前后缀，
这些一律判为不匹配，赛果永远回填不上。

放宽的安全边界是调用方：主客队必须**同时**匹配才认这一行，单边像不算数。
"""

import unittest

from src.domain.sports.football.settlement import _result_team_names_match as matches


class LooseTeamNameMatching(unittest.TestCase):

    def test_single_character_inserted_in_the_middle_still_matches(self):
        self.assertTrue(matches('曼斯菲德', '曼斯菲尔德'))
        self.assertTrue(matches('基马诺克', '基尔马诺克'))
        self.assertTrue(matches('克卢大学', '克卢日大学'))

    def test_single_character_substitution_still_matches(self):
        self.assertTrue(matches('莱斯特城', '莱切斯特城'))
        self.assertTrue(matches('科罗纳', '哥罗纳'))

    def test_existing_affix_tolerance_is_preserved(self):
        self.assertTrue(matches('索尔福德', '索尔福德市'))
        self.assertTrue(matches('温布尔登', 'AFC温布尔登'))
        self.assertTrue(matches('KTP科特卡', '科特卡'))

    def test_two_character_difference_is_still_rejected(self):
        """放宽一个字是权衡，放宽两个字就开始把不同球队并到一起。"""
        self.assertFalse(matches('曼城', '曼联'))
        self.assertFalse(matches('马德里竞技', '马德里体育会'))

    def test_short_names_do_not_get_the_edit_budget(self):
        """两字队名给一个字的预算等于只比一个字，必须挡住。"""
        self.assertFalse(matches('国安', '泰安'))
        self.assertFalse(matches('雷丁', '雷恩'))

    def test_unrelated_teams_still_do_not_match(self):
        self.assertFalse(matches('阿森纳', '切尔西'))
        self.assertFalse(matches('利物浦', '埃弗顿'))
