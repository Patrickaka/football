import unittest
from unittest import mock

from src.domain.sports.basketball import zgzcw_parsing as basketball
from src.domain.sports.basketball.prediction import PredictionService
from src.football.zgzcw_lottery import parse_zgzcw_jczq_schedule


FOOTBALL_ROW = '''
<table><tr id="tr_2041175" mn="周一001" m="芬超" rq="-1">
 <td class="wh-1"><code>周一</code><i>001</i></td>
 <td class="wh-2" title="芬超">芬超</td>
 <td class="wh-3"><span title="比赛时间:2026-09-01 00:00">00:00</span></td>
 <td class="wh-4 t-r"><a title="国际图尔">国际图尔</a></td>
 <td class="wh-5">VS</td>
 <td class="wh-6 t-l"><a title="库普斯">库普斯</a></td>
 <td class="wh-8">
  <div class="tz-area frq" pid="49"><em class="rq">0</em>
   <a id="td_2041175_49_0">2.15</a><a id="td_2041175_49_1">3.20</a><a id="td_2041175_49_2">2.85</a></div>
  <div class="tz-area rqq" pid="22"><em class="rq jian">-1</em>
   <a id="td_2041175_22_0">4.60</a><a id="td_2041175_22_1">3.75</a><a id="td_2041175_22_2">1.55</a></div>
 </td>
 <td class="wh-10" newplayid="4468186"></td>
</tr></table>
'''


BASKET_SF = '''
<table><tr id="tr_2041189" m="WNBA" t="2026-08-29 22:40" rq="-9.5">
 <td class="wh-1"><code>周六</code><i>301</i></td><td class="wh-2">WNBA</td>
 <td class="wh-3"><span title="比赛时间:2026-08-30 01:00">01:00</span></td>
 <td class="wh-4 t-r"><a>天空</a></td><td class="wh-5">VS</td>
 <td class="wh-6 t-l"><a>自由人</a></td>
 <td class="wh-7"><div class="bets-area" pid="26"><em class="total">0</em>
  <a id="td_2041189_26_0">4.08</a><a id="td_2041189_26_1">1.07</a></div>
  <div class="bets-area rqq" pid="27"><em class="total jian">-9.5</em>
  <a id="td_2041189_27_0">1.65</a><a id="td_2041189_27_1">1.75</a></div></td>
 <td class="wh-8" newplayid="3908987"></td>
</tr></table>
'''


BASKET_DX = BASKET_SF.replace(
    '<div class="bets-area" pid="26"><em class="total">0</em>\n  <a id="td_2041189_26_0">4.08</a><a id="td_2041189_26_1">1.07</a></div>\n  <div class="bets-area rqq" pid="27"><em class="total jian">-9.5</em>\n  <a id="td_2041189_27_0">1.65</a><a id="td_2041189_27_1">1.75</a></div>',
    '<div class="bets-area" pid="29"><em class="total">179.5</em>\n  <a id="td_2041189_29_0">1.70</a><a id="td_2041189_29_1">1.70</a></div>')


class ZgzcwSourceTests(unittest.TestCase):
    def test_football_offer_fields_and_market_order(self):
        match = parse_zgzcw_jczq_schedule(FOOTBALL_ROW)[0]
        self.assertEqual(match['zgzcw_id'], '2041175')
        self.assertEqual(match['analysis_id'], '4468186')
        self.assertEqual((match['home'], match['away']), ('国际图尔', '库普斯'))
        self.assertEqual(match['lottery_handicap'], -1)
        self.assertEqual(match['spf_odds'], {'胜': 2.15, '平': 3.2, '负': 2.85})
        self.assertEqual(match['rqspf_odds'], {'让胜': 4.6, '让平': 3.75, '让负': 1.55})

    def test_basketball_reverses_guest_home_display_and_merges_total(self):
        match = basketball.merge_schedule_pages(BASKET_SF, BASKET_DX)[0]
        self.assertEqual((match['home'], match['away']), ('自由人', '天空'))
        self.assertEqual((match['spf_home'], match['spf_away']), (1.07, 4.08))
        self.assertEqual((match['rqspf_home'], match['rqspf_away']), (1.75, 1.65))
        self.assertEqual(match['handicap'], -9.5)
        self.assertEqual((match['dx_over'], match['dx_under'], match['total_line']),
                         (1.7, 1.7, 179.5))

    def test_basketball_empty_primary_reports_500_fallback(self):
        class Analyzer:
            analyze_spf = staticmethod(lambda match, movement: {})
            analyze_rqspf = staticmethod(lambda match, movement: {})
            analyze_daxiao = staticmethod(lambda match, movement: {})

        fallback_match = {'id': '500-1', 'status': 'not_started'}
        service = PredictionService(
            analyzer=Analyzer(),
            schedule_sources={
                'zgzcw': lambda date: [],
                '500': lambda date: [fallback_match],
            })
        result = service.generate(date='2026-08-31', source='zgzcw',
                                  use_movement=False)
        self.assertEqual(result['source'], '500')
        self.assertEqual(result['requested_source'], 'zgzcw')
        self.assertTrue(result['source_fallback'])


if __name__ == '__main__':
    unittest.main()
