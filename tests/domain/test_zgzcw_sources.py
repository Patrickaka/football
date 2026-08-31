import unittest
from unittest import mock

from src.beidan import fetching as beidan_fetching
from src.domain.sports.beidan import parsing as beidan_parsing
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


BEIDAN_ROW = '''
<table><tr id="tr_371" m="墨西超" t="2026-08-31 06:00:00">
 <td class="wh-1"><a>371</a></td><td class="wh-2">墨西超</td>
 <td class="wh-3"><span title="比赛时间:2026-08-31 10:10">10:10</span></td>
 <td class="wh-4 t-r" tn="蒙特雷"><a>蒙特雷</a></td><td class="wh-5">VS</td>
 <td class="wh-6 t-l" tn="圣路易斯"><a>圣路易斯</a></td>
 <td class="wh-8" newplayid="4553848"></td>
 <td class="wh-9"><div><span>1.83</span><span>3.74</span><span>3.82</span></div></td>
</tr></table>
'''


ASIAN_AVERAGE = '''
<table><tr firsttime="2026-08-24 22:14:55"><td>1</td><td>平均*</td>
 <td id="chupan-w-0" data="0.67">0.67</td><td id="chupan-s-0" data="0.25">平/半</td>
 <td id="chupan-l-0" data="1.11">1.11</td>
 <td cid="0" data="0.99">0.99</td><td cid="0" data="0.25">平/半</td>
 <td cid="0" data="0.83">0.83</td></tr></table>
'''


COMPANY_ASIAN_HISTORY = '''
<h2>韦*指数变化</h2><table>
<tr><th>序号</th><th>时间</th><th>更新</th><th>指数</th></tr>
<tr><td>1</td><td>2026-08-30 01:01:05</td><td><span>赛前0分</span></td>
 <td><span>0.81</span></td><td><span>平手</span></td><td><span>0.99</span></td>
 <td>52.37</td><td>47.63</td><td>0.92</td><td>0.98</td><td>0.95</td></tr>
<tr><td>2</td><td>2026-08-29 10:34:32</td><td><span>赛前35分</span></td>
 <td><span>1.05↑</span></td><td><span>半球↓</span></td><td><span>0.83↓</span></td>
 <td>47.16</td><td>52.84</td><td>1.04</td><td>0.90</td><td>0.97</td></tr>
<tr><td>3</td><td>2026-08-24 06:06:05</td><td><span>赛前125时3分</span></td>
 <td><span>0.81</span></td><td><span>平手</span></td><td><span>0.99</span></td>
 <td>52.37</td><td>47.63</td><td>0.95</td><td>0.95</td><td>0.95</td></tr>
</table>
'''


COMPANY_GOALS_HISTORY = COMPANY_ASIAN_HISTORY.replace(
    '<span>平手</span>', '<span>2.5球</span>').replace(
    '<span>半球↓</span>', '<span>2.5/3球↑</span>')


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

    def test_beidan_schedule_uses_analysis_id_and_average_odds(self):
        match = beidan_parsing.parse_zgzcw_schedule(BEIDAN_ROW, '2026-08-31')[0]
        self.assertEqual(match['id'], '4553848')
        self.assertEqual((match['spf_sp'], match['spf_s'], match['spf_f']),
                         (1.83, 3.74, 3.82))
        self.assertEqual(match['status'], 'not_started')

    def test_asian_average_is_two_real_snapshots(self):
        history = beidan_parsing.parse_zgzcw_asian_history(ASIAN_AVERAGE)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['handicap'], '0.25')
        self.assertEqual(history[-1]['away_odds'], 0.83)

    def test_company_asian_history_is_complete_and_chronological(self):
        history = beidan_parsing.parse_zgzcw_asian_company_history(
            COMPANY_ASIAN_HISTORY)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]['time'], '2026-08-24 06:06:05')
        self.assertEqual(history[-1]['time'], '2026-08-30 01:01:05')
        self.assertEqual(history[1]['handicap'], 0.5)
        self.assertEqual(history[1]['home_odds'], 1.05)
        self.assertEqual(history[1]['away_probability'], 52.84)

    def test_company_goals_split_line_uses_midpoint(self):
        history = beidan_parsing.parse_zgzcw_goals_company_history(
            COMPANY_GOALS_HISTORY)
        self.assertEqual(history[1]['line'], 2.75)
        self.assertEqual(history[1]['over_odds'], 1.05)
        self.assertEqual(history[1]['under_odds'], 0.83)

    def test_history_fetch_uses_full_bet365_detail_without_extra_request(self):
        with mock.patch.object(beidan_fetching, 'fetch_zgzcw',
                               return_value=COMPANY_ASIAN_HISTORY) as fetch:
            result = beidan_fetching.fetch_zgzcw_asian_history('4553842')
        self.assertEqual(result['history_source'], 'zgzcw_company_detail')
        self.assertEqual(result['company_id'], '2')
        self.assertEqual(result['company'], '36*')
        self.assertEqual(result['samples'], 3)
        self.assertEqual(fetch.call_count, 1)
        self.assertIn('/4553842/ypdb/zhishu?company_id=2',
                      fetch.call_args.args[0])

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
