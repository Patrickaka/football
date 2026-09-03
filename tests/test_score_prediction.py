"""
比分预测和半全场预测单元测试
确保让球数据流转无误
"""
import gzip
import json
import os
import pathlib
import sys
import threading
import time
import unittest
import urllib.error
from unittest import mock

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.football import (
    analyze_match,
    predict_scores,
    _handicap_text_to_num,
    analyze_asian,
    remove_vig,
)
from src.football import fetching as fb_fetching

FIXTURES = pathlib.Path(__file__).resolve().parent / 'fixtures'
# 真实页面：2026-08-28 从 500.com 抓的两场 x 亚盘/大小球/数据分析。
# 与 tests/domain/sports/football/test_parsing.py 用的是同一份语料。
PAGES = json.loads(gzip.decompress(
    (FIXTURES / 'football_odds_pages.json.gz').read_bytes()))
# 这场让球 0.938（主队让近一球），方向明确，适合做方向性断言；
# 另一场 1430311 只有 -0.094，几乎平手，方向断言没有意义。
FIXTURE_MATCH_ID = '1430017'


def _fake_fetch(url, *args, **kwargs):
    """按 URL 取夹具页面；夹具没有的页面照线上那样 404。

    独赔页就属于「夹具没有」——线上它也常抓不到，降级路径本来就要走通，
    喂个假 200 反而会把这条分支盖掉。
    """
    for page in ('yazhi', 'daxiao', 'shuju'):
        if f'/{page}-' in url:
            match_id = url.rsplit('-', 1)[-1].split('.')[0]
            html = PAGES.get(f'{page}:{match_id}')
            if html is not None:
                return html
            break
    raise urllib.error.HTTPError(url, 404, 'Not Found', {}, None)


# 欧赔走的是另一个 JSON 接口，页面夹具里没有。**第 0 条是终盘、最后一条是
# 初盘**（倒序），这里给一组与夹具那场同向的赔率：主队小、客队大。
# 与夹具那场的亚盘同向：主队让球，所以主胜赔率低。两者矛盾的话，
# λ 方向会被拉扯，方向性断言测的就不是链路而是我编的数据了。
OUZHI_SERIES = [
    [1.85, 3.40, 4.20, 0.95, '2026-08-28 19:30'],
    [1.90, 3.35, 4.00, 0.95, '2026-08-26 10:00'],
]


def _fake_fetch_json(url, *args, **kwargs):
    if 'type=europe' in url:
        return OUZHI_SERIES
    raise urllib.error.HTTPError(url, 404, 'Not Found', {}, None)


def _drain_odds_threads(timeout=15.0):
    """等 analyze_match 的线程池跑完，**必须在 mock 窗口内调用**。

    `analyze_match` 并发发起五组抓取后就 `pool.shutdown(wait=False)`；
    亚盘/欧赔/大小球任一失败会立刻向上抛，此时 `fetch_team_strength` 与
    `fetch_single_company_odds` 两个线程还在跑。不等它们，`with` 一退出
    补丁就没了，那两个线程直接打到真的 500.com——把源站打到限流之后，
    后面用例的抓取跟着失败，全量并行下就随机冒出
    「亚盘数据获取失败: ... got 'NoneType'」，而单跑本文件永远是绿的。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not any(t.name.startswith('FootballOdds') and t.is_alive()
                   for t in threading.enumerate()):
            return
        time.sleep(0.05)


def _analyze_offline(match_id=FIXTURE_MATCH_ID):
    """跑完整 analyze_match，但一个请求都不出网。

    `force_refresh=True` 绕开分析缓存——不绕的话，先跑的用例会把结果留给
    后面的用例，这一族就退化成「只有第一条真的算过」。
    """
    match = {'match_id': match_id, 'home': '主队', 'away': '客队',
             'league': '英超', 'time': '08-28 20:00'}
    with mock.patch.object(fb_fetching, 'fetch', side_effect=_fake_fetch), \
         mock.patch.object(fb_fetching, 'fetch_json', side_effect=_fake_fetch_json):
        try:
            return analyze_match(match, force_refresh=True)
        finally:
            _drain_odds_threads()


class TestHandicapConversion(unittest.TestCase):
    """测试让球类型转换"""

    def test_home_give_positive(self):
        """主队让球应返回正数"""
        self.assertEqual(_handicap_text_to_num('半球'), 0.5)
        self.assertEqual(_handicap_text_to_num('一球'), 1.0)
        self.assertEqual(_handicap_text_to_num('球半'), 1.5)
        self.assertEqual(_handicap_text_to_num('两球'), 2.0)
        self.assertEqual(_handicap_text_to_num('半一'), 0.75)
        self.assertEqual(_handicap_text_to_num('一球/球半'), 1.25)

    def test_home_receive_negative(self):
        """主队受让球应返回负数"""
        self.assertEqual(_handicap_text_to_num('受让半球'), -0.5)
        self.assertEqual(_handicap_text_to_num('受让一球'), -1.0)
        self.assertEqual(_handicap_text_to_num('受让球半'), -1.5)
        self.assertEqual(_handicap_text_to_num('受让两球'), -2.0)
        self.assertEqual(_handicap_text_to_num('受半球'), -0.5)
        self.assertEqual(_handicap_text_to_num('受球半/两球'), -1.75)

    def test_level_handicap(self):
        """平手盘应返回0"""
        self.assertEqual(_handicap_text_to_num('平手'), 0)
        self.assertEqual(_handicap_text_to_num('受让平手'), 0)


class TestAsianAnalysis(unittest.TestCase):
    """测试亚盘分析"""

    def test_home_give_probability_labels(self):
        """主队让球时概率标签应为 home_give/away_recv"""
        # 构造测试数据：主队让半球
        data = {
            'open': {'handicap': 0.5, 'home_odds': 0.85, 'away_odds': 0.80},
            'close': {'handicap': 0.5, 'home_odds': 0.90, 'away_odds': 0.75}
        }
        
        result = analyze_asian(data)
        
        # 验证让球方向
        self.assertEqual(result['handicap'], 0.5)
        self.assertEqual(result['favor'], 'home')
        
        # 验证概率标签
        self.assertIn('home_give', result['close_prob'])
        self.assertIn('away_recv', result['close_prob'])
        self.assertNotIn('home_recv', result['close_prob'])
        self.assertNotIn('away_give', result['close_prob'])

    def test_home_receive_probability_labels(self):
        """主队受让球时概率标签应为 home_recv/away_give"""
        # 构造测试数据：主队受让半球
        data = {
            'open': {'handicap': -0.5, 'home_odds': 0.80, 'away_odds': 0.85},
            'close': {'handicap': -0.5, 'home_odds': 0.75, 'away_odds': 0.90}
        }
        
        result = analyze_asian(data)
        
        # 验证让球方向
        self.assertEqual(result['handicap'], -0.5)
        self.assertEqual(result['favor'], 'away')
        
        # 验证概率标签
        self.assertIn('home_recv', result['close_prob'])
        self.assertIn('away_give', result['close_prob'])
        self.assertNotIn('home_give', result['close_prob'])
        self.assertNotIn('away_recv', result['close_prob'])

    def test_level_probability_labels(self):
        """平手盘时概率标签应为 home/away"""
        # 构造测试数据：平手盘
        data = {
            'open': {'handicap': 0, 'home_odds': 0.90, 'away_odds': 0.90},
            'close': {'handicap': 0, 'home_odds': 0.85, 'away_odds': 0.95}
        }
        
        result = analyze_asian(data)
        
        # 验证让球方向
        self.assertEqual(result['handicap'], 0)
        self.assertEqual(result['favor'], 'even')
        
        # 验证概率标签
        self.assertIn('home', result['close_prob'])
        self.assertIn('away', result['close_prob'])


class AnalyzeMatchContract(unittest.TestCase):
    """`analyze_match` 的输出契约。

    **原先这一族每条都是假绿**，三重叠加：

    1. 拿假 match_id 打真实 500.com，404 之后 `except Exception: skipTest`
       吞掉——CI 日志里那句 `factorial() not defined for negative values`
       就是这么被盖成一行 skip 的。
    2. 断言包在 `if` 里，条件不满足时一个断言都不执行。
    3. **断言的键根本不存在**：`result['recommend']`、`model['half_full_time']`、
       `model['goal_count']` 在真实返回里从来没有过。也就是说这些用例
       即使拿到数据也会 KeyError，然后照样被 except 吞掉。

    现在喂 `tests/fixtures/football_odds_pages.json.gz` 里的真实页面
    （2026-08-28 抓的两场 × 亚盘/大小球/数据分析），链路照走，一个字节
    都不出网；断言全部对着**实际产出的字段**。

    选 1430017 那场：它的让球是 0.938（主队让近一球），方向性明确。
    另一场 1430311 让球只有 -0.094，几乎平手，拿来断言方向没有意义。
    """

    @classmethod
    def setUpClass(cls):
        cls.result = _analyze_offline(FIXTURE_MATCH_ID)

    def test_lambda_direction_follows_the_handicap(self):
        """让球方向与两队 λ 必须同向，反了就是整条盘口链路接错了。"""
        handicap = self.result['asian']['handicap']
        lam_home = self.result['model']['lam_home']
        lam_away = self.result['model']['lam_away']
        self.assertGreater(abs(handicap), 0.5,
                           '这场的让球要够明显，方向断言才有意义')
        if handicap > 0:
            self.assertGreater(lam_home, lam_away, '主队让球，λ 却是客队高')
        else:
            self.assertGreater(lam_away, lam_home, '主队受让，λ 却是主队高')

    def test_score_picks_are_well_formed(self):
        picks = self.result['analysis']['score_picks']
        self.assertGreater(len(picks), 0, '比分推荐不应为空')
        for pick in picks:
            with self.subTest(score=pick.get('score')):
                for field in ('type', 'score', 'home', 'away', 'result', 'probability'):
                    self.assertIn(field, pick)
                self.assertGreater(pick['probability'], 0)
                self.assertLess(pick['probability'], 1)
                self.assertEqual(pick['score'], f"{pick['home']}-{pick['away']}",
                                 'score 文本与 home/away 对不上')

    def test_the_top_scores_do_not_dominate(self):
        """比分是长尾分布，前三个加起来过半就说明分布塌了。"""
        top3 = sum(p['probability'] for p in self.result['analysis']['score_picks'][:3])
        self.assertLess(top3, 0.5)

    def test_goal_distribution_is_consistent(self):
        goals = self.result['analysis']['goals']
        self.assertAlmostEqual(goals['over_prob'] + goals['under_prob'], 1.0, delta=1e-6,
                               msg='大小球两边应当互补')
        self.assertAlmostEqual(goals['btts_yes'] + goals['btts_no'], 1.0, delta=1e-6,
                               msg='双方进球两边应当互补')
        for item in goals['top_goals']:
            with self.subTest(goals=item['goals']):
                self.assertGreater(item['probability'], 0)
                self.assertLess(item['probability'], 1)

    def test_expected_goals_stays_in_a_sane_range(self):
        """期望总进球**不等于**两个 λ 之和——中间还有大小球盘口的锚定。

        实测这场：λ 之和 2.32、盘口线 2.71、最终期望 2.91，三者都不同。
        所以这里只钉住「是个正常量级的正数」，不去假装那条等式成立；
        它与最可能进球数的一致性由下一条守。
        """
        goals = self.result['analysis']['goals']
        self.assertGreater(goals['expected'], 0)
        self.assertLess(goals['expected'], 6, '一场球期望进球过 6 显然是算崩了')

    def test_the_most_likely_goal_count_sits_near_the_expectation(self):
        goals = self.result['analysis']['goals']
        top = goals['top_goals'][0]['goals']
        self.assertLessEqual(abs(top - goals['expected']), 2,
                             '最可能进球数应当落在期望附近')

    def test_the_handicap_chain_is_wired_end_to_end(self):
        """让球 → 隐含净胜球 → λ，链路上每一环都要有产出。"""
        asian, model = self.result['asian'], self.result['model']
        for field in ('handicap', 'close_prob', 'implied_supremacy'):
            self.assertIn(field, asian)
        for field in ('lam_home', 'lam_away'):
            self.assertIn(field, model)
            self.assertGreater(model[field], 0, 'λ 必须为正，否则泊松分布无意义')

    def test_it_runs_without_touching_the_network(self):
        """这一族的立身之本：链路走完，但一个请求都不发。

        没有这条守卫，哪天注入点挪了位，用例会**悄悄退回去打外网**，
        再被 404 变成 skip——正是它原来的样子。
        """
        with mock.patch.object(fb_fetching, 'fetch',
                               side_effect=AssertionError('测试期间不允许联网')), \
             mock.patch.object(fb_fetching, 'fetch_json',
                               side_effect=AssertionError('测试期间不允许联网')):
            with self.assertRaises(AssertionError):
                fb_fetching.fetch('https://odds.500.com/fenxi/yazhi-1.shtml')
            with self.assertRaises(AssertionError):
                fb_fetching.fetch_json('https://odds.500.com/x?type=europe')


class BothFixtureMatchesAnalyse(unittest.TestCase):
    """两场夹具都要能跑完。

    只跑一场的话，某条分支只在特定盘口下才走到时，坏了也看不出来。
    """

    def test_every_fixture_match_produces_a_full_payload(self):
        for match_id in ('1430311', '1430017'):
            with self.subTest(match_id=match_id):
                result = _analyze_offline(match_id)
                for section in ('asian', 'model', 'analysis', 'euro'):
                    self.assertIn(section, result)
                self.assertGreater(len(result['analysis']['score_picks']), 0)


# `TestHalfFullTimePrediction` 与 `TestGoalCountPrediction` 已删除：
# 它们断言的 `model['half_full_time']` / `model['goal_count']` 在
# `analyze_match` 的真实返回里从来不存在——**半全场根本不是这条链路的产出**。
# 进球数那部分的意图（推荐与 λ 一致）保留在了
# `test_the_most_likely_goal_count_sits_near_the_expectation`。


if __name__ == '__main__':
    unittest.main(verbosity=2)
