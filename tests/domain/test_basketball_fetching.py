"""basketball 抓取层接入 foundation/fetch。

抓取层只保留一套重试与熔断（FetchClient 的），transport 只负责领域知识：
按站点定编码、识别人机验证页。识别到验证页就**抛异常**——它是
PermanentFetchError，FetchClient 见到就单次开路，既不重试也不必攒够阈值，
不需要 transport 自己再维护一个「封锁 N 秒」计时器。
"""
import tempfile
import unittest
import unittest.mock

from src.domain.sports.basketball import fetching
from src.domain.sports.basketball.fetching import (
    VerificationPage, ZgzcwTransport, build_fetch_client, dispatch_transport,
)
from src.foundation.fetch import FetchError

SCHEDULE = 'https://cp.zgzcw.com/lottery/jclq.action'
ODDS = 'https://odds.zgzcw.com/basketball/1/'


class _RecordingTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, timeout):
        self.calls.append(url)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class BuildFetchClientTests(unittest.TestCase):
    def test_fetches_through_injected_transport(self):
        transport = _RecordingTransport(['<html>ok</html>'])
        client = build_fetch_client(transport=transport, sleep_fn=lambda _: None)
        self.assertEqual(client.get('https://trade.500.com/jclq/'), '<html>ok</html>')

    def test_rate_limits_are_isolated_per_domain(self):
        """两个源站的限速互不影响：zgzcw 被限速不该拖慢 500.com。"""
        transport = _RecordingTransport(['a', 'b'])
        slept = []
        client = build_fetch_client(transport=transport, sleep_fn=slept.append)
        client.get(SCHEDULE)
        client.get('https://trade.500.com/jclq/')
        self.assertEqual(sum(1 for s in slept if s > 0), 0,
                         '不同域名的首次请求都不该等待')

    def test_zgzcw_is_rate_limited_more_conservatively(self):
        """zgzcw 会弹人机验证，限速须比 500.com 更保守。"""
        client = build_fetch_client(transport=_RecordingTransport([]),
                                    sleep_fn=lambda _: None)
        zgzcw = client.limiters.for_domain('cp.zgzcw.com').rate_per_sec
        wubai = client.limiters.for_domain('trade.500.com').rate_per_sec
        self.assertLess(zgzcw, wubai)

    def test_every_zgzcw_subdomain_is_rate_limited(self):
        """三个子域都要限速——漏掉一个就等于那条路径无限速裸奔。"""
        client = build_fetch_client(transport=_RecordingTransport([]),
                                    sleep_fn=lambda _: None)
        default = client.limiters.for_domain('trade.500.com').rate_per_sec
        for host in fetching.ZGZCW_HOSTS:
            with self.subTest(host=host):
                self.assertLess(client.limiters.for_domain(host).rate_per_sec,
                                default)

    def test_verification_page_counts_as_failure_and_trips_breaker(self):
        """验证页抛异常 → 计入熔断，不需要第二套封锁计时器。"""
        transport = _RecordingTransport([VerificationPage('captcha')] * 10)
        client = build_fetch_client(transport=transport, sleep_fn=lambda _: None,
                                    max_retries=1, failure_threshold=2)
        with self.assertRaises(FetchError):
            client.get(SCHEDULE)
        before = len(transport.calls)
        with self.assertRaises(FetchError):
            client.get(SCHEDULE)
        self.assertEqual(len(transport.calls), before,
                         '熔断开路后不应再发出请求')

    def test_snapshot_serves_as_fallback(self):
        root = tempfile.mkdtemp(prefix='bb-snap-')
        good = build_fetch_client(transport=_RecordingTransport(['cached page']),
                                  sleep_fn=lambda _: None, snapshots_root=root)
        good.get('https://trade.500.com/jclq/')

        broken = build_fetch_client(
            transport=_RecordingTransport([IOError('down')] * 5),
            sleep_fn=lambda _: None, snapshots_root=root, max_retries=2)
        self.assertEqual(broken.get('https://trade.500.com/jclq/'), 'cached page')


class DispatchTransportTests(unittest.TestCase):
    def _dispatch(self, url):
        picked = []
        transport = dispatch_transport(
            zgzcw=lambda u, timeout: picked.append('zgzcw') or 'z',
            default=lambda u, timeout: picked.append('default') or 'd',
        )
        transport(url, 10)
        return picked

    def test_zgzcw_url_goes_to_zgzcw_impl(self):
        self.assertEqual(self._dispatch(SCHEDULE), ['zgzcw'])

    def test_every_known_zgzcw_host_is_dispatched(self):
        """三个子域都必须认得——漏一个就会被当成 500.com 用错编码和 referer。"""
        for host in fetching.ZGZCW_HOSTS:
            with self.subTest(host=host):
                self.assertEqual(self._dispatch(f'https://{host}/x'), ['zgzcw'])

    def test_other_urls_go_to_default_impl(self):
        self.assertEqual(self._dispatch('https://trade.500.com/jclq/'), ['default'])

    def test_dispatch_is_by_host_not_substring(self):
        """按主机名判断，而非 url 里出现源站名就算——
        查询参数里带源站名的链接不该被误分派。"""
        self.assertEqual(
            self._dispatch('https://trade.500.com/jclq/?ref=cp.zgzcw.com'),
            ['default'])


class ZgzcwTransportTests(unittest.TestCase):
    """transport 只做两件领域知识的事：带对 referer、认出验证页。"""

    def _urlopen(self, raw):
        import urllib.request

        captured = {}

        class _Response:
            def read(self):
                return raw

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def _fake(req, timeout=None):
            captured['url'] = req.full_url
            captured['headers'] = req.headers
            return _Response()

        return unittest.mock.patch.object(urllib.request, 'urlopen', _fake), captured

    def _fetch(self, text):
        patcher, captured = self._urlopen(text.encode('utf-8'))
        with patcher:
            body = ZgzcwTransport()(SCHEDULE, 10)
        return body, captured

    def test_decodes_utf8_page(self):
        body, _ = self._fetch('周三301 美职女篮 金州女武神VS太阳')
        self.assertEqual(body, '周三301 美职女篮 金州女武神VS太阳')

    def test_sends_site_referer(self):
        """不带 referer 直连详情页会被判为异常流量。"""
        _, captured = self._fetch('<html>ok</html>')
        self.assertEqual(captured['headers'].get('Referer'),
                         'https://cp.zgzcw.com/')

    def test_verification_page_raises_instead_of_returning_html(self):
        """旧代码把验证页当正文返回，解析器匹不到东西就静静给出 0 场比赛。

        抛出来才有人知道——而且它是 PermanentFetchError，熔断单次开路。
        """
        for marker in ('captcha', '访问验证', '安全验证'):
            with self.subTest(marker=marker):
                patcher, _ = self._urlopen(
                    f'<html><body>{marker}</body></html>'.encode('utf-8'))
                with patcher, self.assertRaises(VerificationPage):
                    ZgzcwTransport()(SCHEDULE, 10)

    def test_marker_detection_is_case_insensitive(self):
        patcher, _ = self._urlopen(b'<html>Please solve the CAPTCHA</html>')
        with patcher, self.assertRaises(VerificationPage):
            ZgzcwTransport()(SCHEDULE, 10)

    def test_normal_page_is_not_mistaken_for_verification(self):
        body, _ = self._fetch('<html>周三301 美职女篮</html>')
        self.assertIn('美职女篮', body)


class VerificationPageIsPermanentTests(unittest.TestCase):
    """撞验证页不重试。

    熔断能替代旧代码手写的「撞墙就跳过 N 秒」开关，但前提是把验证页归为
    PermanentFetchError：否则它要 failure_threshold 次失败才开路，每次失败
    还要先重试 max_retries 遍——在 zgzcw 的 0.5 rps 限速下，这笔账让冷启动
    从 2 秒涨到十几秒。验证页是确定性的（同一出口 IP 再试还是同样结果），
    不该走退避重试那条路。
    """

    def _client(self, **kwargs):
        self.calls = []

        def transport(url, timeout):
            self.calls.append(url)
            raise VerificationPage('中国足彩网返回验证页')

        self.slept = []
        return build_fetch_client(transport=transport, sleep_fn=self.slept.append,
                                  **kwargs)

    def test_verification_page_is_fetched_once_not_three_times(self):
        client = self._client(max_retries=3)
        with self.assertRaises(FetchError):
            client.get(SCHEDULE)
        self.assertEqual(len(self.calls), 1)

    def test_verification_page_does_not_burn_backoff_sleeps(self):
        client = self._client(max_retries=3)
        with self.assertRaises(FetchError):
            client.get(SCHEDULE)
        self.assertEqual([s for s in self.slept if s >= 0.5], [],
                         '为一次注定失败的请求白等了退避')

    def test_one_hit_opens_the_breaker(self):
        """撞一次就够。攒够 failure_threshold 再开路，等于把已知必败的请求
        重复几遍——zgzcw 限速 0.5 rps，默认阈值 5 就是十几秒的冷启动。"""
        client = self._client(max_retries=3)
        with self.assertRaises(FetchError):
            client.get(SCHEDULE)
        self.assertEqual(len(self.calls), 1)
        with self.assertRaises(FetchError):
            client.get(SCHEDULE)
        self.assertEqual(len(self.calls), 1, '熔断开路后仍在白跑请求')


class BreakerKeyTests(unittest.TestCase):
    """熔断按域名隔离，一个上游不该影响另一个。

    **线上真实发生过的故障**：详情页连撞验证页触发熔断，而熔断当时按域名建，
    同域名的赛程页被一起短路——接口返回 200、比赛数 0、走势全空，不报任何错。
    换到 zgzcw 后赛程走 cp.、赔率走 odds./fenxi.，本就是不同子域，
    netloc 天然把它们分开；**前提是 key 保留子域**，退化成注册域就又合流了。
    """

    def test_schedule_and_odds_subdomains_use_different_breakers(self):
        self.assertNotEqual(fetching.breaker_key(SCHEDULE),
                            fetching.breaker_key(ODDS))

    def test_key_keeps_the_subdomain(self):
        self.assertEqual(fetching.breaker_key(SCHEDULE), 'cp.zgzcw.com')
        self.assertEqual(fetching.breaker_key('https://trade.500.com/jclq/'),
                         'trade.500.com')

    def test_blocked_odds_do_not_short_circuit_the_schedule(self):
        calls = []

        def transport(url, timeout):
            calls.append(url)
            if 'odds.' in url:
                raise VerificationPage('中国足彩网返回验证页')
            return '赛程页正文'

        client = build_fetch_client(transport=transport, sleep_fn=lambda _: None)
        for _ in range(4):
            with self.assertRaises(FetchError):
                client.get(ODDS)

        self.assertEqual(client.get(SCHEDULE), '赛程页正文',
                         '赔率页的熔断把赛程页一起打掉了')


class UrllibGetEncodingTests(unittest.TestCase):
    """500.com 的页面是 gbk，必须解对，否则中文全变问号。

    迁移时这里写成了「按候选编码依次 decode(errors=\'replace\')，
    失败就试下一个」——但 errors=\'replace\' **永远不会抛异常**，于是
    第一个候选（utf-8）总是"成功"，gbk 回退一次都没走到。整页变成乱码，
    正则一条也匹不上，接口返回 200 加空列表，看不出任何异常。

    这个 bug 从抓取层迁移那批一直潜伏到端点切换才显形：那批的测试只覆盖了
    澳客的 gb2312 解码，没有一条测 urllib_get；而它当时零消费者，
    也就没有任何运行时信号。
    """

    def _transport(self, raw):
        import urllib.request

        class _Response:
            def read(self):
                return raw

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return unittest.mock.patch.object(
            urllib.request, 'urlopen', lambda *a, **k: _Response())

    def _get(self, raw, **kwargs):
        with self._transport(raw):
            return fetching.urllib_get('https://trade.500.com/jclq/', 10, **kwargs)

    def test_gbk_page_is_decoded_correctly(self):
        text = '周三301 美职女篮 金州女武神VS太阳'
        self.assertEqual(self._get(text.encode('gbk')), text)

    def test_utf8_page_is_decoded_correctly(self):
        text = '周三301 美职女篮 金州女武神VS太阳'
        self.assertEqual(self._get(text.encode('utf-8')), text)

    def test_explicit_encoding_is_tried_first(self):
        text = '中文内容'
        self.assertEqual(self._get(text.encode('gb2312'), encoding='gb2312'), text)

    def test_ascii_page_is_unaffected(self):
        self.assertEqual(self._get(b'<html>plain ascii</html>'),
                         '<html>plain ascii</html>')

    def test_undecodable_bytes_degrade_instead_of_raising(self):
        """哪条编码都解不出来时给出替换字符，不能让整次抓取失败。"""
        result = self._get(b'\xff\xfe\x00\x01 tail')
        self.assertIn('tail', result)

    def test_dirty_gbk_page_still_yields_readable_chinese(self):
        """线上真实情况：500.com 的页面是 gbk，但夹着几十个非法字节，
        **四种编码没有一种能严格解码成功**。

        此时不能随便挑一个——按 utf-8 降级会把整页中文变成问号，正则一条
        也匹不上，接口返回 200 加空列表。改成挑「替换字符最少」的那个：
        gbk 只需替换掉那几十个坏字节，utf-8 则要替换掉每一个中文字。
        """
        page = ('周三301 美职女篮 金州女武神VS太阳 ' * 50).encode('gbk')
        dirty = page[:100] + b'\xff\xfe' + page[100:]
        result = self._get(dirty)
        self.assertIn('美职女篮', result)
        self.assertIn('金州女武神', result)

    def test_dirty_utf8_page_is_not_mistaken_for_gbk(self):
        """反向也要成立：脏的 utf-8 页面不能被判成 gbk。"""
        page = ('周三301 美职女篮 金州女武神VS太阳 ' * 50).encode('utf-8')
        dirty = page[:90] + b'\xff\xfe' + page[90:]
        result = self._get(dirty)
        self.assertIn('美职女篮', result)

    def test_clean_page_still_wins_by_strict_decode(self):
        """干净页面走严格解码，不进入「数替换字符」那条路。"""
        text = '周三301 美职女篮'
        self.assertEqual(self._get(text.encode('gbk')), text)
        self.assertEqual(self._get(text.encode('utf-8')), text)

    def test_unknown_encoding_name_falls_through(self):
        text = '中文内容'
        self.assertEqual(self._get(text.encode('gbk'), encoding='根本不存在的编码'),
                         text)


if __name__ == '__main__':
    unittest.main()
