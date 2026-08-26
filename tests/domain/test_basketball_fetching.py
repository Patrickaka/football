"""basketball 抓取层接入 foundation/fetch。

迁移前有两套并行的重试与熔断：okooo.py 自己维护 max_retries 循环和
「WAF 封锁 60 秒」计时器，而 FetchClient 也有重试与熔断。合并的办法是
让 transport 只负责领域知识（Session 预热、gb2312 编码、WAF 页面识别），
识别到 WAF 就**抛异常**——它自然会被 FetchClient 记为一次失败并计入熔断，
不需要再维护第二个计时器。
"""
import tempfile
import unittest
import unittest.mock

from src.domain.sports.basketball import fetching
from src.domain.sports.basketball.fetching import (
    OkoooTransport, WafBlocked, build_fetch_client, dispatch_transport,
)
from src.foundation.fetch import FetchError


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
        """两个源站的限速互不影响：okooo 被限速不该拖慢 500.com。"""
        transport = _RecordingTransport(['a', 'b'])
        slept = []
        client = build_fetch_client(transport=transport, sleep_fn=slept.append)
        client.get('https://www.okooo.com/jingcailanqiu/hunhe/')
        client.get('https://trade.500.com/jclq/')
        self.assertEqual(sum(1 for s in slept if s > 0), 0,
                         '不同域名的首次请求都不该等待')

    def test_okooo_is_rate_limited_more_conservatively(self):
        """okooo 有 WAF，限速须比 500.com 更保守。"""
        client = build_fetch_client(transport=_RecordingTransport([]),
                                    sleep_fn=lambda _: None)
        okooo = client.limiters.for_domain('www.okooo.com').rate_per_sec
        wubai = client.limiters.for_domain('trade.500.com').rate_per_sec
        self.assertLess(okooo, wubai)

    def test_waf_block_counts_as_failure_and_trips_breaker(self):
        """WAF 页面抛异常 → 计入熔断，不需要第二套 60 秒封锁计时器。"""
        transport = _RecordingTransport([WafBlocked('waf')] * 10)
        client = build_fetch_client(transport=transport, sleep_fn=lambda _: None,
                                    max_retries=1, failure_threshold=2)
        for _ in range(2):
            with self.assertRaises(FetchError):
                client.get('https://www.okooo.com/x')
        before = len(transport.calls)
        with self.assertRaises(FetchError):
            client.get('https://www.okooo.com/x')
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
    def test_okooo_url_goes_to_okooo_impl(self):
        picked = []
        transport = dispatch_transport(
            okooo=lambda url, timeout: picked.append('okooo') or 'o',
            default=lambda url, timeout: picked.append('default') or 'd',
        )
        transport('https://www.okooo.com/basketball/match/', 10)
        self.assertEqual(picked, ['okooo'])

    def test_other_urls_go_to_default_impl(self):
        picked = []
        transport = dispatch_transport(
            okooo=lambda url, timeout: picked.append('okooo') or 'o',
            default=lambda url, timeout: picked.append('default') or 'd',
        )
        transport('https://trade.500.com/jclq/', 10)
        self.assertEqual(picked, ['default'])

    def test_dispatch_is_by_host_not_substring(self):
        """按主机名判断，而非 url 里出现 okooo 字样就算——
        查询参数里带源站名的链接不该被误分派。"""
        picked = []
        transport = dispatch_transport(
            okooo=lambda url, timeout: picked.append('okooo') or 'o',
            default=lambda url, timeout: picked.append('default') or 'd',
        )
        transport('https://trade.500.com/jclq/?ref=okooo.com', 10)
        self.assertEqual(picked, ['default'])


class _FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code
        self.encoding = None
        self.content = text.encode('gb2312', errors='replace')


class _FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.headers = {}
        self.verify = True
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        item = self.responses.pop(0) if self.responses else _FakeResponse('ok')
        if isinstance(item, Exception):
            raise item
        return item


class OkoooTransportTests(unittest.TestCase):
    def test_warms_up_session_before_first_fetch(self):
        """首次抓取前要先访问首页与混合页建立 cookie，否则会被判为异常流量。"""
        session = _FakeSession([_FakeResponse('home'), _FakeResponse('hunhe'),
                                _FakeResponse('<html>data</html>')])
        transport = OkoooTransport(session_factory=lambda: session,
                                   sleep_fn=lambda _: None)
        transport('https://www.okooo.com/basketball/match/', 10)
        self.assertEqual(len(session.calls), 3)
        self.assertIn('okooo.com/', session.calls[0])

    def test_warms_up_only_once(self):
        session = _FakeSession([_FakeResponse('home'), _FakeResponse('hunhe'),
                                _FakeResponse('a'), _FakeResponse('b')])
        transport = OkoooTransport(session_factory=lambda: session,
                                   sleep_fn=lambda _: None)
        transport('https://www.okooo.com/x', 10)
        transport('https://www.okooo.com/y', 10)
        self.assertEqual(len(session.calls), 4, '第二次不应重复预热')

    def test_waf_page_raises_instead_of_returning_none(self):
        """旧代码返回 None 并自己记 60 秒封锁；现在抛异常交给熔断处理。"""
        waf = _FakeResponse('<html>aliyun_waf<title></title></html>')
        session = _FakeSession([_FakeResponse('home'), _FakeResponse('hunhe'), waf])
        transport = OkoooTransport(session_factory=lambda: session,
                                   sleep_fn=lambda _: None)
        with self.assertRaises(WafBlocked):
            transport('https://www.okooo.com/x', 10)

    def test_waf_detection_resets_session(self):
        """撞 WAF 后旧 session 已被污染，下次须换新的。"""
        waf = _FakeResponse('<html>aliyun_waf<title></title></html>')
        sessions = []

        def factory():
            s = _FakeSession([_FakeResponse('home'), _FakeResponse('hunhe'), waf])
            sessions.append(s)
            return s

        transport = OkoooTransport(session_factory=factory, sleep_fn=lambda _: None)
        with self.assertRaises(WafBlocked):
            transport('https://www.okooo.com/x', 10)
        with self.assertRaises(WafBlocked):
            transport('https://www.okooo.com/y', 10)
        self.assertEqual(len(sessions), 2, '撞 WAF 后应重建 session')

    def test_non_200_raises(self):
        """非 200 抛异常，让 FetchClient 的重试与熔断接管。"""
        session = _FakeSession([_FakeResponse('home'), _FakeResponse('hunhe'),
                                _FakeResponse('err', status_code=503)])
        transport = OkoooTransport(session_factory=lambda: session,
                                   sleep_fn=lambda _: None)
        with self.assertRaises(IOError):
            transport('https://www.okooo.com/x', 10)

    def test_decodes_as_gb2312(self):
        session = _FakeSession([_FakeResponse('home'), _FakeResponse('hunhe'),
                                _FakeResponse('中文内容')])
        transport = OkoooTransport(session_factory=lambda: session,
                                   sleep_fn=lambda _: None)
        self.assertEqual(transport('https://www.okooo.com/x', 10), '中文内容')


class WafIsPermanentTests(unittest.TestCase):
    """撞 WAF 不重试。

    迁移时用 FetchClient 的熔断替换了旧代码手写的「撞 WAF 就跳过 60 秒」
    开关。熔断确实能做同一件事，但它要 failure_threshold 次失败才开路，
    而每次失败还要先重试 max_retries 遍——在 okooo 的 0.4 rps 限速下，
    这笔账让冷启动从 0.4 秒涨到 5 秒。WAF 拦截是确定性的（同一出口 IP
    再试还是同样结果），不该走退避重试那条路。
    """

    def _client(self, **kwargs):
        self.calls = []

        def transport(url, timeout):
            self.calls.append(url)
            raise WafBlocked('okooo WAF 拦截')

        self.slept = []
        return build_fetch_client(transport=transport, sleep_fn=self.slept.append,
                                  **kwargs)

    def test_waf_page_is_fetched_once_not_three_times(self):
        client = self._client(max_retries=3)
        with self.assertRaises(FetchError):
            client.get('https://www.okooo.com/basketball/match/1/odds/')
        self.assertEqual(len(self.calls), 1)

    def test_waf_page_does_not_burn_backoff_sleeps(self):
        client = self._client(max_retries=3)
        with self.assertRaises(FetchError):
            client.get('https://www.okooo.com/basketball/match/1/odds/')
        self.assertEqual([s for s in self.slept if s >= 0.5], [],
                         '为一次注定失败的请求白等了退避')

    def test_breaker_still_opens_after_enough_waf_hits(self):
        """不重试不等于不计数——连撞几次照样开路，这正是旧代码那个
        手写 60 秒计时器要达到的效果。"""
        client = self._client(max_retries=3, failure_threshold=2)
        for _ in range(2):
            with self.assertRaises(FetchError):
                client.get('https://www.okooo.com/basketball/match/1/odds/')
        before = len(self.calls)
        with self.assertRaises(FetchError):
            client.get('https://www.okooo.com/basketball/match/1/odds/')
        self.assertEqual(len(self.calls), before, '熔断开路后仍然发了请求')


class BreakerGranularityTests(unittest.TestCase):
    """详情页的熔断不能把同域名的赛程页一起打掉。

    **这是线上真实发生过的故障**：端点切换后，okooo 详情页连撞 WAF 触发
    熔断，而熔断按域名建，于是同属 www.okooo.com 的赛程页也被短路——
    接口返回 200、比赛数 0、走势全空，没有任何报错。赛程页自带的
    rf_trend / dx_trend 是线上唯一活着的走势来源，代价是整份推荐直接空掉。
    """

    SCHEDULE = 'https://www.okooo.com/jingcailanqiu/hunhe/'
    DETAIL = 'https://www.okooo.com/basketball/match/5381400/odds/'

    def test_detail_and_schedule_use_different_breakers(self):
        self.assertNotEqual(fetching.breaker_key(self.DETAIL),
                            fetching.breaker_key(self.SCHEDULE))

    def test_all_three_detail_kinds_share_one_breaker(self):
        keys = {fetching.breaker_key(
            f'https://www.okooo.com/basketball/match/5381400/{kind}/')
            for kind in ('odds', 'ah', 'ou')}
        self.assertEqual(len(keys), 1)

    def test_other_hosts_keep_the_domain_as_key(self):
        self.assertEqual(fetching.breaker_key('https://trade.500.com/jclq/'),
                         'trade.500.com')

    def test_lookalike_paths_are_not_treated_as_detail_pages(self):
        for url in ('https://www.okooo.com/basketball/match/5381400/trends/',
                    'https://www.okooo.com/basketball/league/486/',
                    'https://evil.com/basketball/match/1/odds/'):
            with self.subTest(url=url):
                self.assertNotEqual(fetching.breaker_key(url),
                                    'www.okooo.com#detail')

    def test_blocked_details_do_not_short_circuit_the_schedule(self):
        calls = []

        def transport(url, timeout):
            calls.append(url)
            if '/basketball/match/' in url:
                raise WafBlocked('okooo WAF 拦截')
            return '赛程页正文'

        client = build_fetch_client(transport=transport, sleep_fn=lambda _: None)
        for _ in range(4):
            with self.assertRaises(FetchError):
                client.get(self.DETAIL)

        self.assertEqual(client.get(self.SCHEDULE), '赛程页正文',
                         '详情页的熔断把赛程页一起打掉了')

    def test_one_waf_hit_opens_the_detail_breaker(self):
        """撞一次就够。攒够 failure_threshold 再开路，等于把已知必败的请求
        重复几遍——okooo 限速 0.4 rps，默认阈值 5 就是十几秒的冷启动。"""
        calls = []

        def transport(url, timeout):
            calls.append(url)
            raise WafBlocked('okooo WAF 拦截')

        client = build_fetch_client(transport=transport, sleep_fn=lambda _: None)
        with self.assertRaises(FetchError):
            client.get(self.DETAIL)
        self.assertEqual(len(calls), 1)

        for kind in ('ah', 'ou'):
            with self.assertRaises(FetchError):
                client.get(f'https://www.okooo.com/basketball/match/5381400/{kind}/')
        self.assertEqual(len(calls), 1, '熔断开路后仍在白跑请求')


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
