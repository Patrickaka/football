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
