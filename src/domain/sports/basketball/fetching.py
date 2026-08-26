"""basketball 的抓取层，走 foundation/fetch。

迁移前有两套并行的重试与熔断：okooo.py 自己维护 max_retries 循环和
「WAF 封锁 60 秒」计时器，而 FetchClient 也有重试与熔断。职责这样切开：

- **transport 只管领域知识**：Session 预热、gb2312/gbk 编码回退、WAF 页面识别
- **FetchClient 管通用策略**：按域名限速、退避重试、熔断、响应快照

关键是让 WAF 识别**抛异常**而不是返回 None——它自然会被 FetchClient 记为
一次失败并计入熔断，于是那个手写的 60 秒封锁计时器就不需要了，两套机制
合并成一套。

限速值：okooo 比 500.com 更保守，因为它有 WAF；旧代码完全没有限速，
线上因此吃过 500.com 的大批 503。
"""
import logging
import urllib.error
import urllib.request
from urllib.parse import urlparse

from src.foundation.fetch import DomainRateLimiters, FetchClient

log = logging.getLogger('domain.basketball.fetching')

OKOOO_HOSTS = frozenset({'www.okooo.com', 'okooo.com'})
OKOOO_BASE = 'https://www.okooo.com'
OKOOO_HUNHE_URL = f'{OKOOO_BASE}/jingcailanqiu/hunhe/'

# 每秒请求数。旧代码无限速，线上吃过 500.com 的大批 503。
DEFAULT_RATE = 1.0
RATE_OVERRIDES = {
    'www.okooo.com': 0.4,
    'okooo.com': 0.4,
}

_UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
       '(KHTML, like Gecko) Chrome/120.0 Safari/537.36')


class WafBlocked(Exception):
    """识别到 WAF 拦截页。

    抛异常而非返回 None，是为了让它计入 FetchClient 的熔断——连续撞 WAF
    会开路，自然实现了旧代码里那个手写的「封锁 60 秒」，且不必自己维护计时器。
    """


def dispatch_transport(okooo, default):
    """按主机名把请求分派给对应实现。

    用主机名而非子串匹配：查询参数里带源站名的链接（?ref=okooo.com）
    不该被误分派。
    """
    def _transport(url, timeout):
        host = (urlparse(url).hostname or '').lower()
        impl = okooo if host in OKOOO_HOSTS else default
        return impl(url, timeout)

    return _transport


def urllib_get(url, timeout, encoding='utf-8', referer=None):
    """500.com 系列。该站部分页面是 gbk/gb2312，故按候选编码依次尝试。

    **必须严格解码**（不带 errors）才能让「这个编码不对」表现为异常。
    迁移时这里写成了 `decode(enc, errors='replace')`，而带 replace 的解码
    永远不抛异常——第一个候选总是"成功"，回退一次都走不到。gbk 页面被当作
    utf-8 解出整页乱码，正则一条也匹不上，接口返回 200 加空列表，不报任何错。

    全部候选都失败时才降级到替换字符：拿到乱码总好过整次抓取失败。
    """
    headers = {'User-Agent': _UA}
    if referer:
        headers['Referer'] = referer
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()

    for enc in (encoding, 'gbk', 'gb2312', 'utf-8'):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    log.warning('无法确定页面编码，按 utf-8 降级解码: %s', url)
    return raw.decode('utf-8', errors='replace')


def build_fetch_client(transport, snapshots_root=None, max_retries=3,
                       failure_threshold=5, recovery_timeout=60,
                       sleep_fn=None):
    """装配 basketball 的抓取客户端。

    transport 必须显式传入，不给默认值：真实实现会发网络请求，默认值会让
    测试在忘记注入时静默连上真实源站。
    """
    limiters = DomainRateLimiters(default_rate=DEFAULT_RATE, burst=1,
                                  overrides=RATE_OVERRIDES)
    snapshots = None
    if snapshots_root:
        from src.foundation.fetch import SnapshotStore
        snapshots = SnapshotStore(snapshots_root)

    kwargs = {
        'transport': transport,
        'limiters': limiters,
        'snapshots': snapshots,
        'max_retries': max_retries,
        'failure_threshold': failure_threshold,
        'recovery_timeout': recovery_timeout,
    }
    if sleep_fn is not None:
        kwargs['sleep_fn'] = sleep_fn
    return FetchClient(**kwargs)


class OkoooTransport:
    """okooo 的抓取实现。

    只做领域知识三件事：Session 预热（直接请求详情页会被判为异常流量）、
    gb2312 解码、WAF 页面识别。重试、限速、熔断一概交给 FetchClient——
    旧实现自己维护 max_retries 循环和「WAF 封锁 60 秒」计时器，与
    FetchClient 的同类机制重复，是三套缓存并存那类问题的又一个变种。
    """

    def __init__(self, session_factory=None, sleep_fn=None, warmup_pause=0.3):
        self._session_factory = session_factory or self._default_session
        self._sleep = sleep_fn if sleep_fn is not None else __import__('time').sleep
        self._warmup_pause = warmup_pause
        self._session = None

    @staticmethod
    def _default_session():
        import requests

        session = requests.Session()
        session.headers.update({
            'User-Agent': _UA,
            'Referer': OKOOO_HUNHE_URL,
        })
        # 该站证书链在部分环境下校验失败，沿用迁移前的设置。
        session.verify = False
        return session

    def _ensure_session(self):
        if self._session is not None:
            return self._session
        session = self._session_factory()
        try:
            session.get(OKOOO_BASE + '/', timeout=10)
            self._sleep(self._warmup_pause)
            session.get(OKOOO_HUNHE_URL, timeout=10)
        except Exception as exc:
            log.warning('okooo session 预热失败，继续尝试直接抓取: %s', exc)
        self._session = session
        return session

    def __call__(self, url, timeout):
        session = self._ensure_session()
        resp = session.get(url, timeout=timeout)

        if getattr(resp, 'status_code', 200) != 200:
            raise IOError(f'okooo 返回 {resp.status_code}: {url}')

        try:
            resp.encoding = 'gb2312'
            text = resp.text
        except Exception:
            text = resp.content.decode('gb2312', errors='replace')

        if 'aliyun_waf' in text and '<title></title>' in text:
            # 该 session 已被污染，丢弃后下次重建。
            self._session = None
            raise WafBlocked(f'okooo WAF 拦截: {url}')

        return text
