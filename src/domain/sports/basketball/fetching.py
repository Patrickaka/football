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
import re
import urllib.error
import urllib.request
from urllib.parse import urlparse

from src.foundation.fetch import (
    DomainRateLimiters, FetchClient, PermanentFetchError,
)

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


class WafBlocked(PermanentFetchError):
    """识别到 WAF 拦截页。

    抛异常而非返回 None，是为了让它计入 FetchClient 的熔断——连续撞 WAF
    会开路，自然实现了旧代码里那个手写的「封锁 60 秒」，且不必自己维护计时器。

    继承 PermanentFetchError 是因为 WAF 拦截是**确定性**的：同一个出口 IP
    再试一次还是同样结果。当作暂时故障退避重试，只会把一次失败的代价乘以
    重试次数——在 0.4 rps 的限速下这笔账很贵。
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
    """500.com 系列。该站部分页面是 gbk/gb2312，按候选编码逐个判定。

    两条规则，缺一不可：

    1. **先严格解码**（不带 errors）。带 `errors='replace'` 的解码永远不抛
       异常，写成那样的话第一个候选总是"成功"，回退一次都走不到。迁移时
       正是这么写的，结果 gbk 页面被当作 utf-8 解出整页乱码——接口照样
       返回 200，只是列表空的，不报任何错。

    2. **全部严格解码失败时，选替换字符最少的那个**。线上真实页面就是这种：
       gbk 编码但夹着几十个非法字节，四种候选一个都严格解不出来。此时若随便
       挑一个（比如按 utf-8 降级），整页中文会变成问号；而按替换字符计数，
       gbk 只需替换掉那几十个坏字节，utf-8 要替换掉每一个中文字，差距悬殊，
       判别很稳。
    """
    headers = {'User-Agent': _UA}
    if referer:
        headers['Referer'] = referer
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return decode_page(raw, encoding, url)


def decode_page(raw, encoding='utf-8', url=''):
    candidates = _dedupe((encoding, 'gbk', 'gb2312', 'utf-8'))

    for candidate in candidates:
        try:
            return raw.decode(candidate)
        except (UnicodeDecodeError, LookupError):
            continue

    best, text = _fewest_replacements(raw, candidates)
    log.warning('页面无法严格解码，按替换字符最少的 %s 降级: %s', best, url)
    return text


def _dedupe(values):
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _fewest_replacements(raw, candidates):
    best_name, best_text, best_count = 'utf-8', None, None
    for candidate in candidates:
        try:
            text = raw.decode(candidate, errors='replace')
        except LookupError:
            continue
        count = text.count('\ufffd')
        if best_count is None or count < best_count:
            best_name, best_text, best_count = candidate, text, count
    if best_text is None:
        return 'utf-8', raw.decode('utf-8', errors='replace')
    return best_name, best_text


# 单场详情页：/basketball/match/<id>/odds|ah|ou/。这些页面在部分出口 IP 上
# 被 WAF 拦死，而同域名的赛程页是通的——两者必须各用各的熔断器。
_OKOOO_DETAIL_PATH = re.compile(r'^/basketball/match/\d+/(odds|ah|ou)/?$')
_OKOOO_DETAIL_KEY = 'www.okooo.com#detail'


def breaker_key(url):
    """熔断键。默认按域名，但 okooo 的详情页单列。

    澳客的赛程页正常、详情页被 WAF 拦死，两者同属 www.okooo.com。按域名
    熔断的话，详情页连撞几次就会把赛程页一起打掉——而赛程页自带的
    rf_trend / dx_trend 是线上唯一活着的走势来源，代价是整份推荐直接空掉。
    这不是假设：端点切换后线上就是这么坏的，接口返回 200、比赛数 0。
    """
    parsed = urlparse(url)
    host = (parsed.hostname or '').lower()
    if host in OKOOO_HOSTS and _OKOOO_DETAIL_PATH.match(parsed.path or ''):
        return _OKOOO_DETAIL_KEY
    return parsed.netloc


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
        'breaker_key_fn': breaker_key,
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
        """撞到 WAF 时先换一次 Session 再试，两次都被拦才认定为永久失败。

        WAF 的拦截**不完全是确定性的**：长驻进程里的 Session 用久了会被
        标记，此时换一个干净 Session 往往立刻就通。所以「重建后再试一次」
        与「原样重试一次」是两回事——前者是真正不同的尝试。

        端点切换当天线上就栽在这里：把 WAF 一律当作确定性失败、一次都不
        重试，等于掐掉了这条自愈路径。赛程页因为 Session 老化被拦，熔断
        立刻开路 60 秒，接口返回 200 加 0 场比赛。

        Session 的生命周期是本 transport 自己的事，不该外泄给 FetchClient
        的重试循环——那一层管的是「对端是否暂时不可用」，管不到这里。
        """
        try:
            return self._fetch_once(url, timeout)
        except WafBlocked:
            log.info('okooo 撞 WAF，换一个 Session 重试: %s', url)

        # 上一次失败已经把 Session 丢掉了，这次会重新预热一个
        return self._fetch_once(url, timeout)

    def _fetch_once(self, url, timeout):
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
            # 该 Session 已被污染，丢掉，下次调用会重建
            self._session = None
            raise WafBlocked(f'okooo WAF 拦截: {url}')

        return text
