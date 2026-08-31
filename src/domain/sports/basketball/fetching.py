"""basketball 的抓取层，走 foundation/fetch。

中国足彩网与 500.com 统一走 FetchClient 的限速、重试、熔断和快照机制：

- **transport 只管领域知识**：HTTP 与 utf-8/gbk 编码回退
- **FetchClient 管通用策略**：按域名限速、退避重试、熔断、响应快照

两个站点都使用低频限速，避免批量分析形成突发请求。
"""
import logging
import urllib.request
from urllib.parse import urlparse

from src.foundation.fetch import (
    DomainRateLimiters, FetchClient, PermanentFetchError,
)

log = logging.getLogger('domain.basketball.fetching')

ZGZCW_HOSTS = frozenset({'cp.zgzcw.com', 'fenxi.zgzcw.com', 'odds.zgzcw.com'})

# 每秒请求数。旧代码无限速，线上吃过 500.com 的大批 503。
DEFAULT_RATE = 1.0
RATE_OVERRIDES = {
    'cp.zgzcw.com': 0.5,
    'fenxi.zgzcw.com': 0.5,
    'odds.zgzcw.com': 0.5,
}

_UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
       '(KHTML, like Gecko) Chrome/120.0 Safari/537.36')


class VerificationPage(PermanentFetchError):
    """站点返回人机验证页而不是数据页。"""


def dispatch_transport(zgzcw, default):
    """按主机名把请求分派给对应实现。

    用主机名而非子串匹配：查询参数里带源站名的链接
    不该被误分派。
    """
    def _transport(url, timeout):
        host = (urlparse(url).hostname or '').lower()
        impl = zgzcw if host in ZGZCW_HOSTS else default
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


def breaker_key(url):
    """按域名隔离熔断，避免一个上游影响另一个。"""
    parsed = urlparse(url)
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


class ZgzcwTransport:
    """中国足彩网 transport：复用统一解码并识别验证页。"""

    def __call__(self, url, timeout):
        text = urllib_get(url, timeout, encoding='utf-8',
                          referer='https://cp.zgzcw.com/')
        lowered = text.lower()
        if any(marker in lowered for marker in ('captcha', '访问验证', '安全验证')):
            raise VerificationPage(f'中国足彩网返回验证页: {url}')
        return text
