import logging
import threading
import time
from urllib.parse import urlparse

from .circuit import CircuitBreaker

log = logging.getLogger('foundation.fetch')


class PermanentFetchError(Exception):
    """确定性的失败，重试没有意义。

    退避重试假设失败是暂时的——网络抖动、对端瞬时过载。但有一类失败重试
    多少次都是同样结果：被 WAF 拦、404、鉴权失败。对它们重试只是把一次
    失败的代价乘以重试次数，在限速的域名上尤其昂贵。

    抛出本类（或其子类）时，FetchClient 立刻放弃本次请求，但**仍然计入
    熔断**——正是这类失败最该让熔断尽快开路。
    """


class FetchError(Exception):
    """抓取最终失败（重试耗尽、熔断开路，且无快照可兜底）。"""


class FetchClient:
    """统一抓取入口：限速 → 熔断 → 重试 → 快照兜底。

    transport 为 callable(url, timeout) -> str，注入以便测试不触网。
    """

    def __init__(
        self,
        transport,
        limiters,
        snapshots=None,
        max_retries=3,
        base_backoff=0.5,
        failure_threshold=5,
        recovery_timeout=60,
        sleep_fn=time.sleep,
    ):
        if max_retries < 1:
            raise ValueError('max_retries must be >= 1, got %r' % (max_retries,))
        if base_backoff < 0:
            raise ValueError('base_backoff must be >= 0, got %r' % (base_backoff,))
        if failure_threshold <= 0:
            raise ValueError('failure_threshold must be > 0, got %r' % (failure_threshold,))
        if recovery_timeout <= 0:
            raise ValueError('recovery_timeout must be > 0, got %r' % (recovery_timeout,))
        self.transport = transport
        self.limiters = limiters
        self.snapshots = snapshots
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.sleep_fn = sleep_fn
        self._breakers = {}
        self._breakers_guard = threading.Lock()

    def get(self, url, timeout=20):
        domain = urlparse(url).netloc
        breaker = self._breaker(domain)

        if not breaker.allow():
            # 未被放行：本次调用没有发起任何真实请求，绝不能上报
            # record_success/record_failure —— 否则会凭空消耗掉 half_open
            # 唯一的探针名额，或污染 closed 状态下的失败计数。
            log.warning('熔断开路，跳过请求: domain=%s', domain)
            return self._fallback_or_raise(url, f'{domain} 熔断开路')

        # 一次 allow() 放行对应一次逻辑请求（内部含重试），因此报告也必须
        # 恰好发生一次：用 try/finally 保证无论走成功、重试耗尽还是异常
        # 路径都会上报，且绝不会因为异常提前返回而漏报。重试循环里的每次
        # 失败不单独上报——否则一次 allow() 会触发多次 record_failure，
        # 既打破 allow()/report 的一一配对，又会让失败计数虚高地反映"重试
        # 次数"而不是"失败请求次数"，扭曲 failure_threshold 的语义。
        succeeded = False
        try:
            last_error = None
            for attempt in range(self.max_retries):
                wait = self.limiters.for_domain(domain).acquire()
                if wait > 0:
                    self.sleep_fn(wait)
                try:
                    body = self.transport(url, timeout)
                except PermanentFetchError as exc:
                    # 确定性失败，重试只会把代价乘以重试次数。直接跳出，
                    # 交给 finally 记一次失败——熔断照常推进。
                    log.warning('确定性失败，不重试: url=%s error=%s', url, exc)
                    last_error = exc
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < self.max_retries - 1:
                        self.sleep_fn(self.base_backoff * (2 ** attempt))
                    continue
                succeeded = True
                if self.snapshots is not None:
                    # 快照落盘失败（磁盘满等）不该让一次已经成功拿到 body 的
                    # 请求失败——快照只是可选的降级辅助，不是请求成功与否的
                    # 前提条件，否则兜底机制自己会变成新的故障源。
                    try:
                        self.snapshots.save(url, body)
                    except Exception:
                        log.warning('快照保存失败，忽略：url=%s', url, exc_info=True)
                return body

            log.warning('抓取失败: url=%s error=%s', url, last_error)
            # 快照兜底返回的是历史缓存内容，不是这次请求真正成功，
            # succeeded 保持 False，finally 里仍会上报一次失败。
            return self._fallback_or_raise(url, f'{last_error}')
        finally:
            if succeeded:
                breaker.record_success()
            else:
                breaker.record_failure()

    def _fallback_or_raise(self, url, reason):
        if self.snapshots is not None:
            cached = self.snapshots.load(url)
            if cached is not None:
                log.info('使用快照兜底: url=%s', url)
                return cached
        raise FetchError(f'{url} 抓取失败：{reason}')

    def _breaker(self, domain):
        with self._breakers_guard:
            breaker = self._breakers.get(domain)
            if breaker is None:
                breaker = CircuitBreaker(
                    failure_threshold=self.failure_threshold,
                    recovery_timeout=self.recovery_timeout,
                )
                self._breakers[domain] = breaker
            return breaker
