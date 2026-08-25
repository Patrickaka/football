import threading
import time


class CircuitBreaker:
    """三态熔断器：closed 正常放行，open 直接拒绝，half_open 试探一次。"""

    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = 'closed'
        self._failures = 0
        self._opened_at = None
        self._guard = threading.Lock()

    def allow(self, now=None):
        now = time.time() if now is None else now
        with self._guard:
            if self.state == 'closed':
                return True
            if self.state == 'half_open':
                # half_open 只放行一次试探：把 closed/open -> half_open 的这次
                # allow() 调用本身当作那唯一的探测名额，state 已是 half_open
                # 说明探测名额已被占用，其余并发调用一律拒绝，直到
                # record_success/record_failure 让状态离开 half_open。
                return False
            if now - self._opened_at >= self.recovery_timeout:
                self.state = 'half_open'
                return True
            return False

    def record_success(self):
        with self._guard:
            self._failures = 0
            self._opened_at = None
            self.state = 'closed'

    def record_failure(self, now=None):
        now = time.time() if now is None else now
        with self._guard:
            if self.state == 'half_open':
                self.state = 'open'
                self._opened_at = now
                return
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self.state = 'open'
                self._opened_at = now
