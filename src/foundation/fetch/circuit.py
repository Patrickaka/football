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
                return True
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
