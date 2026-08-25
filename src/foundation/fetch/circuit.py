import threading
import time


class CircuitBreaker:
    """三态熔断器：closed 正常放行，open 直接拒绝，half_open 试探一次。"""

    def __init__(self, failure_threshold=5, recovery_timeout=60, probe_timeout=None):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        # 探针租约：拿到放行名额的调用方若迟迟不上报结果（崩溃/异常路径未覆盖），
        # 租约到期后允许下一次探测，避免熔断器永久卡在 half_open。
        self.probe_timeout = recovery_timeout if probe_timeout is None else probe_timeout
        self.state = 'closed'
        self._failures = 0
        self._opened_at = None
        self._probe_started_at = None
        self._guard = threading.Lock()

    def allow(self, now=None):
        now = time.time() if now is None else now
        with self._guard:
            if self.state == 'closed':
                return True
            if self.state == 'half_open':
                # half_open 期间只放行一个探针：state 变成 half_open 这件事
                # 本身就代表探测名额已被占用，其余并发调用一律拒绝，直到
                # record_success/record_failure 让状态离开 half_open。
                # 但若拿到名额的调用方迟迟不上报（异常路径未覆盖/进程崩溃），
                # 租约到期后视为该次探针已放弃，续租并重新放行下一次探测，
                # 避免永久卡死在 half_open、无法自愈。
                if (self._probe_started_at is not None
                        and now - self._probe_started_at >= self.probe_timeout):
                    self._probe_started_at = now
                    return True
                return False
            if now - self._opened_at >= self.recovery_timeout:
                self.state = 'half_open'
                self._probe_started_at = now
                return True
            return False

    def record_success(self):
        with self._guard:
            self._failures = 0
            self._opened_at = None
            self._probe_started_at = None
            self.state = 'closed'

    def record_failure(self, now=None):
        now = time.time() if now is None else now
        with self._guard:
            if self.state == 'half_open':
                self.state = 'open'
                self._opened_at = now
                self._probe_started_at = None
                return
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self.state = 'open'
                self._opened_at = now
