import logging
import threading
import time

log = logging.getLogger('foundation.cache')


class Cache:
    """两层缓存门面：L1 进程内存，L2 Redis。

    三条硬规则：
    1. 所有读取走单飞，避免缓存过期时并发重算
    2. 过期数据先返回、后台刷新（SWR），请求线程不承担冷计算
    3. invalidate 一次贯穿两层
    """

    def __init__(self, l1, l2, default_ttl=300, lock_timeout=30):
        self.l1 = l1
        self.l2 = l2
        self.default_ttl = default_ttl
        self.lock_timeout = lock_timeout
        self._refresh_threads = []
        self._refresh_guard = threading.Lock()
        self._epoch = 0

    def get(self, key, compute_fn, ttl=None):
        ttl = self.default_ttl if ttl is None else ttl
        now = time.time()

        entry = self.l1.get(key)
        if entry is None:
            entry = self.l2.get(key)
            if entry is not None:
                self.l1.set(key, entry.value, entry.ttl, now=entry.stored_at)

        if entry is not None:
            if entry.is_fresh(now):
                return entry.value
            self._refresh_in_background(key, compute_fn, ttl)
            return entry.value

        return self._compute_with_single_flight(key, compute_fn, ttl)

    def set(self, key, value, ttl=None):
        ttl = self.default_ttl if ttl is None else ttl
        now = time.time()
        self.l2.set(key, value, ttl, now=now)
        self.l1.set(key, value, ttl, now=now)

    def invalidate(self, key):
        """清除该 key 的所有层缓存，并作废所有在途的后台刷新。

        纪元计数器用全局而非 per-key：invalidate 频率很低，全局计数器
        不会有内存增长问题，代价只是"任一 invalidate 会作废所有在途刷新"，
        偏保守但安全——避免 SWR 后台刷新在 invalidate 之后无条件写回，
        让已清空的缓存"自己复活"。
        """
        with self._refresh_guard:
            self._epoch += 1
        self.l1.delete(key)
        self.l2.delete(key)

    def wait_for_refreshes(self, timeout=10):
        """等待后台刷新完成。仅供测试与优雅停机使用。"""
        with self._refresh_guard:
            threads = list(self._refresh_threads)
        for t in threads:
            t.join(timeout=timeout)
        with self._refresh_guard:
            self._refresh_threads = [t for t in self._refresh_threads if t.is_alive()]

    def _compute_with_single_flight(self, key, compute_fn, ttl, poll_interval=0.02):
        """单飞：同一 key 的并发冷启动只允许一个线程计算。

        等待方轮询两件事——值是否出现、锁是否空出。winner 失败时会释放锁但不写值，
        此时等待方必须能立刻接手，否则一次瞬时故障会把所有并发请求拖满 lock_timeout。
        """
        deadline = time.time() + self.lock_timeout
        while True:
            if self.l2.lock(key, timeout=self.lock_timeout):
                try:
                    entry = self.l2.get(key)
                    if entry is not None and entry.is_fresh():
                        return entry.value
                    value = compute_fn()
                    self.set(key, value, ttl)
                    return value
                finally:
                    self.l2.unlock(key)

            entry = self.l2.get(key)
            if entry is not None:
                return entry.value
            if time.time() >= deadline:
                log.warning('等待单飞超时，回退为本地计算: key=%s', key)
                return compute_fn()
            time.sleep(poll_interval)

    def _refresh_in_background(self, key, compute_fn, ttl):
        if not self.l2.lock(key, timeout=self.lock_timeout):
            return

        with self._refresh_guard:
            epoch_at_start = self._epoch

        def _run():
            try:
                value = compute_fn()
                with self._refresh_guard:
                    if self._epoch != epoch_at_start:
                        log.info('刷新期间缓存已被失效，丢弃本次结果: key=%s', key)
                        return
                self.set(key, value, ttl)
            except Exception:
                log.exception('后台刷新失败，保留陈旧值: key=%s', key)
            finally:
                self.l2.unlock(key)

        thread = threading.Thread(target=_run, name=f'CacheRefresh-{key}', daemon=True)
        with self._refresh_guard:
            self._refresh_threads = [t for t in self._refresh_threads if t.is_alive()]
            self._refresh_threads.append(thread)
        thread.start()
