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

    契约：compute_fn 返回值必须 JSON 可序列化——生产态 L2 是 Redis，
    序列化边界在那一层。写入失败会被吞掉并退化为纯 L1（见 `set`
    的实现说明），不会让请求失败，但也意味着不可序列化的值永远
    进不了 L2、每次冷启动都要重新计算。
    """

    def __init__(self, l1, l2, default_ttl=300, lock_timeout=30, wait_timeout=None):
        """wait_timeout 是等待方放弃等待、改为自行计算的上限，与 lock_timeout
        （锁的最长持有时间）是两回事，故单列。

        默认取 lock_timeout：L2 正常时等待方并不靠它退出（winner 释放锁后，
        等待方要么抢到锁、要么读到值），它只在 L2 持续不可用时才生效。若把它
        调得比 compute_fn 的实际耗时还短，正常情况下等待方会集体超时各自开算，
        惊群反而被请回来——所以缩短它之前，先确认它大于该 key 的计算耗时。
        """
        self.l1 = l1
        self.l2 = l2
        self.default_ttl = default_ttl
        self.lock_timeout = lock_timeout
        self.wait_timeout = lock_timeout if wait_timeout is None else wait_timeout
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
        """写入两层缓存。先写 L1 再写 L2：L1（进程内存）几乎不会失败，
        L2（生产态是 Redis）可能因网络抖动、序列化失败等原因写入失败——
        先写 L1 保证即便 L2 写入失败，调用方也能立刻从 L1 读到刚写入的
        值，退化为纯 L1 缓存，而不是"两层都没写成，每次请求都重算"。

        L2 写入额外包一层 try/except：RedisBackend 自身已经把底层
        client 调用都护住了，这里是防御性的第二道保险，避免未来换用
        一个没有做内部防护的 L2 后端时，一次写入失败打穿到调用方。
        """
        ttl = self.default_ttl if ttl is None else ttl
        now = time.time()
        self.l1.set(key, value, ttl, now=now)
        try:
            self.l2.set(key, value, ttl, now=now)
        except Exception:
            log.warning('L2 写入失败，已退化为纯 L1: key=%s', key, exc_info=True)

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
        """等待后台刷新完成。仅供测试与优雅停机使用。

        所有线程共享同一个 deadline。若逐个 join 各自套用完整 timeout，N 个卡住的
        刷新会把停机耗时线性放大到 N × timeout，可能超过编排的 SIGTERM 宽限期而
        被 SIGKILL——那样恰好复现了本方法要避免的 Redis 残留锁。
        """
        with self._refresh_guard:
            threads = list(self._refresh_threads)
        deadline = time.time() + timeout
        for t in threads:
            remaining = deadline - time.time()
            if remaining <= 0:
                log.warning('等待后台刷新超时，仍有 %d 个线程未结束',
                            sum(1 for x in threads if x.is_alive()))
                break
            t.join(timeout=remaining)
        with self._refresh_guard:
            self._refresh_threads = [t for t in self._refresh_threads if t.is_alive()]

    def _compute_with_single_flight(self, key, compute_fn, ttl, poll_interval=0.02):
        """单飞：同一 key 的并发冷启动只允许一个线程计算。

        等待方轮询两件事——值是否出现、锁是否空出。winner 失败时会释放锁但不写值，
        此时等待方必须能立刻接手，否则一次瞬时故障会把所有并发请求拖满 lock_timeout。
        """
        deadline = time.time() + self.wait_timeout
        while True:
            if self.l2.lock(key, timeout=self.lock_timeout):
                try:
                    entry = self.l2.get(key)
                    if entry is not None and entry.is_fresh():
                        return entry.value
                    return self._compute_and_store(key, compute_fn, ttl)
                finally:
                    self.l2.unlock(key)

            entry = self.l2.get(key)
            if entry is not None:
                return entry.value
            if time.time() >= deadline:
                # 走到这里通常意味着 L2 持续不可用：它正常时等待方会靠"抢到锁"
                # 或"读到值"退出，不会耗到 deadline。此时必须照常写缓存——否则
                # L1 永远是空的，接下来每一个请求都要重挨一遍完整等待。
                log.warning('等待单飞超时（L2 可能不可用），回退为本地计算: key=%s', key)
                return self._compute_and_store(key, compute_fn, ttl)
            time.sleep(poll_interval)

    def _compute_and_store(self, key, compute_fn, ttl):
        """计算并写入缓存，写入前核对纪元。

        纪元不一致说明计算期间该 key 被 invalidate 过，此时只把值返回给调用方
        而不写缓存，避免让刚清空的缓存"自己复活"。
        """
        with self._refresh_guard:
            epoch_at_start = self._epoch
        value = compute_fn()
        with self._refresh_guard:
            if self._epoch != epoch_at_start:
                log.info('计算期间缓存已被失效，不写回: key=%s', key)
                return value
        self.set(key, value, ttl)
        return value

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
