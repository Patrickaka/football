"""北单缓存跨重启保留：接上 L2（生产态是 Redis）。

**改这件事的起点是一次事故**：2026-08-28 机器两次冻死，都紧跟在
`systemctl restart` 之后。重启把北单缓存清零，接着是 371 场的全量重算——
实测 12:58→13:23 走了 25 分钟，13:38 那次部署重启后 5 分钟机器就没了。
北单的响应缓存挂在 `src/football/cache_manager.py` 上，那是纯进程内存，
进程一停就全没了。

kl8 与篮球早就走 `foundation/cache`（L1 内存 + L2 Redis）了——线上 Redis
里能看到 `fb:v:basketball:pred:...` 这样的真实键，**只有北单没接**。

这一批要守住的两件事，缺一件这次改动就等于没做：

1. **L1 没了之后还读得到**——这就是「跨重启」的全部含义；
2. **payload 真的进得了 L2**。`RedisBackend.set` 里的 `json.dumps` 一旦抛出
   会被吞成一条 warning，然后退化成纯 L1：**不报错、日志也不显眼、
   而收益是零**（判据 27）。所以要拿真的 JSON 编解码去验，
   不能只验「调用没抛异常」。
"""
import json
import time
import unittest
from unittest.mock import patch

import src.api.runtime.beidan_cache as beidan_cache
from src.foundation.cache import Cache, MemoryBackend
from src.foundation.cache.redis_backend import RedisBackend

# 迁移当时生效的值，写死不 import（判据 4、12）
TTL_SECONDS = 86400
PREFIX = 'beidan:pred'
REDIS_STALE_GRACE_FACTOR = 10
# `RedisBackend` 自己会在键前面加 `fb:v:`（前缀加一位 schema 版本），
# 线上真实的键长这样：`fb:v:basketball:pred:2026-08-28:okooo:dx,rqspf,spf:1`
REDIS_KEY_PREFIX = 'fb:v:'


class _FakeRedis:
    """只实现 `Cache` 用到的那几个命令，**但认真做 JSON 编解码**。

    用 `MemoryBackend` 当 L2 替身测不出序列化问题——它直接存 Python 对象，
    而真正的 Redis 存的是字节。这个假件的价值全在这里。
    """

    def __init__(self):
        self.data = {}
        self.expiry = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None, nx=False, **kwargs):
        if nx and key in self.data:
            return None
        assert isinstance(value, str), '写进 Redis 的必须是字符串'
        self.data[key] = value
        self.expiry[key] = ex
        return True

    def delete(self, key):
        self.data.pop(key, None)
        self.expiry.pop(key, None)


def _payload(matches=1):
    """一份形状与线上一致的响应：三个玩法各带概率、赔率、质量分档。"""
    return {
        'date': '2026-08-28', 'source': 'okooo', 'total_matches': matches,
        'recommendations': [{
            'num': str(index), 'home': '安山小绿人', 'away': '大邱FC',
            'league': 'K2联赛', 'handicap': '(-1)',
            'spf': {'probabilities': {'胜': 0.4, '平': 0.3, '负': 0.3},
                    'odds': {'胜': 2.1, '平': 3.3, '负': 3.6},
                    'prediction': '胜', 'confidence': 0.4,
                    'scores': [{'score': '1-0', 'probability': 0.12,
                                'home_goals': 1, 'away_goals': 0}],
                    'history_calibration': {'applied': False,
                                            'reason': 'insufficient_settled_samples'}},
            'zjq': {'probabilities': {'0': 0.03, '7+': 0.07},
                    'top3': [['4', 0.25], ['3', 0.21], ['2', 0.14]]},
            'rqspf': {'probabilities': {'让胜': 0.3, '让平': 0.18, '让负': 0.52}},
        } for index in range(matches)],
        'history_summary': {'total_records': 500, 'settled_records': 0},
    }


def _cache_with_redis():
    fake = _FakeRedis()
    return Cache(l1=MemoryBackend(), l2=RedisBackend(fake), default_ttl=60), fake


class AcrossRestartTests(unittest.TestCase):
    """L1 清空之后还读得到——「跨重启」就是这一句。"""

    def test_a_written_payload_survives_losing_l1(self):
        cache, _ = _cache_with_redis()
        with patch.object(beidan_cache, 'get_shared_cache', return_value=cache):
            beidan_cache.write_beidan_cache('okooo_today_spf', _payload())
            # 重启 = L1 连同进程一起没了，L2 还在
            cache.l1 = MemoryBackend()
            payload, _ = beidan_cache.read_beidan_cache('okooo_today_spf')
        self.assertIsNotNone(payload, 'L1 没了就读不到，等于没接 L2')
        self.assertEqual(payload['total_matches'], 1)
        self.assertEqual(len(payload['recommendations']), 1)

    def test_reading_from_l2_refills_l1(self):
        """回填之后不该每次都再过一趟 Redis。"""
        cache, fake = _cache_with_redis()
        with patch.object(beidan_cache, 'get_shared_cache', return_value=cache):
            beidan_cache.write_beidan_cache('k', _payload())
            cache.l1 = MemoryBackend()
            beidan_cache.read_beidan_cache('k')
            with patch.object(fake, 'get', side_effect=AssertionError('不该再读 Redis')):
                payload, _ = beidan_cache.read_beidan_cache('k')
        self.assertIsNotNone(payload)

    def test_a_miss_is_reported_as_never_computed(self):
        cache, _ = _cache_with_redis()
        with patch.object(beidan_cache, 'get_shared_cache', return_value=cache):
            self.assertEqual(beidan_cache.read_beidan_cache('没算过'), (None, False))


class SerialisableTests(unittest.TestCase):
    """**payload 进不了 L2 的话，这次改动的收益是零。**"""

    def test_the_payload_really_lands_in_redis(self):
        """不是「写入没抛异常」，是**去 Redis 里把它读出来**（判据 27）。"""
        cache, fake = _cache_with_redis()
        with patch.object(beidan_cache, 'get_shared_cache', return_value=cache):
            beidan_cache.write_beidan_cache('okooo_today_spf', _payload(matches=3))
        stored = fake.data.get(REDIS_KEY_PREFIX + PREFIX + ':okooo_today_spf')
        self.assertIsNotNone(stored, 'Redis 里没有这个键——写入被静默吞掉了')
        decoded = json.loads(stored)
        self.assertEqual(decoded['value']['total_matches'], 3)
        self.assertEqual(len(decoded['value']['recommendations']), 3)

    def test_a_payload_with_tuple_keys_never_reaches_redis(self):
        """**钉住一处已知限制**：比分那一路的概率是元组键的矩阵，
        JSON 的对象键只能是字符串，所以它写不进 L2。

        `Cache.set` 会把这个失败吞成一条 warning 并退化为纯 L1——
        表现是「缓存看着接上了，可这一类请求每次冷启动还是要重算」。
        线上默认请求的三个玩法里没有比分（要 `types=bifen` 才触发），
        所以此刻走不到；这条用例是留给下一个人的路标。
        """
        payload = _payload()
        payload['recommendations'][0]['bifen'] = {
            'probabilities': {(1, 0): 0.12, (1, 1): 0.11}}
        cache, fake = _cache_with_redis()
        with patch.object(beidan_cache, 'get_shared_cache', return_value=cache):
            beidan_cache.write_beidan_cache('okooo_today_bifen', payload)
            # L1 里有——所以同一个进程内还是命中的
            self.assertIsNotNone(beidan_cache.read_beidan_cache('okooo_today_bifen')[0])
            # 但 Redis 里没有，重启就没了
            self.assertIsNone(fake.data.get(REDIS_KEY_PREFIX + PREFIX + ':okooo_today_bifen'))
            cache.l1 = MemoryBackend()
            self.assertEqual(beidan_cache.read_beidan_cache('okooo_today_bifen'),
                             (None, False))


class TtlTests(unittest.TestCase):
    """**一天是内存上限，不是有效期**——有效期由 `_cached_at` 单独决定。"""

    def test_the_logical_ttl_is_a_day(self):
        cache, _ = _cache_with_redis()
        with patch.object(cache, 'set', wraps=cache.set) as setter:
            with patch.object(beidan_cache, 'get_shared_cache', return_value=cache):
                beidan_cache.write_beidan_cache('k', _payload())
        self.assertEqual(setter.call_args.kwargs['ttl'], TTL_SECONDS)

    def test_the_physical_expiry_is_long_enough_to_bridge_a_restart(self):
        """**跨重启保留靠的是物理过期够长。** 拿刷新档位（最短 120 秒）
        当 TTL 的话，物理过期只有 20 分钟——一次部署加冷启动就耗光了，
        而北单一轮全量重算实测就要 25 分钟。"""
        cache, fake = _cache_with_redis()
        with patch.object(beidan_cache, 'get_shared_cache', return_value=cache):
            beidan_cache.write_beidan_cache('k', _payload())
        physical = fake.expiry[REDIS_KEY_PREFIX + PREFIX + ':k']
        self.assertEqual(physical, TTL_SECONDS * REDIS_STALE_GRACE_FACTOR)
        self.assertGreater(physical, 25 * 60)

    def test_an_entry_older_than_the_ttl_is_still_served(self):
        """**跨天的第一个请求靠的就是这条。**

        `beidan_cache_key(None, ...)` 生成的是 `okooo_today_spf`——**这个键
        不含日期，跨天会复用**。前一天写下的条目按一天的 TTL 已经过期，
        而 Redis 按十倍存物理过期，所以它还在。这时必须照样把它返回出去
        （随后后台刷新），否则跨零点后的第一个请求要同步等 25 分钟。

        只用「TTL 之内但 `_cached_at` 很旧」的语料是测不出这条的——
        那种条目按 `is_fresh()` 仍然是新鲜的（判据 23）。
        """
        cache, _ = _cache_with_redis()
        payload = _payload()
        payload['_cached_at'] = time.time() - 25 * 3600
        stored_at = time.time() - 25 * 3600
        cache.l2.set(beidan_cache._shared_key('okooo_today_spf'), payload,
                     ttl=TTL_SECONDS, now=stored_at)
        with patch.object(beidan_cache, 'get_shared_cache', return_value=cache):
            entry = cache.peek(beidan_cache._shared_key('okooo_today_spf'))
            self.assertFalse(entry.is_fresh(), '语料本身要真的越过 TTL')
            stale, fresh = beidan_cache.read_beidan_cache('okooo_today_spf')
        self.assertIsNotNone(stale, '按 TTL 过期就不返回的话，跨天必然 504')
        self.assertFalse(fresh)

    def test_freshness_still_comes_from_cached_at_not_the_ttl(self):
        """存了一天不代表一天都算新鲜：过期与否仍由刷新档位判定。"""
        cache, _ = _cache_with_redis()
        payload = _payload()
        with patch.object(beidan_cache, 'get_shared_cache', return_value=cache):
            beidan_cache.write_beidan_cache('k', payload)
            _, fresh = beidan_cache.read_beidan_cache('k')
            self.assertTrue(fresh)
            payload['_cached_at'] = time.time() - 10 * 3600
            cache.set(beidan_cache._shared_key('k'), payload, ttl=TTL_SECONDS)
            cache.l1 = MemoryBackend()
            stale, fresh = beidan_cache.read_beidan_cache('k')
        self.assertIsNotNone(stale, '过期也要读得到——这是不 504 的关键')
        self.assertFalse(fresh)


class DegradedTests(unittest.TestCase):
    """共享缓存建不起来时退回进程内存。**这条路存在，但它就是问题本身。**"""

    def test_it_falls_back_to_the_in_process_cache(self):
        with patch.object(beidan_cache, 'get_shared_cache', return_value=None):
            with patch.object(beidan_cache, '_memory_set_cache') as setter:
                beidan_cache.write_beidan_cache('k', _payload())
            setter.assert_called_once()
            with patch.object(beidan_cache, '_memory_get_cache',
                              return_value=_payload()) as getter:
                payload, _ = beidan_cache.read_beidan_cache('k')
            getter.assert_called_once()
        self.assertIsNotNone(payload)

    def test_the_fallback_says_so_in_the_log(self):
        """静默降级是这个项目反复吃亏的形状（判据 18）——要留下痕迹。"""
        with patch.object(beidan_cache, 'get_shared_cache', return_value=None):
            with patch.object(beidan_cache, '_memory_get_cache', return_value=None):
                with patch.object(beidan_cache.log, 'warning') as warned:
                    beidan_cache.read_beidan_cache('k')
        self.assertIn('不跨重启', warned.call_args.args[0])


class KeyNamespaceTests(unittest.TestCase):

    def test_keys_are_namespaced(self):
        """L2 与别的业务共用一个 Redis 库，前缀不能省。"""
        self.assertTrue(beidan_cache._shared_key('x').startswith(PREFIX + ':'))

    def test_different_cache_keys_stay_apart(self):
        cache, _ = _cache_with_redis()
        with patch.object(beidan_cache, 'get_shared_cache', return_value=cache):
            beidan_cache.write_beidan_cache('a', _payload(matches=1))
            beidan_cache.write_beidan_cache('b', _payload(matches=2))
            self.assertEqual(beidan_cache.read_beidan_cache('a')[0]['total_matches'], 1)
            self.assertEqual(beidan_cache.read_beidan_cache('b')[0]['total_matches'], 2)


if __name__ == '__main__':
    unittest.main()
