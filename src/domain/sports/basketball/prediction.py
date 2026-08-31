"""赛前预测流程：取赛程 → 取走势 → 三个玩法各出结论 → 汇总成 payload。

**接缓存的收益在这一层**。迁移前 `/api/basketball` 系列端点没有任何缓存，
每个请求都要重跑一遍抓取加分析；并发请求各算各的，算出来的还是同一份东西。
缓存挂在整个 payload 上而不是中间产物：收益主要来自消除重复的全流程，
而分层缓存会让「一次请求触发几次单飞」变得难以推理。

外部数据都以端口注入，本模块只做编排：

- `schedule_sources`：{源名: fn(date) -> [match]}，中国足彩网不可用时回落 500。
- `movement_provider`：fn(matches, source, date) -> {match_id: {spf,rqspf,dx}}。
  走势的采集仍在旧模块（依赖尚未迁移的快照持久化），随赛前赔率追踪一起改造。
- `recorder`：预测记录的落盘，缺失或抛错都不该让请求失败。

**payload 必须是纯 JSON**：生产态 L2 是 Redis，`Cache.set` 会吞掉 L2 的
写入失败。不满足这条时表现不是报错，而是这个 key 永远进不了 L2、每次冷启动
都要重算——缓存看着接上了，收益是零。
"""
import logging

from src.domain.sports.basketball import movement as mv

log = logging.getLogger('domain.basketball.prediction')

PREDICTION_VERSION = '2026-08-20-water-reverse-v3'
BET_TYPES = ('spf', 'rqspf', 'dx')
DEFAULT_SOURCE = '500'

# 缓存有效期。上限来自走势的时效性——line_movement 是给人看的实时信号，
# 缓存久了会出现「页面显示刚变盘，实际是几分钟前的判断」；下限来自收益，
# 太短则命中率不足以摊掉一次冷计算。
DEFAULT_TTL = 120
CACHE_KEY_PREFIX = 'basketball:pred'

VALUE_BET_LIMIT = 20
# 聪明钱确认给的排序增益。只影响推荐次序，不改变概率本身。
SHARP_CONFIRM_BONUS = 0.04


class PredictionService:
    def __init__(self, analyzer, schedule_sources, movement_provider=None,
                 recorder=None, cache=None, ttl=DEFAULT_TTL,
                 version=PREDICTION_VERSION, today_fn=None):
        self._analyzer = analyzer
        self._schedules = dict(schedule_sources)
        self._movement_provider = movement_provider
        self._recorder = recorder
        self._cache = cache
        self._ttl = ttl
        self._version = version
        self._today = today_fn or _today

    def generate(self, date=None, bet_types=None, source=DEFAULT_SOURCE,
                 use_movement=True):
        """出一天的全部推荐。date 省略时取今天。

        日期在建 key 之前就解析成具体值——留着 None 会让跨天后仍命中昨天的结果。
        """
        date = date or self._today()
        bet_types = list(BET_TYPES) if bet_types is None else list(bet_types)

        def compute():
            return self._compute(date, bet_types, source, use_movement)

        if self._cache is None:
            return compute()
        key = self._cache_key(date, bet_types, source, use_movement)
        return self._cache.get(key, compute, ttl=self._ttl)

    def fetch_schedule(self, date=None, source=DEFAULT_SOURCE):
        """只取赛程、不做分析。赛程列表端点用它。

        不走缓存：它比整份 payload 便宜得多，而且调用方要的就是「现在
        场上有哪些比赛」，缓存反而会让刚开售的场次晚几分钟出现。
        """
        return self._fetch_matches(date or self._today(), source)

    @staticmethod
    def _cache_key(date, bet_types, source, use_movement):
        """玩法排序后入 key：顺序不同但内容相同的请求算同一件事。"""
        types = ','.join(sorted(bet_types))
        return f'{CACHE_KEY_PREFIX}:{date}:{source}:{types}:{int(bool(use_movement))}'

    def _compute(self, date, bet_types, source, use_movement):
        matches, actual_source = self._fetch_matches_with_source(date, source)
        movement_map = self._fetch_movements(
            matches, actual_source, date, use_movement)
        results = [self._analyze_match(match, movement_map.get(match.get('id')),
                                       bet_types)
                   for match in matches]

        payload = {
            'date': date,
            'count': len(results),
            'results': results,
            'version': self._version,
            'source': actual_source,
            'requested_source': source,
            'source_fallback': actual_source != source,
            'use_movement': use_movement,
            'movement_stats': _movement_stats(results),
        }
        self._record(payload, date, results)
        return payload

    def _fetch_matches(self, date, source):
        matches, _ = self._fetch_matches_with_source(date, source)
        return matches

    def _fetch_matches_with_source(self, date, source):
        """中国足彩网不可用时回落 500：有一份稍弱的赛程，好过整天没有推荐。"""
        if source in self._schedules and source != DEFAULT_SOURCE:
            try:
                matches = self._schedules[source](date)
                if matches:
                    return matches, source
                log.warning('%s 篮球赛程返回空列表，回退 %s', source, DEFAULT_SOURCE)
            except Exception as exc:
                log.warning('%s 篮球赛程不可用，回退 %s: %s', source, DEFAULT_SOURCE, exc)
        return self._schedules[DEFAULT_SOURCE](date), DEFAULT_SOURCE

    def _fetch_movements(self, matches, source, date, use_movement):
        if not (use_movement and matches and self._movement_provider):
            return {}
        try:
            return self._movement_provider(matches, source, date) or {}
        except Exception as exc:
            log.warning('篮球赔率走势构建失败(回退无走势): %s', exc)
            return {}

    def _analyze_match(self, match, movement, bet_types):
        movement = movement or {}
        analyzers = {
            'spf': self._analyzer.analyze_spf,
            'rqspf': self._analyzer.analyze_rqspf,
            'dx': self._analyzer.analyze_daxiao,
        }
        result = {'match': match}
        for bet_type in BET_TYPES:
            result[bet_type] = (analyzers[bet_type](match, movement.get(bet_type))
                                if bet_type in bet_types else None)

        if match.get('status') == 'in_progress':
            _mark_already_started(result)
        result['market_analysis'] = mv.describe_market_movement(movement, result)
        return result

    def _record(self, payload, date, results):
        """记录预测以便日后回算准确率。写失败只告警——这是旁路，不是主链路。"""
        if self._recorder is None:
            return
        try:
            if results:
                self._recorder.save(date, results, self._version)
            payload['history_stats'] = self._recorder.stats()
        except Exception as exc:
            log.warning('篮球预测记录保存失败: %s', exc)


def _today():
    from datetime import datetime

    return datetime.now().strftime('%Y-%m-%d')


def _mark_already_started(result):
    """开赛后的场次一律撤下推荐：赔率已停更，任何结论都不再可执行。"""
    for bet_type in BET_TYPES:
        section = result.get(bet_type)
        if isinstance(section, dict) and section.get('available'):
            section.update({'playable': False, 'official': False,
                            'skip_reason': 'match_already_started'})


def _movement_stats(results):
    """统计走势命中情况，用于日后评估这套信号到底有没有用。"""
    stats = {'sharp_confirmed': 0, 'with_movement': 0, 'steam': 0, 'movement_led': 0}
    for result in results:
        for bet_type in BET_TYPES:
            bet = result.get(bet_type) or {}
            line_movement = bet.get('line_movement')
            if not line_movement:
                continue
            stats['with_movement'] += 1
            stats['sharp_confirmed'] += int(bool(bet.get('sharp_confirmed')))
            stats['steam'] += int(bool(line_movement.get('steam')))
            stats['movement_led'] += int(bool(bet.get('movement_led')))
    return stats


_VALUE_BET_SPECS = {
    'spf': {'label': '胜负', 'probs': ('home_prob', 'away_prob'),
            'suffix': lambda match: ''},
    'rqspf': {'label': '让分胜负', 'probs': ('home_prob', 'away_prob'),
              'suffix': lambda match: f" ({match['handicap']})"},
    'dx': {'label': '大小分', 'probs': ('over_prob', 'under_prob'),
           'suffix': lambda match: f" (总分{match['total_line']})"},
}


def find_value_bets(results, threshold=0.05):
    """筛出模型边际超过阈值的场次，并让聪明钱确认过的排在前面。

    movement_edge 只用于排序，不回写进概率——走势的作用是增强信心，
    不是改变对胜率的估计。
    """
    value_bets = []
    for result in results:
        match = result['match']
        for bet_type, spec in _VALUE_BET_SPECS.items():
            bet = result.get(bet_type)
            if not (bet and bet['available'] and bet.get('playable', True)):
                continue
            prob = max(bet[spec['probs'][0]], bet[spec['probs'][1]])
            label = f"{match['home']} vs {match['away']}{spec['suffix'](match)}"
            _append_value_bet(value_bets, bet, spec['label'], label, threshold, prob)

    value_bets.sort(key=lambda bet: (-bet['movement_edge'], -bet['edge']))
    return value_bets[:VALUE_BET_LIMIT]


def _append_value_bet(value_bets, bet, label, match_label, threshold, prob):
    edge = prob - 0.5
    if edge <= threshold:
        return
    sharp = bool(bet.get('sharp_confirmed', False))
    value_bets.append({
        'type': label,
        'match': match_label,
        'recommendation': bet.get('recommendation'),
        'edge': round(edge, 4),
        'movement_edge': round(edge + (SHARP_CONFIRM_BONUS if sharp else 0.0), 4),
        'prob': round(prob, 4),
        'sharp_confirmed': sharp,
        'movement': bet.get('line_movement'),
    })
