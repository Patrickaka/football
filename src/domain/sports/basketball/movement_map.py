"""为一天的比赛构建走势映射：{match_id: {spf, rqspf, dx}}。

三个来源，优先级由「谁的信息更完整」决定：

1. **澳客赛程页自带的 rf_trend / dx_trend** —— 一行里就带着完整盘路历史，
   最完整，让分与大小分优先用它。
2. **澳客详情页的各家共识** —— 胜负那一路只有这里有。线上它常年被 WAF
   拦着，所以实际很少落到。
3. **自己攒的 500 源快照序列** —— 只有我们轮询到的那些时刻，粒度最粗，
   但它是唯一在任何情况下都拿得到的，用来补前两者的缺口。

500 源那条路上夹着一次**跨源按队名匹配**：500 与澳客对同一场比赛用不同的
主键，只能靠队名对上，而澳客的队名带 `[西2]` 这类排名标记，必须先剥掉。

所有外部依赖（采集、澳客赛程、详情页、快照库）一律注入，且**全部允许为 None**：
走势是增强项，任何一个不可用都只该让结论退化，不该让整天的推荐失败。
"""
import logging
import re

from src.domain.sports.basketball import movement as mv

log = logging.getLogger('domain.basketball.movement_map')

# 澳客队名里的排名标记，如「金州女武神[西2]」
_RANK_TAG = re.compile(r'\[.*?\]')

BET_TYPES = ('spf', 'rqspf', 'dx')
OKOOO_SOURCE = 'okooo'


def normalize_team(name):
    """跨源匹配用的队名规范化。"""
    return _RANK_TAG.sub('', str(name or '')).strip()


class MovementMapBuilder:
    def __init__(self, history_store=None, tracker=None, okooo_schedule=None,
                 bundle_fetcher=None, now_fn=None):
        self._history_store = history_store
        self._tracker = tracker
        self._okooo_schedule = okooo_schedule
        self._bundles = bundle_fetcher
        self._now = now_fn

    def __call__(self, matches, source, date):
        if source == OKOOO_SOURCE:
            return self._from_okooo_source(matches)
        return self._from_500_source(matches, date)

    # ---------- 澳客源 ----------

    def _from_okooo_source(self, matches):
        """比赛自带 rf_trend / dx_trend，只需再补一份详情页共识给胜负。"""
        bundles = self._fetch_bundles([m.get('id') for m in matches])
        return {
            m.get('id'): mv.build_movement_for_match(
                m, okooo_bundle=bundles.get(str(m.get('id'))))
            for m in matches
        }

    # ---------- 500 源 ----------

    def _from_500_source(self, matches, date):
        """先采一轮快照再算——否则一天里的第一个请求永远看不到最新那次变盘。"""
        self._capture(date)
        history = self._load_history()
        enriched = self._enrich_from_okooo(matches, date)

        out = {}
        for match in matches:
            match_id = match.get('id')
            out[match_id] = self._merge(match, enriched.get(match_id), history)
        return out

    def _merge(self, match, okooo_movement, history):
        """澳客的更完整，优先用；它缺的玩法再用自己的快照补齐。"""
        if not okooo_movement:
            return mv.build_movement_for_match(match, history=history,
                                               now_fn=self._now)
        if not history:
            return okooo_movement

        fallback = mv.build_movement_for_match(match, history=history,
                                               now_fn=self._now)
        for bet_type in BET_TYPES:
            if not okooo_movement.get(bet_type) and fallback.get(bet_type):
                okooo_movement[bet_type] = fallback[bet_type]
        return okooo_movement

    def _enrich_from_okooo(self, matches, date):
        """尽力用澳客的盘路补强 500 源的比赛，按队名匹配。

        整段失败都退回空字典：这是锦上添花，拿不到就用自己的快照。
        """
        if not self._okooo_schedule:
            return {}
        try:
            okooo_matches = self._okooo_schedule(date)
        except Exception as exc:
            log.warning('澳客走势增强失败(回退kv历史): %s', exc)
            return {}
        if not okooo_matches:
            return {}

        by_teams = {_team_key(m): m for m in okooo_matches}
        bundles = self._fetch_bundles([m.get('id') for m in okooo_matches])

        enriched = {}
        for match in matches:
            counterpart = by_teams.get(_team_key(match))
            if not counterpart:
                continue
            enriched[match.get('id')] = mv.build_movement_for_match(
                counterpart,
                okooo_bundle=bundles.get(str(counterpart.get('id'))))
        return enriched

    # ---------- 注入依赖的安全调用 ----------

    def _capture(self, date):
        if not self._tracker:
            return
        try:
            self._tracker.track(date)
        except Exception as exc:
            log.warning('篮球赔率采集失败(继续用已有历史): %s', exc)

    def _load_history(self):
        if not self._history_store:
            return None
        try:
            return self._history_store.load()
        except Exception as exc:
            log.warning('赔率历史读取失败: %s', exc)
            return None

    def _fetch_bundles(self, match_ids):
        ids = [str(i) for i in match_ids if i]
        if not (self._bundles and ids):
            return {}
        try:
            return self._bundles.fetch_many(ids) or {}
        except Exception as exc:
            log.warning('澳客 bundle 抓取失败: %s', exc)
            return {}


def _team_key(match):
    return (normalize_team(match.get('home')), normalize_team(match.get('away')))
