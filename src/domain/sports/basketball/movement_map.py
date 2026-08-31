"""为一天的比赛构建走势映射：{match_id: {spf, rqspf, dx}}。

三个来源，优先级由「谁的信息更完整」决定：

1. **中国足彩网当前官方赔率** —— 准确提供胜负、让分与大小分当前值。
2. **自己低频积累的快照序列** —— 用于计算真实走势，不把单个当前值伪装成历史。

500 源那条路上夹着一次**跨源按队名匹配**：两个站点对同一场比赛使用不同
主键，只能靠规范化后的主客队名对上。

所有外部依赖（采集、中国足彩网赛程、快照库）一律注入，且**全部允许为 None**：
走势是增强项，任何一个不可用都只该让结论退化，不该让整天的推荐失败。
"""
import logging
import re

from src.domain.sports.basketball import movement as mv

log = logging.getLogger('domain.basketball.movement_map')

# 队名里的排名标记，如「金州女武神[西2]」
_RANK_TAG = re.compile(r'\[.*?\]')

BET_TYPES = ('spf', 'rqspf', 'dx')
ZGZCW_SOURCE = 'zgzcw'


def normalize_team(name):
    """跨源匹配用的队名规范化。"""
    return _RANK_TAG.sub('', str(name or '')).strip()


class MovementMapBuilder:
    def __init__(self, history_store=None, tracker=None, zgzcw_schedule=None,
                 bundle_fetcher=None, now_fn=None):
        self._history_store = history_store
        self._tracker = tracker
        self._zgzcw_schedule = zgzcw_schedule
        self._bundles = bundle_fetcher
        self._now = now_fn

    def __call__(self, matches, source, date):
        if source == ZGZCW_SOURCE:
            return self._from_zgzcw_source(matches)
        return self._from_500_source(matches, date)

    # ---------- 中国足彩网源 ----------

    def _from_zgzcw_source(self, matches):
        """当前赔率由赛程页提供，走势由本地快照按需补齐。"""
        bundles = self._fetch_bundles([m.get('id') for m in matches])
        return {
            m.get('id'): mv.build_movement_for_match(
                m, source_bundle=bundles.get(str(m.get('id'))))
            for m in matches
        }

    # ---------- 500 源 ----------

    def _from_500_source(self, matches, date):
        """先采一轮快照再算——否则一天里的第一个请求永远看不到最新那次变盘。"""
        self._capture(date)
        history = self._load_history()
        enriched = self._enrich_from_zgzcw(matches, date)

        out = {}
        for match in matches:
            match_id = match.get('id')
            out[match_id] = self._merge(match, enriched.get(match_id), history)
        return out

    def _merge(self, match, source_movement, history):
        """源站增强缺失的玩法由自己的快照补齐。"""
        if not source_movement:
            return mv.build_movement_for_match(match, history=history,
                                               now_fn=self._now)
        if not history:
            return source_movement

        fallback = mv.build_movement_for_match(match, history=history,
                                               now_fn=self._now)
        for bet_type in BET_TYPES:
            if not source_movement.get(bet_type) and fallback.get(bet_type):
                source_movement[bet_type] = fallback[bet_type]
        return source_movement

    def _enrich_from_zgzcw(self, matches, date):
        """尽力用中国足彩网当前赔率补强 500 源的比赛，按队名匹配。

        整段失败都退回空字典：这是锦上添花，拿不到就用自己的快照。
        """
        if not self._zgzcw_schedule:
            return {}
        try:
            zgzcw_matches = self._zgzcw_schedule(date)
        except Exception as exc:
            log.warning('中国足彩网走势增强失败(回退kv历史): %s', exc)
            return {}
        if not zgzcw_matches:
            return {}

        by_teams = {_team_key(m): m for m in zgzcw_matches}
        bundles = self._fetch_bundles([m.get('id') for m in zgzcw_matches])

        enriched = {}
        for match in matches:
            counterpart = by_teams.get(_team_key(match))
            if not counterpart:
                continue
            enriched[match.get('id')] = mv.build_movement_for_match(
                counterpart,
                source_bundle=bundles.get(str(counterpart.get('id'))))
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
            log.warning('中国足彩网赔率增强失败: %s', exc)
            return {}


def _team_key(match):
    return (normalize_team(match.get('home')), normalize_team(match.get('away')))
