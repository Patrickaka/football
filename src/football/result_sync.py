#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
赛后比分同步模块
================

功能：
1. 保存预测记录（赛前）
2. 定时扫描未结算比赛
3. 自动抓取实际比分
4. 更新校准库、盘口库、ELO、命中率统计

同步状态：
- pending: 等待比赛结束
- ready: 可以同步
- synced: 已回填
- retry: 等待重试
- failed: 多次失败，不再重试
- ignored: 不参与回填

重试策略：
- 失败1次：2小时后再试
- 失败2次：6小时后再试
- 失败3次：24小时后再试
- 失败5次：标记为failed

数据结构：
{
    "match_id": "123456",
    "league": "英超",
    "home": "阿森纳",
    "away": "切尔西",
    "match_time": "2026-06-12 22:00:00",
    "asian": -0.5,
    "total_line": 2.5,
    "predicted_scores": {"1-1": 0.112, "2-1": 0.094},
    "predicted_1x2": {"home": 0.46, "draw": 0.27, "away": 0.27},
    "actual_score": null,
    "actual_result": null,
    "settled": false,
    "sync_status": "pending",      # 新增：同步状态
    "sync_attempts": 0,            # 新增：同步尝试次数
    "last_sync_at": null,          # 新增：上次同步时间
    "last_sync_error": null,       # 新增：上次同步错误
    "next_sync_at": null,          # 新增：下次同步时间
    "created_at": "..."
}
"""

import os
import re
import json
import time
import math
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from threading import Thread, RLock

from ..common import repositories

from ..domain.sports.football import settlement as _st

# 27 个纯计算转发给领域层（时间解析的"当前年"由调用方注入）
normalize_1x2_probs = _st.normalize_1x2_probs
calculate_logloss = _st.calculate_logloss
calculate_brier_score = _st.calculate_brier_score
calculate_hit = _st.calculate_hit
_score_to_result = _st._score_to_result
_parse_score_result = _st._parse_score_result
_parse_score_string = _st._parse_score_string
_extract_score_text = _st._extract_score_text
_parse_shuju_score = _st._parse_shuju_score
_parse_live_row_final_score = _st._parse_live_row_final_score
_parse_live_row_score = _st._parse_live_row_score
_is_valid_match_id = _st._is_valid_match_id


def _calibration_sample_weight(record):
    """样本质量评估住在本层，注入给领域层（判据 16）"""
    from .sample_quality import assess_record_quality
    return _st._calibration_sample_weight(record, assess_record_quality)

_is_result_quality_usable = _st._is_result_quality_usable
fuse_probabilities = _st.fuse_probabilities
evaluate_ml_prediction = _st.evaluate_ml_prediction
time_layer_weight = _st.time_layer_weight
check_ml_fusion_eligibility = _st.check_ml_fusion_eligibility
get_ml_fusion_weight = _st.get_ml_fusion_weight
_prediction_decision_snapshot = _st._prediction_decision_snapshot
_audited_decision_snapshot = _st._audited_decision_snapshot
_prediction_content_sig = _st._prediction_content_sig
_parse_match_datetime = _st._parse_match_datetime
_is_match_settle_due = _st._is_match_settle_due
_assess_result_quality = _st._assess_result_quality
infer_time_layer = _st.infer_time_layer
_live_query_dates = _st._live_query_dates


log = logging.getLogger('football')

PRODUCTION_MODEL_VERSION = 'football-v2026.08.20-audited-upset-gated-11'
# 3,504-match chronological validation (2024/25 -> 2025/26): the 0.65 gate
# held 76.86% -> 77.82% accuracy at 19.98% -> 15.70% coverage.  Margin 0.10
# remains explicit for auditability; at a normalized 65% top probability the
# probability threshold is normally the binding constraint.
ACTIONABLE_MIN_PROBABILITY = 0.65
ACTIONABLE_MIN_MARGIN = 0.10
ACTIONABLE_POLICY_VERSION = 'selective-1x2-v4-accuracy-first'






DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
HISTORY_FILE = os.path.join(DATA_DIR, 'prediction_history.json')


# ==================== 评估指标计算函数 ====================



























def _append_market_timeline(
    timeline,
    *,
    match_time,
    layer,
    odds_data,
    predicted_1x2,
    predicted_rqspf,
    model_version,
    professional_snapshot=None,
    limit=80,
):
    """Append an immutable, timestamped market/prediction snapshot if changed."""
    rows = list(timeline or [])
    captured_at = datetime.now()
    match_dt = _parse_match_datetime(match_time)
    seconds_to_kickoff = (
        int((match_dt - captured_at).total_seconds()) if match_dt else None
    )
    is_prematch = seconds_to_kickoff is None or seconds_to_kickoff >= 0
    try:
        payload = json.loads(json.dumps({
            'odds': odds_data,
            'predicted_1x2': predicted_1x2,
            'predicted_rqspf': predicted_rqspf,
            'professional_snapshot': professional_snapshot,
        }, ensure_ascii=False, sort_keys=True, default=str))
    except Exception:
        payload = {
            'odds': odds_data,
            'predicted_1x2': predicted_1x2,
            'predicted_rqspf': predicted_rqspf,
            'professional_snapshot': professional_snapshot,
        }
    signature_source = json.dumps(
        [layer, payload, model_version],
        ensure_ascii=False, sort_keys=True, default=str,
    )
    signature = hashlib.sha256(signature_source.encode('utf-8')).hexdigest()
    if rows and rows[-1].get('signature') == signature:
        return rows, None
    snapshot = {
        'captured_at': captured_at.isoformat(timespec='seconds'),
        'layer': layer,
        'seconds_to_kickoff': seconds_to_kickoff,
        'is_prematch': is_prematch,
        'model_version': model_version,
        **payload,
        'signature': signature,
    }
    rows.append(snapshot)
    return rows[-limit:], snapshot


def _complete_offered_lottery_predictions(
    predicted_scores,
    predicted_1x2,
    predicted_rqspf,
    lottery_handicap,
    lottery_snapshot,
):
    """保证每个已开售竞彩玩法都有一份可保存、可展示的概率预测。

    正常路径会直接传入两种预测；这里主要修复历史记录和缓存升级场景：
    玩法已经核验为开售，但旧版本漏存了其中一种预测时，从同一份比分分布
    重新边际化，不能继续把空值渲染成“暂无预测”。
    """
    lottery_snapshot = lottery_snapshot or {}
    handicap_snapshot = lottery_snapshot.get('handicap') or {}
    resolved_handicap = (
        lottery_handicap
        if lottery_handicap is not None
        else handicap_snapshot.get('handicap')
    )
    if not lottery_snapshot.get('offer_matched'):
        return predicted_1x2, predicted_rqspf, resolved_handicap

    spf_was_offered = bool(
        lottery_snapshot.get('spf_available')
        and lottery_snapshot.get('spf_odds')
    )
    rqspf_was_offered = bool(
        lottery_snapshot.get('rqspf_available')
        and lottery_snapshot.get('rqspf_odds')
    )
    if ((not spf_was_offered or predicted_1x2)
            and (not rqspf_was_offered or predicted_rqspf)):
        return predicted_1x2, predicted_rqspf, resolved_handicap

    candidates = []
    for score, probability in (predicted_scores or {}).items():
        score_match = re.match(r'^\s*(\d+)\s*[-:：]\s*(\d+)\s*$', str(score))
        if not score_match:
            continue
        try:
            probability = float(probability)
        except (TypeError, ValueError):
            continue
        if probability > 0:
            candidates.append(((int(score_match.group(1)), int(score_match.group(2))),
                               probability))
    if not candidates:
        return predicted_1x2, predicted_rqspf, resolved_handicap

    from ..domain.sports.football.lottery import lottery_market_probabilities

    completed = lottery_market_probabilities(
        candidates,
        resolved_handicap,
        spf_odds=lottery_snapshot.get('spf_odds'),
        rqspf_odds=lottery_snapshot.get('rqspf_odds'),
    )
    if spf_was_offered and not predicted_1x2:
        standard = ((completed.get('standard') or {}).get('probabilities') or {})
        predicted_1x2 = {
            'H': standard.get('胜', 0.0),
            'D': standard.get('平', 0.0),
            'A': standard.get('负', 0.0),
        }
    if rqspf_was_offered and not predicted_rqspf:
        predicted_rqspf = (
            (completed.get('handicap') or {}).get('probabilities') or {}
        )
    return predicted_1x2, predicted_rqspf, resolved_handicap


def _lottery_match_key(match_num, match_time):
    """竞彩编号+开赛时间构成的跨源业务键；任一为空则不可用于对齐。"""
    num = str(match_num or '').replace(' ', '')
    when = str(match_time or '').strip()
    return (num, when) if num and when else None


class PredictionHistory:
    """预测历史记录管理器"""

    # 多场比赛并发分析时 add_prediction 会同时读写 records 并做单条 UPSERT，
    # 用类级可重入锁把「查重—更新—落库」串起来（实例可能绕过 __init__ 构造）。
    _records_lock = RLock()

    def __init__(self):
        self.records: List[Dict] = []
        self._load()
    
    def _load(self):
        """从 MySQL 加载记录"""
        try:
            self.records = repositories.football_prediction_load()
            log.debug("已加载 %d 条预测历史记录", len(self.records))
        except Exception as e:
            log.error(f"加载预测历史失败: {e}")
            self.records = []

    def _save(self):
        """保存记录到 MySQL（整表重写）。仅用于批量操作（audit/repair 等）。

        每请求级的单条变更请用 _save_record，避免整表 DELETE+INSERT 把
        binlog/磁盘写爆。
        """
        try:
            repositories.football_prediction_save(self.records)
        except Exception as e:
            log.error(f"保存预测历史失败: {e}")

    def _save_record(self, record):
        """仅 UPSERT 单条记录，把每请求写入量从 O(表行数) 降到 O(1)。"""
        try:
            backend = repositories.football_prediction_upsert(record)
            if backend == 'fallback':
                log.warning(
                    "MySQL预测记录写入失败，已降级本地存储: match_id=%s",
                    record.get('match_id'),
                )
            return backend
        except Exception as e:
            log.error(f"保存预测记录失败: {e}")
            return 'failed'
    
    def add_prediction(self, *args, **kwargs):
        """添加预测记录（并发安全入口）"""
        with self._records_lock:
            return self._add_prediction(*args, **kwargs)

    def _find_existing_record(self, match_id: str, match_num: str,
                              match_time: str) -> Optional[Dict]:
        """先按 match_id 找，再按竞彩编号+开赛时间找。

        换源会整体改写 match_id 的命名空间（500 的数字 fid → 竞彩官网的
        `sporttery_*`），只认 match_id 会给同一场比赛再建一条记录，页面上
        就成了同场两条、赛果与竞彩玩法各在一条上。竞彩编号+开赛时间是两个
        源都稳定的业务键，队名与联赛名则两边写法不一致，不能用来对齐。
        """
        for record in self.records:
            if record.get('match_id') == match_id:
                return record
        key = _lottery_match_key(match_num, match_time)
        if key is None:
            return None
        for record in self.records:
            if _lottery_match_key(
                    record.get('match_num'), record.get('match_time')) == key:
                return record
        return None

    @staticmethod
    def _register_alias_match_id(record: Dict, match_id: str) -> bool:
        """把跨源命中的新 match_id 记成别名，返回是否是新增的别名。

        身份先到先得：改写 match_id 就得删旧行，反而多一种不一致。
        """
        if not match_id or record.get('match_id') == match_id:
            return False
        aliases = list(record.get('alias_match_ids') or [])
        if match_id in aliases:
            return False
        aliases.append(match_id)
        record['alias_match_ids'] = aliases
        return True

    def _add_prediction(self, match_id: str, league: str, home: str, away: str,
                       match_time: str, predicted_scores: Dict[str, float],
                       predicted_1x2: Dict[str, float], asian: float = None,
                       total_line: float = None, odds_data: Dict = None,
                       predicted_half_full: Dict[str, float] = None,
                       # 影子预测相关字段
                       base_1x2: Dict[str, float] = None,
                       ml_1x2: Dict[str, float] = None,
                       ml_model_version: str = None,
                       ml_available: bool = False,
                       ml_feature_snapshot: Dict = None,
                       lottery_handicap: int = None,
                       predicted_rqspf: Dict[str, float] = None,
                       goal_count: Dict = None,
                       professional_snapshot: Dict = None,
                       model_version: str = PRODUCTION_MODEL_VERSION,
                       match_num: str = None):
        """
        添加预测记录
        
        参数：
            match_id: 比赛ID
            league: 联赛名称
            home: 主队名称
            away: 客队名称
            match_time: 比赛时间
            match_num: 竞彩比赛编号，如“周一001”
            predicted_scores: 预测比分概率 {"1-1": 0.108, ...}
            predicted_1x2: 预测胜平负 {"home": 0.46, "draw": 0.27, "away": 0.27}
            asian: 亚盘让球
            total_line: 大小球盘口
            odds_data: 原始赔率数据（可选）
            predicted_half_full: 预测半全场概率 {"HH": 0.24, "DH": 0.19, ...}（可选）
            base_1x2: 基础模型胜平负预测 {"H": 0.48, "D": 0.27, "A": 0.25}
            ml_1x2: ML模型胜平负预测 {"H": 0.45, "D": 0.29, "A": 0.26}
            ml_model_version: ML模型版本
            ml_available: ML模型是否可用
            ml_feature_snapshot: ML特征快照
        """
        # 已核验到官方竞彩场次时，只保存实际开售玩法的预测。部分数据源会
        # 短暂留下 ``*_available=True``，但对应赔率已经为空；没有完整赔率
        # 就不能视为开售，尤其不能把模型内部的普通胜平负结果记成官方预测。
        lottery_snapshot = (
            (odds_data or {}).get('lottery')
            if isinstance(odds_data, dict) else None
        ) or {}
        predicted_1x2, predicted_rqspf, lottery_handicap = (
            _complete_offered_lottery_predictions(
                predicted_scores,
                predicted_1x2,
                predicted_rqspf,
                lottery_handicap,
                lottery_snapshot,
            )
        )
        if lottery_snapshot.get('offer_matched'):
            spf_was_offered = bool(
                lottery_snapshot.get('spf_available')
                and lottery_snapshot.get('spf_odds')
            )
            rqspf_was_offered = bool(
                lottery_snapshot.get('rqspf_available')
                and lottery_snapshot.get('rqspf_odds')
            )
            if not spf_was_offered:
                predicted_1x2 = {}
                base_1x2 = {} if base_1x2 is not None else None
                ml_1x2 = {} if ml_1x2 is not None else None
            if not rqspf_was_offered:
                predicted_rqspf = {}

        # 检查是否已存在
        existing = self._find_existing_record(match_id, match_num, match_time)
        if existing is not None:
            record = existing
            newly_aliased = self._register_alias_match_id(record, match_id)
            # 跳过无变化的重复写入：缓存命中时同一场比赛会被反复「预测」，
            # 但内容与时间层其实一字未变。此时直接返回，不写库、不更新时间戳，
            # 消灭每请求整表重写的写入风暴。
            layer = infer_time_layer(match_time)
            new_sig = _prediction_content_sig(
                predicted_scores, predicted_1x2, asian, total_line,
                odds_data, predicted_half_full, model_version,
                professional_snapshot,
                lottery_handicap=lottery_handicap,
                predicted_rqspf=predicted_rqspf,
            )
            existing_layers = record.get('time_layers') or {}
            if (
                new_sig is not None
                and record.get('_pred_sig') == new_sig
                and existing_layers.get(layer) is not None
                and not record.get('settled')
                and (not match_num or record.get('match_num') == match_num)
                and not newly_aliased
            ):
                return {'saved': False, 'persistence_backend': 'unchanged'}

            # 更新现有记录
            update_data = {
                'league': league,
                'predicted_scores': predicted_scores,
                'predicted_1x2': predicted_1x2,
                'asian': asian,
                'total_line': total_line,
                'updated_at': datetime.now().isoformat(),
                'odds_snapshot': odds_data,
                'model_version': model_version,
                'decision_snapshot': _audited_decision_snapshot(
                    predicted_1x2, professional_snapshot,
                ),
                'professional_snapshot': professional_snapshot,
                '_pred_sig': new_sig,
            }
            if match_num:
                update_data['match_num'] = match_num
            if predicted_half_full:
                update_data['predicted_half_full'] = predicted_half_full
            # 添加影子预测字段
            if base_1x2 is not None:
                update_data['base_1x2'] = base_1x2
            if ml_1x2 is not None:
                update_data['ml_1x2'] = ml_1x2
            if ml_model_version:
                update_data['ml_model_version'] = ml_model_version
            update_data['ml_available'] = ml_available
            if ml_feature_snapshot:
                update_data['ml_feature_snapshot'] = ml_feature_snapshot
            update_data['lottery_handicap'] = lottery_handicap
            update_data['predicted_rqspf'] = predicted_rqspf
            if goal_count:
                update_data['goal_count'] = goal_count
            record.update(update_data)

            # 更新对应时间层的预测
            if 'time_layers' not in record:
                record['time_layers'] = {}
            record['time_layers']['final'] = predicted_scores  # 始终更新最终预测
            # 只在该层为None时才更新（保留更早时间点的预测）
            if record['time_layers'].get(layer) is None:
                record['time_layers'][layer] = predicted_scores

            # 更新赔率分层记录
            if 'odds_layers' not in record:
                record['odds_layers'] = {}
            record['odds_layers'][layer] = odds_data
            record['odds_layers']['final'] = odds_data

            record['market_timeline'], market_snapshot = _append_market_timeline(
                record.get('market_timeline'),
                match_time=match_time,
                layer=layer,
                odds_data=odds_data,
                predicted_1x2=predicted_1x2,
                predicted_rqspf=predicted_rqspf,
                model_version=model_version,
                professional_snapshot=professional_snapshot,
            )
            if market_snapshot and market_snapshot.get('is_prematch'):
                record['last_prematch_odds_snapshot'] = odds_data
                record['last_prematch_snapshot_at'] = market_snapshot['captured_at']
                if layer in {'T-15min', 'final'}:
                    record['closing_odds_snapshot'] = odds_data
                    record['closing_odds_source'] = 'last_observed_prematch_proxy'

            backend = self._save_record(record)
            return {'saved': True, 'persistence_backend': backend}
        
        # 新增记录
        # 时间分层预测记录
        time_layers = {
            'T-24h': None,  # 赛前24小时预测
            'T-6h': None,   # 赛前6小时预测
            'T-1h': None,   # 赛前1小时预测
            'T-15min': None, # 赛前15分钟预测
            'final': predicted_scores,  # 最终预测
        }
        
        # 当前时间层也记录预测（与下方 odds_layers 对齐），
        # 否则按值去重时同一层会被重复预测一次
        layer = infer_time_layer(match_time)
        if layer in time_layers:
            time_layers[layer] = predicted_scores

        # 赔率分层记录
        odds_layers = {
            'T-24h': None,
            'T-6h': None,
            'T-1h': None,
            'T-15min': None,
            'final': odds_data,
        }
        odds_layers[layer] = odds_data
        market_timeline, market_snapshot = _append_market_timeline(
            [],
            match_time=match_time,
            layer=layer,
            odds_data=odds_data,
            predicted_1x2=predicted_1x2,
            predicted_rqspf=predicted_rqspf,
            model_version=model_version,
            professional_snapshot=professional_snapshot,
        )
        
        record_data = {
            'match_id': match_id,
            'league': league,
            'home': home,
            'away': away,
            'match_time': match_time,
            'match_num': match_num,
            'asian': asian,
            'total_line': total_line,
            'predicted_scores': predicted_scores,
            'predicted_1x2': predicted_1x2,
            'model_version': model_version,
            'decision_snapshot': _audited_decision_snapshot(
                predicted_1x2, professional_snapshot,
            ),
            'professional_snapshot': professional_snapshot,
            'lottery_handicap': lottery_handicap,
            'predicted_rqspf': predicted_rqspf,
            'goal_count': goal_count,
            'predicted_half_full': predicted_half_full,  # 新增：半全场预测
            'time_layers': time_layers,  # 新增：时间分层预测记录
            'odds_layers': odds_layers,  # 新增：赔率分层记录
            'market_timeline': market_timeline,
            'last_prematch_odds_snapshot': (
                odds_data if market_snapshot and market_snapshot.get('is_prematch') else None
            ),
            'last_prematch_snapshot_at': (
                market_snapshot.get('captured_at') if market_snapshot and market_snapshot.get('is_prematch') else None
            ),
            'closing_odds_snapshot': (
                odds_data if market_snapshot and market_snapshot.get('is_prematch')
                and layer in {'T-15min', 'final'} else None
            ),
            'closing_odds_source': (
                'last_observed_prematch_proxy'
                if market_snapshot and market_snapshot.get('is_prematch')
                and layer in {'T-15min', 'final'} else None
            ),
            # 影子预测字段
            'base_1x2': base_1x2,
            'ml_1x2': ml_1x2,
            'ml_model_version': ml_model_version,
            'ml_available': ml_available,
            'ml_feature_snapshot': ml_feature_snapshot,
            # 赛后评估字段（结算时填充）
            'evaluation': None,
            'actual_score': None,
            'actual_result': None,
            'actual_half_score': None,   # 新增：实际半场比分
            'actual_half_result': None,  # 新增：实际半场结果
            'actual_half_full': None,    # 新增：实际半全场结果
            'settled': False,
            # 同步状态字段
            'sync_status': 'pending',
            'sync_attempts': 0,
            'last_sync_at': None,
            'last_sync_error': None,
            'next_sync_at': None,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'odds_snapshot': odds_data,
            '_pred_sig': _prediction_content_sig(
                predicted_scores, predicted_1x2, asian, total_line,
                odds_data, predicted_half_full, model_version,
                professional_snapshot,
                lottery_handicap=lottery_handicap,
                predicted_rqspf=predicted_rqspf,
            ),
        }
        self.records.append(record_data)
        backend = self._save_record(record_data)
        log.info(f"添加预测记录: {home} vs {away} (match_id={match_id})")
        return {'saved': True, 'persistence_backend': backend}
    
    def get_record(self, match_id: str) -> Optional[Dict]:
        """按比赛ID获取单条记录，无则返回 None"""
        return next((r for r in self.records if r.get('match_id') == match_id), None)

    def get_unsettled(self) -> List[Dict]:
        """获取未结算的记录"""
        return [r for r in self.records if not r.get('settled', False)]
    
    def get_settled(self, limit: int = None) -> List[Dict]:
        """获取已结算的记录"""
        records = [r for r in self.records if r.get('settled', False)]
        if limit:
            records = records[-limit:]
        return records
    
    def get_ready_to_settle(self, minutes: int = 180) -> List[Dict]:
        """
        获取可以结算的记录（比赛时间已过）
        
        参数：
            minutes: 比赛开始后等待分钟数（默认180分钟=3小时）
        """
        ready = []
        now = datetime.now()
        
        for record in self.records:
            if record.get('settled', False):
                continue
            
            match_time_str = record.get('match_time')
            if not match_time_str:
                continue
            
            try:
                if _is_match_settle_due(match_time_str, minutes=minutes, now=now):
                    ready.append(record)
                    
            except Exception:
                continue
        
        return ready
    
    def update_time_layer(self, match_id: str, time_layer: str, predicted_scores: Dict[str, float]):
        """
        更新时间分层预测记录
        
        参数：
            match_id: 比赛ID
            time_layer: 时间层标识 ('T-24h', 'T-6h', 'T-1h', 'T-15min', 'final')
            predicted_scores: 该时间点的预测比分概率
        """
        for record in self.records:
            if record.get('match_id') == match_id:
                if 'time_layers' not in record:
                    record['time_layers'] = {}
                record['time_layers'][time_layer] = predicted_scores
                record['updated_at'] = datetime.now().isoformat()
                self._save_record(record)
                log.info(f"更新时间分层预测: {match_id} -> {time_layer}")
                return True
        return False
    
    def _calculate_hit_flags(self, record: Dict) -> Dict:
        """计算命中标志和失败原因"""
        actual_score = record.get('actual_score')
        actual_result = record.get('actual_result')
        predicted_scores = record.get('predicted_scores', {})
        predicted_1x2 = normalize_1x2_probs(record.get('predicted_1x2', {}))
        
        # 半全场相关
        actual_half_full = record.get('actual_half_full')
        predicted_half_full = record.get('predicted_half_full', {})
        
        sorted_scores = sorted(predicted_scores.items(), key=lambda x: -x[1])
        top1 = sorted_scores[0][0] if sorted_scores else None
        top3 = [s for s, _ in sorted_scores[:3]]
        top5 = [s for s, _ in sorted_scores[:5]]
        top10 = [s for s, _ in sorted_scores[:10]]
        top20 = [s for s, _ in sorted_scores[:20]]
        top30 = [s for s, _ in sorted_scores[:30]]
        
        # 计算真实比分的排名和概率
        actual_score_rank = None
        actual_score_prob = predicted_scores.get(actual_score, 0)
        
        for i, (score, prob) in enumerate(sorted_scores):
            if score == actual_score:
                actual_score_rank = i + 1  # 排名从1开始
                break
        
        pred_result = max(predicted_1x2.items(), key=lambda x: x[1])[0] if predicted_1x2 else None
        
        hit_top1 = actual_score == top1
        hit_top3 = actual_score in top3
        hit_top5 = actual_score in top5
        hit_top10 = actual_score in top10
        hit_top20 = actual_score in top20
        hit_top30 = actual_score in top30
        hit_1x2 = (pred_result == actual_result) if pred_result and actual_result else None

        actual_rqspf = None
        hit_rqspf = None
        predicted_rqspf = record.get('predicted_rqspf') or {}
        lottery_handicap = record.get('lottery_handicap')
        if predicted_rqspf and lottery_handicap is not None and actual_score:
            try:
                actual_home, actual_away = (int(value) for value in actual_score.split('-'))
                adjusted_margin = actual_home + int(lottery_handicap) - actual_away
                actual_rqspf = '让胜' if adjusted_margin > 0 else '让负' if adjusted_margin < 0 else '让平'
                predicted_rqspf_result = max(predicted_rqspf, key=predicted_rqspf.get)
                hit_rqspf = predicted_rqspf_result == actual_rqspf
            except (TypeError, ValueError):
                pass
        
        # 半全场命中计算
        hit_half_full_top1 = False
        hit_half_full_top3 = False
        if predicted_half_full and actual_half_full:
            sorted_htf = sorted(predicted_half_full.items(), key=lambda x: -x[1])
            htf_top1 = sorted_htf[0][0] if sorted_htf else None
            htf_top3 = [s[0] for s in sorted_htf[:3]]
            hit_half_full_top1 = actual_half_full == htf_top1
            hit_half_full_top3 = actual_half_full in htf_top3
        
        # 半场胜平负命中
        actual_half_result = record.get('actual_half_result')
        hit_half_1x2 = False
        if actual_half_result and predicted_half_full:
            # 从半全场预测中推断半场结果概率
            half_probs = {}
            for key, prob in predicted_half_full.items():
                half_res = key[0]  # 取第一个字符作为半场结果
                half_probs[half_res] = half_probs.get(half_res, 0) + prob
            if half_probs:
                pred_half_result = max(half_probs.items(), key=lambda x: x[1])[0]
                hit_half_1x2 = pred_half_result == actual_half_result
        
        fail_reasons = []
        if not hit_top3:
            fail_reasons = self._analyze_fail_reasons(record, sorted_scores, predicted_1x2)
        
        return {
            'hit_top1': hit_top1,
            'hit_top3': hit_top3,
            'hit_top5': hit_top5,
            'hit_top10': hit_top10,
            'hit_top20': hit_top20,
            'hit_top30': hit_top30,
            'hit_1x2': hit_1x2,
            'hit_rqspf': hit_rqspf,
            'actual_rqspf': actual_rqspf,
            # 半全场命中指标
            'hit_half_full_top1': hit_half_full_top1,
            'hit_half_full_top3': hit_half_full_top3,
            'hit_half_1x2': hit_half_1x2,
            'actual_score_rank': actual_score_rank,
            'actual_score_prob': actual_score_prob,
            'fail_reasons': fail_reasons,
        }
    
    def _analyze_fail_reasons(self, record: Dict, sorted_scores: List[Tuple[str, float]], 
                             predicted_1x2: Dict[str, float]) -> List[str]:
        """分析失败原因"""
        reasons = []
        actual_score = record.get('actual_score', '')
        actual_result = record.get('actual_result', '')
        
        if not actual_score or not actual_result:
            return reasons
        
        try:
            parts = actual_score.split('-')
            home_goals = int(parts[0])
            away_goals = int(parts[1])
            actual_goals = home_goals + away_goals
        except:
            return reasons
        
        pred_total = 0.0
        for score, prob in sorted_scores:
            try:
                h, a = map(int, score.split('-'))
                pred_total += (h + a) * prob
            except:
                pass
        
        pred_max = max(predicted_1x2.items(), key=lambda x: x[1])[0] if predicted_1x2 else None
        
        if pred_total < 2.5 and actual_goals >= 3:
            reasons.append('lambda_error_high')
        elif pred_total >= 2.5 and actual_goals <= 1:
            reasons.append('lambda_error_low')
        
        if pred_max == 'H' and actual_result == 'A':
            reasons.append('supremacy_error')
        elif pred_max == 'A' and actual_result == 'H':
            reasons.append('supremacy_error')
        
        draw_prob = predicted_1x2.get('D', 0)
        if actual_result == 'D' and draw_prob < 0.25:
            reasons.append('draw_underestimated')
        
        away_prob = predicted_1x2.get('A', 0)
        if actual_result == 'A' and away_prob < 0.2:
            reasons.append('away_underestimated')
        
        if actual_goals >= 4:
            has_high_score = False
            for score, prob in sorted_scores[:3]:
                try:
                    h, a = map(int, score.split('-'))
                    if h + a >= 4:
                        has_high_score = True
                        break
                except:
                    pass
            if not has_high_score:
                reasons.append('high_score_missed')
        
        market_weight = record.get('model_params', {}).get('market_weight', 0)
        if market_weight > 0.3 and actual_score not in [s for s, _ in sorted_scores[:5]]:
            reasons.append('market_prior_error')
        
        top3_list = [s for s, _ in sorted_scores[:3]]
        if sorted_scores and sorted_scores[0][1] > 0.4 and actual_score not in top3_list:
            reasons.append('bayes_overcorrect')
        
        steam_signals = record.get('steam_signals', [])
        if steam_signals:
            steam_bias = sum(s.get('bias', 0) for s in steam_signals)
            if steam_bias > 0.5 and actual_result == 'A':
                reasons.append('steam_misread')
            elif steam_bias < -0.5 and actual_result == 'H':
                reasons.append('steam_misread')
        
        return reasons
    
    def get_fail_reason_statistics(self, limit: int = 100) -> Dict[str, int]:
        """获取失败原因统计"""
        statistics = {}
        recent_records = self.get_settled(limit)
        
        for record in recent_records:
            fail_reasons = record.get('fail_reasons', [])
            for reason in fail_reasons:
                statistics[reason] = statistics.get(reason, 0) + 1
        
        return dict(sorted(statistics.items(), key=lambda x: -x[1]))
    
    def print_fail_reason_report(self, limit: int = 100) -> Dict[str, int]:
        """打印失败原因报告"""
        statistics = self.get_fail_reason_statistics(limit)
        
        print(f"\n{'='*60}")
        print(f"最近 {limit} 场失败原因统计")
        print(f"{'='*60}")
        
        if not statistics:
            print("  暂无失败记录")
            return statistics
        
        total_failures = sum(statistics.values())
        print(f"  总失败场次: {total_failures}")
        print(f"{'原因':<25} | {'场次':^8} | {'占比':^10}")
        print(f"{'-'*60}")
        
        reason_descriptions = {
            'lambda_error_high': '总进球高估',
            'lambda_error_low': '总进球低估',
            'supremacy_error': '强弱方向错',
            'draw_underestimated': '平局低估',
            'away_underestimated': '客队低估',
            'high_score_missed': '高比分漏掉',
            'market_prior_error': '盘口先验拉偏',
            'bayes_overcorrect': '贝叶斯校准过度',
            'steam_misread': '资金流误判',
        }
        
        for reason, count in statistics.items():
            desc = reason_descriptions.get(reason, reason)
            percentage = (count / total_failures) * 100
            print(f"  {desc:<25} | {count:^8} | {percentage:^10.1f}%")
        
        print(f"{'='*60}\n")
        
        return statistics
    
    def update_result(self, match_id: str, actual_score: str, actual_result: str,
                      actual_half_score: str = None, error: str = None,
                      source: str = None, now: datetime = None):
        """
        更新比赛结果
        
        参数：
            match_id: 比赛ID
            actual_score: 实际比分 "2-1"
            actual_result: 实际结果 "H"/"D"/"A"
            actual_half_score: 实际半场比分 "1-0"（可选）
            error: 同步错误信息（可选）
            now: 当前时间，**只为可测**。读时钟是副作用，藏在函数里就没法测
                「拒绝提前回填」这条守卫——测试只能写死一个当时属于未来的日期，
                而那种测试会在那天到来之后自己变红（这里踩过一次）。
        """
        for record in self.records:
            if record.get('match_id') == match_id:
                if actual_score and actual_result:
                    if not _is_match_settle_due(record.get('match_time'), minutes=180, now=now):
                        record['sync_status'] = 'pending'
                        record['last_sync_error'] = '比赛尚未到结算时间，拒绝提前回填'
                        record['last_sync_at'] = datetime.now().isoformat()
                        log.warning(
                            f"拒绝提前回填: {record.get('home')} vs {record.get('away')} "
                            f"match_time={record.get('match_time')} score={actual_score}"
                        )
                        self._save_record(record)
                        return False

                    result_quality = _assess_result_quality(
                        record,
                        actual_score,
                        actual_result,
                        source=source,
                        actual_half_score=actual_half_score,
                    )
                    if result_quality['grade'] == 'reject':
                        record['sync_status'] = 'retry'
                        record['last_sync_error'] = f"赛果可信度过低，拒绝回填: {result_quality['reasons']}"
                        record['last_sync_at'] = datetime.now().isoformat()
                        record['next_sync_at'] = (datetime.now() + timedelta(hours=6)).isoformat()
                        self._save_record(record)
                        return False

                    # 成功结算
                    record['actual_score'] = actual_score
                    record['actual_result'] = actual_result
                    record['result_quality'] = result_quality
                    record['settled'] = True
                    settled_at = datetime.now().isoformat()
                    record['settled_at'] = settled_at
                    record['last_sync_at'] = settled_at
                    record['sync_status'] = 'synced'
                    
                    # 处理半场比分
                    if actual_half_score:
                        record['actual_half_score'] = actual_half_score
                        # 计算半场结果
                        try:
                            half_h, half_a = map(int, actual_half_score.split('-'))
                            half_res = 'H' if half_h > half_a else 'A' if half_h < half_a else 'D'
                            record['actual_half_result'] = half_res
                            # 计算半全场结果
                            record['actual_half_full'] = f"{half_res}{actual_result}"
                            # 标记数据质量
                            record['half_time_data_quality'] = 'real'
                        except:
                            record['half_time_data_quality'] = 'invalid'
                    else:
                        record['half_time_data_quality'] = 'missing'
                    
                    # 计算命中结果（包含半全场）
                    record.update(self._calculate_hit_flags(record))
                    
                    # 计算 ML 评估指标
                    record['evaluation'] = evaluate_ml_prediction(record)
                    
                    # 更新各模块。跨源重复记录（换源期间同一场比赛留下的另一条）
                    # 的派生库已由先结算的那条写过，再灌一次会让 ELO 多走一步、
                    # 校准与盘口库多收一份同场样本。
                    if record.get('skip_training_ingest'):
                        log.info(
                            "跨源重复记录，仅回填赛果不重复灌入派生库: "
                            f"{record.get('home')} vs {record.get('away')} "
                            f"match_id={record.get('match_id')}"
                        )
                    elif _is_result_quality_usable(record):
                        self._update_calibrator(record)
                        self._update_market_db(record)
                        self._update_score_frequency_db(record)
                        self._update_elo_ratings(record)
                        self._update_half_time_stats(record)  # 新增：更新半场统计
                        self._update_goal_count_stats(record)  # 新增：更新总进球校准闭环

                        # 最后写入盘口变化库
                        self._update_market_change_db(record)
                    else:
                        log.warning(
                            f"赛果质量不足，仅保存结果不更新训练库: "
                            f"{record.get('home')} vs {record.get('away')} {record.get('result_quality')}"
                        )
                    
                    log.info(f"结算比赛: {record['home']} vs {record['away']} -> {actual_score} ({actual_result})")
                else:
                    # 同步失败
                    self._handle_sync_failure(record, error or '无法获取赛果')

                self._save_record(record)
                return True
        return False
    
    def _handle_sync_failure(self, record: Dict, error: str):
        """处理同步失败"""
        attempts = record.get('sync_attempts', 0) + 1
        record['sync_attempts'] = attempts
        record['last_sync_at'] = datetime.now().isoformat()
        record['last_sync_error'] = error
        
        # 计算下次重试时间
        retry_intervals = {
            1: 2,    # 2小时
            2: 6,    # 6小时
            3: 24,   # 24小时
            4: 48,   # 48小时
        }
        
        if attempts >= 5:
            record['sync_status'] = 'failed'
            record['next_sync_at'] = None
            log.warning(f"同步失败超过5次，标记为失败: {record.get('home')} vs {record.get('away')}")
        else:
            hours = retry_intervals.get(attempts, 24)
            record['sync_status'] = 'retry'
            record['next_sync_at'] = (datetime.now() + timedelta(hours=hours)).isoformat()
            log.debug(f"同步失败，等待 {hours} 小时后重试: {record.get('home')} vs {record.get('away')}")
    
    def get_ready_to_sync(self, minutes: int = 180) -> List[Dict]:
        """
        获取可以同步的记录（比赛结束且未结算）
        
        参数：
            minutes: 比赛开始后等待分钟数（默认180分钟=3小时）
        """
        ready = []
        now = datetime.now()
        
        for record in self.records:
            if record.get('settled', False):
                continue
            
            sync_status = record.get('sync_status', 'pending')
            if sync_status in ('synced', 'failed', 'ignored'):
                continue
            
            # 检查是否在重试等待中
            if sync_status == 'retry':
                next_sync = record.get('next_sync_at')
                if next_sync:
                    try:
                        next_time = datetime.fromisoformat(next_sync)
                        if now < next_time:
                            continue
                    except:
                        pass
            
            # 检查比赛是否已结束
            match_time_str = record.get('match_time')
            if not match_time_str:
                continue
            
            try:
                if _is_match_settle_due(match_time_str, minutes=minutes, now=now):
                    record['sync_status'] = 'ready'
                    ready.append(record)
                elif record.get('sync_status') == 'ready':
                    record['sync_status'] = 'pending'
                    
            except Exception:
                continue
        
        return ready
    
    def get_sync_status_summary(self) -> Dict:
        """获取同步状态汇总"""
        pending = 0
        ready = 0
        synced = 0
        retry = 0
        failed = 0
        ignored = 0
        
        last_sync = None
        last_settled = None
        
        for record in self.records:
            status = record.get('sync_status', 'pending')
            
            if status == 'pending':
                pending += 1
            elif status == 'ready':
                ready += 1
            elif status == 'synced':
                synced += 1
            elif status == 'retry':
                retry += 1
            elif status == 'failed':
                failed += 1
            elif status == 'ignored':
                ignored += 1
            
            last_sync_at = record.get('last_sync_at')
            if last_sync_at:
                try:
                    sync_time = datetime.fromisoformat(last_sync_at)
                    if last_sync is None or sync_time > last_sync:
                        last_sync = sync_time
                except:
                    pass

            if status == 'synced':
                settled_at = record.get('settled_at') or record.get('last_sync_at')
                if settled_at:
                    try:
                        settled_time = datetime.fromisoformat(settled_at)
                        if last_settled is None or settled_time > last_settled:
                            last_settled = settled_time
                    except:
                        pass
        
        return {
            'total': len(self.records),
            'settled': synced,
            'pending_sync': pending + ready,
            'retry': retry,
            'failed': failed,
            'ignored': ignored,
            'last_sync_at': last_sync.isoformat() if last_sync else None,
            'last_settled_at': last_settled.isoformat() if last_settled else None,
        }

    def repair_future_settlements(self, minutes: int = 180,
                                  now: datetime = None) -> Dict:
        """Reset records that were settled before kickoff plus wait window.

        `now` 只为可测：读时钟是副作用，藏在里面就只能靠「写一个当时还没到的
        日期」来测，而那种用例到了那天会自己变红。
        """
        repaired = []
        now = now or datetime.now()
        fields_to_clear = [
            'actual_score', 'actual_result', 'actual_half_score', 'actual_half_result',
            'actual_half_full', 'settled_at', 'evaluation', 'hit_top1', 'hit_top3',
            'hit_top5', 'hit_top10', 'hit_top20', 'hit_top30', 'hit_1x2',
            'actual_score_rank', 'actual_score_prob',
        ]

        for record in self.records:
            if not record.get('settled') and record.get('sync_status') != 'synced':
                continue
            match_time = record.get('match_time')
            if not match_time:
                continue
            if _is_match_settle_due(match_time, minutes=minutes, now=now):
                continue

            for field in fields_to_clear:
                if field in record:
                    record[field] = None
            record['settled'] = False
            record['sync_status'] = 'pending'
            record['last_sync_error'] = '已撤销提前回填，等待比赛结束后重新同步'
            record['updated_at'] = now.isoformat()
            repaired.append({
                'match_id': record.get('match_id'),
                'home': record.get('home'),
                'away': record.get('away'),
                'match_time': match_time,
            })

        if repaired:
            self._save()
        return {'repaired': len(repaired), 'records': repaired}

    def audit_prediction_history(self, repair: bool = False, minutes: int = 180,
                                 now: datetime = None) -> Dict:
        """Audit historical records for unsafe calibration/backtest samples.

        `now` 只为可测，理由同 `repair_future_settlements`。
        """
        issues = []
        repaired = []
        now = now or datetime.now()

        def add_issue(record, code, severity='warning', detail=None):
            item = {
                'match_id': record.get('match_id'),
                'home': record.get('home'),
                'away': record.get('away'),
                'match_time': record.get('match_time'),
                'code': code,
                'severity': severity,
            }
            if detail is not None:
                item['detail'] = detail
            issues.append(item)

        for record in self.records:
            actual_score = record.get('actual_score')
            settled = bool(record.get('settled'))
            sync_status = record.get('sync_status')
            match_time = record.get('match_time')

            is_future_settled = False
            if (settled or sync_status == 'synced') and match_time:
                try:
                    is_future_settled = not _is_match_settle_due(match_time, minutes=minutes, now=now)
                except Exception:
                    is_future_settled = False

            if is_future_settled:
                add_issue(record, 'future_settlement', 'error')
                if repair:
                    for field in (
                        'actual_score', 'actual_result', 'actual_half_score', 'actual_half_result',
                        'actual_half_full', 'settled_at', 'evaluation', 'hit_top1', 'hit_top3',
                        'hit_top5', 'hit_top10', 'hit_top20', 'hit_top30', 'hit_1x2',
                        'actual_score_rank', 'actual_score_prob',
                    ):
                        if field in record:
                            record[field] = None
                    record['settled'] = False
                    record['sync_status'] = 'pending'
                    record['audit_repaired_at'] = now.isoformat()
                    repaired.append({'match_id': record.get('match_id'), 'action': 'reset_future_settlement'})
                continue

            if settled and actual_score:
                try:
                    home_goals, away_goals = map(int, str(actual_score).split('-'))
                    if home_goals < 0 or away_goals < 0 or home_goals > 15 or away_goals > 15:
                        add_issue(record, 'implausible_actual_score', 'error', actual_score)
                except Exception:
                    add_issue(record, 'invalid_actual_score', 'error', actual_score)

            result_quality = record.get('result_quality') or {}
            grade = result_quality.get('grade')
            if settled and not result_quality:
                add_issue(record, 'missing_result_quality', 'warning')
            elif grade in {'reject', 'low'}:
                add_issue(record, f'result_quality_{grade}', 'error' if grade == 'reject' else 'warning')
                if repair:
                    record['exclude_from_calibration'] = True
                    record['audit_repaired_at'] = now.isoformat()
                    repaired.append({'match_id': record.get('match_id'), 'action': 'exclude_from_calibration'})

            if record.get('half_time_data_quality') == 'invalid':
                add_issue(record, 'invalid_half_time_data', 'warning')
                if repair:
                    record['actual_half_score'] = None
                    record['actual_half_result'] = None
                    record['actual_half_full'] = None
                    record['half_time_data_quality'] = 'missing'
                    record['audit_repaired_at'] = now.isoformat()
                    repaired.append({'match_id': record.get('match_id'), 'action': 'clear_invalid_half_time'})

            if settled and _calibration_sample_weight(record) <= 0:
                add_issue(record, 'zero_calibration_weight', 'warning')
                if repair:
                    record['exclude_from_calibration'] = True
                    record['audit_repaired_at'] = now.isoformat()

        issue_counts = {}
        for issue in issues:
            issue_counts[issue['code']] = issue_counts.get(issue['code'], 0) + 1

        if repair and repaired:
            self._save()

        return {
            'checked': len(self.records),
            'issue_count': len(issues),
            'issue_counts': dict(sorted(issue_counts.items())),
            'issues': issues[:50],
            'repaired_count': len(repaired),
            'repaired': repaired[:50],
            'repair': repair,
        }
    
    def _update_calibrator(self, record: Dict):
        """更新贝叶斯校准库"""
        try:
            from .bayesian_calibration import get_calibrator
            
            calibrator = get_calibrator()
            predicted_scores = record.get('predicted_scores', {})
            actual_score = record.get('actual_score', '')
            league = record.get('league', '')
            total_line = record.get('total_line')
            asian = record.get('asian')
            sample_weight = _calibration_sample_weight(record)
            if sample_weight <= 0:
                return
            
            for score, prob in predicted_scores.items():
                is_correct = (score == actual_score)
                # 添加市场环境信息
                calibrator.add_record(score, prob, is_correct, league, total_line or 2.5, asian or 0.0, sample_weight)
            
            calibrator.save()
            log.debug(f"已更新贝叶斯校准库")
        except Exception as e:
            log.debug(f"更新贝叶斯校准库失败: {e}")
    
    def _update_market_db(self, record: Dict):
        """更新盘口聚类库"""
        try:
            from .market_clustering import get_cluster
            
            cluster = get_cluster()
            
            asian = record.get('asian')
            total_line = record.get('total_line')
            actual_score = record.get('actual_score', '')
            
            if asian is not None and total_line is not None and actual_score:
                cluster.add_match(asian, total_line, actual_score)
                cluster.save()
                log.debug(f"已更新盘口聚类库")
        except Exception as e:
            log.debug(f"更新盘口聚类库失败: {e}")
    
    def _update_score_frequency_db(self, record: Dict):
        """更新盘口比分频率库"""
        try:
            from .market_db import MarketScoreDB
            
            db = MarketScoreDB()
            
            asian = record.get('asian')
            total_line = record.get('total_line')
            actual_score = record.get('actual_score', '')
            
            if asian is not None and total_line is not None and actual_score:
                db.add_match_result(asian, total_line, actual_score)
                db.save()
                log.debug(f"已更新盘口比分频率库")
        except Exception as e:
            log.debug(f"更新盘口比分频率库失败: {e}")
    
    def _update_elo_ratings(self, record: Dict):
        """更新ELO评分"""
        try:
            from .elo import get_elo_system
            
            elo = get_elo_system()
            actual_score = record.get('actual_score', '')
            
            if actual_score:
                parts = actual_score.split('-')
                if len(parts) == 2:
                    home_goals, away_goals = map(int, parts)
                    elo.update_ratings(
                        home_team=record['home'],
                        away_team=record['away'],
                        home_score=home_goals,
                        away_score=away_goals,
                        league_type=record.get('league', '联赛')
                    )
                    log.debug(f"已更新ELO评分: {record['home']} vs {record['away']}")
        except Exception as e:
            log.debug(f"更新ELO评分失败: {e}")

    def _update_half_time_stats(self, record: Dict):
        """更新半场比分统计数据库"""
        try:
            from .half_time_stats import record_half_time_result

            if record.get('half_time_data_quality') != 'real' or not _is_result_quality_usable(record):
                return
            
            league = record.get('league', '')
            total_line = record.get('total_line')
            handicap = record.get('asian')
            actual_score = record.get('actual_score', '')
            actual_half_score = record.get('actual_half_score', '')
            
            # 只有当有真实半场比分时才记录
            if actual_half_score and actual_score and total_line is not None:
                try:
                    half_h, half_a = map(int, actual_half_score.split('-'))
                    full_h, full_a = map(int, actual_score.split('-'))
                    
                    # 判断比赛类型
                    match_type = 'league'
                    if league:
                        league_lower = league.lower()
                        if '杯' in league or 'cup' in league_lower or 'tournament' in league_lower:
                            match_type = 'cup'
                        elif '友谊' in league or 'friendly' in league_lower:
                            match_type = 'friendly'
                    
                    record_half_time_result(
                        league=league,
                        total_line=total_line,
                        handicap=handicap or 0.0,
                        match_type=match_type,
                        half_home=half_h,
                        half_away=half_a,
                        full_home=full_h,
                        full_away=full_a,
                        sample_weight=_calibration_sample_weight(record)
                    )
                    log.debug(f"已更新半场统计数据库")
                except Exception as e:
                    log.debug(f"解析半场比分失败: {e}")
        except Exception as e:
            log.debug(f"更新半场统计数据库失败: {e}")

    def _update_goal_count_stats(self, record: Dict):
        """更新总进球数校准数据库（赛后回填闭环）。

        此前 GoalCountCalibrator 的写入函数在生产代码从未被调用，导致校准表恒空、
        校准恒等返回。这里由预测比分分布边缘化出「预测进球数分布」与「期望总进球」，
        与真实总进球一起回填，逐步积累后校准器才能真正生效。
        """
        try:
            from .goal_count_calibrator import record_goal_count_result

            predicted_scores = record.get('predicted_scores') or {}
            actual_score = record.get('actual_score', '')
            total_line = record.get('total_line')
            if not predicted_scores or not actual_score or total_line is None:
                return

            goal_dist = {}
            expected_total = 0.0
            prob_sum = 0.0
            for score, prob in predicted_scores.items():
                try:
                    h, a = map(int, str(score).split('-'))
                    p = float(prob)
                except (ValueError, TypeError):
                    continue
                if p <= 0:
                    continue
                g = h + a
                goal_dist[g] = goal_dist.get(g, 0.0) + p
                expected_total += g * p
                prob_sum += p
            if prob_sum <= 0:
                return
            goal_dist = {g: p / prob_sum for g, p in goal_dist.items()}
            expected_total /= prob_sum

            try:
                full_h, full_a = map(int, actual_score.split('-'))
            except (ValueError, TypeError):
                return
            actual_total = full_h + full_a

            record_goal_count_result(
                league=record.get('league', ''),
                total_line=total_line,
                predicted_goal_dist=goal_dist,
                actual_total_goals=actual_total,
                expected_total_goals=expected_total,
                asian=record.get('asian') or 0.0,
                sample_weight=_calibration_sample_weight(record),
            )
            log.debug("已更新总进球校准数据库")
        except Exception as e:
            log.debug(f"更新总进球校准数据库失败: {e}")

    def _first_not_none(self, *values):
        for v in values:
            if v is not None:
                return v
        return None

    def _update_market_change_db(self, record: Dict):
        """赛后回填成功后，写入盘口变化数据库"""
        try:
            # 防止同一场重复结算导致重复写入
            if record.get('market_change_updated'):
                return

            from .market_db import MarketChangeDB, normalize_asian, normalize_ou

            odds = record.get('odds_snapshot') or {}
            asian_data = odds.get('asian') or {}
            total_data = odds.get('total') or {}

            # 兼容 analyze_asian 后结构
            asian_from = self._first_not_none(
                asian_data.get('open_handicap'),
                asian_data.get('open', {}).get('handicap')
            )

            asian_to = self._first_not_none(
                asian_data.get('handicap'),
                asian_data.get('close', {}).get('handicap'),
                record.get('asian')
            )

            # 兼容 analyze_total 后结构
            ou_from = self._first_not_none(
                total_data.get('open_line'),
                total_data.get('open', {}).get('line')
            )

            ou_to = self._first_not_none(
                total_data.get('close_line'),
                total_data.get('line'),
                total_data.get('close', {}).get('line'),
                record.get('total_line')
            )

            actual_score = record.get('actual_score')

            if asian_from is None or asian_to is None:
                log.debug(f"盘口变化库跳过：缺少亚盘开终盘 match_id={record.get('match_id')}")
                return

            if ou_from is None or ou_to is None:
                log.debug(f"盘口变化库跳过：缺少大小球开终盘 match_id={record.get('match_id')}")
                return

            if not actual_score:
                return

            asian_from_n = normalize_asian(asian_from)
            asian_to_n = normalize_asian(asian_to)
            ou_from_n = normalize_ou(ou_from)
            ou_to_n = normalize_ou(ou_to)

            if asian_from_n is None or asian_to_n is None or ou_from_n is None or ou_to_n is None:
                return

            db = MarketChangeDB()
            db.add_record(
                asian_from_n,
                asian_to_n,
                ou_from_n,
                ou_to_n,
                actual_score
            )
            db.save()

            record['market_change_updated'] = True
            record['market_change_updated_at'] = datetime.now().isoformat()
            record['market_change_key'] = f"{asian_from_n:.2f}→{asian_to_n:.2f}_{ou_from_n:.2f}→{ou_to_n:.2f}"

            log.info(
                f"盘口变化库已更新: "
                f"{record.get('home')} vs {record.get('away')} | "
                f"亚盘 {asian_from_n}->{asian_to_n}, "
                f"大小球 {ou_from_n}->{ou_to_n}, "
                f"比分 {actual_score}"
            )

        except Exception as e:
            log.debug(f"更新盘口变化数据库失败: {e}")
    
    def get_stats(self) -> Dict:
        """获取统计信息（包含时间分层统计）"""
        total = len(self.records)
        settled = len([r for r in self.records if r.get('settled', False)])
        unsettled = total - settled
        
        # 时间分层统计
        time_layers = ['T-24h', 'T-6h', 'T-1h', 'T-15min', 'final']
        layer_stats = {
            layer: {
                'correct_top1': 0,
                'correct_top3': 0,
                'correct_top5': 0,
                'total': 0,
                'weighted_correct_top1': 0.0,
                'weighted_correct_top3': 0.0,
                'weighted_correct_top5': 0.0,
                'weighted_total': 0.0,
            }
            for layer in time_layers
        }
        
        # 计算命中率
        correct_top1 = 0
        correct_top3 = 0
        correct_top5 = 0
        correct_1x2 = 0
        valid_score_predictions = 0
        valid_1x2_predictions = 0
        actionable_total = 0
        actionable_correct = 0
        version_1x2 = {}
        
        for record in self.records:
            if not record.get('settled', False):
                continue
            
            actual_score = record.get('actual_score', '')
            predicted_scores = record.get('predicted_scores', {})
            actual_result = record.get('actual_result', '')
            predicted_1x2 = normalize_1x2_probs(record.get('predicted_1x2', {}))
            time_layers_data = record.get('time_layers', {})
            
            if not predicted_scores or not actual_score:
                continue
            valid_score_predictions += 1
            
            # 统计各时间层命中率
            for layer in time_layers:
                layer_pred = time_layers_data.get(layer)
                if layer == 'final' and not layer_pred:
                    layer_pred = predicted_scores
                if not layer_pred:
                    continue
                layer_weight = time_layer_weight(layer)
                
                sorted_scores = sorted(layer_pred.items(), key=lambda x: -x[1])
                layer_stats[layer]['total'] += 1
                layer_stats[layer]['weighted_total'] += layer_weight
                
                if sorted_scores and sorted_scores[0][0] == actual_score:
                    layer_stats[layer]['correct_top1'] += 1
                    layer_stats[layer]['weighted_correct_top1'] += layer_weight
                
                top3_scores = [s[0] for s in sorted_scores[:3]]
                if actual_score in top3_scores:
                    layer_stats[layer]['correct_top3'] += 1
                    layer_stats[layer]['weighted_correct_top3'] += layer_weight
                
                top5_scores = [s[0] for s in sorted_scores[:5]]
                if actual_score in top5_scores:
                    layer_stats[layer]['correct_top5'] += 1
                    layer_stats[layer]['weighted_correct_top5'] += layer_weight
            
            # 最终预测统计
            sorted_scores = sorted(predicted_scores.items(), key=lambda x: -x[1])
            
            if sorted_scores and sorted_scores[0][0] == actual_score:
                correct_top1 += 1
            
            top3_scores = [s[0] for s in sorted_scores[:3]]
            if actual_score in top3_scores:
                correct_top3 += 1
            
            top5_scores = [s[0] for s in sorted_scores[:5]]
            if actual_score in top5_scores:
                correct_top5 += 1
            
            # 胜平负
            if actual_result and actual_result in predicted_1x2:
                valid_1x2_predictions += 1
                pred_result = max(predicted_1x2.items(), key=lambda x: x[1])[0]
                if pred_result == actual_result:
                    correct_1x2 += 1
                version = record.get('model_version') or 'legacy-unversioned'
                version_stats = version_1x2.setdefault(version, {'total': 0, 'correct': 0})
                version_stats['total'] += 1
                if pred_result == actual_result:
                    version_stats['correct'] += 1
                decision = record.get('decision_snapshot') or _prediction_decision_snapshot(predicted_1x2)
                if decision.get('eligible'):
                    actionable_total += 1
                    if pred_result == actual_result:
                        actionable_correct += 1

        hit_rate_top1 = correct_top1 / valid_score_predictions if valid_score_predictions > 0 else 0
        hit_rate_top3 = correct_top3 / valid_score_predictions if valid_score_predictions > 0 else 0
        hit_rate_top5 = correct_top5 / valid_score_predictions if valid_score_predictions > 0 else 0
        hit_rate_1x2 = correct_1x2 / valid_1x2_predictions if valid_1x2_predictions > 0 else 0

        # 计算各时间层命中率
        layer_hit_rates = {}
        for layer in time_layers:
            total_layer = layer_stats[layer]['total']
            if total_layer > 0:
                layer_hit_rates[layer] = {
                    'hit_rate_top1': layer_stats[layer]['correct_top1'] / total_layer,
                    'hit_rate_top3': layer_stats[layer]['correct_top3'] / total_layer,
                    'hit_rate_top5': layer_stats[layer]['correct_top5'] / total_layer,
                    'correct_top1': layer_stats[layer]['correct_top1'],
                    'correct_top3': layer_stats[layer]['correct_top3'],
                    'correct_top5': layer_stats[layer]['correct_top5'],
                    'total': total_layer,
                    'weight': time_layer_weight(layer),
                    'weighted_hit_rate_top1': (
                        layer_stats[layer]['weighted_correct_top1'] / layer_stats[layer]['weighted_total']
                        if layer_stats[layer]['weighted_total'] > 0 else 0.0
                    ),
                    'weighted_hit_rate_top3': (
                        layer_stats[layer]['weighted_correct_top3'] / layer_stats[layer]['weighted_total']
                        if layer_stats[layer]['weighted_total'] > 0 else 0.0
                    ),
                    'weighted_hit_rate_top5': (
                        layer_stats[layer]['weighted_correct_top5'] / layer_stats[layer]['weighted_total']
                        if layer_stats[layer]['weighted_total'] > 0 else 0.0
                    ),
                    'weighted_total': round(layer_stats[layer]['weighted_total'], 3),
                }
            else:
                layer_hit_rates[layer] = {
                    'hit_rate_top1': 0.0,
                    'hit_rate_top3': 0.0,
                    'hit_rate_top5': 0.0,
                    'correct_top1': 0,
                    'correct_top3': 0,
                    'correct_top5': 0,
                    'total': 0,
                    'weight': time_layer_weight(layer),
                    'weighted_hit_rate_top1': 0.0,
                    'weighted_hit_rate_top3': 0.0,
                    'weighted_hit_rate_top5': 0.0,
                    'weighted_total': 0.0,
                }
        
        return {
            'total_predictions': total,
            'settled': settled,
            'unsettled': unsettled,
            'hit_rate_top1': hit_rate_top1,
            'hit_rate_top3': hit_rate_top3,
            'hit_rate_top5': hit_rate_top5,
            'hit_rate_1x2': hit_rate_1x2,
            'by_time_layer': layer_hit_rates,
            'correct_top1': correct_top1,
            'correct_top3': correct_top3,
            'correct_top5': correct_top5,
            'correct_1x2': correct_1x2,
            'valid_score_predictions': valid_score_predictions,
            'valid_1x2_predictions': valid_1x2_predictions,
            'actionable_1x2': {
                'policy_version': ACTIONABLE_POLICY_VERSION,
                'total': actionable_total,
                'correct': actionable_correct,
                'hit_rate': actionable_correct / actionable_total if actionable_total else 0.0,
                'coverage': actionable_total / valid_1x2_predictions if valid_1x2_predictions else 0.0,
                'min_probability': ACTIONABLE_MIN_PROBABILITY,
                'min_margin': ACTIONABLE_MIN_MARGIN,
            },
            'by_model_version': {
                version: {
                    **values,
                    'hit_rate_1x2': values['correct'] / values['total'] if values['total'] else 0.0,
                }
                for version, values in version_1x2.items()
            },
        }
    
    def get_ml_evaluation_stats(self, min_samples: int = 45) -> Dict:
        """
        获取 ML 模型评估统计（按维度）
        
        参数：
            min_samples: 最小样本数阈值
        
        返回：
            按维度统计的评估结果
        """
        # 五大联赛列表
        top_leagues = ['英超', '西甲', '德甲', '意甲', '法甲']
        
        # 初始化统计结构
        stats = {
            'overall': {
                'sample_count': 0,
                'base_1x2_logloss': [],
                'base_1x2_brier': [],
                'base_1x2_hit': [],
                'ml_1x2_logloss': [],
                'ml_1x2_brier': [],
                'ml_1x2_hit': [],
                'fused_5pct_logloss': [],
                'fused_5pct_brier': [],
                'fused_10pct_logloss': [],
                'fused_10pct_brier': [],
            },
            'by_league': {},
            'by_handicap_type': {
                'strong_favorite': {},  # 让球 >= 1.0
                'balanced': {},         # -0.5 < 让球 < 0.5
                'weak_favorite': {},    # 让球 <= -1.0
            },
            'by_total_line': {
                'low': {},              # <= 2.25
                'medium': {},           # 2.25 < x < 3.0
                'high': {},             # >= 3.0
            },
            'by_result': {
                'H': {},
                'D': {},
                'A': {},
            },
        }
        
        # 初始化联赛统计
        for league in top_leagues:
            stats['by_league'][league] = {
                'sample_count': 0,
                'base_1x2_logloss': [],
                'base_1x2_brier': [],
                'base_1x2_hit': [],
                'ml_1x2_logloss': [],
                'ml_1x2_brier': [],
                'ml_1x2_hit': [],
                'fused_5pct_logloss': [],
                'fused_5pct_brier': [],
                'fused_10pct_logloss': [],
                'fused_10pct_brier': [],
            }
        
        # 初始化其他维度统计
        for dim in ['by_handicap_type', 'by_total_line', 'by_result']:
            for key in stats[dim]:
                stats[dim][key] = {
                    'sample_count': 0,
                    'base_1x2_logloss': [],
                    'base_1x2_brier': [],
                    'base_1x2_hit': [],
                    'ml_1x2_logloss': [],
                    'ml_1x2_brier': [],
                    'ml_1x2_hit': [],
                    'fused_5pct_logloss': [],
                    'fused_5pct_brier': [],
                    'fused_10pct_logloss': [],
                    'fused_10pct_brier': [],
                }
        
        # 遍历记录收集数据
        for record in self.records:
            if not record.get('settled', False):
                continue
            
            actual_result = record.get('actual_result')
            if actual_result not in ['H', 'D', 'A']:
                continue
            
            evaluation = record.get('evaluation', {})
            league = record.get('league', '')
            handicap = record.get('asian', 0.0)
            total_line = record.get('total_line', 2.5)
            
            # 确定维度分类
            if league in top_leagues:
                league_key = league
            else:
                league_key = None
            
            # 让球盘类型
            if handicap >= 1.0:
                handicap_key = 'strong_favorite'
            elif handicap <= -1.0:
                handicap_key = 'weak_favorite'
            else:
                handicap_key = 'balanced'
            
            # 大小球类型
            if total_line <= 2.25:
                total_key = 'low'
            elif total_line >= 3.0:
                total_key = 'high'
            else:
                total_key = 'medium'
            
            # 结果类型
            result_key = actual_result
            
            # 收集评估数据到各维度
            dimensions = [('overall', None), 
                          ('by_handicap_type', handicap_key),
                          ('by_total_line', total_key),
                          ('by_result', result_key)]
            
            # 只有当联赛在五大联赛列表中时才添加联赛维度
            if league_key is not None:
                dimensions.insert(1, ('by_league', league_key))
            
            for dim_key, key in dimensions:
                if key is None:
                    target = stats[dim_key]
                elif key in stats[dim_key]:
                    target = stats[dim_key][key]
                else:
                    continue
                
                target['sample_count'] += 1
                
                # 添加评估指标
                for metric in ['base_1x2_logloss', 'base_1x2_brier', 'base_1x2_hit',
                               'ml_1x2_logloss', 'ml_1x2_brier', 'ml_1x2_hit',
                               'fused_5pct_logloss', 'fused_5pct_brier',
                               'fused_10pct_logloss', 'fused_10pct_brier']:
                    value = evaluation.get(metric)
                    if value is not None and not math.isnan(value):
                        target[metric].append(value)
        
        # 计算汇总统计
        def summarize_dimension(dim_stats):
            result = {}
            for key, data in dim_stats.items():
                if data['sample_count'] == 0:
                    result[key] = {
                        'sample_count': 0,
                        'base_1x2_logloss': None,
                        'base_1x2_brier': None,
                        'base_1x2_hit_rate': None,
                        'ml_1x2_logloss': None,
                        'ml_1x2_brier': None,
                        'ml_1x2_hit_rate': None,
                        'fused_5pct_logloss': None,
                        'fused_5pct_brier': None,
                        'fused_10pct_logloss': None,
                        'fused_10pct_brier': None,
                        'qualified': False,
                    }
                    continue
                
                # 计算均值
                result[key] = {
                    'sample_count': data['sample_count'],
                    'base_1x2_logloss': sum(data['base_1x2_logloss']) / len(data['base_1x2_logloss']) if data['base_1x2_logloss'] else None,
                    'base_1x2_brier': sum(data['base_1x2_brier']) / len(data['base_1x2_brier']) if data['base_1x2_brier'] else None,
                    'base_1x2_hit_rate': sum(data['base_1x2_hit']) / len(data['base_1x2_hit']) if data['base_1x2_hit'] else None,
                    'ml_1x2_logloss': sum(data['ml_1x2_logloss']) / len(data['ml_1x2_logloss']) if data['ml_1x2_logloss'] else None,
                    'ml_1x2_brier': sum(data['ml_1x2_brier']) / len(data['ml_1x2_brier']) if data['ml_1x2_brier'] else None,
                    'ml_1x2_hit_rate': sum(data['ml_1x2_hit']) / len(data['ml_1x2_hit']) if data['ml_1x2_hit'] else None,
                    'fused_5pct_logloss': sum(data['fused_5pct_logloss']) / len(data['fused_5pct_logloss']) if data['fused_5pct_logloss'] else None,
                    'fused_5pct_brier': sum(data['fused_5pct_brier']) / len(data['fused_5pct_brier']) if data['fused_5pct_brier'] else None,
                    'fused_10pct_logloss': sum(data['fused_10pct_logloss']) / len(data['fused_10pct_logloss']) if data['fused_10pct_logloss'] else None,
                    'fused_10pct_brier': sum(data['fused_10pct_brier']) / len(data['fused_10pct_brier']) if data['fused_10pct_brier'] else None,
                    'qualified': data['sample_count'] >= min_samples,
                }
            return result
        
        return {
            'overall': summarize_dimension({'overall': stats['overall']})['overall'],
            'by_league': summarize_dimension(stats['by_league']),
            'by_handicap_type': summarize_dimension(stats['by_handicap_type']),
            'by_total_line': summarize_dimension(stats['by_total_line']),
            'by_result': summarize_dimension(stats['by_result']),
            'min_samples_required': min_samples,
        }


# 全局实例
_global_history = PredictionHistory()


# ==================== ML 融合门槛判断 ====================





# ==================== 便捷函数 ====================

def save_prediction(match_id: str, league: str, home: str, away: str,
                   match_time: str, predicted_scores: Dict[str, float],
                   predicted_1x2: Dict[str, float], asian: float = None,
                   total_line: float = None, odds_data: Dict = None,
                   predicted_half_full: Dict[str, float] = None,
                   # 影子预测相关字段
                   base_1x2: Dict[str, float] = None,
                   ml_1x2: Dict[str, float] = None,
                   ml_model_version: str = None,
                   ml_available: bool = False,
                   ml_feature_snapshot: Dict = None,
                   lottery_handicap: int = None,
                   predicted_rqspf: Dict[str, float] = None,
                   goal_count: Dict = None,
                   professional_snapshot: Dict = None,
                   model_version: str = PRODUCTION_MODEL_VERSION,
                   match_num: str = None):
    """保存预测记录"""
    return _global_history.add_prediction(
        match_id, league, home, away, match_time,
        predicted_scores, predicted_1x2, asian, total_line, odds_data,
        predicted_half_full=predicted_half_full,
        base_1x2=base_1x2,
        ml_1x2=ml_1x2,
        ml_model_version=ml_model_version,
        ml_available=ml_available,
        ml_feature_snapshot=ml_feature_snapshot,
        lottery_handicap=lottery_handicap,
        predicted_rqspf=predicted_rqspf,
        goal_count=goal_count,
        professional_snapshot=professional_snapshot,
        model_version=model_version,
        match_num=match_num,
    )


def sync_results():
    """同步比赛结果"""
    return auto_sync_results()


def auto_sync_results():
    """
    自动同步比赛结果（逐层兜底）
    1. 竞彩官网记录（sporttery_*）：按 matchId 查官网开奖接口
    2. 500 数字 fid：match_id 对应赛果页面
    3. 主队 + 客队 + 比赛日期模糊匹配
    4. 都没有：记一次失败，按退避重试，超过 5 次标记 failed
    """
    ready = _global_history.get_ready_to_sync()
    
    if not ready:
        return {'synced': 0, 'failed': 0, 'message': '没有需要同步的比赛'}
    
    synced = 0
    failed = 0
    
    for record in ready:
        match_id = record['match_id']
        home = record['home']
        away = record['away']
        match_time = record.get('match_time', '')
        league = record.get('league', '')

        try:
            # 竞彩官网来的记录（sporttery_*）先按 matchId 查官网开奖接口：
            # 竞彩简称与 500 的队名经常对不上（迈季宽广/迈季迈阿宽广），
            # 队名兜底对这些场次永远失败。非 500 数字 fid 在
            # fetch_result_by_match_id 内部直接返回 None，落到队名+日期兜底，
            # 不能在这里整条跳过——那会让这些记录永远停在「准备同步」。
            result = fetch_result_by_sporttery_id(match_id, match_time)
            if not result:
                result = fetch_result_by_match_id(match_id, match_time)
            if not result:
                result = fetch_result_by_team_and_date(home, away, match_time)

            if result:
                if _global_history.update_result(
                    match_id,
                    result['score'],
                    result['result'],
                    actual_half_score=result.get('half_score'),
                    source=result.get('source'),
                ):
                    synced += 1
                    log.info(f"同步成功: {home} vs {away} -> {result['score']}")
                else:
                    failed += 1
            else:
                _global_history.update_result(match_id, None, None, error='未找到赛果')
                failed += 1
                log.warning(f"无法获取比赛结果: {home} vs {away}")
                
        except Exception as e:
            _global_history.update_result(match_id, None, None, error=str(e))
            failed += 1
            log.error(f"同步比赛结果异常: {home} vs {away} - {e}")
    
    return {
        'synced': synced,
        'failed': failed,
        'total': len(ready),
        'message': f'结算了 {synced}/{len(ready)} 场比赛，失败 {failed} 场'
    }






def _fetch_match_html(match_id: str) -> str:
    """复用足球模块抓取逻辑，保持与预测数据同源"""
    from . import fetch as fetch_html
    return fetch_html(f'https://odds.500.com/fenxi/shuju-{match_id}.shtml')




def fetch_result_by_match_id(match_id: str, match_time: str = '') -> Optional[Dict]:
    """
    通过 match_id 抓取赛果：
    1. live.500.com 按 fid + 动态日期（竞彩官方赛果页）
    2. odds.500.com 赛事数据页（兜底）
    """
    if not _is_valid_match_id(match_id):
        return None

    if match_time:
        score = _fetch_live_score_by_fid(match_id, match_time)
        if score:
            result = _parse_score_string(score)
            if result:
                result['source'] = 'live_fid'
            return result

    try:
        html = _fetch_match_html(match_id)
        score = _parse_shuju_score(html, match_id)
        if score:
            log.info(f"通过 shuju 页面抓取赛果: match_id={match_id} -> {score}")
            result = _parse_score_string(score)
            if result:
                result['source'] = 'shuju'
            return result
    except Exception as e:
        log.debug(f"shuju 页面抓取失败: {e}")

    return None








def _fetch_live_html(search_date: str) -> str:
    from . import fetch as fetch_html
    return fetch_html(f'https://live.500.com/?e={search_date}')


def _fetch_finished_html(search_date: str) -> str:
    """抓取指定自然日的全部完场比赛，而不是仅限竞彩场次。"""
    from . import fetch as fetch_html
    return fetch_html(f'https://live.500.com/wanchang.php?e={search_date}')


def _finished_query_dates(match_time: str) -> List[str]:
    """完场页按实际开球自然日归档；兼容跨午夜记录偏差一天。"""
    match_dt = _parse_match_datetime(match_time)
    if not match_dt:
        return []
    return [
        (match_dt.date() + timedelta(days=offset)).strftime('%Y-%m-%d')
        for offset in (0, -1, 1)
    ]




def _fetch_live_score_by_fid(match_id: str, match_time: str) -> Optional[str]:
    """在 live.500.com 按 fid 查找赛果，日期动态推算"""
    for search_date in _finished_query_dates(match_time):
        try:
            html = _fetch_finished_html(search_date)
        except Exception as e:
            log.debug(f"完场页面抓取失败 e={search_date}: {e}")
            continue

        row_m = re.search(
            rf'<tr[^>]*(?:\bid|\bfid)=["\'](?:a)?{re.escape(match_id)}["\'][^>]*>.*?</tr>',
            html,
            re.DOTALL,
        )
        if row_m:
            score = _parse_live_row_final_score(row_m.group(0))
            if score:
                log.info(
                    f"通过完场页面(fid)抓取赛果: match_id={match_id}, "
                    f"e={search_date} -> {score}")
                return score

    for search_date in _live_query_dates(match_time):
        try:
            html = _fetch_live_html(search_date)
        except Exception as e:
            log.debug(f"live 页面抓取失败 e={search_date}: {e}")
            continue

        row_m = re.search(
            rf'<tr[^>]*\bfid="{re.escape(match_id)}"[^>]*>.*?</tr>',
            html,
            re.DOTALL,
        )
        if not row_m:
            continue

        score = _parse_live_row_final_score(row_m.group(0))
        if score:
            log.info(f"通过 live 页面(fid)抓取赛果: match_id={match_id}, e={search_date} -> {score}")
            return score

    return None




SPORTTERY_ID_PREFIX = 'sporttery_'


def _sporttery_fetch_json(url: str, referer: str = None) -> Dict:
    from .fetching import fetch_json
    return fetch_json(url, referer=referer)


def _fetch_sporttery_results(begin_date: str, end_date: str) -> Dict[str, Dict]:
    """拉取日期窗口内竞彩官网的全部完场赛果，翻完所有分页后按 matchId 合并。

    底层 fetch 带 TTL 缓存，同一轮同步里同一天的几十场只会真正请求一次。
    """
    from .sporttery import (
        SPORTTERY_RESULT_REFERER, parse_sporttery_results, sporttery_result_url,
    )

    merged: Dict[str, Dict] = {}
    page_no = 1
    while True:
        payload = _sporttery_fetch_json(
            sporttery_result_url(begin_date, end_date, page_no=page_no),
            referer=SPORTTERY_RESULT_REFERER,
        )
        merged.update(parse_sporttery_results(payload))
        pages = int((payload.get('value') or {}).get('pages') or 1)
        if page_no >= pages:
            return merged
        page_no += 1


def fetch_result_by_sporttery_id(match_id: str, match_time: str) -> Optional[Dict]:
    """竞彩官网来的记录按 matchId 查官网开奖接口，主客队名完全不参与。

    日期窗口取比赛日前后各一天：接口按 matchDate 归档，跨午夜场次两边
    的记法可能差一天。
    """
    match_id = str(match_id or '')
    if not match_id.startswith(SPORTTERY_ID_PREFIX):
        return None
    sporttery_id = match_id[len(SPORTTERY_ID_PREFIX):]
    match_dt = _parse_match_datetime(match_time)
    if not sporttery_id or not match_dt:
        return None

    window = [
        (match_dt.date() + timedelta(days=offset)).strftime('%Y-%m-%d')
        for offset in (-1, 1)
    ]
    hit = _fetch_sporttery_results(window[0], window[1]).get(sporttery_id)
    if not hit:
        return None
    result = _parse_score_string(hit['score'])
    if not result:
        return None
    result['half_score'] = hit.get('half_score')
    result['source'] = 'sporttery'
    log.info(
        f"通过竞彩官网开奖接口抓取赛果: {match_id} "
        f"{hit.get('match_num')} -> {hit['score']}"
    )
    return result


def fetch_result_by_team_and_date(home: str, away: str, match_time: str) -> Optional[Dict]:
    """
    第二优先：通过球队名和比赛时间在 live.500.com 模糊匹配抓取赛果
    """
    try:
        # `/?e=` 是竞彩销售日页面，只包含少量竞彩场次；北单历史覆盖的赛事
        # 远多于竞彩。先查按自然日归档的完场页，才不会把真实完赛误判成无赛果。
        for search_date in _finished_query_dates(match_time):
            try:
                html = _fetch_finished_html(search_date)
            except Exception as e:
                log.debug(f"完场页面抓取失败 e={search_date}: {e}")
                continue

            for row in re.finditer(r'<tr[^>]*>.*?</tr>', html, re.DOTALL):
                score = _parse_live_row_score(row.group(0), home, away)
                if score:
                    log.info(
                        f"通过完场页面(球队)抓取赛果: {home} vs {away}, "
                        f"e={search_date} -> {score}")
                    result = _parse_score_string(score)
                    if result:
                        result['source'] = 'live_team'
                    return result

        for search_date in _live_query_dates(match_time):
            try:
                html = _fetch_live_html(search_date)
            except Exception as e:
                log.debug(f"live 页面抓取失败 e={search_date}: {e}")
                continue

            for row in re.finditer(r'<tr[^>]*>.*?</tr>', html, re.DOTALL):
                score = _parse_live_row_score(row.group(0), home, away)
                if score:
                    log.info(f"通过 live 页面(球队)抓取赛果: {home} vs {away}, e={search_date} -> {score}")
                    result = _parse_score_string(score)
                    if result:
                        result['source'] = 'live_team'
                    return result

    except Exception as e:
        log.debug(f"通过球队名+日期抓取失败: {e}")

    return None






def get_history_stats() -> Dict:
    """获取历史统计"""
    return _global_history.get_stats()


def get_sync_status_summary() -> Dict:
    """获取同步状态汇总"""
    return _global_history.get_sync_status_summary()


def repair_future_settlements(minutes: int = 180, now: datetime = None) -> Dict:
    """撤销尚未到结算时间却已经回填的记录。"""
    return _global_history.repair_future_settlements(minutes=minutes, now=now)


def audit_prediction_history(repair: bool = False, minutes: int = 180,
                             now: datetime = None) -> Dict:
    return _global_history.audit_prediction_history(repair=repair, minutes=minutes, now=now)


def get_prediction_records(include_hidden: bool = False,
                           now: datetime = None) -> List[Dict]:
    """
    获取预测记录列表

    参数：
        include_hidden: 是否包含已失败的记录
        now: 当前时间，**只为可测**（理由同 `repair_future_settlements`）
    """
    records = []
    for record in _global_history.records:
        if not include_hidden:
            if record.get('sync_status') == 'failed':
                continue
        
        is_future_settled = (
            (record.get('settled') or record.get('sync_status') == 'synced')
            and record.get('match_time')
            and not _is_match_settle_due(record.get('match_time'), minutes=180, now=now)
        )
        lottery_snapshot = (record.get('odds_snapshot') or {}).get('lottery') or {}
        lottery_snapshot_present = bool(lottery_snapshot)
        lottery_offer_matched = bool(lottery_snapshot.get('offer_matched'))
        predicted_1x2, predicted_rqspf, lottery_handicap = (
            _complete_offered_lottery_predictions(
                record.get('predicted_scores'),
                record.get('predicted_1x2'),
                record.get('predicted_rqspf'),
                record.get('lottery_handicap'),
                lottery_snapshot,
            )
        )
        # 旧数据完全没有体彩快照时维持兼容；已经明确记录为抓取失败/未核验的
        # 场次，不能把内部模型 SPF 冒充成已开售的官方胜平负。
        spf_was_offered = (
            not lottery_snapshot_present
            or bool(
                lottery_offer_matched
                and lottery_snapshot.get('spf_available')
                and lottery_snapshot.get('spf_odds')
            )
        )
        rqspf_was_offered = (
            not lottery_snapshot_present
            or bool(
                lottery_offer_matched
                and lottery_snapshot.get('rqspf_available')
                and lottery_snapshot.get('rqspf_odds')
                and lottery_handicap not in (None, 0)
            )
        )

        records.append({
            'match_id': record.get('match_id'),
            'league': record.get('league'),
            'home': record.get('home'),
            'away': record.get('away'),
            'match_time': record.get('match_time'),
            'match_num': record.get('match_num'),
            'created_at': record.get('created_at'),
            'lottery_offer_matched': (
                lottery_offer_matched if lottery_snapshot_present else None
            ),
            'lottery_unavailable_reason': lottery_snapshot.get('unavailable_reason'),
            'lottery_spf_available': spf_was_offered if lottery_snapshot_present else None,
            'lottery_rqspf_available': (
                rqspf_was_offered if lottery_snapshot_present else None
            ),
            'settled': False if is_future_settled else record.get('settled', False),
            'actual_score': None if is_future_settled else record.get('actual_score'),
            'sync_status': 'pending' if is_future_settled else record.get('sync_status', 'pending'),
            'sync_attempts': record.get('sync_attempts', 0),
            'last_sync_error': (
                '比赛尚未到结算时间，已隐藏提前回填结果'
                if is_future_settled else record.get('last_sync_error')
            ),
            'next_sync_at': record.get('next_sync_at'),
            'hit_top1': None if is_future_settled else record.get('hit_top1'),
            'hit_top3': None if is_future_settled else record.get('hit_top3'),
            # 预测记录页以两个竞彩赛果市场为主。精确比分仍保留在存储和
            # 完整导出中，列表只在赛后输出 actual_score。
            'predicted_1x2': predicted_1x2 if spf_was_offered else {},
            'predicted_rqspf': predicted_rqspf if rqspf_was_offered else {},
            'lottery_handicap': lottery_handicap,
            'actual_result': None if is_future_settled else record.get('actual_result'),
            'actual_rqspf': None if is_future_settled else record.get('actual_rqspf'),
            'hit_1x2': (
                None if is_future_settled or not spf_was_offered else record.get('hit_1x2')
            ),
            'hit_rqspf': None if is_future_settled else record.get('hit_rqspf'),
        })
    
    # 按比赛时间倒序排列
    records.sort(key=lambda x: x.get('match_time', ''), reverse=True)
    return records


def get_prediction_export() -> Dict:
    """返回可用于离线回测/校准的完整预测记录（不包含数据库配置）。"""
    export_fields = (
        'match_id', 'league', 'home', 'away', 'match_time',
        'match_num',
        'created_at', 'updated_at', 'settled_at', 'model_version',
        'prediction_logic_version', 'asian', 'total_line',
        'predicted_scores', 'predicted_1x2', 'predicted_rqspf', 'goal_count',
        'lottery_handicap', 'predicted_half_full',
        'time_layers', 'odds_layers', 'odds_snapshot', 'market_timeline',
        'last_prematch_odds_snapshot', 'last_prematch_snapshot_at',
        'closing_odds_snapshot', 'closing_odds_source',
        'professional_snapshot',
        'base_1x2', 'ml_1x2', 'ml_model_version', 'ml_available',
        'ml_feature_snapshot', 'actual_score', 'actual_result',
        'actual_half_score', 'actual_half_result', 'actual_half_full',
        'settled', 'sync_status', 'evaluation', 'hit_top1', 'hit_top3',
        'hit_top5', 'hit_1x2', 'hit_rqspf', 'actual_rqspf',
        'actual_score_rank', 'actual_score_prob',
    )
    records = [
        {key: record.get(key) for key in export_fields if key in record}
        for record in _global_history.records
    ]
    records.sort(key=lambda item: item.get('match_time', ''))
    return {
        'schema_version': 'football-prediction-export-v2',
        'exported_at': datetime.now().astimezone().isoformat(),
        'record_count': len(records),
        'settled_count': sum(
            1 for record in records
            if record.get('settled') or record.get('actual_score')
        ),
        'stats': _global_history.get_stats(),
        'records': records,
    }


def hide_failed_records():
    """隐藏所有失败记录（标记为 ignored）"""
    for record in _global_history.records:
        if record.get('sync_status') == 'failed':
            record['sync_status'] = 'ignored'
    _global_history._save()
    log.info("已隐藏所有失败记录")


def predict_at_time_layer(match: Dict, time_layer: str) -> bool:
    """
    在指定时间层对比赛进行预测并落库

    参数：
        match: fetch_match_list 的比赛字典（含 match_id/home/away/league/time）
        time_layer: 时间层标识

    返回：
        是否成功
    """
    match_id = match.get('match_id') or match.get('mid')
    try:
        from . import analyze_match

        log.info(f"正在进行时间分层预测: match_id={match_id}, time_layer={time_layer}")

        # analyze_match 内部会保存预测记录；传完整字段以便记录含队名/联赛
        analyze_match({
            'match_id': match_id,
            'home': match.get('home', ''),
            'away': match.get('away', ''),
            'league': match.get('league', ''),
            'time': match.get('time', match.get('match_time', '')),
            'num': match.get('num') or match.get('match_num'),
        }, force_refresh=True)

        log.info(f"时间分层预测成功: match_id={match_id}, time_layer={time_layer}")
        return True

    except Exception as e:
        log.error(f"时间分层预测异常: match_id={match_id}, time_layer={time_layer}, error={e}")
        return False


def scan_and_predict_time_layers() -> Dict[str, int]:
    """
    扫描未来比赛并在时间分层点进行预测
    
    返回：
        统计结果 {'T-24h': 数量, 'T-6h': 数量, 'T-1h': 数量, 'T-15min': 数量}
    """
    result = {'T-24h': 0, 'T-6h': 0, 'T-1h': 0, 'T-15min': 0}
    
    try:
        from .data_loader import fetch_future_matches
        
        matches = fetch_future_matches()
        
        for match in matches:
            match_id = match.get('mid', match.get('match_id'))
            match_time_str = match.get('time', match.get('match_time', ''))
            
            if not match_id or not match_time_str:
                continue
            
            # 推断当前应该属于哪个时间层
            time_layer = infer_time_layer(match_time_str)
            
            # 检查是否需要在这个时间层进行预测
            if time_layer in result:
                # 检查是否已经在这个时间层预测过（避免重复）。
                # 用共享实例：每场比赛重建一次就是一次整表读。
                existing = _global_history.get_record(match_id)
                
                if existing:
                    time_layers = existing.get('time_layers', {})
                    if time_layers.get(time_layer) is not None:
                        log.debug(f"已在 {time_layer} 层预测过: {match_id}")
                        continue
                
                # 执行预测
                if predict_at_time_layer(match, time_layer):
                    result[time_layer] += 1
        
        log.info(f"时间分层扫描完成: T-24h={result['T-24h']}, T-6h={result['T-6h']}, T-1h={result['T-1h']}, T-15min={result['T-15min']}")
        
    except Exception as e:
        log.error(f"时间分层扫描异常: {e}")
    
    return result


def get_history() -> PredictionHistory:
    """获取全局预测历史管理器实例"""
    return _global_history


# 两个周期任务的间隔（秒）
SYNC_INTERVAL_SECONDS = 7200        # 赛后回填，两小时一轮
TIME_LAYER_INTERVAL_SECONDS = 600   # 时间分层扫描，十分钟一轮


def register_football_tasks(submit, interval_seconds=SYNC_INTERVAL_SECONDS):
    """把足球的两个周期任务登记到进程级调度器。

    **迁移前这里用 APScheduler，而 `apscheduler` 不在 `requirements.txt` 里。**
    线上碰巧装着（3.11.3），所以走的是 APScheduler 那条路；环境一旦重建、
    它不在了，代码会静默走进 `except ImportError` 的降级分支——那条分支把
    两个任务塞进同一个 `while` 循环、共用一个 `sleep(7200)`，于是时间分层
    扫描从十分钟一轮变成两小时一轮。而 T-15min 层的窗口只有 45 分钟
    （`infer_time_layer` 分的是区间不是时刻），两小时一轮必然整层漏掉。
    **不报错、不告警，只是那一层的预测再也不会产生。**

    `foundation/tasks` 是纯线程实现，没有可选依赖，两个任务各自独立间隔——
    这个隐患随之消失。

    与 kl8 那批同样的语义差别：APScheduler 的 interval 是「从启动时刻起每 N
    秒」，这里是「上一轮结束后再等 N 秒」，任务耗时会累积成漂移。这两个都是
    「看当前状态、需要才做」的任务（分层扫描还带幂等：同一层预测过就跳过），
    最窄的 T-15min 也有 45 分钟窗口容得下漂移，所以无害。
    """
    tasks = (
        ('football_result_sync', auto_sync_results, interval_seconds),
        ('football_time_layer_scan', scan_and_predict_time_layers,
         TIME_LAYER_INTERVAL_SECONDS),
    )
    registered = [name for name, fn, interval in tasks
                  if submit(name, fn, interval)]
    log.info('足球后台任务已登记: %s', ', '.join(registered) or '（无）')
    return registered


# ==================== 测试 ====================

def main():
    print("=== 预测历史模块测试 ===")
    
    # 查看统计
    stats = get_history_stats()
    print(f"统计信息: {stats}")
    
    # 手动同步
    result = sync_results()
    print(f"同步结果: {result}")


if __name__ == '__main__':
    main()
