# -*- coding: utf-8 -*-
"""快乐8主分析器 KL8Analyzer"""

import math
import json
import time
import hashlib
import uuid
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple
from itertools import combinations
from pathlib import Path

from src.common.paths import data_path
from src.common.repositories import doc_store
from src.common.logger import setup_logger
from src.domain.numeric import statistics as stats
from src.domain.numeric.kl8 import pools, scoring, voting

log = setup_logger('kl8')
from . import strategies as _strategies_mod
from . import config as _cfg

from .config import (
    CANDIDATE_STRATEGIES, FEATURE_CONFIG, FUSHI_CONFIG, FUSHI_PLAY_KEYS, KL8_DEFAULT_HISTORY, KL8_DRAW_COUNT, KL8_EXPECTED_GAP, KL8_MIN_PREDICTION_PERIODS, KL8_NUM_RANGE, KL8_PREDICTOR_VERSION, MODEL_CONFIG, REFERENCE_STRATEGY, SELECT_CONFIG, SELECT_PLAY_KEYS, SELECT_TYPES,
)
from .strategies import (
    get_active_feature_weights, get_active_model_weights, is_prediction_ready,
)
from .stats import (
    _hit_rate_priority_thresholds, _play_accuracy_profile, _prize_tier_thresholds,
)
from .candidates import (
    _adaptive_repeat_cap, _adaptive_repeat_target, _clean_pick_numbers, _diversify_candidate_pool, _enforce_minimum_repeats, _high_tier_chase_candidate_pool, _prize_floor_candidate_pool, _select_final_candidate_pool, _shape_balanced_candidate_pool, _shape_penalty, _shape_profile, _shape_targets, _zone_spread_candidate_pool,
)
from .records import (
    _build_recent_settlement_performance, _build_strategy_health, _checksum_numbers, _compute_next_issue, _compute_prediction_changes, _load_last_snapshot, _strategy_fingerprint, check_data_integrity, load_prize_table, normalize_record, save_conflict_to_queue,
)

# 候选池整形已在领域层（3-10）。这里仍然显式注入而不是让 voting 直接
# import：投票要的是「一种整形办法」，不是「kl8 的那一种」，
# 换成别的彩票时换的就是这个参数。
_POOL_SHAPER = voting.PoolShaper(
    diversify=pools.diversify,
    select_final=pools.select_final,
)

# kl8 的号码空间与几个分区参数。写死在统计函数里的话，换一种彩票就用不上。
KL8_SPACE = stats.NumberSpace(low=1, high=80)
KL8_ZONE_SIZE = 5           # 16 个 5 码区，与 position_residual 粒度一致
KL8_BIG_SMALL_THRESHOLD = 40  # 大于 40 为大；40 本身算小


class KL8Analyzer:
    """快乐8预测分析器（v5: 严格三段式+预测就绪判断+纯参数化回测）"""

    def __init__(self, history_file: Optional[str] = None):
        self.history_file = history_file or data_path('kl8_history.json')
        self.using_simulated_data = False
        self._data_mtime = 0
        self.history_data = self._load_history()
        self.statistics = {}
        self.update_statistics()

    # ─── 数据加载（v5: 排序而非假设顺序+完整性检查）───

    def _load_history(self) -> List[Dict]:
        """加载历史开奖数据（v5: 合并后排序+完整性检查+冲突审核队列）"""

        source_records = {}

        # 来源0: foundation/store。放在最前是因为抓取路径每次都会镜像进来，
        # 它是唯一保证完整的那个；库不可用时下面两个来源照常兜底。
        store_records = {}
        try:
            from .store_sync import load_from_store

            for r in load_from_store():
                normed = normalize_record(r, keep_meta=True)
                if normed:
                    store_records[normed['issue']] = normed
        except Exception as e:
            log.warning(f'快乐8: 库加载失败: {e}')

        # 来源1: doc_store
        try:
            raw_records = doc_store._fallback_load_all('kl8_history')
            if raw_records:
                for r in raw_records:
                    normed = normalize_record(r, keep_meta=True)
                    if not normed:
                        continue
                    issue = normed['issue']
                    if issue in source_records:
                        old = source_records[issue]
                        if old['numbers'] != normed['numbers']:
                            log.error(f'doc_store内期号{issue}号码冲突，保留第一条')
                            save_conflict_to_queue({
                                'source': 'doc_store_internal',
                                'issue': issue,
                                'old_numbers': old['numbers'],
                                'new_numbers': normed['numbers'],
                                'action': 'kept_old',
                            })
                            continue
                    source_records[issue] = normed
                log.info(f'快乐8: doc_store加载了{len(source_records)}期有效数据')
        except Exception as e:
            log.warning(f'快乐8: doc_store加载失败: {e}')

        # 来源2: JSON文件
        file_records = {}
        path = Path(self.history_file)
        try:
            if path.exists():
                self._data_mtime = path.stat().st_mtime
                raw = json.loads(path.read_text(encoding='utf-8'))
                if isinstance(raw, dict):
                    source_list = raw.get('results', raw.get('data', []))
                else:
                    source_list = raw

                for r in source_list:
                    normed = normalize_record(r, keep_meta=True)
                    if not normed:
                        continue
                    issue = normed['issue']
                    if issue in file_records:
                        old = file_records[issue]
                        if old['numbers'] != normed['numbers']:
                            log.error(f'JSON文件内期号{issue}号码冲突')
                            save_conflict_to_queue({
                                'source': 'json_file_internal',
                                'issue': issue,
                                'old_numbers': old['numbers'],
                                'new_numbers': normed['numbers'],
                                'action': 'kept_old',
                            })
                            continue
                    file_records[issue] = normed
                log.info(f'快乐8: 文件加载了{len(file_records)}期有效数据')
        except Exception as e:
            log.warning(f'快乐8: 文件加载失败: {e}')

        # 合并两个来源，按期号去重，冲突时报错不覆盖
        merged = {}
        for source_name, records in [('store', store_records),
                                     ('doc_store', source_records),
                                     ('json', file_records)]:
            for issue, record in records.items():
                if issue in merged:
                    old = merged[issue]
                    if old['numbers'] != record['numbers']:
                        log.error(
                            f'期号{issue}多源号码冲突: '
                            f'{source_name}={record["numbers"]}, '
                            f'已有={old["numbers"]}, 保留旧值待人工确认'
                        )
                        save_conflict_to_queue({
                            'source': source_name,
                            'issue': issue,
                            'old_numbers': old['numbers'],
                            'new_numbers': record['numbers'],
                            'old_source': old.get('source', ''),
                            'new_source': record.get('source', ''),
                            'action': 'kept_old',
                        })
                        continue
                merged[issue] = record

        if not merged:
            self.using_simulated_data = True
            log.error('快乐8: 无真实历史数据')
            return []

        # v5: 排序而非假设顺序（读取后确保按期号降序排列）
        data = sorted(merged.values(), key=lambda x: x['issue'], reverse=True)
        self.using_simulated_data = False

        # v5: 数据完整性检查
        integrity = check_data_integrity(data)
        if integrity.get('missing_count', 0) > 0:
            log.warning(f'快乐8: 发现{integrity["missing_count"]}个缺失期号')
        if integrity.get('conflict_count', 0) > 0:
            log.warning(f'快乐8: 发现{integrity["conflict_count"]}个日期期号不一致')

        log.info(f'快乐8: 多源合并后共{len(data)}期有效数据')
        return data

    def _check_data_mtime(self) -> bool:
        """检查数据文件mtime是否变化"""
        path = Path(self.history_file)
        try:
            if path.exists():
                current_mtime = path.stat().st_mtime
                if current_mtime != self._data_mtime:
                    log.info(f'快乐8: 数据文件mtime变化，需重新加载')
                    self._data_mtime = current_mtime
                    return True
        except Exception:
            pass
        return False

    def reload_if_needed(self) -> bool:
        """如果数据文件已更新则重新加载"""
        if self._check_data_mtime():
            self.history_data = self._load_history()
            self.update_statistics()
            return True
        return False

    # ─── 统计计算 ───

    def update_statistics(self):
        """更新所有统计量。

        具体算法都在 `domain/numeric/statistics.py`——它们对任何数字彩票
        都是同一批概念（频率、遗漏、冷热、共现、跨期转移、区间分布），
        变的只是号码空间。这里只负责把 kl8 的空间参数喂进去。
        """
        if not self.history_data:
            self.statistics = {}
            return

        recent = min(len(self.history_data), KL8_DEFAULT_HISTORY)
        recent_data = self.history_data[:recent]
        draws = [record['numbers'] for record in recent_data]

        freq = stats.number_frequency(draws, space=KL8_SPACE)
        pairs = stats.pair_cooccurrence(draws)
        transition, transition_support = stats.transition_probability(
            draws, space=KL8_SPACE)

        self.statistics = {
            'frequency': freq,
            'gap': stats.gaps(draws, space=KL8_SPACE),
            'trend': stats.trend(draws, space=KL8_SPACE),
            'pair_cooccurrence': pairs,
            'avg_cooccurrence': stats.average_cooccurrence(pairs, space=KL8_SPACE),
            'next_transition_probability': transition,
            'next_transition_support': transition_support,
            'adjacent_freq': stats.adjacent_frequency(freq, space=KL8_SPACE),
            'total_periods': recent,
            'expected_freq': recent * KL8_DRAW_COUNT / KL8_NUM_RANGE,
            'expected_gap': KL8_EXPECTED_GAP,
            'last_numbers': set(draws[0]) if draws else set(),
            'freq_by_zone': stats.zone_frequency(draws, size=KL8_ZONE_SIZE,
                                                 space=KL8_SPACE),
            'freq_by_road': stats.road_frequency(draws),
            'freq_by_odd_even': stats.parity_frequency(draws),
            'freq_by_big_small': stats.high_low_frequency(
                draws, threshold=KL8_BIG_SMALL_THRESHOLD),
        }

    # ─── 对称评分函数 ───

    @staticmethod
    def balance_score(actual_ratio: float, target_ratio: float, is_target: bool) -> float:
        """对称平衡评分"""
        imbalance = target_ratio - actual_ratio
        delta = 0.30 * (imbalance if is_target else -imbalance)
        return max(0.2, min(0.8, 0.5 + delta))

    # ─── 特征评分 ───

    def _calculate_feature_score(self, num: int, **options) -> Dict[str, float]:
        """号码 num 的各特征得分。算法在 `domain/numeric/kl8/scoring.py`。"""
        return scoring.feature_scores(
            num, self.statistics, based_on_issue=self._based_on_issue(), **options)

    def _based_on_issue(self) -> str:
        """当前最新一期期号。种子随机分靠它做到「同期稳定、跨期变化」。"""
        return self.history_data[0].get('issue', '') if self.history_data else ''

    # ─── 排名模型（v5: 纯参数化，接受外部feature_weights）───

    def get_ensemble_ranking(self, top_n: int = 20,
                             feature_weights: Optional[Dict[str, float]] = None,
                             **options) -> List[Dict]:
        """按加权和排名。权重缺省时取全局活跃权重。

        权重从哪儿来是配置问题，留在这里；怎么算分是领域问题，在 scoring 里。
        """
        return scoring.ensemble_ranking(
            self.statistics, feature_weights or get_active_feature_weights(),
            top_n=top_n, based_on_issue=self._based_on_issue(), **options)

    # ─── 多模型投票 ───

    def multi_model_voting(
        self,
        pick_n: int = 5,
        top_n: int = 20,
        feature_weights: Optional[Dict[str, float]] = None,
        model_weights: Optional[Dict[str, float]] = None,
        repeat_direction: str = 'neutral',
        repeat_avoid_score: float = 0.10,
        repeat_non_avoid_score: float = 0.85,
        repeat_follow_score: float = 0.90,
        repeat_non_follow_score: float = 0.50,
        pool_diversify: bool = True,
        pool_max_last_numbers: Optional[int] = None,
        frequency_mode: str = 'mean_reversion',
        final_selection_mode: str = 'balanced',
    ) -> Dict:
        """跑投票管道。权重缺省时取全局活跃权重。

        权重从哪儿来、版本号是什么，都是配置问题，留在这里；怎么投、
        怎么定候选池，是领域问题，在 `domain/numeric/kl8/voting` 里。
        """
        return voting.vote(
            self.statistics,
            feature_weights or get_active_feature_weights(),
            model_weights or get_active_model_weights(),
            _POOL_SHAPER,
            version=KL8_PREDICTOR_VERSION,
            based_on_issue=self._based_on_issue(),
            pick_n=pick_n,
            top_n=top_n,
            pool_diversify=pool_diversify,
            pool_max_last_numbers=pool_max_last_numbers,
            final_selection_mode=final_selection_mode,
            repeat_direction=repeat_direction,
            repeat_avoid_score=repeat_avoid_score,
            repeat_non_avoid_score=repeat_non_avoid_score,
            repeat_follow_score=repeat_follow_score,
            repeat_non_follow_score=repeat_non_follow_score,
            frequency_mode=frequency_mode,
        )

    # ─── 预测快照 ───

    def _save_prediction_snapshot(self, prediction_result: Dict) -> Optional[str]:
        """保存预测快照（v9: 唯一约束 + is_experiment标记）

        v9改动:
        - 同一策略+同一目标期只保留一份正式快照
        - 其他重复快照标记 is_experiment=True，不进入正式命中率统计
        """
        if not self.history_data:
            return None

        snapshot_dir = Path(_cfg.KL8_SNAPSHOT_DIR)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # v9: 策略指纹 — 用于唯一约束
        # 使用 select_5 的策略作为基准指纹（所有玩法当前使用同一策略）
        resolved = prediction_result.get('resolved_strategies', {})
        base_strategy = resolved.get('select_5', {})
        strategy_fp = _strategy_fingerprint(base_strategy) if base_strategy else 'no_strategy'

        # v9: 检查是否已有同一目标期+策略指纹的正式快照
        target_issue_val = _compute_next_issue(self.history_data[0]['issue'], self.history_data)
        snapshot_key = f'{target_issue_val}_{strategy_fp}'
        is_experiment = False

        # 扫描已有快照
        for existing_file in snapshot_dir.glob('snapshot_*.json'):
            try:
                existing_data = json.loads(existing_file.read_text(encoding='utf-8'))
                existing_key = f'{existing_data.get("target_issue", "")}_{_strategy_fingerprint(existing_data.get("resolved_strategies", {}).get("select_5", {}))}'
                if existing_key == snapshot_key and not existing_data.get('is_experiment', False):
                    # 已有正式快照 → 新快照标记为实验
                    is_experiment = True
                    log.info(f'快乐8: 同期同策略已有正式快照，新快照标记为实验预测')
                    break
            except Exception:
                continue

        # 全窗口SHA256指纹
        recent = min(len(self.history_data), KL8_DEFAULT_HISTORY)
        history_window = self.history_data[:recent]
        history_fingerprint = hashlib.sha256(
            json.dumps(
                [
                    {
                        'issue': r['issue'],
                        'numbers': r['numbers'],
                        'date': r.get('date', ''),
                        'checksum': r.get('checksum', ''),
                    }
                    for r in history_window
                ],
                ensure_ascii=False,
                sort_keys=True,
                separators=(',', ':'),
            ).encode()
        ).hexdigest()

        latest_issue = self.history_data[0]['issue']
        # v9: target_issue 推算改进 + target_type 严格校验
        # 不再简单用 int(latest_issue)+1，而是从历史数据推导下一期期号模式
        # 同时保存 target_type='next_draw_after_based_on'，结算时验证actual是based_on的直接下一期
        target_issue = _compute_next_issue(latest_issue, self.history_data)
        target_type = 'next_draw_after_based_on'
        snapshot_id = uuid.uuid4().hex
        snapshot = {
            'snapshot_id': snapshot_id,
            'target_issue': target_issue,  # v9: 从历史模式推算（用于调度器匹配）
            'target_type': target_type,     # v9: 结算时验证actual是based_on的直接下一期
            'based_on_issue': latest_issue,
            'predicted_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'history_window_size': recent,
            'history_start_issue': history_window[-1]['issue'] if history_window else '',
            'history_end_issue': latest_issue,
            'history_fingerprint': history_fingerprint,
            'strategy_fingerprint': strategy_fp,  # v9: 策略指纹
            'is_experiment': is_experiment,  # v9: 实验预测标记
            'version': KL8_PREDICTOR_VERSION,
            'feature_config': {k: dict(v) for k, v in FEATURE_CONFIG.items()},
            'model_config': {k: dict(v) for k, v in MODEL_CONFIG.items()},
            'active_strategies': {k: dict(v) for k, v in _cfg.ACTIVE_STRATEGIES.items()},
            'reference_strategy': dict(REFERENCE_STRATEGY),
            'candidate_strategies': {k: dict(v) for k, v in CANDIDATE_STRATEGIES.items()},
            # v7.1: 每个玩法记录strategy_id和prediction_mode
            'play_strategies': {
                key: prediction_result.get(key, {}).get('strategy_id', '')
                for key in (*SELECT_PLAY_KEYS, *FUSHI_PLAY_KEYS)
            },
            'prediction_modes': {
                key: prediction_result.get(key, {}).get('prediction_mode', '')
                for key in (*SELECT_PLAY_KEYS, *FUSHI_PLAY_KEYS)
            },
            # v9: 保存每种玩法当时实际使用的完整策略配置
            'resolved_strategies': prediction_result.get('resolved_strategies', {}),
            'ranking': prediction_result.get('ranking', []),
        }
        for select_type in SELECT_TYPES:
            key = f'select_{select_type}'
            snapshot[key] = prediction_result.get(key, {}).get('numbers', [])
            # v9.6: 保存多注组合，便于赛后统计组合层面命中率
            multi = prediction_result.get(key, {}).get('multi_slips')
            if multi:
                snapshot[f'{key}_multi_slips'] = multi
        for fushi_key, fushi_cfg in FUSHI_CONFIG.items():
            snapshot[fushi_key] = prediction_result.get(fushi_key, {}).get(
                fushi_cfg['numbers_field'],
                prediction_result.get(fushi_key, {}).get('core_numbers', []),
            )

        snapshot_file = snapshot_dir / f'snapshot_{snapshot_id}.json'

        try:
            with snapshot_file.open('x', encoding='utf-8') as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
            log.info(f'快乐8: 预测快照已保存 -> {snapshot_file.name}')
            return snapshot_file.name
        except FileExistsError:
            log.error(f'快乐8: 快照UUID冲突 {snapshot_file.name}')
            return None
        except Exception as e:
            log.error(f'快乐8: 保存快照失败: {e}')
            return None

    def settle_prediction(self, snapshot_file: str, actual_issue: str, actual_numbers: List[int], force: bool = False) -> Dict:
        """赛后结算（v6: 无号码不扣投注+ROI统一+期号校验）

        Args:
            force: 为 True 时强制覆盖已存在的结算文件（用于奖金表更新后重算历史结算）。


        v6改动:
        1. 只有len(numbers)==select_type时才视为placed（已投注）
        2. 未placed的玩法bet=0, prize=0
        3. ROI统一为return_multiple和profit_roi两个字段
        4. 复式7码每期全部21注组合都计算投注和奖金
        5. actual_issue必须晚于based_on_issue
        """

        # 路径安全
        snapshot_file = Path(snapshot_file).name
        path = Path(_cfg.KL8_SNAPSHOT_DIR) / snapshot_file
        if not path.exists():
            return {'error': f'快照文件不存在: {snapshot_file}'}

        # 校验开奖号码
        actual_normed = normalize_record({
            'issue': actual_issue,
            'numbers': actual_numbers,
        })
        if not actual_normed:
            return {'error': '实际开奖号码非法'}

        # 读取原始快照
        try:
            snapshot = json.loads(path.read_text(encoding='utf-8'))
        except Exception as e:
            return {'error': f'读取快照失败: {e}'}

        # v6: 期号校验 — actual_issue必须晚于based_on_issue
        based_on_issue = snapshot.get('based_on_issue', '')
        target_issue = snapshot.get('target_issue', '')

        # v9: 严格校验 — target_issue必须与实际开奖期号一致
        # 不再只检查"actual > based_on"，而是检查actual == target_issue
        # 或者actual是based_on_issue的直接下一期（target_type校验）
        if target_issue:
            if str(target_issue) != str(actual_normed['issue']):
                # 容许：actual_issue是based_on_issue的直接下一期（跨年/停开/补期场景）
                target_type = snapshot.get('target_type', '')
                if target_type == 'next_draw_after_based_on':
                    # 验证actual_issue确实是based_on_issue在历史数据中的直接下一期
                    analyzer = get_kl8_analyzer()
                    history_asc = sorted(analyzer.history_data, key=lambda x: x['issue'])
                    # 找到based_on_issue的位置
                    based_on_idx = None
                    for idx, rec in enumerate(history_asc):
                        if rec['issue'] == based_on_issue:
                            based_on_idx = idx
                            break
                    if based_on_idx is not None and based_on_idx + 1 < len(history_asc):
                        next_issue = history_asc[based_on_idx + 1]['issue']
                        if str(next_issue) == str(actual_normed['issue']):
                            pass  # 验证通过：actual确实是based_on的直接下一期
                        else:
                            return {
                                'error': f'快照目标期号{target_issue}与实际期号{actual_normed["issue"]}不一致'
                                         f'(based_on的直接下一期是{next_issue})'
                            }
                    else:
                        return {
                            'error': f'快照目标期号{target_issue}与实际期号{actual_normed["issue"]}不一致'
                        }
                else:
                    return {
                        'error': f'快照目标期号{target_issue}与实际期号{actual_normed["issue"]}不一致'
                    }

        # 旧版兜底：如果没有target_issue，仍保留 based_on < actual 的宽松校验
        if not target_issue and based_on_issue:
            try:
                actual_int = int(actual_normed['issue'])
                based_on_int = int(based_on_issue)
                if actual_int <= based_on_int:
                    return {'error': f'实际开奖期号{actual_normed["issue"]}必须晚于预测基准期号{based_on_issue}'}
            except (ValueError, TypeError):
                if str(actual_normed['issue']) <= str(based_on_issue):
                    return {'error': f'实际开奖期号必须晚于预测基准期号'}

        # 检查是否已结算
        settlements_dir = Path(_cfg.KL8_SETTLEMENT_DIR)
        settlements_dir.mkdir(parents=True, exist_ok=True)

        existing_settlement = settlements_dir / f'settlement_{snapshot.get("snapshot_id", "")}.json'
        if existing_settlement.exists() and not force:
            try:
                old = json.loads(existing_settlement.read_text(encoding='utf-8'))
                return {'error': '快照已结算，不可重复结算', 'settlement': old}
            except Exception:
                pass

        snapshot_sha256 = hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()
        actual_set = set(actual_normed['numbers'])

        # v6: 奖金结算 — 只有placed=true才计投注和奖金
        prize_table = load_prize_table()

        prize_settlement = {}
        cumulative_bet = 0
        cumulative_prize = 0

        for select_type in SELECT_TYPES:
            numbers = _clean_pick_numbers(
                snapshot.get(f'select_{select_type}', []),
                select_type,
            )
            prize_key = f'select_{select_type}'
            prize_info = prize_table.get(prize_key, {})

            # v6: 只有号码完整时才视为placed
            placed = len(numbers) == select_type
            bet = prize_info.get('bet', 2) if placed else 0
            hits = len(set(numbers) & actual_set) if placed else 0
            prize = prize_info.get(str(hits), 0) if placed else 0

            cumulative_bet += bet
            cumulative_prize += prize

            # v6: ROI统一为两个字段
            return_multiple = prize / max(bet, 1) if placed else 0
            profit_roi = (prize - bet) / max(bet, 1) if placed else 0

            prize_settlement[prize_key] = {
                'placed': placed,
                'hits': hits,
                'bet': bet,
                'prize': prize,
                'return_multiple': round(return_multiple, 4),
                'profit_roi': round(profit_roi, 4),
            }

        # v9.6: 多注组合结算（记录组合层面的最佳命中，用于后续命中率统计）
        multi_slip_settlement = {}
        for select_type in SELECT_TYPES:
            key = f'select_{select_type}'
            prize_key = key
            prize_info = prize_table.get(prize_key, {})
            slips = snapshot.get(f'{key}_multi_slips') or []
            if not slips:
                continue
            pick_size = len(slips[0]) if slips else select_type
            slip_hits = []
            total_bet = 0
            total_prize = 0
            bets_per_slip = math.comb(pick_size, select_type)
            bet_per_bet = prize_info.get('bet', 2)
            for slip in slips:
                h = len(set(slip) & actual_set)
                slip_hits.append(h)
                total_bet += bets_per_slip * bet_per_bet
                total_prize += prize_info.get(str(h), 0)
            best = max(slip_hits) if slip_hits else 0
            distribution = dict(Counter(slip_hits))
            multi_slip_settlement[prize_key] = {
                'placed': True,
                'pick_size': pick_size,
                'slip_count': len(slips),
                'slip_hits': slip_hits,
                'best_hits': best,
                'hit_distribution': distribution,
                'ge3': int(best >= 3),
                'ge4': int(best >= 4),
                'ge5': int(best >= 5),
                'ge6': int(best >= 6),
                'total_bet': total_bet,
                'total_prize': total_prize,
                'return_multiple': round(total_prize / max(total_bet, 1), 4),
                'profit_roi': round((total_prize - total_bet) / max(total_bet, 1), 4),
            }

        # 复式玩法ROI — 每期全部组合都计算
        fushi_settlement = {}
        for fushi_key, fushi_cfg in FUSHI_CONFIG.items():
            pool_size = fushi_cfg['pool_size']
            base_pick = fushi_cfg['base_pick']
            core_numbers = _clean_pick_numbers(snapshot.get(fushi_key, []), pool_size)
            placed = len(core_numbers) == pool_size

            fushi_prize_info = prize_table.get(fushi_key, {})
            prize_key = fushi_prize_info.get('prize_key', fushi_cfg.get('prize_key', fushi_key))
            combo_prize_info = prize_table.get(prize_key, fushi_prize_info)
            bet_per_combo = fushi_prize_info.get('bet_per_combo', combo_prize_info.get('bet', 2))

            pool_hits = 0
            combo_hits = []
            total_bet = 0
            total_prize = 0

            if placed and core_numbers:
                pool_hits = len(set(core_numbers) & actual_set)
                total_bet = math.comb(pool_size, base_pick) * bet_per_combo
                for combo in combinations(core_numbers, base_pick):
                    combo_h = len(set(combo) & actual_set)
                    combo_hits.append(combo_h)
                    total_prize += combo_prize_info.get(str(combo_h), 0)

            cumulative_bet += total_bet
            cumulative_prize += total_prize

            fushi_settlement[fushi_key] = {
                'placed': placed,
                'pool_hits': pool_hits,
                'max_combo_hits': max(combo_hits, default=0),
                'hit_distribution': dict(Counter(combo_hits)) if combo_hits else {},
                'combo_pick': base_pick,
                'pool_size': pool_size,
                'total_combinations': math.comb(pool_size, base_pick) if placed else 0,
                'total_bet': total_bet,
                'total_prize': total_prize,
            }

        # v6: 累计ROI统一
        cumulative_return_multiple = cumulative_prize / max(cumulative_bet, 1)
        cumulative_profit_roi = (cumulative_prize - cumulative_bet) / max(cumulative_bet, 1)

        settlement = {
            'snapshot_id': snapshot.get('snapshot_id', ''),
            'snapshot_file': snapshot_file,
            'snapshot_sha256': snapshot_sha256,
            'settled_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'actual_issue': actual_normed['issue'],
            'actual_numbers': actual_normed['numbers'],
            'actual_checksum': _checksum_numbers(actual_normed['numbers']),
            'based_on_issue': snapshot.get('based_on_issue', ''),
            'strategy_ids': dict(snapshot.get('play_strategies', {})),  # v9: 从快照读取
            'prediction_modes': dict(snapshot.get('prediction_modes', {})),  # v9: 从快照读取
            'resolved_strategies': dict(snapshot.get('resolved_strategies', {})),  # v9: 完整策略快照
            **{
                f'hit_select_{select_type}': prize_settlement.get(f'select_{select_type}', {}).get('hits', 0)
                for select_type in SELECT_TYPES
            },
            'prize_settlement': prize_settlement,
            'multi_slip_settlement': multi_slip_settlement,
            'fushi_settlement': fushi_settlement,
            'fu_shi_7_pool_hits': fushi_settlement.get('fu_shi_7', {}).get('pool_hits', 0),
            'hit_fu_shi_7_max': fushi_settlement.get('fu_shi_7', {}).get('max_combo_hits', 0),
            'fu_shi_7_hit_distribution': fushi_settlement.get('fu_shi_7', {}).get('hit_distribution', {}),
            'fu_shi_10_11_pool_hits': fushi_settlement.get('fu_shi_10_11', {}).get('pool_hits', 0),
            'hit_fu_shi_10_11_max': fushi_settlement.get('fu_shi_10_11', {}).get('max_combo_hits', 0),
            'fu_shi_10_11_hit_distribution': fushi_settlement.get('fu_shi_10_11', {}).get('hit_distribution', {}),
            'cumulative_bet': cumulative_bet,
            'cumulative_prize': cumulative_prize,
            'cumulative_return_multiple': round(cumulative_return_multiple, 4),
            'cumulative_profit_roi': round(cumulative_profit_roi, 4),
        }

        try:
            mode = 'w' if force else 'x'
            with existing_settlement.open(mode, encoding='utf-8') as f:
                json.dump(settlement, f, ensure_ascii=False, indent=2)
            log.info(f'快乐8: 结算完成 -> {existing_settlement.name}')
            return {'success': True, 'settlement': settlement}
        except FileExistsError:
            return {'error': '结算文件已存在，不可重复结算'}
        except Exception as e:
            return {'error': f'写入结算失败: {e}'}

    # ─── 窗口分析器构造（v7.1新增）───

    def _build_window_analyzer(self, window_size: int):
        """构造临时分析器（与回测逻辑完全一致）

        策略指定window_size时，必须创建临时分析器并调用update_statistics()
        确保线上预测和回测使用相同的统计窗口，而不是只临时计算freq。

        参数:
            window_size: 统计窗口大小，0或None时使用KL8_DEFAULT_HISTORY
        """
        recent = min(len(self.history_data), window_size or KL8_DEFAULT_HISTORY)

        temp = KL8Analyzer.__new__(KL8Analyzer)
        temp.history_data = self.history_data[:recent]
        temp.using_simulated_data = False
        temp.history_file = self.history_file
        temp._data_mtime = self._data_mtime
        temp.statistics = {}
        temp.update_statistics()

        return temp

    # ─── 统一候选池（v9新增）───

    def build_candidate_pool(self) -> Tuple[Dict, Dict]:
        """统一候选池（仅供回测管道使用）

        v9.1改动: 内部调用 build_pool_by_strategy
        线上预测不再使用此方法，改为各玩法独立生成候选池
        """
        strategy = _strategies_mod.resolve_play_strategy('select_5')
        pool_result = self.build_pool_by_strategy(strategy, pool_size=20)
        return pool_result, strategy

    # ─── 按策略独立生成候选池 ───

    def build_pool_by_strategy(self, strategy: Dict, pool_size: int = 20) -> Dict:
        """按指定策略生成候选池

        v9.1改动:
        - 每个玩法可以按自己的策略独立生成候选池
        - 不再强制共用 select_5 的 Top20
        - 当所有玩法使用同一参考策略时，输出自然一致
        - 当未来不同玩法有不同验证策略时，候选池才会真正不同
        """
        repeat_direction = strategy.get('repeat_direction', 'neutral')
        repeat_avoid_score = strategy.get('repeat_avoid_score', 0.10)
        repeat_non_avoid_score = strategy.get('repeat_non_avoid_score', 0.85)
        repeat_follow_score = strategy.get('repeat_follow_score', 0.90)
        repeat_non_follow_score = strategy.get('repeat_non_follow_score', 0.50)

        predictor = self._build_window_analyzer(
            strategy.get('window_size', KL8_DEFAULT_HISTORY)
        )

        return predictor.multi_model_voting(
            pick_n=pool_size,
            top_n=pool_size,
            feature_weights=strategy['feature_weights'],
            model_weights=strategy['model_weights'],
            repeat_direction=repeat_direction,
            repeat_avoid_score=repeat_avoid_score,
            repeat_non_avoid_score=repeat_non_avoid_score,
            repeat_follow_score=repeat_follow_score,
            repeat_non_follow_score=repeat_non_follow_score,
            pool_diversify=strategy.get('pool_diversify', True),
            pool_max_last_numbers=(
                strategy.get('pool_max_last_numbers')
                if strategy.get('pool_max_last_numbers') is not None
                else _adaptive_repeat_cap(predictor.history_data, pool_size)
            ),
            frequency_mode=strategy.get('frequency_mode', 'mean_reversion'),
            final_selection_mode=strategy.get('final_selection_mode', 'balanced'),
        )

    def _score_exclude_recalculation_pool(
        self,
        selected: List[Tuple[int, float]],
        candidates: List[Tuple[int, float]],
        target_size: int,
        repeat_cap: int,
    ) -> Dict:
        """Quality score for temporary exclude recalculation candidates."""
        numbers = [num for num, _ in selected]
        if len(numbers) < target_size:
            return {'quality_score': -1.0, 'reason': 'insufficient_numbers'}

        score_lookup = {num: float(score) for num, score in candidates}
        selected_score = sum(score_lookup.get(num, 0.0) for num in numbers)
        top_score = sum(float(score) for _, score in candidates[:target_size]) or 1.0
        score_ratio = max(0.0, min(1.2, selected_score / top_score))

        zone_counts = Counter((num - 1) // 5 + 1 for num in numbers)
        road_counts = Counter(num % 3 for num in numbers)
        dominant_zone = max(zone_counts.values()) / max(1, target_size)
        dominant_road = max(road_counts.values()) / max(1, target_size)
        zone_balance = 1.0 - max(0.0, dominant_zone - 0.42)
        road_balance = 1.0 - max(0.0, dominant_road - 0.55)

        last_numbers = self.statistics.get('last_numbers', set())
        repeat_count = sum(1 for num in numbers if num in last_numbers)
        repeat_fit = 1.0 - min(1.0, abs(repeat_count - repeat_cap) / max(1, target_size))
        shape_penalty = _shape_penalty(numbers, target_size, last_numbers, repeat_cap)
        shape_fit = max(0.0, 1.0 - shape_penalty / max(1.0, target_size * 1.6))

        quality_score = (
            score_ratio * 60.0
            + max(0.0, zone_balance) * 8.0
            + max(0.0, road_balance) * 6.0
            + repeat_fit * 8.0
            + shape_fit * 18.0
        )
        return {
            'quality_score': round(quality_score, 4),
            'score_ratio': round(score_ratio, 4),
            'zone_count': len(zone_counts),
            'road_count': len(road_counts),
            'repeat_count': repeat_count,
            'repeat_cap': repeat_cap,
            'shape_fit': round(shape_fit, 4),
            'shape_profile': _shape_profile(numbers, last_numbers),
        }

    def _best_exclude_recalculation_pool(
        self,
        candidates: List[Tuple[int, float]],
        target_size: int,
        repeat_cap: int,
        selection_mode: str = 'best_variant',
    ) -> Tuple[List[Tuple[int, float]], Dict]:
        """Try several deterministic recalculation shapes and keep the best one."""
        last_numbers = self.statistics.get('last_numbers', set())
        score_lookup = {num: score for num, score in candidates}
        variants: List[Tuple[str, List[Tuple[int, float]]]] = []
        seen = set()

        def add(label: str, pool: List[Tuple[int, float]]):
            nums = tuple(num for num, _ in pool[:target_size])
            if len(nums) < target_size or nums in seen:
                if len(nums) >= target_size and label == selection_mode:
                    variants.append((label, pool[:target_size]))
                return
            seen.add(nums)
            variants.append((label, pool[:target_size]))

        # 每种模式的重号上限该加还是该减，由 `pools.MODE_BUILDERS` 说了算。
        # 在这里把那套加减重写一遍，等于给同一条规则开第二个定义——而两处
        # 走偏了不会报错，只会让这两个入口给出不一样的推荐。
        for label in ('concentrated', 'balanced', 'repeat_follow',
                      'low_repeat', 'prize_floor'):
            add(label, pools.build_pool(label, candidates, target_size,
                                        last_numbers, repeat_cap))

        zone_spread_nums = []
        zone_counts = Counter()
        max_zone = max(1, math.ceil(target_size / 16))
        for num, _ in candidates[:max(target_size * 4, 20)]:
            zone = (num - 1) // 5 + 1
            if zone_counts[zone] >= max_zone:
                continue
            zone_spread_nums.append(num)
            zone_counts[zone] += 1
            if len(zone_spread_nums) >= target_size:
                break
        for num, _ in candidates:
            if len(zone_spread_nums) >= target_size:
                break
            if num not in zone_spread_nums:
                zone_spread_nums.append(num)
        add('zone_spread', [(num, score_lookup.get(num, 0.0)) for num in zone_spread_nums])
        add('shape_balanced', _shape_balanced_candidate_pool(
            candidates,
            target_size,
            last_numbers,
            max_last_numbers=repeat_cap,
        ))

        scored = []
        for label, pool in variants:
            quality = self._score_exclude_recalculation_pool(
                pool,
                candidates,
                target_size,
                repeat_cap,
            )
            scored.append((quality.get('quality_score', -1.0), label, pool, quality))

        scored.sort(key=lambda item: (-item[0], item[1]))
        if selection_mode != 'best_variant':
            selected = next((item for item in scored if item[1] == selection_mode), None)
        else:
            selected = None
        _, label, best_pool, quality = selected or scored[0]
        quality['selection_mode'] = label
        quality['requested_selection_mode'] = selection_mode
        quality['variant_count'] = len(scored)
        quality['variants'] = [
            {
                'mode': item_label,
                'numbers': [num for num, _ in item_pool],
                'quality_score': item_quality.get('quality_score', -1.0),
            }
            for _, item_label, item_pool, item_quality in scored[:4]
        ]
        return best_pool, quality

    def recalculate_play_excluding(
        self,
        play_type: str,
        exclude_numbers: List[int],
        record_context: Optional[Dict] = None,
    ) -> Dict:
        """临时剔除指定号码后重算单个玩法，不写入正式快照/缓存。"""
        if not self.history_data or self.using_simulated_data:
            return {'error': '历史数据不足，无法重新计算'}

        excluded = sorted({
            int(n) for n in (exclude_numbers or [])
            if isinstance(n, int) or str(n).isdigit()
        })
        excluded = [n for n in excluded if 1 <= n <= KL8_NUM_RANGE]
        excluded_set = set(excluded)

        if play_type in SELECT_PLAY_KEYS:
            try:
                pick_n = int(play_type.split('_')[1])
            except (ValueError, IndexError):
                return {'error': f'无效玩法: {play_type}'}

            strategy = _strategies_mod.resolve_play_strategy(play_type)
            if strategy is None:
                return {'error': '当前玩法没有可用策略'}

            pool_result = self.build_pool_by_strategy(
                strategy,
                pool_size=min(KL8_NUM_RANGE, max(40, pick_n + len(excluded) + 20)),
            )
            candidates = [
                (num, score)
                for num, score in pool_result.get('candidates', [])
                if num not in excluded_set
            ]
            if len(candidates) < pick_n:
                result = {
                    'error': f'剔除后候选号码不足，{play_type} 至少需要 {pick_n} 个号码，当前仅剩 {len(candidates)} 个',
                    'play_type': play_type,
                    'excluded_numbers': excluded,
                    'remaining_count': len(candidates),
                    'required_count': pick_n,
                }
                self._save_exclude_recalculation(result, status='exhausted', record_context=record_context)
                return result
            adaptive_cap = _adaptive_repeat_cap(self.history_data, pick_n)
            repeat_cap = (
                max(0, min(pick_n, int(strategy.get('pool_max_last_numbers'))))
                if strategy.get('pool_max_last_numbers') is not None
                else adaptive_cap
            )
            final_pool, quality = self._best_exclude_recalculation_pool(
                candidates,
                pick_n,
                repeat_cap,
                strategy.get('final_selection_mode', 'best_variant'),
            )
            output_numbers = sorted(num for num, _ in final_pool)
            result = {
                'play_type': play_type,
                'numbers': output_numbers,
                'excluded_numbers': excluded,
                'candidates': candidates[:12],
                'quality': quality,
                'strategy_id': strategy.get('strategy_id', ''),
                'prediction_mode': strategy.get('prediction_mode', ''),
                'is_validated': strategy.get('is_validated', False),
                'source': 'exclude_recalculate',
                'version': KL8_PREDICTOR_VERSION,
            }
            result['recalculation_record'] = self._save_exclude_recalculation(
                result, record_context=record_context,
            )
            return result

        if play_type in FUSHI_CONFIG:
            fushi_cfg = FUSHI_CONFIG[play_type]
            strategy = _strategies_mod.resolve_play_strategy(play_type)
            if strategy is None:
                return {'error': '当前玩法没有可用策略'}

            pool_size = fushi_cfg['pool_size']
            base_pick = fushi_cfg['base_pick']
            pool_result = self.build_pool_by_strategy(
                strategy,
                pool_size=min(KL8_NUM_RANGE, max(40, pool_size + len(excluded) + 20)),
            )
            candidates = [
                (num, score)
                for num, score in pool_result.get('candidates', [])
                if num not in excluded_set
            ]
            if len(candidates) < pool_size:
                result = {
                    'error': f'剔除后候选号码不足，{play_type} 至少需要 {pool_size} 个号码，当前仅剩 {len(candidates)} 个',
                    'play_type': play_type,
                    'excluded_numbers': excluded,
                    'remaining_count': len(candidates),
                    'required_count': pool_size,
                }
                self._save_exclude_recalculation(result, status='exhausted', record_context=record_context)
                return result
            adaptive_cap = _adaptive_repeat_cap(self.history_data, pool_size)
            repeat_cap = (
                max(0, min(pool_size, int(strategy.get('pool_max_last_numbers'))))
                if strategy.get('pool_max_last_numbers') is not None
                else adaptive_cap
            )
            final_pool, quality = self._best_exclude_recalculation_pool(
                candidates,
                pool_size,
                repeat_cap,
                strategy.get('final_selection_mode', 'best_variant'),
            )
            core_numbers = sorted(num for num, _ in final_pool)
            combo_list = [sorted(c) for c in combinations(core_numbers, base_pick)] if len(core_numbers) == pool_size else []
            result = {
                'play_type': play_type,
                fushi_cfg['numbers_field']: core_numbers,
                'core_numbers': core_numbers,
                'excluded_numbers': excluded,
                'candidates': candidates[:12],
                'quality': quality,
                'combinations': combo_list,
                'total_combinations': len(combo_list),
                'combo_pick': base_pick,
                'pool_size': pool_size,
                'strategy_id': strategy.get('strategy_id', ''),
                'prediction_mode': strategy.get('prediction_mode', ''),
                'is_validated': strategy.get('is_validated', False),
                'source': 'exclude_recalculate',
                'version': KL8_PREDICTOR_VERSION,
            }
            result['recalculation_record'] = self._save_exclude_recalculation(
                result, record_context=record_context,
            )
            return result

        return {'error': f'无效玩法: {play_type}'}

    def _save_exclude_recalculation(
        self,
        result: Dict,
        status: str = 'generated',
        record_context: Optional[Dict] = None,
    ) -> Dict:
        """Persist one exclude/recalculate round without mutating the formal snapshot."""
        if not self.history_data:
            return {}

        play_type = str(result.get('play_type') or '')
        excluded = sorted({int(n) for n in result.get('excluded_numbers', [])})
        numbers = result.get('numbers') or result.get('core_numbers') or []
        numbers = sorted({int(n) for n in numbers})
        based_on_issue = str(self.history_data[0].get('issue') or '')
        target_issue = str(_compute_next_issue(based_on_issue, self.history_data) or '')
        context = dict(record_context or {})
        source_snapshot_id = str(context.get('source_snapshot_id') or '')
        source_version = str(context.get('source_version') or KL8_PREDICTOR_VERSION)
        generation_mode = str(context.get('generation_mode') or 'manual')
        initial_numbers = sorted({int(n) for n in context.get('initial_numbers', [])})
        directory = Path(_cfg.KL8_RECALCULATION_DIR)
        directory.mkdir(parents=True, exist_ok=True)

        identity = hashlib.sha256(json.dumps({
            'target_issue': target_issue,
            'source_snapshot_id': source_snapshot_id,
            'play_type': play_type,
            'excluded_numbers': excluded,
        }, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:20]
        path = directory / f'recalculation_{identity}.json'

        existing = []
        for candidate in directory.glob('recalculation_*.json'):
            try:
                item = json.loads(candidate.read_text(encoding='utf-8'))
                if (
                    str(item.get('target_issue') or '') == target_issue
                    and str(item.get('source_snapshot_id') or '') == source_snapshot_id
                    and item.get('play_type') == play_type
                ):
                    existing.append(item)
            except Exception:
                continue

        previous = next((item for item in existing if item.get('record_id') == identity), None)
        if previous:
            return previous

        record = {
            'record_id': identity,
            'target_issue': target_issue,
            'based_on_issue': based_on_issue,
            'source_snapshot_id': source_snapshot_id,
            'source_version': source_version,
            'generation_mode': generation_mode,
            'initial_numbers': initial_numbers,
            'play_type': play_type,
            'round': 1 + max((int(item.get('round', 0)) for item in existing), default=0),
            'excluded_numbers': excluded,
            'numbers': numbers,
            'status': status,
            'remaining_count': result.get('remaining_count'),
            'required_count': result.get('required_count'),
            'strategy_id': result.get('strategy_id', ''),
            'selection_mode': (result.get('quality') or {}).get('selection_mode', ''),
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'version': source_version,
        }
        try:
            with path.open('x', encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            return record
        except FileExistsError:
            try:
                return json.loads(path.read_text(encoding='utf-8'))
            except Exception:
                return record
        except Exception as exc:
            log.error(f'快乐8: 保存删号重算记录失败: {exc}')
            return {}

    def generate_exclude_recalculation_chain(
        self,
        play_type: str,
        initial_numbers: List[int],
        max_rounds: int = 20,
        source_snapshot_id: str = '',
        source_version: str = '',
    ) -> Dict:
        """Automatically replay cumulative exclusions until no full pick remains."""
        current = sorted({int(n) for n in (initial_numbers or []) if 1 <= int(n) <= KL8_NUM_RANGE})
        if play_type in SELECT_PLAY_KEYS:
            required = int(play_type.split('_')[1])
        elif play_type in FUSHI_CONFIG:
            required = int(FUSHI_CONFIG[play_type]['pool_size'])
        else:
            return {'error': f'无效玩法: {play_type}'}
        if len(current) != required:
            return {'error': f'{play_type} 初始号码必须为{required}个'}

        excluded = set()
        record_context = {
            'source_snapshot_id': source_snapshot_id,
            'source_version': source_version or KL8_PREDICTOR_VERSION,
            'generation_mode': 'automatic',
            'initial_numbers': current,
        }
        generated = []
        exhausted = None
        for _ in range(max(1, max_rounds)):
            excluded.update(current)
            result = self.recalculate_play_excluding(
                play_type,
                sorted(excluded),
                record_context=record_context,
            )
            if result.get('error'):
                exhausted = {
                    'excluded_numbers': result.get('excluded_numbers', sorted(excluded)),
                    'remaining_count': result.get('remaining_count'),
                    'required_count': result.get('required_count', required),
                }
                break
            current = result.get('numbers') or result.get('core_numbers') or []
            if len(current) != required:
                break
            generated.append(result.get('recalculation_record') or {})

        return {
            'play_type': play_type,
            'generated_rounds': len(generated),
            'generation_mode': 'automatic',
            'records': generated,
            'exhausted': exhausted is not None,
            'terminal': exhausted,
        }

    def _candidate_variants(
        self,
        candidates: List[Tuple[int, float]],
        target_size: int,
        repeat_cap: int,
    ) -> Dict[str, List[int]]:
        """Build a few practical alternatives from the same candidate pool."""
        if not candidates or target_size <= 0:
            return {}

        last_numbers = self.statistics.get('last_numbers', set())
        def pool_numbers(mode):
            return sorted(num for num, _ in pools.build_pool(
                mode, candidates, target_size, last_numbers, repeat_cap))

        concentrated = pool_numbers('concentrated')
        high_tier_chase = pool_numbers('high_tier_chase')
        balanced = pool_numbers('balanced')
        low_repeat = pool_numbers('low_repeat')
        repeat_follow = pool_numbers('repeat_follow')
        zone_spread = sorted(num for num, _ in _zone_spread_candidate_pool(candidates, target_size))
        prize_floor = sorted(
            num for num, _ in _prize_floor_candidate_pool(
                candidates,
                target_size,
                last_numbers,
                max_last_numbers=repeat_cap,
            )
        )
        shape_balanced = sorted(
            num for num, _ in _shape_balanced_candidate_pool(
                candidates,
                target_size,
                last_numbers,
                max_last_numbers=repeat_cap,
            )
        )

        return {
            'high_tier_chase': high_tier_chase,
            'balanced': balanced,
            'concentrated': concentrated,
            'low_repeat': low_repeat,
            'repeat_follow': repeat_follow,
            'zone_spread': zone_spread,
            'prize_floor': prize_floor,
            'shape_balanced': shape_balanced,
        }

    # ─── 综合预测（v9.1: 各玩法独立候选池 + 本期变化对比）───

    def predict_all(self) -> Dict:
        """生成所有选型的预测结果

        v9.1改动:
        - 各玩法按自己的策略独立生成候选池（不再强制共用 select_5 的 Top20）
        - 本期变化对比: 与上期快照对比，展示候选池变化数和推荐号码替换详情
        - resolved_strategies: 快照保存每种玩法当时的完整策略配置
        """
        if not self.history_data or self.using_simulated_data:
            return {
                'error': '历史数据不足，无法进行有效预测。请先抓取真实数据。',
                'using_simulated_data': True,
            }
        if len(self.history_data) < KL8_MIN_PREDICTION_PERIODS:
            return {
                'error': f'历史数据不足，至少需要{KL8_MIN_PREDICTION_PERIODS}期真实数据后再预测。',
                'using_simulated_data': False,
                'data_quality': {
                    'valid': False,
                    'total_records': len(self.history_data),
                    'min_required': KL8_MIN_PREDICTION_PERIODS,
                    'reason': 'insufficient_history',
                },
            }

        prediction_ready = is_prediction_ready()

        results = {}
        resolved_strategies = {}  # v9: 保存每种玩法当时的完整策略
        all_candidate_pools = {}  # v9.1: 各玩法独立候选池

        # ── 加载上期快照，用于本期变化对比 ──
        last_snapshot = _load_last_snapshot()

        # v9.2: 各玩法按自己的策略独立生成候选池
        # _cfg.VERIFY_ONLY_MODE: 没有validated策略时不输出号码
        for select_type in SELECT_TYPES:
            config = SELECT_CONFIG[select_type]
            s_key = f'select_{select_type}'

            strategy = _strategies_mod.resolve_play_strategy(s_key)

            # v9.2: _cfg.VERIFY_ONLY_MODE — 没有已验证策略时不输出号码
            if strategy is None:
                results[s_key] = {
                    'desc': config['desc'],
                    'pick': config['pick'],
                    'numbers': [],
                    'status': 'verification_pending',
                    'prediction_mode': 'not_verified',
                    'is_validated': False,
                    'warning': '当前没有通过验证的策略，本玩法暂不输出推荐号码。',
                }
                resolved_strategies[s_key] = {
                    'strategy_id': '',
                    'prediction_mode': 'not_verified',
                    'is_validated': False,
                }
                continue

            # v9.1: 保存完整策略配置到 resolved_strategies
            resolved_strategies[s_key] = {
                'strategy_id': strategy['strategy_id'],
                'feature_weights': strategy['feature_weights'],
                'model_weights': strategy['model_weights'],
                'window_size': strategy.get('window_size', KL8_DEFAULT_HISTORY),
                'repeat_direction': strategy.get('repeat_direction', 'neutral'),
                'repeat_avoid_score': strategy.get('repeat_avoid_score', 0.10),
                'repeat_non_avoid_score': strategy.get('repeat_non_avoid_score', 0.85),
                'repeat_follow_score': strategy.get('repeat_follow_score', 0.90),
                'repeat_non_follow_score': strategy.get('repeat_non_follow_score', 0.50),
                'pool_diversify': strategy.get('pool_diversify', True),
                'pool_max_last_numbers': strategy.get('pool_max_last_numbers'),
                'final_max_last_numbers': strategy.get('final_max_last_numbers'),
                'final_min_last_numbers': strategy.get('final_min_last_numbers', 0),
                'frequency_mode': strategy.get('frequency_mode', 'mean_reversion'),
                'final_selection_mode': strategy.get('final_selection_mode', 'balanced'),
                'chain_objective': strategy.get('chain_objective'),
                'chain_audit_rounds': strategy.get('chain_audit_rounds'),
                'target_hits': strategy.get('target_hits'),
                'prediction_mode': strategy['prediction_mode'],
                'is_validated': strategy['is_validated'],
            }

            # v9.1: 按策略独立生成候选池
            # 选5/6的遗漏特征可能把上期号码全部压到Top20之外。内部保留完整排名，
            # 让后续重号下限约束始终有候选可用；对外候选池仍只展示Top20。
            internal_pool_size = KL8_NUM_RANGE if select_type in (5, 6) else 20
            pool_result = self.build_pool_by_strategy(strategy, pool_size=internal_pool_size)
            pool_top = pool_result.get('selected', [])[:20]
            selection_candidates = pool_result.get('candidates', [])[:internal_pool_size]
            pool_candidates = selection_candidates[:20]
            all_candidate_pools[s_key] = {
                'top20': pool_top,
                'candidates': pool_candidates,
                'strategy_id': strategy['strategy_id'],
            }

            final_repeat_cap = strategy.get(
                'final_max_last_numbers',
                min(
                    strategy.get('pool_max_last_numbers', _adaptive_repeat_cap(self.history_data, select_type))
                    or _adaptive_repeat_cap(self.history_data, select_type),
                    _adaptive_repeat_cap(self.history_data, select_type),
                )
            )
            final_pool, selected_mode = _select_final_candidate_pool(
                pool_candidates,
                select_type,
                self.statistics.get('last_numbers', set()),
                max_last_numbers=final_repeat_cap,
                selection_mode=strategy.get('final_selection_mode', 'balanced'),
            )
            repeat_profile = _adaptive_repeat_target(
                self.history_data,
                select_type,
                strategy.get('final_min_last_numbers', 0),
            )
            # v9.5: 默认不再为了“形态好看”强制换入上期重号。严格 walk-forward
            # 回测显示该后处理会降低选5/选6的平均命中；仅当某个已验证策略显式配置
            # final_min_last_numbers > 0 时才应用，避免启发式覆盖真实排名信号。
            configured_repeat_min = strategy.get('final_min_last_numbers')
            repeat_constraint_applied = (
                configured_repeat_min is not None and configured_repeat_min > 0
            )
            if repeat_constraint_applied:
                final_pool = _enforce_minimum_repeats(
                    final_pool,
                    selection_candidates,
                    self.statistics.get('last_numbers', set()),
                    configured_repeat_min,
                )
            repeat_profile['constraint_applied'] = repeat_constraint_applied
            repeat_profile['configured_minimum'] = configured_repeat_min
            numbers = sorted(num for num, _ in final_pool)
            variants = self._candidate_variants(pool_candidates, select_type, final_repeat_cap)
            shape_profile = _shape_profile(numbers, self.statistics.get('last_numbers', set()))
            accuracy_profile = _play_accuracy_profile(
                s_key,
                select_type,
                selected_mode,
                variants,
                target_hits=strategy.get('target_hits'),
            )

            results[s_key] = {
                'desc': config['desc'],
                'pick': config['pick'],
                'numbers': numbers,
                'shape_profile': shape_profile,
                'repeat_profile': repeat_profile,
                'accuracy_profile': accuracy_profile,
                'candidates': pool_candidates[:10],
                'prize_hit_thresholds': _prize_tier_thresholds(s_key),
                'hit_rate_priority_thresholds': _hit_rate_priority_thresholds(s_key),
                'target_hits': strategy.get('target_hits'),
                'chain_objective': strategy.get('chain_objective'),
                'chain_audit_rounds': strategy.get('chain_audit_rounds'),
                'strategy_id': strategy['strategy_id'],
                'prediction_mode': strategy['prediction_mode'],
                'is_validated': strategy['is_validated'],
                'baseline_type': strategy.get('baseline_type', ''),
                'strategy_evidence': strategy.get('strategy_evidence'),
                'final_selection_mode': selected_mode,
                'warning': (
                    '' if strategy['is_validated']
                    else '近100期热度仅显示微弱历史优势，未通过显著性验证，不代表下期概率必然提高。'
                    if str(strategy.get('baseline_type', '')).startswith('single_hot')
                    else '公平单组基线：不使用未经验证的冷热、遗漏或趋势猜号。'
                ),
            }


        # 复式玩法（v9.2: 也按自己的策略独立验证）
        for fushi_key, fushi_cfg in FUSHI_CONFIG.items():
            strategy = _strategies_mod.resolve_play_strategy(fushi_key)
            numbers_field = fushi_cfg['numbers_field']
            pool_size = fushi_cfg['pool_size']
            base_pick = fushi_cfg['base_pick']

            # v9.2: _cfg.VERIFY_ONLY_MODE — 没有已验证策略时不输出号码
            if strategy is None:
                results[fushi_key] = {
                    numbers_field: [],
                    'core_numbers': [],
                    'total_combinations': 0,
                    'combinations': [],
                    'combo_pick': base_pick,
                    'pool_size': pool_size,
                    'desc': fushi_cfg['desc'],
                    'status': 'verification_pending',
                    'prediction_mode': 'not_verified',
                    'is_validated': False,
                    'warning': '当前没有通过验证的策略，本玩法暂不输出推荐号码。',
                }
                resolved_strategies[fushi_key] = {
                    'strategy_id': '',
                    'prediction_mode': 'not_verified',
                    'is_validated': False,
                }
                continue

            resolved_strategies[fushi_key] = {
                'strategy_id': strategy['strategy_id'],
                'feature_weights': strategy['feature_weights'],
                'model_weights': strategy['model_weights'],
                'window_size': strategy.get('window_size', KL8_DEFAULT_HISTORY),
                'repeat_direction': strategy.get('repeat_direction', 'neutral'),
                'repeat_avoid_score': strategy.get('repeat_avoid_score', 0.10),
                'repeat_non_avoid_score': strategy.get('repeat_non_avoid_score', 0.85),
                'repeat_follow_score': strategy.get('repeat_follow_score', 0.90),
                'repeat_non_follow_score': strategy.get('repeat_non_follow_score', 0.50),
                'pool_diversify': strategy.get('pool_diversify', True),
                'pool_max_last_numbers': strategy.get('pool_max_last_numbers'),
                'frequency_mode': strategy.get('frequency_mode', 'mean_reversion'),
                'final_selection_mode': strategy.get('final_selection_mode', 'balanced'),
                'prediction_mode': strategy['prediction_mode'],
                'is_validated': strategy['is_validated'],
            }

            fu_pool_result = self.build_pool_by_strategy(strategy, pool_size=max(20, pool_size))
            fu_candidates = fu_pool_result.get('candidates', [])[:20]
            fushi_repeat_cap = (
                strategy.get('pool_max_last_numbers')
                if strategy.get('pool_max_last_numbers') is not None
                else _adaptive_repeat_cap(self.history_data, pool_size)
            )
            final_pool, selected_mode = _select_final_candidate_pool(
                fu_candidates,
                pool_size,
                self.statistics.get('last_numbers', set()),
                max_last_numbers=fushi_repeat_cap,
                selection_mode=strategy.get('final_selection_mode', 'balanced'),
            )
            core_numbers = sorted(num for num, _ in final_pool)
            variants = self._candidate_variants(fu_candidates, pool_size, fushi_repeat_cap)
            shape_profile = _shape_profile(core_numbers, self.statistics.get('last_numbers', set()))

            all_candidate_pools[fushi_key] = {
                f'top{pool_size}': core_numbers,
                'core_numbers': core_numbers,
                'strategy_id': strategy['strategy_id'],
            }

            if len(core_numbers) == pool_size:
                combo_list = [sorted(c) for c in combinations(core_numbers, base_pick)]
            else:
                combo_list = []

            results[fushi_key] = {
                numbers_field: core_numbers,
                'core_numbers': core_numbers,
                'shape_profile': shape_profile,
                'prize_hit_thresholds': _prize_tier_thresholds(fushi_key),
                'hit_rate_priority_thresholds': _hit_rate_priority_thresholds(fushi_key),
                'total_combinations': len(combo_list),
                'combinations': combo_list,
                'combo_pick': base_pick,
                'pool_size': pool_size,
                'desc': fushi_cfg['desc'],
                'strategy_id': strategy['strategy_id'],
                'prediction_mode': strategy['prediction_mode'],
                'is_validated': strategy['is_validated'],
                'baseline_type': strategy.get('baseline_type', ''),
                'final_selection_mode': selected_mode,
                'warning': '' if strategy['is_validated']
                    else '公平单组基线：不使用未经验证的冷热、遗漏或趋势猜号。',
            }

        # v9.1: 本期变化对比
        change_info = _compute_prediction_changes(results, last_snapshot, all_candidate_pools)
        results['change_info'] = change_info

        # v9: 保存 resolved_strategies 到 results，以便 _save_prediction_snapshot 使用
        results['resolved_strategies'] = resolved_strategies

        latest_issue = self.history_data[0]['issue'] if self.history_data else ''
        target_issue = _compute_next_issue(latest_issue, self.history_data) if latest_issue else ''
        results['based_on_issue'] = latest_issue
        results['target_issue'] = target_issue
        results['prediction_generated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')

        recent = self.history_data[:10] if self.history_data else []
        results['recent_results'] = [
            {'issue': r['issue'], 'numbers': r['numbers'], 'date': r['date']}
            for r in recent
        ]

        # v9.2: 状态分三种: validated / verification_pending / no_data
        # 判断整体预测模式
        all_modes = [results.get(f'select_{st}', {}).get('prediction_mode', '') for st in SELECT_TYPES]
        all_modes.extend(results.get(key, {}).get('prediction_mode', '') for key in FUSHI_PLAY_KEYS)

        if any(m == 'validated' for m in all_modes):
            overall_status = 'validated'
        elif any(m == 'not_verified' for m in all_modes):
            overall_status = 'verification_pending'
        elif any(m == 'reference_unvalidated' for m in all_modes):
            overall_status = 'reference_unvalidated'  # 仅在allow_reference时出现
        else:
            overall_status = 'no_data'

        # v9.2: 补数进度信息
        from src.kl8.fetch import count_valid_history_periods, KL8_BACKFILL_MIN_PERIODS
        current_periods = count_valid_history_periods()
        backfill_target = KL8_BACKFILL_MIN_PERIODS

        stats = self.statistics
        recent_performance = _build_recent_settlement_performance()
        results['statistics'] = {
            'total_periods': stats.get('total_periods', 0),
            'min_prediction_periods': KL8_MIN_PREDICTION_PERIODS,
            'based_on_issue': latest_issue,
            'target_issue': target_issue,
            'prediction_generated_at': results['prediction_generated_at'],
            'expected_freq': round(stats.get('expected_freq', 2), 2),
            'expected_gap': round(stats.get('expected_gap', 1), 1),
            'last_numbers': sorted(list(stats.get('last_numbers', set()))),
            'shape_targets': {
                str(pick): _shape_targets(pick)
                for pick in list(SELECT_TYPES) + [
                    cfg['pool_size'] for cfg in FUSHI_CONFIG.values()
                ]
            },
            'version': KL8_PREDICTOR_VERSION,
            'feature_config': FEATURE_CONFIG,
            'active_feature_weights': get_active_feature_weights(),
            'model_config': MODEL_CONFIG,
            'active_model_weights': get_active_model_weights(),
            'active_strategies': _cfg.ACTIVE_STRATEGIES,
            'reference_strategy': REFERENCE_STRATEGY,
            'candidate_strategies': CANDIDATE_STRATEGIES,
            'is_prediction_ready': prediction_ready,
            'signal_status': overall_status,
            'verify_only_mode': _cfg.VERIFY_ONLY_MODE,
            'fairness_disclaimer': (
                '快乐8为公平均匀摇奖(80选20)。任意选号的命中数期望恒为 pick_n×0.25，'
                '与选哪些号无关——任何"预测"都无法系统性提高命中率。未通过验证的参考号码'
                '与随机选号期望命中完全相同，仅供参考，不预示中奖，请理性购彩。'
            ),
            'backfill_progress': {
                'current_periods': current_periods,
                'target_periods': backfill_target,
                'progress_pct': round(min(100, current_periods / backfill_target * 100), 1),
                'is_complete': current_periods >= backfill_target,
            },
            'recent_settlement_performance': recent_performance,
            'strategy_health': _build_strategy_health(recent_performance),
            'note': (
                '当前启用策略已通过回测验证。'
                if overall_status == 'validated'
                else '验证中：历史数据不足或尚无通过验证的策略，本玩法暂不输出推荐号码。'
                if overall_status == 'verification_pending'
                else '参考号码：基于频率/间隔等启发式，未通过回测验证。快乐8为公平摇奖，'
                '此类号码与随机选号的期望命中完全相同，仅供参考、不预示中奖。'
                if overall_status == 'reference_unvalidated'
                else '历史数据不足，无法进行预测。'
            ),
        }

        # v9.1: ranking 使用 select_5 的候选池（展示排名最完整的池）
        ranking_pool = all_candidate_pools.get('select_5', {}).get('candidates', [])
        results['ranking'] = [
            {
                'num': num,
                'ranking_score': score,
                'score_type': 'candidate_pool_vote',
                'is_probability': False,
            }
            for num, score in ranking_pool
        ]

        results['all_candidate_pools'] = all_candidate_pools  # v9.1: 保存各玩法候选池详情

        results['using_simulated_data'] = self.using_simulated_data

        # 数据完整性信息
        if self.history_data:
            integrity = check_data_integrity(self.history_data)
            results['data_integrity'] = integrity

        snapshot_name = self._save_prediction_snapshot(results)
        if snapshot_name:
            results['snapshot_file'] = snapshot_name
            select6_numbers = results.get('select_6', {}).get('numbers', [])
            if len(select6_numbers) == 6:
                results['select_6_recalculation_chain'] = self.generate_exclude_recalculation_chain(
                    'select_6',
                    select6_numbers,
                    source_snapshot_id=Path(snapshot_name).stem.replace('snapshot_', '', 1),
                    source_version=KL8_PREDICTOR_VERSION,
                )

        return results


_analyzer_instance = None


def get_kl8_analyzer() -> KL8Analyzer:
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = KL8Analyzer()
    return _analyzer_instance


def build_candidate_pool(analyzer: Optional[KL8Analyzer] = None) -> Tuple[Dict, Dict]:
    """统一候选池入口函数（v9遗留，仅回测使用）

    v9.1: 线上预测已改为各玩法独立生成候选池，此函数仅供回测管道使用
    """
    analyzer = analyzer or get_kl8_analyzer()
    return analyzer.build_candidate_pool()


