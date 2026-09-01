# -*- coding: utf-8 -*-
"""快乐8记录读写：期号/奖表/持久化/快照结算读取/策略健康"""

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

log = setup_logger('kl8')
from . import config as _cfg

from .config import (
    FUSHI_CONFIG, FUSHI_PLAY_KEYS, KL8_ACTIVE_STRATEGIES_FILE, KL8_CONFLICT_QUEUE_FILE, KL8_FINAL_TEST_REPORT_FILE, KL8_PREDICTOR_VERSION, KL8_PRIZE_TABLE_FILE, KL8_STRATEGY_TRIAL_FILE, SELECT_PLAY_KEYS, SELECT_TYPES,
)
from .stats import (
    hypergeom_expected,
)

def normalize_record(record, keep_meta: bool = False) -> Optional[Dict]:
    """校验并标准化单条记录

    v5加固:
    - 坏JSON字符串 -> 返回None(不再抛异常)
    - 坏数据类型 -> 返回None
    - 20个号码必须唯一且在1-80范围
    - 期号不为空
    - keep_meta=True时保留source/fetched_at/checksum溯源字段
    """
    if not isinstance(record, dict):
        return None

    nums = record.get('numbers') or record.get('draw_numbers')

    if isinstance(nums, str):
        try:
            nums = json.loads(nums)
        except (json.JSONDecodeError, TypeError):
            return None

    if not nums:
        return None

    if not isinstance(nums, (list, tuple, set)):
        return None

    try:
        nums = sorted(int(x) for x in nums)
    except (ValueError, TypeError):
        return None

    if len(nums) != 20:
        return None
    if len(set(nums)) != 20:
        return None
    if any(n < 1 or n > 80 for n in nums):
        return None

    issue = str(record.get('issue', '')).strip()
    if not issue:
        return None

    result = {
        'issue': issue,
        'numbers': nums,
        'date': record.get('date') or record.get('draw_date', ''),
    }

    if keep_meta:
        result.update({
            'source': record.get('source', ''),
            'fetched_at': record.get('fetched_at', ''),
            'checksum': record.get('checksum', _checksum_numbers(nums)),
        })

    return result


def _checksum_numbers(nums: List[int]) -> str:
    """号码列表的短校验码"""
    s = json.dumps(sorted(nums), separators=(',', ':'))
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _compute_next_issue(latest_issue: str, history_data: List[Dict]) -> str:
    """从历史数据推导下一期期号（不再简单int+1）

    策略:
    1. 从历史数据中找相邻期号的差值模式
    2. 使用最常见的差值推算下一期
    3. 跨年(如2026365→2027001)和停开期间能正确处理
    """
    if not history_data:
        return f'next_after_{latest_issue}'

    # 收集相邻期号差值
    history_asc = sorted(history_data, key=lambda x: x['issue'])
    diffs = []
    start_idx = max(0, len(history_asc) - 21)
    for i in range(start_idx, len(history_asc) - 1):
        try:
            curr = int(history_asc[i + 1]['issue'])
            prev = int(history_asc[i]['issue'])
            diff = curr - prev
            if diff > 0:
                diffs.append(diff)
        except (ValueError, TypeError):
            continue

    if not diffs:
        return f'next_after_{latest_issue}'

    # 使用最常见的差值
    from collections import Counter as _Counter
    most_common_diff = _Counter(diffs).most_common(1)[0][0]

    try:
        latest_int = int(latest_issue)
        return str(latest_int + most_common_diff)
    except (ValueError, TypeError):
        return f'next_after_{latest_issue}'


def load_prize_table() -> Dict:
    """加载可配置奖金表

    格式: {
        "select_3": {"3": 25, "2": 5, "1": 0, "0": 0, "bet": 2},
        "select_4": {"4": 100, "3": 20, ...},
        ...
        "fu_shi_7": {"5": 10000, "4": 500, ...}
    }

    每个玩法包含: 命中档位奖金 + "bet"单注金额
    如果文件不存在，返回默认值
    """
    path = Path(KL8_PRIZE_TABLE_FILE)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception as e:
            log.warning(f'奖金表加载失败: {e}')

    # 默认奖金表（中国快乐8官方奖金，单位:元）
    # 来源: 福彩快乐8官方规则（2026年现行版）
    return {
        'select_1': {'1': 4.5, 'bet': 2},
        'select_2': {'2': 19, 'bet': 2},
        'select_3': {'3': 52, '2': 3, 'bet': 2},
        'select_4': {'4': 93, '3': 5, '2': 3, 'bet': 2},
        'select_5': {'5': 1000, '4': 20, '3': 3, 'bet': 2},
        'select_6': {'6': 2880, '5': 30, '4': 10, '3': 3, 'bet': 2},
        'select_7': {'7': 8500, '6': 300, '5': 30, '4': 4, '0': 2, 'bet': 2},
        'select_8': {'8': 50000, '7': 800, '6': 80, '5': 10, '4': 3, '0': 2, 'bet': 2},
        'select_9': {'9': 250000, '8': 2000, '7': 225, '6': 22, '5': 5, '4': 3, '0': 2, 'bet': 2},
        'select_10': {'10': 5000000, '9': 8000, '8': 720, '7': 80, '6': 5, '5': 3, '0': 2, 'bet': 2},
        'fu_shi_7': {'5': 1000, '4': 20, '3': 3, 'bet_per_combo': 2},
        'fu_shi_4': {'4': 93, '3': 5, '2': 3, 'bet_per_combo': 2},
        'fu_shi_10_11': {'prize_key': 'select_10', 'base_pick': 10, 'pool_size': 11, 'bet_per_combo': 2},
    }


def _strategy_fingerprint(strategy: Dict) -> str:
    """策略指纹 — 包含所有影响预测结果的字段

    v9.2扩展:
    - 新增: 候选池后处理配置(pool_diversify_enabled)和代码版本
    - 当前号码池分散化(_diversify_candidate_pool)是全局强制开启的
    - 如果后处理逻辑变了但策略ID没变，会导致回测和线上不一致
    - 因此把后处理配置和代码版本也纳入指纹
    """
    fp_data = {
        'feature_weights': strategy.get('feature_weights', {}),
        'model_weights': strategy.get('model_weights', {}),
        'window_size': strategy.get('window_size', 0),
        'repeat_direction': strategy.get('repeat_direction', 'neutral'),
        'repeat_avoid_score': strategy.get('repeat_avoid_score', 0.10),
        'repeat_non_avoid_score': strategy.get('repeat_non_avoid_score', 0.85),
        'repeat_follow_score': strategy.get('repeat_follow_score', 0.90),
        'repeat_non_follow_score': strategy.get('repeat_non_follow_score', 0.50),
        'pool_diversify_enabled': strategy.get('pool_diversify', True),
        'pool_max_last_numbers': strategy.get('pool_max_last_numbers'),
        'frequency_mode': strategy.get('frequency_mode', 'mean_reversion'),
        'final_selection_mode': strategy.get('final_selection_mode', 'balanced'),
        'final_max_last_numbers': strategy.get('final_max_last_numbers'),
        'final_min_last_numbers': strategy.get('final_min_last_numbers', 0),
        'chain_objective': strategy.get('chain_objective'),
        'chain_audit_rounds': strategy.get('chain_audit_rounds'),
        'target_hits': strategy.get('target_hits'),
        'code_version': KL8_PREDICTOR_VERSION,
    }
    return hashlib.sha256(
        json.dumps(fp_data, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()[:12]


def _resolved_strategies_fingerprint(strategies: Dict) -> str:
    """整份预测实际使用策略的稳定指纹，而不是只观察某一个玩法。"""
    play_fingerprints = {
        key: _strategy_fingerprint(strategy)
        for key, strategy in sorted((strategies or {}).items())
        if isinstance(strategy, dict)
    }
    if not play_fingerprints:
        return 'no_strategy'
    return hashlib.sha256(
        json.dumps(
            play_fingerprints,
            sort_keys=True,
            separators=(',', ':'),
        ).encode()
    ).hexdigest()[:12]


def _prediction_config_fingerprint() -> str:
    """所有会改变当前预测输出的运行时策略配置指纹。"""
    return hashlib.sha256(
        json.dumps(
            {
                'active_strategies': _cfg.ACTIVE_STRATEGIES,
                'reference_strategy': _cfg.REFERENCE_STRATEGY,
                'candidate_strategies': _cfg.CANDIDATE_STRATEGIES,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode()
    ).hexdigest()[:16]


def _persist_trial_results():
    """持久化策略试验结果 — 追加、去重、原子写入

    v9新增: _cfg.STRATEGY_TRIAL_RESULTS 不再只在内存，服务重启后能恢复
    """
    path = Path(KL8_STRATEGY_TRIAL_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 去重: strategy_id + play_type + tournament_round + tested_at 组合唯一
    unique_trials = []
    seen_keys = set()
    for trial in _cfg.STRATEGY_TRIAL_RESULTS:
        key = f"{trial.get('strategy_id', '')}_{trial.get('play_type', '')}_{trial.get('tournament_round', '')}_{trial.get('tested_at', '')}"
        if key not in seen_keys:
            seen_keys.add(key)
            unique_trials.append(trial)

    # 原子写入
    temp_path = path.with_suffix('.json.tmp')
    try:
        temp_path.write_text(
            json.dumps(unique_trials, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        temp_path.replace(path)
    except Exception as e:
        log.warning(f'持久化策略试验结果失败: {e}')
        if temp_path.exists():
            temp_path.unlink()


def _load_trial_results():
    """加载持久化的策略试验结果"""
    path = Path(KL8_STRATEGY_TRIAL_FILE)
    if not path.exists():
        return []

    try:
        trials = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(trials, list):
            return trials
    except Exception as e:
        log.warning(f'加载策略试验结果失败: {e}')

    return []


def _persist_active_strategies():
    """持久化已激活策略 — 服务启动时自动加载"""
    path = Path(KL8_ACTIVE_STRATEGIES_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 原子写入
    temp_path = path.with_suffix('.json.tmp')
    try:
        temp_path.write_text(
            json.dumps(_cfg.ACTIVE_STRATEGIES, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        temp_path.replace(path)
    except Exception as e:
        log.warning(f'持久化已激活策略失败: {e}')
        if temp_path.exists():
            temp_path.unlink()


def _load_active_strategies():
    """加载持久化的已激活策略"""
    path = Path(KL8_ACTIVE_STRATEGIES_FILE)
    if not path.exists():
        return None

    try:
        loaded = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(loaded, dict):
            return loaded
    except Exception as e:
        log.warning(f'加载已激活策略失败: {e}')

    return None


def _persist_final_test_report(report: Dict):
    """持久化最终测试报告 — 只允许写入一次

    v9新增: 最终测试结果锁定，不允许重复写入
    """
    path = Path(KL8_FINAL_TEST_REPORT_FILE)
    if path.exists():
        log.warning('最终测试报告已存在，不允许重复写入')
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        return True
    except Exception as e:
        log.warning(f'持久化最终测试报告失败: {e}')
        return False


_cfg.STRATEGY_TRIAL_RESULTS.extend(_load_trial_results())


loaded_strategies = _load_active_strategies()


if loaded_strategies:
    for play_type, strategy in loaded_strategies.items():
        if play_type in _cfg.ACTIVE_STRATEGIES:
            _cfg.ACTIVE_STRATEGIES[play_type] = strategy
    log.info(f'快乐8: 已从持久化文件加载{len(loaded_strategies)}个已激活策略')


def save_conflict_to_queue(conflict_info: Dict):
    """将数据冲突记录保存到审核队列（不自动覆盖，等待人工确认）"""
    path = Path(KL8_CONFLICT_QUEUE_FILE)
    queue = []
    if path.exists():
        try:
            queue = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            queue = []

    conflict_info['queued_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    queue.append(conflict_info)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding='utf-8')


def list_conflict_queue() -> List[Dict]:
    """查看冲突审核队列"""
    path = Path(KL8_CONFLICT_QUEUE_FILE)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return []


def check_data_integrity(data: List[Dict]) -> Dict:
    """检查历史数据完整性

    检查项:
    1. 数据是否按期号排序（读取后排序而非假设顺序）
    2. 期号连续性（缺失期号报告）
    3. 日期与期号一致性
    4. 号码范围和唯一性
    """
    if not data:
        return {'valid': False, 'error': '数据为空'}

    # 1. 确保排序（读取后排序，不假设顺序）
    data_sorted = sorted(data, key=lambda x: x['issue'], reverse=True)

    issues = [r['issue'] for r in data_sorted]
    total = len(issues)

    # 2. 期号连续性检查
    # 快乐8期号格式: 通常是年份+3位序号(如2026001~2026365)
    missing_issues = []
    issue_ints = []
    for issue in issues:
        try:
            issue_ints.append(int(issue))
        except (ValueError, TypeError):
            continue

    if issue_ints:
        issue_ints_sorted = sorted(issue_ints)
        for i in range(len(issue_ints_sorted) - 1):
            # 检查相邻期号差值
            diff = issue_ints_sorted[i + 1] - issue_ints_sorted[i]
            if diff > 1:
                # 报告缺失期号（但只报告小的gap，跨年的大gap可能正常）
                if diff <= 5:
                    for j in range(1, diff):
                        missing_issues.append(str(issue_ints_sorted[i] + j))

    # 3. 日期与期号一致性（简单检查: 同一天应该有相似期号前缀）
    date_issue_conflicts = []
    seen_dates = {}
    for r in data_sorted:
        date = r.get('date', '')
        issue = r['issue']
        if date:
            year_prefix = issue[:4] if len(issue) >= 4 else ''
            date_year = date[:4] if len(date) >= 4 else ''
            if year_prefix and date_year and year_prefix != date_year:
                date_issue_conflicts.append({
                    'issue': issue,
                    'date': date,
                    'reason': f'期号年份{year_prefix}与日期年份{date_year}不匹配',
                })

    return {
        'valid': True,
        'total_records': total,
        'latest_issue': issues[0] if issues else '',
        'earliest_issue': issues[-1] if issues else '',
        'missing_issues': missing_issues[:20],  # 只报告前20个缺失
        'missing_count': len(missing_issues),
        'date_issue_conflicts': date_issue_conflicts[:10],
        'conflict_count': len(date_issue_conflicts),
    }


def _load_last_snapshot() -> Optional[Dict]:
    """加载最近的正式预测快照（用于本期变化对比）

    v9.1新增: 读取最近一份非实验快照，用于与当前预测对比变化
    """
    snapshot_dir = Path(_cfg.KL8_SNAPSHOT_DIR)
    if not snapshot_dir.exists():
        return None

    # 按文件修改时间排序，找最近的非实验快照
    candidates = []
    for f in snapshot_dir.glob('snapshot_*.json'):
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
            if not data.get('is_experiment', False):
                candidates.append((f, data))
        except Exception:
            continue

    if not candidates:
        return None

    # 按predicted_at时间排序（而不是文件名UUID）
    candidates.sort(key=lambda x: x[1].get('predicted_at', ''), reverse=True)
    return candidates[0][1]


def _load_recent_settlements(limit: int = 100) -> List[Dict]:
    settlements_dir = Path(_cfg.KL8_SETTLEMENT_DIR)
    if not settlements_dir.exists():
        return []

    items = []
    for path in settlements_dir.glob('settlement_*.json'):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            sort_key = data.get('settled_at') or data.get('actual_issue') or ''
            items.append((sort_key, path.stat().st_mtime, data))
        except Exception:
            continue

    items.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [data for _, _, data in items[:limit]]


def _summarize_settlement_window(settlements: List[Dict], window_size: int) -> Dict:
    window = settlements[:window_size]
    play_stats = {}

    for select_type in SELECT_TYPES:
        play_type = f'select_{select_type}'
        rows = [
            s.get('prize_settlement', {}).get(play_type, {})
            for s in window
        ]
        rows = [row for row in rows if row.get('placed')]
        count = len(rows)
        total_hits = sum(int(row.get('hits', 0)) for row in rows)
        expected_hits = hypergeom_expected(select_type)
        avg_hits = total_hits / count if count else 0.0
        total_bet = sum(float(row.get('bet', 0)) for row in rows)
        total_prize = sum(float(row.get('prize', 0)) for row in rows)

        play_stats[play_type] = {
            'play_type': play_type,
            'label': f'选{select_type}',
            'settled_count': count,
            'avg_hits': round(avg_hits, 4),
            'random_expected_hits': round(expected_hits, 4),
            'hit_delta_vs_random': round(avg_hits - expected_hits, 4),
            'total_bet': round(total_bet, 2),
            'total_prize': round(total_prize, 2),
            'profit_roi': round((total_prize - total_bet) / total_bet, 4) if total_bet else 0.0,
        }

        # v9.6: 多注组合层面统计
        ms_rows = [
            s.get('multi_slip_settlement', {}).get(play_type, {})
            for s in window
        ]
        ms_rows = [row for row in ms_rows if row.get('placed')]
        if ms_rows:
            ms_count = len(ms_rows)
            avg_best = sum(int(r.get('best_hits', 0)) for r in ms_rows) / ms_count
            avg_total_bet = sum(float(r.get('total_bet', 0)) for r in ms_rows) / ms_count
            avg_total_prize = sum(float(r.get('total_prize', 0)) for r in ms_rows) / ms_count
            play_stats[play_type]['multi_slip'] = {
                'settled_count': ms_count,
                'avg_best_hits': round(avg_best, 4),
                'ge3_rate': round(sum(r.get('ge3', 0) for r in ms_rows) / ms_count, 4),
                'ge4_rate': round(sum(r.get('ge4', 0) for r in ms_rows) / ms_count, 4),
                'ge5_rate': round(sum(r.get('ge5', 0) for r in ms_rows) / ms_count, 4),
                'ge6_rate': round(sum(r.get('ge6', 0) for r in ms_rows) / ms_count, 4),
                'avg_bet': round(avg_total_bet, 2),
                'avg_prize': round(avg_total_prize, 2),
                'profit_roi': round(
                    (sum(float(r.get('total_prize', 0)) for r in ms_rows) -
                     sum(float(r.get('total_bet', 0)) for r in ms_rows)) /
                    sum(float(r.get('total_bet', 0)) for r in ms_rows), 4
                ) if ms_rows and sum(float(r.get('total_bet', 0)) for r in ms_rows) else 0.0,
            }

    for fushi_key, fushi_cfg in FUSHI_CONFIG.items():
        pool_size = fushi_cfg['pool_size']
        rows = [
            s.get('fushi_settlement', {}).get(fushi_key, {})
            for s in window
        ]
        rows = [row for row in rows if row.get('placed')]
        count = len(rows)
        total_hits = sum(int(row.get('pool_hits', 0)) for row in rows)
        expected_hits = hypergeom_expected(pool_size)
        avg_hits = total_hits / count if count else 0.0
        total_bet = sum(float(row.get('total_bet', 0)) for row in rows)
        total_prize = sum(float(row.get('total_prize', 0)) for row in rows)

        play_stats[fushi_key] = {
            'play_type': fushi_key,
            'label': fushi_cfg['desc'],
            'settled_count': count,
            'avg_hits': round(avg_hits, 4),
            'random_expected_hits': round(expected_hits, 4),
            'hit_delta_vs_random': round(avg_hits - expected_hits, 4),
            'total_bet': round(total_bet, 2),
            'total_prize': round(total_prize, 2),
            'profit_roi': round((total_prize - total_bet) / total_bet, 4) if total_bet else 0.0,
        }

    return {
        'window_size': window_size,
        'settled_count': len(window),
        'play_stats': play_stats,
    }


def _build_recent_settlement_performance(windows: Tuple[int, ...] = (30, 100)) -> Dict:
    max_window = max(windows) if windows else 100
    settlements = _load_recent_settlements(max_window)
    summaries = [
        _summarize_settlement_window(settlements, window)
        for window in windows
    ]
    return {
        'available_count': len(settlements),
        'windows': summaries,
        'note': '实际命中与随机理论期望对照；快乐8为公平摇奖，短期高低可能只是随机波动。',
    }


def _build_strategy_health(performance: Optional[Dict] = None) -> Dict:
    performance = performance or _build_recent_settlement_performance()
    windows = performance.get('windows', []) if isinstance(performance, dict) else []
    available_windows = [w for w in windows if w.get('settled_count', 0) > 0]
    window = (
        next((w for w in available_windows if int(w.get('window_size', 0)) == 30), None)
        or (available_windows[0] if available_windows else {})
    )
    play_stats = window.get('play_stats', {}) if isinstance(window, dict) else {}
    health_by_play = {}

    def make_label(play_type: str) -> str:
        if play_type.startswith('select_'):
            return f'选{play_type.split("_")[-1]}'
        return FUSHI_CONFIG.get(play_type, {}).get('desc', play_type)

    for play_type in list(SELECT_PLAY_KEYS) + list(FUSHI_PLAY_KEYS):
        strategy = _cfg.ACTIVE_STRATEGIES.get(play_type, {}) or {}
        strategy_id = strategy.get('strategy_id', '')
        is_validated = bool(strategy_id and strategy.get('is_validated', False))
        report = strategy.get('validation_report', {}) if isinstance(strategy.get('validation_report', {}), dict) else {}
        stat = play_stats.get(play_type, {})
        settled_count = int(stat.get('settled_count', 0) or 0)

        validation_lift = report.get('validation_lift')
        final_test_lift = report.get('final_test_lift')
        hit_delta = float(stat.get('hit_delta_vs_random', 0) or 0)
        roi = float(stat.get('profit_roi', 0) or 0)

        score = 50
        reasons = []

        if not strategy_id:
            health_by_play[play_type] = {
                'play_type': play_type,
                'label': make_label(play_type),
                'status': 'unverified',
                'status_label': '未验证',
                'score': 0,
                'settled_count': settled_count,
                'strategy_id': '',
                'reasons': ['当前玩法没有激活的已验证策略'],
            }
            continue

        if is_validated:
            score += 20
            reasons.append('策略已通过验证')
        else:
            score -= 10
            reasons.append('策略未标记为已验证')

        if isinstance(validation_lift, (int, float)):
            score += 10 if validation_lift > 0 else -8
            reasons.append(f'验证Lift={round(validation_lift, 4)}')
        if isinstance(final_test_lift, (int, float)):
            score += 5 if final_test_lift > 0 else -5
            reasons.append(f'封存Lift={round(final_test_lift, 4)}')

        if settled_count < 5:
            health_by_play[play_type] = {
                'play_type': play_type,
                'label': make_label(play_type),
                'status': 'pending',
                'status_label': '待结算',
                'score': max(0, min(100, score)),
                'settled_count': settled_count,
                'strategy_id': strategy_id,
                'validation_lift': validation_lift,
                'final_test_lift': final_test_lift,
                'hit_delta_vs_random': hit_delta,
                'profit_roi': roi,
                'reasons': reasons + ['结算样本少于5期，暂不判断实测表现'],
            }
            continue

        if settled_count >= 30:
            score += 5
        elif settled_count < 10:
            score -= 5

        if hit_delta >= 0.2:
            score += 15
            reasons.append(f'近期命中高于随机 {round(hit_delta, 2)}')
        elif hit_delta >= 0:
            score += 8
            reasons.append(f'近期命中略高于随机 {round(hit_delta, 2)}')
        elif hit_delta >= -0.2:
            score -= 5
            reasons.append(f'近期命中略低于随机 {round(hit_delta, 2)}')
        else:
            score -= 15
            reasons.append(f'近期命中低于随机 {round(hit_delta, 2)}')

        if roi > 0:
            score += 10
            reasons.append(f'近期ROI为正 {round(roi * 100, 1)}%')
        elif roi < -0.75:
            score -= 10
            reasons.append(f'近期ROI偏低 {round(roi * 100, 1)}%')

        score = max(0, min(100, round(score)))
        if score >= 75:
            status, status_label = 'healthy', '健康'
        elif score >= 55:
            status, status_label = 'watch', '观察'
        else:
            status, status_label = 'cool_down', '降温'

        health_by_play[play_type] = {
            'play_type': play_type,
            'label': make_label(play_type),
            'status': status,
            'status_label': status_label,
            'score': score,
            'settled_count': settled_count,
            'strategy_id': strategy_id,
            'validation_lift': validation_lift,
            'final_test_lift': final_test_lift,
            'hit_delta_vs_random': round(hit_delta, 4),
            'profit_roi': round(roi, 4),
            'reasons': reasons[:4],
        }

    return {
        'window_size': window.get('window_size', 0) if isinstance(window, dict) else 0,
        'settled_count': window.get('settled_count', 0) if isinstance(window, dict) else 0,
        'health_by_play': health_by_play,
        'note': '策略健康度只用于观察，不会自动启用、降级或替换策略。',
    }


def _compute_prediction_changes(
    current_results: Dict,
    last_snapshot: Optional[Dict],
    all_candidate_pools: Dict,
) -> Dict:
    """计算本期预测相较上期的变化

    v9.1新增:
    - 候选池变化数（Top20中替换了多少号码）
    - 各玩法推荐号码的替换详情
    - 变化原因说明
    - 如果变化为0，如实显示"候选池延续"

    不添加任何随机扰动，如实反映数据更新后的真实变化
    """
    if not last_snapshot:
        return {
            'has_previous': False,
            'previous_based_on': '',
            'previous_target_issue': '',
            'change_summary': '首次预测，无上期快照可对比',
        }

    prev_based_on = last_snapshot.get('based_on_issue', '')
    prev_target = last_snapshot.get('target_issue', '')

    current_based_on = current_results.get('based_on_issue', '')

    # ── 候选池变化（以 select_5 的 Top20 为基准）───
    current_top20 = set(all_candidate_pools.get('select_5', {}).get('top20', []))

    # 从上期快照中获取候选池
    prev_candidates = []
    if last_snapshot.get('ranking'):
        # ranking 字段保存了上期的候选池排名
        prev_candidates = [r['num'] for r in last_snapshot['ranking'][:20]]
    elif last_snapshot.get('select_5', []):
        # fallback: 从 select_5 的候选池字段获取
        sel5_data = last_snapshot.get('select_5', {})
        if isinstance(sel5_data, dict) and sel5_data.get('candidates'):
            prev_candidates = [c[0] if isinstance(c, (list, tuple)) else c for c in sel5_data['candidates'][:20]]
        elif isinstance(sel5_data, list):
            # 旧格式：select_5 直接是号码列表
            prev_candidates = sel5_data

    prev_top20_set = set(prev_candidates) if prev_candidates else set()
    pool_changed = len(current_top20 - prev_top20_set)
    pool_unchanged = len(current_top20 & prev_top20_set)

    # ── 各玩法推荐号码变化 ──
    play_changes = {}
    for play_type in SELECT_PLAY_KEYS:
        current_nums = set(current_results.get(play_type, {}).get('numbers', []))
        prev_nums = set()

        # 从快照中获取上期推荐号码
        prev_play_nums = last_snapshot.get(play_type, [])
        if isinstance(prev_play_nums, list):
            prev_nums = set(prev_play_nums)
        elif isinstance(prev_play_nums, dict) and prev_play_nums.get('numbers'):
            prev_nums = set(prev_play_nums['numbers'])

        added = sorted(current_nums - prev_nums)
        removed = sorted(prev_nums - current_nums)

        play_changes[play_type] = {
            'added': added,
            'removed': removed,
            'changed_count': len(added) + len(removed),
            'unchanged_count': len(current_nums & prev_nums),
        }

    # 复式玩法
    for fushi_key, fushi_cfg in FUSHI_CONFIG.items():
        field = fushi_cfg['numbers_field']
        current_fu = set(current_results.get(fushi_key, {}).get(field, current_results.get(fushi_key, {}).get('core_numbers', [])))
        prev_fu = set()
        prev_fu_nums = last_snapshot.get(fushi_key, [])
        if isinstance(prev_fu_nums, list):
            prev_fu = set(prev_fu_nums)
        elif isinstance(prev_fu_nums, dict):
            prev_fu = set(prev_fu_nums.get(field, prev_fu_nums.get('core_numbers', [])))

        play_changes[fushi_key] = {
            'added': sorted(current_fu - prev_fu),
            'removed': sorted(prev_fu - current_fu),
            'changed_count': len(current_fu - prev_fu) + len(prev_fu - current_fu),
            'unchanged_count': len(current_fu & prev_fu),
        }

    # ── 变化原因 ──
    if pool_changed == 0 and all(pc['changed_count'] == 0 for pc in play_changes.values()):
        change_reason = '候选池延续：当前数据更新后，前20排名未发生变化'
    else:
        reasons = []
        if pool_changed > 0:
            reasons.append(f'候选池 {pool_changed}/20 号码变化')
        # 检查是否有重号方向变化
        current_strategy = current_results.get('resolved_strategies', {}).get('select_5', {})
        prev_resolved = last_snapshot.get('resolved_strategies', {})
        prev_strategy = prev_resolved.get('select_5', {}) if isinstance(prev_resolved, dict) else {}

        if current_strategy.get('repeat_direction') != prev_strategy.get('repeat_direction'):
            reasons.append(f'重号方向: {prev_strategy.get("repeat_direction","?")}→{current_strategy.get("repeat_direction","?")}')
        if current_strategy.get('window_size') != prev_strategy.get('window_size'):
            reasons.append(f'窗口: {prev_strategy.get("window_size","?")}→{current_strategy.get("window_size","?")}')

        # 如果只是频率微调导致排名变化
        if not reasons or (pool_changed > 0 and pool_changed <= 3):
            reasons.append('数据更新后短窗口频率微调')

        change_reason = '；'.join(reasons)

    # ── 置信等级 ──
    overall_status = current_results.get('statistics', {}).get('signal_status', 'reference_unvalidated')
    confidence_level = '低' if overall_status == 'reference_unvalidated' else ('中' if overall_status == 'validated' else '无数据')
    strategy_status = '参考预测，未通过回测验证' if overall_status == 'reference_unvalidated' else (
        '已通过回测验证' if overall_status == 'validated' else '数据不足'
    )

    return {
        'has_previous': True,
        'previous_based_on': prev_based_on,
        'previous_target_issue': prev_target,
        'current_based_on': current_based_on,
        'pool_changes': {
            'total_pool_size': 20,
            'changed': pool_changed,
            'unchanged': pool_unchanged,
        },
        'play_changes': play_changes,
        'change_reason': change_reason,
        'confidence_level': confidence_level,
        'strategy_status': strategy_status,
    }


