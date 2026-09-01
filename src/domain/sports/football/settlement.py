# -*- coding: utf-8 -*-
"""赛果判定、命中统计、结算质量与 ML 融合资格。

纯计算——**历史库的读写、网络抓取与调度留在 `src/football/result_sync.py`**。
时间解析的"当前年"由调用方注入（判据 16），不注入的话黄金跨年就红。

**football 的「预测 → 赛果回填 → 校准」这条链是接上的**——线上 239 条历史
里 234 条 `settled=True`、有实际比分。这与北单不同：北单那条从投产起就没接上
（500 条 `settled` 全是 False，交接文档 §四）。所以这里的判定逻辑是活的。
"""

import hashlib
import json
import logging
import math
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger('domain.football.settlement')

PRODUCTION_MODEL_VERSION = 'football-v2026.08.20-audited-upset-gated-11'
ACTIONABLE_MIN_PROBABILITY = 0.65
ACTIONABLE_MIN_MARGIN = 0.10
ACTIONABLE_POLICY_VERSION = 'selective-1x2-v4-accuracy-first'
SYNC_INTERVAL_SECONDS = 7200        # 赛后回填，两小时一轮
TIME_LAYER_INTERVAL_SECONDS = 600   # 时间分层扫描，十分钟一轮


def normalize_1x2_probs(probs: Dict[str, float]) -> Dict[str, float]:
    """Normalize 1X2 probability keys to H/D/A."""
    if not probs:
        return {}

    normalized = {
        'H': probs.get('H', probs.get('home', 0.0)),
        'D': probs.get('D', probs.get('draw', 0.0)),
        'A': probs.get('A', probs.get('away', 0.0)),
    }
    total = sum(normalized.values())
    if total > 0:
        normalized = {key: value / total for key, value in normalized.items()}
    return normalized


def calculate_logloss(probs: Dict[str, float], actual_result: str) -> float:
    """
    计算 LogLoss（对数损失）
    
    参数：
        probs: 预测概率 {'H': 0.48, 'D': 0.27, 'A': 0.25}
        actual_result: 实际结果 'H', 'D', 或 'A'
    
    返回：
        LogLoss 值，越小越好
    """
    if actual_result not in ['H', 'D', 'A']:
        return float('nan')

    probs = normalize_1x2_probs(probs)
    
    p = probs.get(actual_result, 0.0)
    p = max(min(p, 1 - 1e-15), 1e-15)  # 防止 log(0)
    return -math.log(p)


def calculate_brier_score(probs: Dict[str, float], actual_result: str) -> float:
    """
    计算 Brier Score（布瑞尔分数）
    
    参数：
        probs: 预测概率 {'H': 0.48, 'D': 0.27, 'A': 0.25}
        actual_result: 实际结果 'H', 'D', 或 'A'
    
    返回：
        Brier Score 值，越小越好
    """
    if actual_result not in ['H', 'D', 'A']:
        return float('nan')

    probs = normalize_1x2_probs(probs)
    
    # 创建真实标签向量
    true_label = {'H': 0.0, 'D': 0.0, 'A': 0.0}
    true_label[actual_result] = 1.0
    
    # 计算 Brier Score
    score = 0.0
    for key in ['H', 'D', 'A']:
        score += (probs.get(key, 0.0) - true_label[key]) ** 2
    
    return score


def calculate_hit(probs: Dict[str, float], actual_result: str) -> bool:
    """
    判断是否命中（预测概率最高的结果是否等于实际结果）
    
    参数：
        probs: 预测概率 {'H': 0.48, 'D': 0.27, 'A': 0.25}
        actual_result: 实际结果 'H', 'D', 或 'A'
    
    返回：
        True 如果命中，False 否则
    """
    if actual_result not in ['H', 'D', 'A']:
        return None

    probs = normalize_1x2_probs(probs)
    
    # 找到概率最高的结果
    max_prob = -1
    predicted = None
    for key in ['H', 'D', 'A']:
        p = probs.get(key, 0.0)
        if p > max_prob:
            max_prob = p
            predicted = key
    
    return predicted == actual_result


def _score_to_result(score: str) -> Optional[str]:
    try:
        home_goals, away_goals = map(int, str(score).split('-'))
    except Exception:
        return None
    if home_goals > away_goals:
        return 'H'
    if home_goals < away_goals:
        return 'A'
    return 'D'


def _parse_score_result(score_match) -> Optional[Dict]:
    """解析比分匹配结果"""
    home_goals = int(score_match.group(1))
    away_goals = int(score_match.group(2))
    return _parse_score_string(f"{home_goals}-{away_goals}")


def _parse_score_string(score_str: str) -> Optional[Dict]:
    """解析比分字符串"""
    try:
        parts = score_str.split('-')
        if len(parts) != 2:
            return None
        
        home_goals = int(parts[0])
        away_goals = int(parts[1])
        
        if home_goals > away_goals:
            result = 'H'
        elif home_goals < away_goals:
            result = 'A'
        else:
            result = 'D'
        
        return {'score': score_str, 'result': result}
    except:
        return None


def _extract_score_text(raw: str) -> Optional[str]:
    """将页面比分文本规范为 home-away 格式，未开赛返回 None"""
    text = (raw or '').strip().replace('：', ':')
    if not text or text.upper() == 'VS':
        return None

    text = re.sub(r'\s+', '', text)
    if ':' in text:
        home_goals, away_goals = text.split(':', 1)
    elif '-' in text:
        home_goals, away_goals = text.split('-', 1)
    else:
        return None

    if not (home_goals.isdigit() and away_goals.isdigit()):
        return None

    home_goals = int(home_goals)
    away_goals = int(away_goals)
    if home_goals > 15 or away_goals > 15:
        return None

    return f"{home_goals}-{away_goals}"


def _parse_shuju_score(html: str, match_id: str) -> Optional[str]:
    """从 odds.500.com 赛事数据页解析终场比分"""
    patterns = [
        rf'shuju-{re.escape(match_id)}\.shtml[^>]*>.*?<em class="l">[^<]*</em><span class="gray">([^<]+)</span><em class="r">[^<]*</em>',
        rf'<em class="l">[^<]*</em><span class="gray">([^<]+)</span><em class="r">[^<]*</em>',
    ]
    for pat in patterns:
        m = re.search(pat, html, re.DOTALL)
        if m:
            score = _extract_score_text(m.group(1))
            if score:
                return score
    return None


def _parse_live_row_final_score(row: str) -> Optional[str]:
    """
    从 live 表格行解析全场比分。

    live.500.com 列结构：
    - <div class="pk"> 中 clt1 / clt3 = 全场比分（如 1-1）
    - 其后 class="red" 的 td = 半场比分（如 0-1），不能当作终场
    """
    pk_m = re.search(
        r'<div class="pk">.*?class="clt1"[^>]*>\s*(\d+)\s*</a>.*?class="clt3"[^>]*>\s*(\d+)\s*</a>',
        row,
        re.DOTALL,
    )
    if pk_m:
        score = _extract_score_text(f"{pk_m.group(1)}-{pk_m.group(2)}")
        if score:
            return score
    return None


def _parse_live_row_score(row: str, home: str, away: str) -> Optional[str]:
    """从 live.500.com 单行比赛记录提取终场比分"""
    if home not in row or away not in row:
        return None

    score = _parse_live_row_final_score(row)
    if score:
        return score

    fid_m = re.search(r'fid="(\d+)"', row)
    if fid_m:
        # 行内已有 fid，直接解析比分列，避免重复请求
        return None

    home_idx = row.find(home)
    away_idx = row.find(away)
    if home_idx < 0 or away_idx < 0:
        return None

    start = min(home_idx, away_idx)
    end = max(home_idx, away_idx) + max(len(home), len(away))
    segment = row[start:end]

    for pat in (
        r'>(\d{1,2})\s*[-:：]\s*(\d{1,2})<',
        r'(\d{1,2})\s*[-:：]\s*(\d{1,2})',
    ):
        m = re.search(pat, segment)
        if m:
            score = _extract_score_text(f"{m.group(1)}-{m.group(2)}")
            if score:
                return score
    return None


def _is_valid_match_id(match_id: str) -> bool:
    """仅对 500.com 数字型 fid 尝试抓取赛果"""
    return bool(match_id) and str(match_id).isdigit()


def _calibration_sample_weight(record: Dict, assess_record_quality=None) -> float:
    """样本在校准里的权重。

    **`assess_record_quality` 由调用方注入**（判据 16）：它住在适配层的
    `sample_quality` 模块里。不注入时走下面那条按来源分档的兜底——
    迁移前那条 `from .sample_quality import` 是延迟 import，搬进领域包后
    解析不到会**静默落到兜底**，双跑差分抓到了（旧 0.0 / 新 0.7）。
    """
    if record.get('exclude_from_calibration'):
        return 0.0
    try:
        if assess_record_quality is None:
            raise ImportError('未注入 assess_record_quality')
        quality = assess_record_quality(record)
        return max(0.0, min(1.0, float(quality.get('calibration_weight', 0.0))))
    except Exception:
        result_quality = record.get('result_quality') or {}
        if result_quality.get('grade') in {'reject', 'low'}:
            return 0.0
        source = result_quality.get('source')
        if source == 'live_fid':
            return 1.0
        if source == 'live_team':
            return 0.85
        if source == 'shuju':
            return 0.60
        return 0.70


def _is_result_quality_usable(record: Dict, min_grade: str = 'medium') -> bool:
    rank = {'reject': 0, 'low': 1, 'medium': 2, 'high': 3}
    quality = record.get('result_quality') or {}
    return rank.get(quality.get('grade'), 0) >= rank.get(min_grade, 2)


def fuse_probabilities(base_probs: Dict[str, float], ml_probs: Dict[str, float], 
                      ml_weight: float = 0.05) -> Dict[str, float]:
    """
    融合基础模型和 ML 模型的概率
    
    参数：
        base_probs: 基础模型概率 {'H': 0.48, 'D': 0.27, 'A': 0.25}
        ml_probs: ML 模型概率 {'H': 0.45, 'D': 0.29, 'A': 0.26}
        ml_weight: ML 模型权重（默认0.05）
    
    返回：
        融合后的概率（已归一化）
    """
    fused = {}
    total = 0.0
    
    for key in ['H', 'D', 'A']:
        fused[key] = (1 - ml_weight) * base_probs.get(key, 0.0) + ml_weight * ml_probs.get(key, 0.0)
        total += fused[key]
    
    # 归一化
    if total > 0:
        for key in ['H', 'D', 'A']:
            fused[key] /= total
    
    return fused


def evaluate_ml_prediction(record: Dict) -> Dict:
    """
    评估 ML 模型预测结果
    
    参数：
        record: 预测记录
    
    返回：
        评估结果字典
    """
    evaluation = {}
    
    actual_result = record.get('actual_result')
    if actual_result not in ['H', 'D', 'A']:
        return evaluation
    
    # 基础模型评估
    base_1x2 = record.get('base_1x2')
    if base_1x2:
        evaluation['base_1x2_logloss'] = calculate_logloss(base_1x2, actual_result)
        evaluation['base_1x2_brier'] = calculate_brier_score(base_1x2, actual_result)
        evaluation['base_1x2_hit'] = calculate_hit(base_1x2, actual_result)
    
    # ML 模型评估
    ml_1x2 = record.get('ml_1x2')
    if ml_1x2 and record.get('ml_available', False):
        evaluation['ml_1x2_logloss'] = calculate_logloss(ml_1x2, actual_result)
        evaluation['ml_1x2_brier'] = calculate_brier_score(ml_1x2, actual_result)
        evaluation['ml_1x2_hit'] = calculate_hit(ml_1x2, actual_result)
        
        # 模拟融合评估（5% ML权重）
        if base_1x2:
            fused_5pct = fuse_probabilities(base_1x2, ml_1x2, ml_weight=0.05)
            evaluation['fused_5pct_logloss'] = calculate_logloss(fused_5pct, actual_result)
            evaluation['fused_5pct_brier'] = calculate_brier_score(fused_5pct, actual_result)
            
            # 模拟融合评估（10% ML权重）
            fused_10pct = fuse_probabilities(base_1x2, ml_1x2, ml_weight=0.10)
            evaluation['fused_10pct_logloss'] = calculate_logloss(fused_10pct, actual_result)
            evaluation['fused_10pct_brier'] = calculate_brier_score(fused_10pct, actual_result)
    
    return evaluation


def time_layer_weight(time_layer: str) -> float:
    """Information weight for prediction snapshots at different pre-match layers."""
    weights = {
        'T-24h': 0.35,
        'T-6h': 0.55,
        'T-1h': 0.75,
        'T-15min': 0.90,
        'final': 1.00,
    }
    return weights.get(time_layer, 0.50)


def check_ml_fusion_eligibility(ml_stats: Dict, test_set_samples: int = 0) -> Dict:
    """
    检查 ML 模型是否满足参与正式融合的门槛
    
    参数：
        ml_stats: ML 评估统计（来自 get_ml_evaluation_stats）
        test_set_samples: 测试集样本数
    
    返回：
        包含是否合格及原因的字典
    """
    overall = ml_stats.get('overall', {})
    shadow_samples = overall.get('sample_count', 0)
    
    conditions = {
        'test_set_samples': {
            'passed': test_set_samples >= 200,
            'actual': test_set_samples,
            'required': 200,
            'reason': '严格样本外测试集 >= 200 场',
            'required_for_fusion': True,
        },
        'shadow_samples': {
            'passed': shadow_samples >= 100,
            'actual': shadow_samples,
            'required': 100,
            'reason': '影子实盘样本 >= 100 场',
            'required_for_fusion': True,
        },
        'ml_logloss_better': {
            'passed': False,
            'actual': None,
            'required': None,
            'reason': 'ML LogLoss < 基础模型 LogLoss',
            'required_for_fusion': False,
        },
        'ml_brier_not_worse': {
            'passed': False,
            'actual': None,
            'required': None,
            'reason': 'ML Brier Score <= 基础模型 Brier Score',
            'required_for_fusion': False,
        },
        'fused_5pct_logloss_better': {
            'passed': False,
            'actual': None,
            'required': None,
            'reason': '5% ML 融合后的 LogLoss < 基础模型 LogLoss',
            'required_for_fusion': True,
        },
        'fused_5pct_brier_not_worse': {
            'passed': False,
            'actual': None,
            'required': None,
            'reason': '5% ML 融合后的 Brier Score 不变差',
            'required_for_fusion': True,
        },
    }
    
    # Raw-ML metrics remain diagnostic, but even the initial 5% blend must
    # improve LogLoss without worsening Brier before production participation.
    base_logloss = overall.get('base_1x2_logloss')
    base_brier = overall.get('base_1x2_brier')
    ml_logloss = overall.get('ml_1x2_logloss')
    ml_brier = overall.get('ml_1x2_brier')
    fused_5pct_logloss = overall.get('fused_5pct_logloss')
    fused_5pct_brier = overall.get('fused_5pct_brier')
    
    if base_logloss is not None and ml_logloss is not None:
        conditions['ml_logloss_better']['passed'] = ml_logloss < base_logloss
        conditions['ml_logloss_better']['actual'] = f"{ml_logloss:.4f} vs {base_logloss:.4f}"
    
    if base_brier is not None and ml_brier is not None:
        conditions['ml_brier_not_worse']['passed'] = ml_brier <= base_brier
        conditions['ml_brier_not_worse']['actual'] = f"{ml_brier:.4f} vs {base_brier:.4f}"
    
    if base_logloss is not None and fused_5pct_logloss is not None:
        conditions['fused_5pct_logloss_better']['passed'] = fused_5pct_logloss < base_logloss
        conditions['fused_5pct_logloss_better']['actual'] = f"{fused_5pct_logloss:.4f} vs {base_logloss:.4f}"
    
    if base_brier is not None and fused_5pct_brier is not None:
        conditions['fused_5pct_brier_not_worse']['passed'] = fused_5pct_brier <= base_brier
        conditions['fused_5pct_brier_not_worse']['actual'] = f"{fused_5pct_brier:.4f} vs {base_brier:.4f}"
    
    eligible = all(
        cond['passed']
        for cond in conditions.values()
        if cond.get('required_for_fusion')
    )
    metrics_passed = (
        conditions['fused_5pct_logloss_better']['passed']
        and conditions['fused_5pct_brier_not_worse']['passed']
    )
    
    return {
        'eligible': eligible,
        'metrics_passed': metrics_passed,
        'conditions': conditions,
        'test_set_samples': test_set_samples,
        'shadow_samples': shadow_samples,
        'stats': overall,
    }


def get_ml_fusion_weight(eligible: bool, shadow_samples: int, 
                        current_weight: float = 0.0) -> float:
    """
    根据资格和样本数确定 ML 融合权重
    
    参数：
        eligible: 是否满足融合门槛
        shadow_samples: 影子实盘样本数
        current_weight: 当前权重
    
    返回：
        建议的 ML 融合权重
    """
    if not eligible:
        return 0.0
    
    # 根据样本数逐步提升权重
    max_weight = 0.15
    
    if shadow_samples >= 500:
        # 500+ 场可以考虑更高权重，但不超过 0.15
        if current_weight < 0.10:
            return min(0.10, max_weight)
        elif current_weight < 0.15:
            return min(0.15, max_weight)
        return current_weight
    elif shadow_samples >= 300:
        # 300-500 场，最高 0.10
        return min(0.10, max_weight)
    elif shadow_samples >= 45:
        # 45-300 场，初始权重 0.05
        return min(0.05, max_weight)
    else:
        return 0.0


def _prediction_decision_snapshot(predicted_1x2: Dict[str, float]) -> Dict:
    """Freeze the pre-match rule used to measure selective recommendations."""
    probs = normalize_1x2_probs(predicted_1x2)
    ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    top_probability = ranked[0][1] if ranked else 0.0
    margin = top_probability - ranked[1][1] if len(ranked) > 1 else 0.0
    return {
        'policy_version': ACTIONABLE_POLICY_VERSION,
        'eligible': top_probability >= ACTIONABLE_MIN_PROBABILITY and margin >= ACTIONABLE_MIN_MARGIN,
        'prediction': ranked[0][0] if ranked else None,
        'top_probability': round(top_probability, 6),
        'margin': round(margin, 6),
        'min_probability': ACTIONABLE_MIN_PROBABILITY,
        'min_margin': ACTIONABLE_MIN_MARGIN,
    }


def _audited_decision_snapshot(
    predicted_1x2: Dict[str, float],
    professional_snapshot: Dict = None,
) -> Dict:
    """Prefer the persisted match/league gate over the legacy global rule."""
    decision = _prediction_decision_snapshot(predicted_1x2)
    spf = (((professional_snapshot or {}).get('accuracy_gate') or {}).get('spf') or {})
    if not spf:
        return decision
    candidate = {'胜': 'H', '平': 'D', '负': 'A'}.get(
        spf.get('candidate'), spf.get('candidate'),
    )
    decision.update({
        'eligible': spf.get('selected') is True,
        'prediction': candidate,
        'top_probability': spf.get('probability'),
        'market_probability': spf.get('market_probability'),
        'margin': spf.get('margin'),
        'market_margin': spf.get('market_margin'),
        'min_probability': spf.get('minimum_probability'),
        'threshold_scope': spf.get('threshold_scope'),
        'validation_status': spf.get('validation_status'),
        'reasons': list(spf.get('reasons') or []),
        'policy_version': 'league-validated-spf-v1',
    })
    return decision


def _prediction_content_sig(predicted_scores, predicted_1x2, asian, total_line,
                            odds_data, predicted_half_full, model_version,
                            professional_snapshot=None, *,
                            lottery_handicap=None, predicted_rqspf=None):
    """预测的「有意义内容」签名，用于跳过无变化的重复写入。

    只覆盖影响预测结果的字段，刻意排除 updated_at 等时间戳——否则缓存命中时
    每次内容相同却因时间戳不同而反复写库（整表重写风暴的根源之一）。
    """
    try:
        payload = json.dumps(
            [predicted_scores, predicted_1x2, asian, total_line,
             odds_data, predicted_half_full, model_version, professional_snapshot,
             lottery_handicap, predicted_rqspf],
            ensure_ascii=False, sort_keys=True, default=str,
        )
    except Exception:
        # 任意不可序列化内容都视作「已变化」，从而照常写入，绝不吞掉真实更新。
        return None
    return hashlib.md5(payload.encode('utf-8')).hexdigest()


def _parse_match_datetime(match_time: str, now: datetime = None) -> Optional[datetime]:
    """解析比赛时间，兼容 MM-DD HH:MM 与完整日期格式

    **`now` 由调用方注入**（判据 16）：不带年的时间串要补"当前年"，
    而且跨年时还要把 12 月/1 月的边界修正回来——那是时钟依赖。
    不注入的话黄金**跨年就红**。
    """
    if not match_time:
        return None

    now = now or datetime.now()
    text = str(match_time).strip()
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M'):
        try:
            match_dt = datetime.strptime(text, fmt)
            return match_dt
        except ValueError:
            continue
    try:
        match_dt = datetime.strptime(f"{now.year}-{text}", '%Y-%m-%d %H:%M')
        if now.month == 12 and match_dt.month == 1:
            match_dt = match_dt.replace(year=now.year + 1)
        elif now.month == 1 and match_dt.month == 12:
            match_dt = match_dt.replace(year=now.year - 1)
        return match_dt
    except ValueError:
        pass
    return None


def _is_match_settle_due(match_time: str, minutes: int = 180, now: datetime = None) -> bool:
    """Return True only after kickoff plus the settlement wait window."""
    match_dt = _parse_match_datetime(match_time, now)
    if not match_dt:
        return False
    now = now or datetime.now()
    return now >= match_dt + timedelta(minutes=minutes)


def _assess_result_quality(record: Dict,
                           actual_score: str,
                           actual_result: str,
                           source: str = None,
                           actual_half_score: str = None,
                           now: datetime = None) -> Dict:
    """Assess whether a fetched result is trustworthy enough for calibration."""
    reasons = []
    score = 1.0

    extracted_result = _score_to_result(actual_score)
    if extracted_result is None:
        reasons.append('invalid_score_format')
        score -= 0.60
    elif extracted_result != actual_result:
        reasons.append('result_mismatch')
        score -= 0.45

    try:
        home_goals, away_goals = map(int, str(actual_score).split('-'))
        if home_goals > 12 or away_goals > 12:
            reasons.append('implausible_score')
            score -= 0.50
    except Exception:
        pass

    if not _is_match_settle_due(record.get('match_time'), minutes=180, now=now):
        reasons.append('not_settle_due')
        score -= 0.70

    source = source or 'unknown'
    if source == 'live_fid':
        score += 0.05
    elif source == 'live_team':
        score -= 0.05
    elif source == 'shuju':
        score -= 0.12
    else:
        reasons.append('unknown_source')
        score -= 0.15

    if actual_score in {'0-0', '1-1'} and source not in {'live_fid', 'live_team'}:
        reasons.append('low_information_score_without_live_source')
        score -= 0.18

    if actual_half_score:
        if _score_to_result(actual_half_score) is None:
            reasons.append('invalid_half_score_format')
            score -= 0.12

    if not record.get('match_id'):
        reasons.append('missing_match_id')
        score -= 0.10

    score = max(0.0, min(1.0, score))
    if score >= 0.82:
        grade = 'high'
    elif score >= 0.60:
        grade = 'medium'
    elif score >= 0.35:
        grade = 'low'
    else:
        grade = 'reject'

    return {
        'score': round(score, 3),
        'grade': grade,
        'source': source,
        'reasons': reasons,
        'usable_for_calibration': grade in {'high', 'medium'},
    }


def infer_time_layer(match_time_str: str, now: datetime = None) -> str:
    """
    根据比赛时间推断当前预测应该记录到哪个时间层
    
    参数：
        match_time_str: 比赛时间字符串（格式："06-14 09:00"）
    
    返回：
        时间层标识: 'T-24h', 'T-6h', 'T-1h', 'T-15min', 'final'
    """
    try:
        now = now or datetime.now()
        match_time = _parse_match_datetime(match_time_str, now)
        if not match_time:
            return 'final'
        
        diff_minutes = (match_time - now).total_seconds() / 60
        
        if diff_minutes >= 24 * 60:
            return 'T-24h'
        if diff_minutes >= 6 * 60:
            return 'T-6h'
        if diff_minutes >= 60:
            return 'T-1h'
        if diff_minutes >= 15:
            return 'T-15min'
        return 'final'
    except Exception as e:
        log.debug(f"推断时间层失败: {e}")
        return 'final'


def _live_query_dates(match_time: str, now: datetime = None) -> List[str]:
    """
    动态计算 live.500.com 的 ?e= 查询日期。

    竞彩赛果页规则（见 https://live.500.com/?e=YYYY-MM-DD ）：
    - e=某日 的页面展示该「开售日」对应场次的赛果
    - 开球时间为 06-13 03:00 的比赛，出现在 e=2026-06-12 页面（开球日前一天）
    - 因此优先查 kickoff_date - 1，再查当日及邻近日，最后兜底今天/昨天
    """
    now = now or datetime.now()
    today = now.date()
    candidates = []

    match_dt = _parse_match_datetime(match_time, now)
    if match_dt:
        kickoff_date = match_dt.date()
        candidates.extend([
            kickoff_date - timedelta(days=1),
            kickoff_date,
            kickoff_date - timedelta(days=2),
            kickoff_date + timedelta(days=1),
        ])

    candidates.extend([today, today - timedelta(days=1), today - timedelta(days=2)])

    seen = set()
    dates = []
    for day in candidates:
        key = day.strftime('%Y-%m-%d')
        if key not in seen:
            seen.add(key)
            dates.append(key)
    return dates
