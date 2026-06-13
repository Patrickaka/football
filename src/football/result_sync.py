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
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from threading import Thread


log = logging.getLogger('football')


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
HISTORY_FILE = os.path.join(DATA_DIR, 'prediction_history.json')


def infer_time_layer(match_time_str: str) -> str:
    """
    根据比赛时间推断当前预测应该记录到哪个时间层
    
    参数：
        match_time_str: 比赛时间字符串（格式："06-14 09:00"）
    
    返回：
        时间层标识: 'T-24h', 'T-6h', 'T-1h', 'T-15min', 'final'
    """
    try:
        now = datetime.now()
        for fmt in ['%m-%d %H:%M', '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M']:
            try:
                match_time = datetime.strptime(match_time_str, fmt)
                if match_time.year == 1900:
                    match_time = match_time.replace(year=now.year)
                break
            except ValueError:
                continue
        else:
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


class PredictionHistory:
    """预测历史记录管理器"""
    
    def __init__(self):
        self.records: List[Dict] = []
        self._load()
    
    def _load(self):
        """从文件加载记录"""
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    self.records = json.load(f)
                log.info(f"已加载 {len(self.records)} 条预测历史记录")
            except Exception as e:
                log.error(f"加载预测历史失败: {e}")
                self.records = []
    
    def _save(self):
        """保存记录到文件"""
        os.makedirs(DATA_DIR, exist_ok=True)
        try:
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.error(f"保存预测历史失败: {e}")
    
    def add_prediction(self, match_id: str, league: str, home: str, away: str,
                       match_time: str, predicted_scores: Dict[str, float],
                       predicted_1x2: Dict[str, float], asian: float = None,
                       total_line: float = None, odds_data: Dict = None):
        """
        添加预测记录
        
        参数：
            match_id: 比赛ID
            league: 联赛名称
            home: 主队名称
            away: 客队名称
            match_time: 比赛时间
            predicted_scores: 预测比分概率 {"1-1": 0.108, ...}
            predicted_1x2: 预测胜平负 {"home": 0.46, "draw": 0.27, "away": 0.27}
            asian: 亚盘让球
            total_line: 大小球盘口
            odds_data: 原始赔率数据（可选）
        """
        # 检查是否已存在
        for record in self.records:
            if record.get('match_id') == match_id:
                # 更新现有记录
                record.update({
                    'predicted_scores': predicted_scores,
                    'predicted_1x2': predicted_1x2,
                    'asian': asian,
                    'total_line': total_line,
                    'updated_at': datetime.now().isoformat(),
                    'odds_snapshot': odds_data,
                })
                
                # 更新对应时间层的预测
                layer = infer_time_layer(match_time)
                if 'time_layers' not in record:
                    record['time_layers'] = {}
                record['time_layers']['final'] = predicted_scores  # 始终更新最终预测
                # 只在该层为None时才更新（保留更早时间点的预测）
                if record['time_layers'].get(layer) is None:
                    record['time_layers'][layer] = predicted_scores
                
                self._save()
                return
        
        # 新增记录
        # 时间分层预测记录
        time_layers = {
            'T-24h': None,  # 赛前24小时预测
            'T-6h': None,   # 赛前6小时预测
            'T-1h': None,   # 赛前1小时预测
            'T-15min': None, # 赛前15分钟预测
            'final': predicted_scores,  # 最终预测
        }
        
        self.records.append({
            'match_id': match_id,
            'league': league,
            'home': home,
            'away': away,
            'match_time': match_time,
            'asian': asian,
            'total_line': total_line,
            'predicted_scores': predicted_scores,
            'predicted_1x2': predicted_1x2,
            'time_layers': time_layers,  # 新增：时间分层预测记录
            'actual_score': None,
            'actual_result': None,
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
        })
        self._save()
        log.info(f"添加预测记录: {home} vs {away} (match_id={match_id})")
    
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
                # 尝试解析时间
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%m-%d %H:%M']:
                    try:
                        match_time = datetime.strptime(match_time_str, fmt)
                        if fmt == '%m-%d %H:%M':
                            match_time = match_time.replace(year=now.year)
                        break
                    except ValueError:
                        continue
                else:
                    continue
                
                # 比赛结束时间 = 比赛开始时间 + 150分钟
                settle_time = match_time + timedelta(minutes=minutes)
                
                if now >= settle_time:
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
                self._save()
                log.info(f"更新时间分层预测: {match_id} -> {time_layer}")
                return True
        return False
    
    def _calculate_hit_flags(self, record: Dict) -> Dict:
        """计算命中标志"""
        actual_score = record.get('actual_score')
        actual_result = record.get('actual_result')
        predicted_scores = record.get('predicted_scores', {})
        predicted_1x2 = record.get('predicted_1x2', {})
        
        sorted_scores = sorted(predicted_scores.items(), key=lambda x: -x[1])
        top1 = sorted_scores[0][0] if sorted_scores else None
        top3 = [s for s, _ in sorted_scores[:3]]
        top5 = [s for s, _ in sorted_scores[:5]]
        
        pred_result = max(predicted_1x2.items(), key=lambda x: x[1])[0] if predicted_1x2 else None
        
        return {
            'hit_top1': actual_score == top1,
            'hit_top3': actual_score in top3,
            'hit_top5': actual_score in top5,
            'hit_1x2': pred_result == actual_result,
        }
    
    def update_result(self, match_id: str, actual_score: str, actual_result: str, error: str = None):
        """
        更新比赛结果
        
        参数：
            match_id: 比赛ID
            actual_score: 实际比分 "2-1"
            actual_result: 实际结果 "H"/"D"/"A"
            error: 同步错误信息（可选）
        """
        for record in self.records:
            if record.get('match_id') == match_id:
                if actual_score and actual_result:
                    # 成功结算
                    record['actual_score'] = actual_score
                    record['actual_result'] = actual_result
                    record['settled'] = True
                    record['settled_at'] = datetime.now().isoformat()
                    record['sync_status'] = 'synced'
                    
                    # 计算命中结果
                    record.update(self._calculate_hit_flags(record))
                    
                    # 更新各模块
                    self._update_calibrator(record)
                    self._update_market_db(record)
                    self._update_score_frequency_db(record)
                    
                    log.info(f"结算比赛: {record['home']} vs {record['away']} -> {actual_score} ({actual_result})")
                else:
                    # 同步失败
                    self._handle_sync_failure(record, error or '无法获取赛果')
                
                self._save()
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
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%m-%d %H:%M']:
                    try:
                        match_time = datetime.strptime(match_time_str, fmt)
                        if fmt == '%m-%d %H:%M':
                            match_time = match_time.replace(year=now.year)
                        break
                    except ValueError:
                        continue
                else:
                    continue
                
                settle_time = match_time + timedelta(minutes=minutes)
                
                if now >= settle_time:
                    record['sync_status'] = 'ready'
                    ready.append(record)
                    
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
        
        return {
            'total': len(self.records),
            'settled': synced,
            'pending_sync': pending + ready,
            'retry': retry,
            'failed': failed,
            'ignored': ignored,
            'last_sync_at': last_sync.isoformat() if last_sync else None,
        }
    
    def _update_calibrator(self, record: Dict):
        """更新贝叶斯校准库"""
        try:
            from .bayesian_calibration import get_calibrator
            
            calibrator = get_calibrator()
            predicted_scores = record.get('predicted_scores', {})
            actual_score = record.get('actual_score', '')
            
            for score, prob in predicted_scores.items():
                is_correct = (score == actual_score)
                calibrator.add_record(score, prob, is_correct)
            
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
    
    def get_stats(self) -> Dict:
        """获取统计信息（包含时间分层统计）"""
        total = len(self.records)
        settled = len([r for r in self.records if r.get('settled', False)])
        unsettled = total - settled
        
        # 时间分层统计
        time_layers = ['T-24h', 'T-6h', 'T-1h', 'T-15min', 'final']
        layer_stats = {layer: {'correct_top1': 0, 'correct_top3': 0, 'correct_top5': 0, 'total': 0} 
                      for layer in time_layers}
        
        # 计算命中率
        correct_top1 = 0
        correct_top3 = 0
        correct_top5 = 0
        correct_1x2 = 0
        
        for record in self.records:
            if not record.get('settled', False):
                continue
            
            actual_score = record.get('actual_score', '')
            predicted_scores = record.get('predicted_scores', {})
            actual_result = record.get('actual_result', '')
            predicted_1x2 = record.get('predicted_1x2', {})
            time_layers_data = record.get('time_layers', {})
            
            if not predicted_scores or not actual_score:
                continue
            
            # 统计各时间层命中率
            for layer in time_layers:
                layer_pred = time_layers_data.get(layer) or predicted_scores
                if not layer_pred:
                    continue
                
                sorted_scores = sorted(layer_pred.items(), key=lambda x: -x[1])
                layer_stats[layer]['total'] += 1
                
                if sorted_scores and sorted_scores[0][0] == actual_score:
                    layer_stats[layer]['correct_top1'] += 1
                
                top3_scores = [s[0] for s in sorted_scores[:3]]
                if actual_score in top3_scores:
                    layer_stats[layer]['correct_top3'] += 1
                
                top5_scores = [s[0] for s in sorted_scores[:5]]
                if actual_score in top5_scores:
                    layer_stats[layer]['correct_top5'] += 1
            
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
                pred_result = max(predicted_1x2.items(), key=lambda x: x[1])[0]
                if pred_result == actual_result:
                    correct_1x2 += 1
        
        hit_rate_top1 = correct_top1 / settled if settled > 0 else 0
        hit_rate_top3 = correct_top3 / settled if settled > 0 else 0
        hit_rate_top5 = correct_top5 / settled if settled > 0 else 0
        hit_rate_1x2 = correct_1x2 / settled if settled > 0 else 0
        
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
        }


# 全局实例
_global_history = PredictionHistory()


# ==================== 便捷函数 ====================

def save_prediction(match_id: str, league: str, home: str, away: str,
                   match_time: str, predicted_scores: Dict[str, float],
                   predicted_1x2: Dict[str, float], asian: float = None,
                   total_line: float = None, odds_data: Dict = None):
    """保存预测记录"""
    return _global_history.add_prediction(
        match_id, league, home, away, match_time,
        predicted_scores, predicted_1x2, asian, total_line, odds_data
    )


def sync_results():
    """同步比赛结果"""
    return auto_sync_results()


def auto_sync_results():
    """
    自动同步比赛结果（三层兜底）
    1. 第一优先：match_id 对应赛果页面
    2. 第二优先：主队 + 客队 + 比赛日期模糊匹配
    3. 第三优先：放弃自动同步，标记 failed
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

        if not _is_valid_match_id(match_id):
            log.debug(f"跳过非数字 match_id 的同步: {home} vs {away} ({match_id})")
            continue
        
        try:
            # 三层兜底抓取赛果
            result = fetch_result_by_match_id(match_id, match_time)
            if not result:
                result = fetch_result_by_team_and_date(home, away, match_time)
            
            if result:
                _global_history.update_result(match_id, result['score'], result['result'])
                synced += 1
                log.info(f"同步成功: {home} vs {away} -> {result['score']}")
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


def _is_valid_match_id(match_id: str) -> bool:
    """仅对 500.com 数字型 fid 尝试抓取赛果"""
    return bool(match_id) and str(match_id).isdigit()


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


def _fetch_match_html(match_id: str) -> str:
    """复用足球模块抓取逻辑，保持与预测数据同源"""
    from . import fetch as fetch_html
    return fetch_html(f'https://odds.500.com/fenxi/shuju-{match_id}.shtml')


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
            return _parse_score_string(score)

    try:
        html = _fetch_match_html(match_id)
        score = _parse_shuju_score(html, match_id)
        if score:
            log.info(f"通过 shuju 页面抓取赛果: match_id={match_id} -> {score}")
            return _parse_score_string(score)
    except Exception as e:
        log.debug(f"shuju 页面抓取失败: {e}")

    return None


def _parse_match_datetime(match_time: str) -> Optional[datetime]:
    """解析比赛时间，兼容 MM-DD HH:MM 与完整日期格式"""
    if not match_time:
        return None

    now = datetime.now()
    text = match_time.strip()
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%m-%d %H:%M'):
        try:
            match_dt = datetime.strptime(text, fmt)
            if fmt == '%m-%d %H:%M':
                match_dt = match_dt.replace(year=now.year)
                # 跨年：记录月份比当前月大很多，说明是去年
                if match_dt > now + timedelta(days=2):
                    match_dt = match_dt.replace(year=now.year - 1)
                elif now.month == 12 and match_dt.month == 1:
                    match_dt = match_dt.replace(year=now.year + 1)
            return match_dt
        except ValueError:
            continue
    return None


def _live_query_dates(match_time: str) -> List[str]:
    """
    动态计算 live.500.com 的 ?e= 查询日期。

    竞彩赛果页规则（见 https://live.500.com/?e=YYYY-MM-DD ）：
    - e=某日 的页面展示该「开售日」对应场次的赛果
    - 开球时间为 06-13 03:00 的比赛，出现在 e=2026-06-12 页面（开球日前一天）
    - 因此优先查 kickoff_date - 1，再查当日及邻近日，最后兜底今天/昨天
    """
    today = datetime.now().date()
    candidates = []

    match_dt = _parse_match_datetime(match_time)
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


def _fetch_live_html(search_date: str) -> str:
    from . import fetch as fetch_html
    return fetch_html(f'https://live.500.com/?e={search_date}')


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


def _fetch_live_score_by_fid(match_id: str, match_time: str) -> Optional[str]:
    """在 live.500.com 按 fid 查找赛果，日期动态推算"""
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


def fetch_result_by_team_and_date(home: str, away: str, match_time: str) -> Optional[Dict]:
    """
    第二优先：通过球队名和比赛时间在 live.500.com 模糊匹配抓取赛果
    """
    try:
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
                    return _parse_score_string(score)

    except Exception as e:
        log.debug(f"通过球队名+日期抓取失败: {e}")

    return None


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


def get_history_stats() -> Dict:
    """获取历史统计"""
    return _global_history.get_stats()


def get_sync_status_summary() -> Dict:
    """获取同步状态汇总"""
    return _global_history.get_sync_status_summary()


def get_prediction_records(include_hidden: bool = False) -> List[Dict]:
    """
    获取预测记录列表
    
    参数：
        include_hidden: 是否包含已失败的记录
    """
    records = []
    for record in _global_history.records:
        if not include_hidden:
            if record.get('sync_status') == 'failed':
                continue
        
        records.append({
            'match_id': record.get('match_id'),
            'league': record.get('league'),
            'home': record.get('home'),
            'away': record.get('away'),
            'match_time': record.get('match_time'),
            'settled': record.get('settled', False),
            'actual_score': record.get('actual_score'),
            'sync_status': record.get('sync_status', 'pending'),
            'sync_attempts': record.get('sync_attempts', 0),
            'last_sync_error': record.get('last_sync_error'),
            'next_sync_at': record.get('next_sync_at'),
            'hit_top1': record.get('hit_top1'),
            'hit_top3': record.get('hit_top3'),
            'predicted_scores': record.get('predicted_scores'),
        })
    
    # 按比赛时间倒序排列
    records.sort(key=lambda x: x.get('match_time', ''), reverse=True)
    return records


def hide_failed_records():
    """隐藏所有失败记录（标记为 ignored）"""
    for record in _global_history.records:
        if record.get('sync_status') == 'failed':
            record['sync_status'] = 'ignored'
    _global_history._save()
    log.info("已隐藏所有失败记录")


def start_background_sync(interval_seconds: int = 7200):
    """
    启动后台定时同步线程（使用 APScheduler）
    
    参数：
        interval_seconds: 同步间隔（秒），默认2小时
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.schedulers.blocking import BlockingScheduler
        
        scheduler = BlockingScheduler(timezone="Asia/Shanghai")
        
        # 每2小时同步一次
        scheduler.add_job(
            auto_sync_results,
            'interval',
            seconds=interval_seconds,
            id='football_result_sync',
            replace_existing=True
        )
        
        scheduler.start()
        log.info(f"已启动后台同步调度器，间隔 {interval_seconds} 秒")
        return scheduler
        
    except ImportError:
        # 如果没有 APScheduler，使用简单线程
        log.warning("APScheduler 未安装，使用简单线程调度")
        
        def sync_loop():
            while True:
                try:
                    result = auto_sync_results()
                    if result['synced'] > 0 or result['failed'] > 0:
                        log.info(f"后台同步: {result['message']}")
                except Exception as e:
                    log.error(f"后台同步异常: {e}")
                
                time.sleep(interval_seconds)
        
        thread = Thread(target=sync_loop, daemon=True)
        thread.start()
        log.info(f"已启动后台同步线程，间隔 {interval_seconds} 秒")
        return thread


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
