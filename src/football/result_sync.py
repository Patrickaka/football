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
    "created_at": "..."
}
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from threading import Thread


log = logging.getLogger('football')


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
HISTORY_FILE = os.path.join(DATA_DIR, 'prediction_history.json')


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
    
    def get_ready_to_settle(self, minutes: int = 150) -> List[Dict]:
        """
        获取可以结算的记录（比赛时间已过）
        
        参数：
            minutes: 比赛结束后等待分钟数（默认150分钟，给足补时）
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
    
    def update_result(self, match_id: str, actual_score: str, actual_result: str):
        """
        更新比赛结果
        
        参数：
            match_id: 比赛ID
            actual_score: 实际比分 "2-1"
            actual_result: 实际结果 "H"/"D"/"A"
        """
        for record in self.records:
            if record.get('match_id') == match_id:
                record['actual_score'] = actual_score
                record['actual_result'] = actual_result
                record['settled'] = True
                record['settled_at'] = datetime.now().isoformat()
                self._save()
                
                # 更新贝叶斯校准库
                self._update_calibrator(record)
                
                # 更新盘口聚类库
                self._update_market_db(record)
                
                # 更新盘口比分频率库
                self._update_score_frequency_db(record)
                
                log.info(f"结算比赛: {record['home']} vs {record['away']} -> {actual_score} ({actual_result})")
                return True
        return False
    
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
            from .market_db import MarketScoreDB
            
            db = MarketScoreDB()
            
            asian = record.get('asian')
            total_line = record.get('total_line')
            actual_score = record.get('actual_score', '')
            
            if asian is not None and total_line is not None and actual_score:
                db.add_match_result(asian, total_line, actual_score)
                db.save()
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
    ready = _global_history.get_ready_to_settle()
    
    if not ready:
        return {'synced': 0, 'message': '没有需要结算的比赛'}
    
    synced = 0
    
    for record in ready:
        match_id = record['match_id']
        league = record['league']
        home = record['home']
        
        try:
            # 尝试从网络抓取实际比分
            actual_score, actual_result = _fetch_result(match_id, league, home)
            
            if actual_score:
                _global_history.update_result(match_id, actual_score, actual_result)
                synced += 1
            else:
                log.warning(f"无法获取比赛结果: {home} vs {record['away']}")
                
        except Exception as e:
            log.error(f"同步比赛结果失败: {e}")
    
    return {
        'synced': synced,
        'total': len(ready),
        'message': f'结算了 {synced}/{len(ready)} 场比赛'
    }


def _fetch_result(match_id: str, league: str, home: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从网络抓取比赛结果
    
    返回：(actual_score, actual_result)
    """
    try:
        # 尝试从 odds.500.com 获取结果
        import urllib.request
        import re
        
        # 使用比赛ID或球队名搜索
        url = f"https://odds.500.com/close/?id={match_id}"
        
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')
            
            # 解析比分
            # 格式示例: <span class="score">2-1</span>
            score_match = re.search(r'<[^>]*class=["\']score["\'][^>]*>([\d]+)-([\d]+)</span>', html)
            
            if score_match:
                home_goals = int(score_match.group(1))
                away_goals = int(score_match.group(2))
                actual_score = f"{home_goals}-{away_goals}"
                
                # 计算结果
                if home_goals > away_goals:
                    actual_result = 'H'
                elif home_goals < away_goals:
                    actual_result = 'A'
                else:
                    actual_result = 'D'
                
                return actual_score, actual_result
        
    except Exception as e:
        log.debug(f"抓取比赛结果失败: {e}")
    
    return None, None


def get_history_stats() -> Dict:
    """获取历史统计"""
    return _global_history.get_stats()


def start_background_sync(interval_seconds: int = 300):
    """
    启动后台定时同步线程
    
    参数：
        interval_seconds: 同步间隔（秒），默认5分钟
    """
    def sync_loop():
        while True:
            try:
                result = sync_results()
                if result['synced'] > 0:
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
