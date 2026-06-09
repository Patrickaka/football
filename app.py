#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
足球比分预测Web应用
=====================

集成盘口比分频率数据库和泊松预测模型
"""

import os
import sys
import json
from flask import Flask, render_template, request, jsonify

# 添加模块路径
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.football import predict_scores
from src.football.market_db import get_market_score_prob, blend_predictions

app = Flask(__name__)

# 默认配置
DEFAULT_WEIGHTS = {
    'poisson': 0.8,
    'market': 0.2,
}

# 联赛列表
LEAGUES = {
    'E0': '英超',
    'SP1': '西甲',
    'D1': '德甲',
    'I1': '意甲',
    'F1': '法甲',
}


@app.route('/')
def index():
    """首页"""
    return render_template('index.html', leagues=LEAGUES)


@app.route('/predict', methods=['POST'])
def predict():
    """比分预测API"""
    try:
        # 获取请求参数
        data = request.get_json()
        
        # 解析盘口数据
        asian_handicap = float(data.get('asian_handicap', 0))
        over_under = float(data.get('over_under', 2.5))
        home_odds = float(data.get('home_odds', 2.0))
        draw_odds = float(data.get('draw_odds', 3.0))
        away_odds = float(data.get('away_odds', 3.5))
        
        # 权重配置
        weights = {
            'poisson': float(data.get('weight_poisson', DEFAULT_WEIGHTS['poisson'])),
            'market': float(data.get('weight_market', DEFAULT_WEIGHTS['market'])),
        }
        
        league = data.get('league', 'E0')
        
        # 构建输入数据结构
        asian = {
            'handicap': asian_handicap,
            'close_prob': {
                'home_recv': 1.0 / home_odds if home_odds > 0 else 0.33,
                'away_give': 1.0 / away_odds if away_odds > 0 else 0.33,
            },
            'open_prob': {
                'home_recv': 1.0 / home_odds if home_odds > 0 else 0.33,
                'away_give': 1.0 / away_odds if away_odds > 0 else 0.33,
            },
        }
        
        euro = {
            'close': {
                'home': 1.0 / home_odds if home_odds > 0 else 0.33,
                'draw': 1.0 / draw_odds if draw_odds > 0 else 0.33,
                'away': 1.0 / away_odds if away_odds > 0 else 0.33,
            },
        }
        
        total = {
            'close_line': over_under,
            'close_prob': {
                'over': 0.5,
                'under': 0.5,
            },
        }
        
        # 获取联赛配置（暂不使用）
        league_profile = None
        
        # 使用泊松模型预测
        try:
            candidates, lam_home, lam_away, meta = predict_scores(
                asian, euro, total,
                league_profile=league_profile,
                model_type='poisson',
                enable_draw_calibration=True,
            )
            
            # 转换为字典格式
            poisson_probs = {score: prob for score, prob in candidates}
        except Exception as e:
            # 如果预测失败，使用简化的泊松计算
            poisson_probs = {}
            lam_home = (over_under + asian_handicap) / 2
            lam_away = (over_under - asian_handicap) / 2
        
        # 获取历史市场概率
        market_result = get_market_score_prob(asian_handicap, over_under)
        market_probs = market_result.get('probabilities', {})
        sample_count = market_result.get('sample_count', 0)
        
        # 融合预测
        if poisson_probs:
            blended_probs = blend_predictions(poisson_probs, market_probs, weights)
        else:
            blended_probs = market_probs
        
        # 获取TOP10比分
        top_scores = sorted(blended_probs.items(), key=lambda x: -x[1])[:10]
        
        # 计算期望值
        expected_goals_home = lam_home if lam_home else sum(
            int(s.split('-')[0]) * p for s, p in blended_probs.items()
        )
        expected_goals_away = lam_away if lam_away else sum(
            int(s.split('-')[1]) * p for s, p in blended_probs.items()
        )
        
        # 生成响应
        response = {
            'success': True,
            'asian_handicap': market_result.get('asian_handicap', asian_handicap),
            'over_under': market_result.get('over_under', over_under),
            'sample_count': sample_count,
            'expected_goals': {
                'home': round(expected_goals_home, 2),
                'away': round(expected_goals_away, 2),
                'total': round(expected_goals_home + expected_goals_away, 2),
            },
            'top_scores': [
                {'score': score, 'probability': round(prob * 100, 1), 'raw_prob': prob}
                for score, prob in top_scores
            ],
            'weights': weights,
            'meta': {
                'lam_home': round(lam_home, 2) if lam_home else None,
                'lam_away': round(lam_away, 2) if lam_away else None,
            },
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
        })


@app.route('/api/market/prob', methods=['GET'])
def get_market_prob():
    """获取历史盘口概率API"""
    try:
        asian = float(request.args.get('asian', 0))
        ou = float(request.args.get('ou', 2.5))
        
        result = get_market_score_prob(asian, ou)
        
        return jsonify({
            'success': True,
            'asian_handicap': result['asian_handicap'],
            'over_under': result['over_under'],
            'sample_count': result['sample_count'],
            'top_scores': [
                {'score': score, 'probability': round(prob * 100, 1)}
                for score, prob in result['top_scores'][:10]
            ],
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
        })


if __name__ == '__main__':
    # 确保templates目录存在
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
