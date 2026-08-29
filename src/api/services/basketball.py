# -*- coding: utf-8 -*-
"""篮球接口的业务装配。

每个函数收 `parse_qs` 形状的 `params`（`{键: [值]}`）、返回可直接序列化的
dict——**新旧两个入口共用同一份**，迁移期间各写一份必然会漂（判据 11）。
旧的 `src/webapp/basketball_api.py` 已改为转发到这里。

`params` 保持 `{键: [值]}` 这个别扭形状，是为了让新旧双跑差分能喂同一批
输入。切换完成、旧入口删除后再换成具名参数。

**错误一律吞成 `{'error': ...}`**，照搬旧行为：这些接口的调用方是网页，
抛异常会让整页空掉，返回错误串至少能显示出来。
"""

import logging

from src.api.runtime.basketball_service import get_context

log = logging.getLogger('api.services.basketball')


def basketball_payload(params):
    """获取篮球推荐预测"""
    try:
        date = params.get('date', [None])[0]
        bet_types = params.get('types', ['spf,rqspf,dx'])[0].split(',')
        source = params.get('source', ['okooo'])[0]
        if source not in ('okooo', '500'):
            source = 'okooo'
        
        log.info(f'篮球推荐请求: date={date}, types={bet_types}, source={source}')
        
        result = get_context().prediction.generate(
            date=date, bet_types=bet_types, source=source, use_movement=True
        )
        
        if 'error' in result:
            return result
        
        matches = []
        for r in result.get('results', []):
            match_data = r.get('match', {})
            match_item = {
                'home': match_data.get('home', ''),
                'away': match_data.get('away', ''),
                'league': match_data.get('league', ''),
                'time': match_data.get('time', ''),
                'status': match_data.get('status', ''),
                'official_open': match_data.get('status') == 'not_started',
                'market_analysis': r.get('market_analysis'),
            }
            
            spf = r.get('spf')
            if spf and spf.get('available'):
                match_item['spf'] = {
                    'prediction': spf.get('recommendation'),
                    'probabilities': {
                        '主胜': spf.get('home_prob'),
                        '客胜': spf.get('away_prob'),
                    },
                    'odds': {
                        '主胜': spf.get('home_odds'),
                        '客胜': spf.get('away_odds'),
                    },
                    'line_movement': spf.get('line_movement'),
                    'sharp_confirmed': spf.get('sharp_confirmed'),
                }
            else:
                match_item['spf'] = {'error': spf.get('reason') if spf else 'no_data'}

            rqspf = r.get('rqspf')
            if rqspf and rqspf.get('available'):
                match_item['rqspf'] = {
                    'prediction': rqspf.get('recommendation'),
                    'handicap': rqspf.get('handicap'),
                    'probabilities': {
                        '主胜': rqspf.get('home_prob'),
                        '客胜': rqspf.get('away_prob'),
                    },
                    'odds': {
                        '主胜': rqspf.get('home_odds'),
                        '客胜': rqspf.get('away_odds'),
                    },
                    'line_movement': rqspf.get('line_movement'),
                    'water_inference': rqspf.get('water_inference'),
                    'movement_led': rqspf.get('movement_led'),
                    'sharp_confirmed': rqspf.get('sharp_confirmed'),
                    'official': rqspf.get('official'),
                    'skip_reason': rqspf.get('skip_reason'),
                }
            else:
                match_item['rqspf'] = {'error': rqspf.get('reason') if rqspf else 'no_data'}

            daxiao = r.get('dx')
            if daxiao and daxiao.get('available'):
                match_item['daxiao'] = {
                    'prediction': daxiao.get('recommendation'),
                    'total': daxiao.get('total_line'),
                    'probabilities': {
                        '大分': daxiao.get('over_prob'),
                        '小分': daxiao.get('under_prob'),
                    },
                    'odds': {
                        '大分': daxiao.get('over_odds'),
                        '小分': daxiao.get('under_odds'),
                    },
                    'line_movement': daxiao.get('line_movement'),
                    'water_inference': daxiao.get('water_inference'),
                    'movement_led': daxiao.get('movement_led'),
                    'sharp_confirmed': daxiao.get('sharp_confirmed'),
                    'official': daxiao.get('official'),
                    'skip_reason': daxiao.get('skip_reason'),
                }
            else:
                match_item['daxiao'] = {'error': daxiao.get('reason') if daxiao else 'no_data'}
            
            matches.append(match_item)
        
        return {'result': {
            'date': result.get('date'),
            'total_matches': len(matches),
            'matches': matches,
            'version': result.get('version'),
            'source': result.get('source'),
            'movement_stats': result.get('movement_stats'),
        }}
    except Exception as e:
        log.error('篮球推荐失败', exc_info=True)
        return {'error': f'篮球推荐失败: {str(e)}'}


def basketball_matches_payload(params):
    """获取篮球比赛列表"""
    try:
        date = params.get('date', [None])[0]
        matches = get_context().prediction.fetch_schedule(date=date)
        return {'matches': matches}
    except Exception as e:
        log.error('篮球比赛列表获取失败', exc_info=True)
        return {'error': f'获取比赛列表失败: {str(e)}'}


def basketball_value_payload(params):
    """获取篮球价值投注推荐"""
    try:
        date = params.get('date', [None])[0]
        threshold = float(params.get('threshold', [0.05])[0])

        from src.domain.sports.basketball.prediction import find_value_bets

        recommendations = get_context().prediction.generate(date=date)
        value_bets = find_value_bets(recommendations.get('results', []),
                                     threshold=threshold)

        return {'result': value_bets}
    except Exception as e:
        log.error('篮球价值投注失败', exc_info=True)
        return {'error': f'价值投注分析失败: {str(e)}'}


def basketball_track_payload(params):
    """触发一次实时赔率轮询，累积盘路快照。"""
    try:
        date = params.get('date', [None])[0]
        tracker = get_context().tracker
        if tracker is None:
            return {'error': '赔率追踪不可用：数据库未连接'}
        count = tracker.track(date)
        return {'result': {'tracked': count, 'date': date}}
    except Exception as e:
        log.error('篮球赔率追踪失败', exc_info=True)
        return {'error': f'赔率追踪失败: {str(e)}'}


def basketball_movement_payload(params):
    """汇总当前累积的赔率走势信号。"""
    try:
        match_id = params.get('match_id', [None])[0]
        store = get_context().history
        if store is None:
            return {'error': '走势汇总不可用：数据库未连接'}
        if match_id:
            return {'result': {'match_id': match_id,
                               'snapshots': store.history_for(match_id)}}
        history = store.load()
        # 汇总每场的走势统计
        summary = []
        for mid, snaps in history.items():
            if not snaps:
                continue
            valid = [s for s in snaps if s.get('spf_home') and s.get('spf_away')]
            first = valid[0] if valid else None
            last = valid[-1] if valid else None
            entry = {'match_id': mid, 'samples': len(snaps)}
            if first and last:
                entry['spf_home_move'] = round((last['spf_home'] - first['spf_home']), 4)
                entry['spf_away_move'] = round((last['spf_away'] - first['spf_away']), 4)
            summary.append(entry)
        return {'result': {'matches': len(summary), 'detail': summary}}
    except Exception as e:
        log.error('篮球走势汇总失败', exc_info=True)
        return {'error': f'走势汇总失败: {str(e)}'}
