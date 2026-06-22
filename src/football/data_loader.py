"""比赛数据加载适配层。

`get_live_league_profile`（联赛实时画像）与 `scan_and_predict_time_layers`
（时间分层扫描）历史上直接 `from .data_loader import ...`，但本模块此前缺失，
线上运行时持续报 `No module named 'src.football.data_loader'`，导致两处功能降级。
这里把它们桥接到已有的真实数据源，不引入新的抓取逻辑。
"""
from typing import Dict, List


def fetch_league_matches(league_name: str, limit: int = 200) -> List[Dict]:
    """返回指定联赛最近的已完赛比赛，供联赛画像统计。

    元素含 `score`（"主-客" 比分字符串），仅包含已回填实际结果的比赛。
    """
    from .prediction_records import get_historical_data

    records = get_historical_data(league_name, limit)
    return [
        {'home': r.get('home'), 'away': r.get('away'), 'score': r.get('actual_score')}
        for r in records
        if r.get('actual_score')
    ]


def fetch_future_matches() -> List[Dict]:
    """返回今日/未来赛程，元素含 `match_id` 与 `time`，供时间分层扫描。"""
    from . import fetch_match_list

    return fetch_match_list()
