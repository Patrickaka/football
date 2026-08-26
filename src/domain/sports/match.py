from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Match:
    """一场比赛。充血模型：结算判定等行为属于领域对象自身。

    to_dict 保证纯 JSON 类型（datetime 转 ISO 8601），这是进缓存的前置条件——
    L2 用 json.dumps，datetime 会抛 TypeError，Cache 会静默退化为纯 L1，
    该 key 从此每次冷启动都要重算。
    """

    match_id: str
    league: str
    home: str
    away: str
    start_time: datetime
    home_score: Optional[int] = None
    away_score: Optional[int] = None

    def is_settled(self):
        """双方比分都存在才算结算。

        用 is not None 而非真值判断——0:0 是合法比分，真值判断会把它当成未结算。
        """
        return self.home_score is not None and self.away_score is not None

    def to_dict(self):
        return {
            'match_id': self.match_id,
            'league': self.league,
            'home': self.home,
            'away': self.away,
            'start_time': self.start_time.isoformat(),
            'home_score': self.home_score,
            'away_score': self.away_score,
        }
