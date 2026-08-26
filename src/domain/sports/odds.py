from dataclasses import dataclass


def odds_to_prob(odds):
    """欧赔转隐含概率。

    非正赔率无意义，直接拒绝而非返回 0——静默返回 0 会让上游的归一化除零，
    或者算出一个看起来正常、实际错误的概率。
    """
    if odds <= 0:
        raise ValueError(f'赔率必须为正，收到 {odds!r}')
    return 1.0 / odds


@dataclass
class Odds:
    """两路赔率。to_dict 保证纯 JSON 类型，可直接进缓存。"""

    home: float
    away: float

    def implied_probs(self):
        """归一化后的隐含概率，已剔除博彩公司抽水。"""
        raw_home = odds_to_prob(self.home)
        raw_away = odds_to_prob(self.away)
        total = raw_home + raw_away
        return (raw_home / total, raw_away / total)

    def to_dict(self):
        return {'home': float(self.home), 'away': float(self.away)}
