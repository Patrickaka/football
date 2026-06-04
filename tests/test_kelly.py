"""
凯利指数单元测试
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import football


def test_kelly_near_return_rate():
    """三项凯利应聚集在返还率附近（公平市场）"""
    h, d, a = 1.52, 3.88, 6.11
    ph, pd, pa = football.remove_vig(h, d, a)
    k = football.kelly_index_triple(h, d, a, ph, pd, pa)
    rr = 92.6
    for v in k.values():
        assert abs(v - rr) < 2.5, (k, rr)


def test_analyze_kelly_structure():
    data = {
        'open': {'home': 1.50, 'draw': 3.80, 'away': 5.50, 'return_rate': 93.0},
        'close': {'home': 1.65, 'draw': 3.60, 'away': 5.00, 'return_rate': 93.0},
    }
    ph, pd, pa = football.remove_vig(1.65, 3.60, 5.00)
    kelly = football.analyze_kelly(data, (0.55, 0.25, 0.20), (ph, pd, pa))
    assert kelly['hardest'] in ('home', 'draw', 'away')
    assert kelly['favored'] in ('home', 'draw', 'away')
    assert kelly['spread'] >= 0
    assert kelly['summary']


def test_fetch_ouzhi_return_rate():
    series = [
        [1.52, 3.88, 6.11, 92.6, '2026-06-03'],
        [1.46, 4.03, 6.39, 91.8, '2026-05-29'],
    ]
    football.fetch_json = lambda url, referer=None: series
    oz = football.fetch_ouzhi('x')
    assert oz['close']['return_rate'] == 92.6
    euro = football.analyze_euro(oz)
    assert 'kelly' in euro
    assert euro['kelly']['close']['home'] > 0


if __name__ == '__main__':
    test_kelly_near_return_rate()
    test_analyze_kelly_structure()
    test_fetch_ouzhi_return_rate()
    print('✓ 凯利指数测试全部通过')
