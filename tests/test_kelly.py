"""
凯利指数单元测试
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.football as football
from src.football import fetching as fb_fetching


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
    # 'neutral' 是后来加的第四种取值：三项凯利离散度太小时，说不出哪项更难。
    # 原先这里只允许三种，于是这组本就该判 neutral 的赔率把用例判红了
    assert kelly['hardest'] in ('home', 'draw', 'away', 'neutral')
    assert kelly['favored'] in ('home', 'draw', 'away', 'neutral')
    assert kelly['spread'] >= 0
    assert kelly['summary']


def test_analyze_kelly_neutral_when_spread_is_small():
    """三项凯利挤在一起时判 neutral——**下游靠这个值决定要不要用凯利信号**
    （`modeling.py` 与 `upset.py` 都显式判 `!= 'neutral'`）。"""
    flat = {
        'open': {'home': 3.00, 'draw': 3.05, 'away': 3.10, 'return_rate': 93.0},
        'close': {'home': 3.00, 'draw': 3.05, 'away': 3.10, 'return_rate': 93.0},
    }
    ph, pd, pa = football.remove_vig(3.00, 3.05, 3.10)
    kelly = football.analyze_kelly(flat, (0.33, 0.33, 0.34), (ph, pd, pa))
    assert kelly['spread'] < 1.0
    assert kelly['hardest'] == 'neutral'
    assert kelly['favored'] == 'neutral'


def test_kelly_spread_is_always_zero_with_matching_probabilities():
    """**离散度恒为 0，跟赔率长什么样无关。**

    凯利指数 = 赔率 × 去水概率。而去水概率就是 `(1/赔率) / Σ(1/赔率)`，
    于是三项的乘积恒等于 `1/Σ(1/赔率)`——**同一个数**。只要传进来的概率是
    这组赔率自己的去水概率（`analyze_euro` 是唯一调用方，传的正是它），
    离散度在数学上就不可能非 0。

    后果：`hardest` 与 `favored` 恒为 `neutral`，而下游三处消费点
    （`modeling.py` 的 `!= 'neutral'`、`upset.py` 的 `== favorite_key`、
    `scoring.py` 的 `hard`）因此永远不生效——整段凯利离散度分析是死的。

    这条用例**钉住现状**，不是认可它。真要让离散度有意义，得传一组独立于
    这些赔率的概率（模型自己的预测），那是一次产品行为改动，不该夹在
    「修测试基线」里顺手做。改对之后这条会红，那正是提醒。
    """
    for h, d, a in [(1.20, 6.00, 15.0), (3.00, 3.05, 3.10), (1.50, 3.80, 5.50)]:
        data = {
            'open': {'home': h, 'draw': d, 'away': a, 'return_rate': 93.0},
            'close': {'home': h, 'draw': d, 'away': a, 'return_rate': 93.0},
        }
        kelly = football.analyze_kelly(data, football.remove_vig(h, d, a),
                                       football.remove_vig(h, d, a))
        # 用容差而不是 `== 0`：三项在浮点精度内相等，求和顺序不同会留下
        # 1e-14 量级的尾巴。这个尾巴比判定门槛（1.0）小十几个数量级
        assert kelly['spread'] < 1e-9, (h, d, a, kelly['spread'])
        assert kelly['hardest'] == 'neutral'
        assert kelly['favored'] == 'neutral'


def test_kelly_spread_becomes_positive_with_independent_probabilities():
    """传一组**独立于赔率**的概率时，离散度才会非 0——证明这段逻辑本身是对的，
    死掉的是调用方喂给它的东西。"""
    data = {
        'open': {'home': 1.50, 'draw': 3.80, 'away': 5.50, 'return_rate': 93.0},
        'close': {'home': 1.50, 'draw': 3.80, 'away': 5.50, 'return_rate': 93.0},
    }
    kelly = football.analyze_kelly(data, (0.75, 0.15, 0.10), (0.75, 0.15, 0.10))
    assert kelly['spread'] > 1.0
    assert kelly['hardest'] in ('home', 'draw', 'away')
    assert kelly['favored'] in ('home', 'draw', 'away')


def test_fetch_ouzhi_return_rate():
    series = [
        [1.52, 3.88, 6.11, 92.6, '2026-06-03'],
        [1.46, 4.03, 6.39, 91.8, '2026-05-29'],
    ]
    fb_fetching.fetch_json = lambda url, referer=None: series
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
