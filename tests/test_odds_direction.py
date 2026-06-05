"""
回归测试：初盘/终盘方向
======================
历史 bug：500.com "平均值"行第一组为初盘、第二组为终盘，
但原代码把第一组当成终盘(close)，导致初/终盘抓反，模型实际用了初盘。
欧赔 JSON 接口按时间降序（首条=终盘），亦在此锁定。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.football as football


def test_ouzhi_json_first_is_close():
    # 时间序列首条=最新=终盘，末条=最早=初盘
    series = [
        [1.52, 3.88, 6.11, 92.6, '2026-06-03 16:06'],
        [1.49, 3.95, 6.20, 92.1, '2026-06-01 10:00'],
        [1.46, 4.03, 6.39, 91.8, '2026-05-29 00:05'],
    ]
    football.fetch_json = lambda url, referer=None: series
    oz = football.fetch_ouzhi('x')
    assert oz['close']['home'] == 1.52 and oz['close']['return_rate'] == 92.6
    assert oz['open']['home'] == 1.46 and oz['open']['return_rate'] == 91.8


def test_daxiao_first_triple_is_open():
    nums = [0.90, 2.31, 0.88, 0.87, 2.38, 0.89]  # [初盘组][终盘组]
    football._fetch_avg_page = lambda mid, page: (None, nums)
    dx = football.fetch_daxiao('x')
    assert dx['open']['line'] == 2.31, dx['open']
    assert dx['close']['line'] == 2.38, dx['close']


def test_yazhi_first_triple_is_open():
    nums = [0.899, -0.969, 0.89, 0.863, -1.063, 0.911]
    fake_html = '平均值 0.899 -0.969 0.89 0.863 -1.063 0.911'
    football._fetch_avg_page = lambda mid, page: (fake_html, nums)
    ya = football.fetch_yazhi('x')
    # 取反后正=主让；初盘让球 0.969 < 终盘 1.063（主队走强）
    assert round(ya['open']['handicap'], 3) == 0.969, ya['open']
    assert round(ya['close']['handicap'], 3) == 1.063, ya['close']
    assert ya['open']['home_odds'] == 0.899, ya['open']
    assert ya['close']['home_odds'] == 0.863, ya['close']


if __name__ == '__main__':
    test_ouzhi_json_first_is_close()
    test_daxiao_first_triple_is_open()
    test_yazhi_first_triple_is_open()
    print('✓ 初/终盘方向测试全部通过')
