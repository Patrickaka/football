# -*- coding: utf-8 -*-
"""已下线数字玩法的回归守卫。"""
from pathlib import Path

from src.api.app import create_app


RETIRED_ROUTES = {
    '/ssq',
    '/api/pailie5',
    '/api/3d',
    '/api/3d-ml',
    '/api/3d-refresh',
    '/api/ssq',
    '/api/ssq-refresh',
    '/api/lottery',
    '/api/lottery-refresh',
    '/api/lottery/task-status',
    '/api/lottery/recommend',
    '/api/lottery/rank',
    '/api/lottery/ensemble',
    '/api/lottery/cycles',
    '/api/lottery/contribution',
    '/api/lottery/backtest',
    '/api/lottery/fetch',
    '/api/lottery/ml',
    '/api/lottery/ml-refresh',
}


def test_retired_routes_are_not_registered():
    paths = {route.path for route in create_app().routes if hasattr(route, 'path')}
    assert RETIRED_ROUTES.isdisjoint(paths)


def test_retired_source_packages_are_gone():
    for path in ('src/lottery', 'src/lottery3d', 'src/ssq',
                 'src/domain/numeric/lottery3d'):
        assert not Path(path).exists(), f'{path} 不应继续存在'


def test_web_has_no_retired_tabs_or_requests():
    html = Path('web/index.html').read_text(encoding='utf-8')
    for token in ('lottery3d', 'pailie5', '福彩3D', '大乐透', '双色球', '排列五',
                  '/api/3d', '/api/ssq', '/api/lottery', '/api/pailie5'):
        assert token not in html
