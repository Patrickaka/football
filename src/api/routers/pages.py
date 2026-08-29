# -*- coding: utf-8 -*-
"""网页入口。

`/` 是单页应用本体，`/ssq` 是它的一个锚点的历史入口。
"""

import pathlib

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(tags=['pages'], include_in_schema=False)

INDEX_FILE = pathlib.Path(__file__).resolve().parents[3] / 'web' / 'index.html'

#: 单页应用的入口必须每次都取新的——缓存住了，用户就会一直拿到旧版本的
#: 前端去打新版本的接口。旧入口三个头一个不少，这里照抄。
NO_CACHE = {
    'Cache-Control': 'no-cache, no-store, must-revalidate',
    'Pragma': 'no-cache',
    'Expires': '0',
}


@router.get('/')
async def index():
    """单页应用入口。"""
    try:
        body = INDEX_FILE.read_text(encoding='utf-8')
    except OSError:
        return HTMLResponse('index.html 缺失', status_code=500,
                            media_type='text/plain; charset=utf-8')
    return HTMLResponse(body, headers=NO_CACHE)


@router.get('/ssq')
async def ssq_redirect():
    """双色球的历史入口，跳到单页应用的对应锚点。

    **用相对地址 `./#ssq`**：线上反代把服务挂在 `/football/` 下并剥掉了
    前缀，应用自己看不到这一段。写死 `/#ssq` 会把访问者甩到站点根目录。
    旧入口靠嗅探 `route.path` 前缀来补，那在剥了前缀的新入口下永远为假。
    """
    return RedirectResponse('./#ssq', status_code=302)
