"""登录、登出、当前身份。"""

import logging

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from src.api.auth import SESSION_COOKIE, AuthSettings
from src.foundation.auth import verify_password

log = logging.getLogger('api.auth.routes')

router = APIRouter(tags=['auth'])


class LoginRequest(BaseModel):
    user: str
    password: str


class Identity(BaseModel):
    user: str | None
    auth_enabled: bool


@router.post('/auth/login')
async def login(request: Request, payload: LoginRequest, response: Response):
    """校验账密，签发会话 Cookie。

    **失败时不区分"用户不存在"与"密码错误"**：两种情况同一句话、同一个
    状态码。分开说等于免费送出一个用户名枚举接口。
    """
    settings: AuthSettings = request.app.state.auth
    if not settings.enabled:
        return JSONResponse({'detail': '本服务未启用鉴权'}, status_code=400)

    if not verify_password(payload.user, payload.password, settings.credentials):
        log.warning('登录失败 user=%s', payload.user)
        return JSONResponse({'detail': '用户名或密码错误'}, status_code=401)

    session_id = request.app.state.sessions.create(payload.user)
    response = JSONResponse({'user': payload.user})
    response.set_cookie(
        SESSION_COOKIE, session_id,
        max_age=settings.session_ttl,
        httponly=True,          # JS 读不到，XSS 偷不走
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        path=settings.cookie_path,
    )
    log.info('登录成功 user=%s', payload.user)
    return response


@router.post('/auth/logout')
async def logout(request: Request):
    """撤销会话并清 Cookie。**对未登录的请求也返回 200**——登出是幂等的。"""
    settings: AuthSettings = request.app.state.auth
    request.app.state.sessions.revoke(request.cookies.get(SESSION_COOKIE))
    response = JSONResponse({'detail': '已登出'})
    response.delete_cookie(SESSION_COOKIE, path=settings.cookie_path)
    return response


@router.get('/auth/me', response_model=Identity)
async def me(request: Request):
    """当前身份。未启用鉴权时 `user` 为 null、`auth_enabled` 为 false——
    前端据此决定要不要显示登录入口。
    """
    settings: AuthSettings = request.app.state.auth
    return Identity(user=getattr(request.state, 'user', None),
                    auth_enabled=settings.enabled)


@router.get('/login', response_class=HTMLResponse)
async def login_page(request: Request):
    """登录页。已登录的直接跳回首页，不必再看一次表单。"""
    if getattr(request.state, 'user', None):
        return RedirectResponse('./', status_code=303)
    return HTMLResponse(request.app.state.login_page)
