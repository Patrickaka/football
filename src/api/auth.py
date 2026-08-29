"""入口的鉴权装配：设置、会话依赖、默认全拦的中间件。

**默认全拦，白名单豁免**，不是逐个路由挂 `Depends`。理由：漏挂一个
`Depends` 不会有任何东西报错，那条路由就此裸奔——而漏加一条豁免会立刻
以 401 暴露出来。两种错误的代价不对称，所以选会响的那种。

豁免清单本身有测试盯着（`tests/api/test_auth.py`），加路由时别默默扩它。
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse

from src.foundation.auth import SessionManager, parse_credentials

log = logging.getLogger('api.auth')

SESSION_COOKIE = 'fb_session'

#: 无需登录即可访问的路径。**只放健康探针与鉴权自身**，别的一律拦。
#:
#: - `/auth/me` 的用途就是让前端判断"该不该显示登录入口"。拦下来的话
#:   未登录的前端只能拿到 401，分不清"服务没开鉴权"和"我没登录"。
#:   它也不泄露任何东西：未登录时只回一句 `{"user": null}`。
#: - `/auth/logout` 豁免是为了让登出**幂等**：会话刚好过期时前端调它
#:   应该得到"已登出"，而不是一个需要特殊处理的 401。它不泄露信息，
#:   也只能撤销请求自己带来的那个会话。跨站强制登出被 `SameSite=Lax`
#:   挡住（跨站 POST 不带 Cookie），危害本身也仅止于"被登出"。
PUBLIC_PATHS = frozenset({'/healthz', '/auth/login', '/auth/logout', '/auth/me',
                          '/login', '/login.html'})


@dataclass
class AuthSettings:
    """鉴权配置。

    `credentials` 为空 → **不启用鉴权**，与旧入口的约定一致
    （线上靠 systemd 的 `Environment=FOOTBALL_USERS=...` 打开）。
    """

    credentials: Dict[str, str] = field(default_factory=dict)
    session_ttl: int = 7 * 24 * 3600
    #: Cookie 的 Path。线上反代把服务挂在 `/football/` 下并剥掉了前缀，
    #: 应用自己看不到这一段，所以必须由配置告诉它——否则 Cookie 会
    #: 以 `Path=/` 下发给整个域，同域下的其它应用都会收到它。
    cookie_path: str = '/'
    #: 是否给 Cookie 加 `Secure`。线上是 https（反代 443 + 证书），
    #: 本地开发是 http——加了 Secure 浏览器就不发了，所以不能写死。
    cookie_secure: bool = False
    cookie_samesite: str = 'lax'

    @property
    def enabled(self) -> bool:
        return bool(self.credentials)

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        return cls(
            credentials=parse_credentials(env),
            session_ttl=int(env.get('FOOTBALL_SESSION_TTL', str(7 * 24 * 3600))),
            cookie_path=env.get('FOOTBALL_COOKIE_PATH', '/'),
            cookie_secure=env.get('FOOTBALL_COOKIE_SECURE', '').lower() in {'1', 'true', 'yes'},
            cookie_samesite=env.get('FOOTBALL_COOKIE_SAMESITE', 'lax'),
        )


def build_session_manager(backend, settings: AuthSettings) -> SessionManager:
    """会话存 L2（生产态是 Redis），这样跨进程重启保留、撤销是真的撤销。"""
    return SessionManager(backend, ttl=settings.session_ttl)


def current_user(request: Request) -> Optional[str]:
    """当前请求的登录用户；未启用鉴权时返回 `None`（不是报错）。"""
    return getattr(request.state, 'user', None)


def install_auth(app):
    """挂上「默认全拦」的中间件。"""

    @app.middleware('http')
    async def _require_session(request: Request, call_next):
        settings: AuthSettings = request.app.state.auth

        # **先认领、后判断**，豁免路径也要认领。否则 `/auth/me` 永远回
        # `user: null`（前端因此以为自己没登录），`/login` 也不会把已登录的
        # 人送回首页——两处都是"看着能用、其实没在工作"。
        request.state.user = (
            request.app.state.sessions.resolve(request.cookies.get(SESSION_COOKIE))
            if settings.enabled else None)

        if not settings.enabled or _is_public(request) or request.state.user:
            return await call_next(request)

        log.warning('未登录访问 %s %s（来自 %s）',
                    request.method, request.url.path, _client_hint(request))
        return _unauthenticated(request)

    return app


def _is_public(request: Request) -> bool:
    """CORS 预检必须放行——浏览器发 OPTIONS 时不带 Cookie，
    拦下来的话所有跨域请求都会在预检阶段就失败。旧入口同样放行 OPTIONS。
    """
    if request.method == 'OPTIONS':
        return True
    # `/` 是业务首页，**不在豁免之列**——旧入口同样要求它带凭据。
    return (request.url.path.rstrip('/') or '/') in PUBLIC_PATHS


def _unauthenticated(request: Request):
    """页面请求跳登录页，接口请求返回 401 JSON。

    靠 `Accept` 区分：浏览器地址栏访问带 `text/html`，`fetch`/脚本不带。
    分开是因为给 XHR 返回一个 302 到 HTML 登录页，前端只会拿到一坨
    HTML 而不知道自己该重新登录。
    """
    accepts_html = 'text/html' in (request.headers.get('accept') or '')
    if accepts_html:
        settings: AuthSettings = request.app.state.auth
        return RedirectResponse(login_url(settings), status_code=303)
    return JSONResponse({'detail': '未登录或会话已过期'}, status_code=401)


def login_url(settings: AuthSettings) -> str:
    """登录页的**完整路径**，带上应用在反代下的挂载前缀。

    **不能写死 `/login`**：线上反代是
    `location /football/ { proxy_pass http://127.0.0.1:9000/; }`，
    前缀被剥掉了，应用自己看不到这一段。跳 `/login` 会把浏览器送到
    `https://域名/login`——那条路径在反代上没有对应 location，
    结果是一个 **openresty 的 404**，用户根本进不来。

    **也不能用相对地址 `./login`**：从 `/football/api/xxx` 被拦下时，
    相对解析的结果是 `/football/api/login`，同样不存在。

    挂载前缀就是 `cookie_path`——两者在任何反代部署下都必须相同
    （Cookie 要覆盖整个应用），所以不另设一个配置项让它们有机会不一致。
    """
    return settings.cookie_path.rstrip('/') + '/login'


def _client_hint(request: Request) -> str:
    """日志用的来源标识。**只取反代加的那一跳**，不信任整条链——
    `X-Forwarded-For` 是客户端可以随便伪造的，取第一个等于让人随便写。
    """
    real_ip = request.headers.get('x-real-ip')
    if real_ip:
        return real_ip
    return request.client.host if request.client else 'unknown'
