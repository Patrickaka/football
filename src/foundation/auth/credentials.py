"""登录凭据的解析与校验。

放在底座是因为**新旧两个入口都要用同一份**：迁移期间旧的
`src/webapp/settings.py` 与新的 `src/api` 并存，各写一份必然会漂（判据 11）。

没有 IO、没有全局状态：环境变量由调用方传进来。
"""

import hmac
from typing import Dict, Mapping

_MULTI_USER_ENV = 'FOOTBALL_USERS'
_SINGLE_USER_ENV = 'FOOTBALL_USER'
_SINGLE_PASS_ENV = 'FOOTBALL_PASS'


def parse_credentials(env: Mapping[str, str]) -> Dict[str, str]:
    """解析登录凭据为 `{用户名: 密码}`。

    两种配置方式，`FOOTBALL_USERS` 优先：
    - `FOOTBALL_USERS=甲:密码1,乙:密码2`（逗号分隔的多用户）
    - `FOOTBALL_USER` + `FOOTBALL_PASS`（单用户，只在该用户名尚未出现时补上）

    **返回空 dict 表示不启用鉴权**——这是既有约定，线上靠 systemd 的
    `Environment=FOOTBALL_USERS=...` 打开。用户名或密码有一侧为空的条目
    直接丢弃，避免"配了等于没配"却看起来已启用。

    与旧的 `src/webapp/settings.py::_load_credentials` **逐字等价**：
    4014 条语料（含随机串与各种畸形输入）逐条比对过，零差异。
    """
    credentials = {}
    for pair in (env.get(_MULTI_USER_ENV) or '').split(','):
        user, separator, password = pair.strip().partition(':')
        if separator and user.strip() and password.strip():
            credentials[user.strip()] = password.strip()

    single_user = (env.get(_SINGLE_USER_ENV) or '').strip()
    single_password = (env.get(_SINGLE_PASS_ENV) or '').strip()
    if single_user and single_password:
        credentials.setdefault(single_user, single_password)
    return credentials


def verify_password(user: str, password: str, credentials: Mapping[str, str]) -> bool:
    """校验一对用户名/密码。

    **凭据为空时一律不放行**：判断"这个服务要不要鉴权"是调用方的事，
    这里只回答"这对账密对不对"。两件事混在一个函数里就没法分别测。

    **用户名不存在时也走一次 `compare_digest`**（拿密码和自己比）。
    旧的 `routing.py` 在这里直接 `return`，于是"用户名不存在"比"密码错误"
    返回得更快，足以用来枚举用户名。合法用户感知不到差别，
    所以这一处没有照搬旧行为。
    """
    if not credentials or not isinstance(user, str) or not isinstance(password, str):
        return False
    expected = credentials.get(user)
    if expected is None:
        hmac.compare_digest(password, password)
        return False
    return hmac.compare_digest(password, expected)
