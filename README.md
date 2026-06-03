# 足球比分预测

基于 odds.500.com 多家博彩公司赔率（欧赔走 JSON 接口、亚盘/大小球走 HTML），
用**泊松概率模型**预测比分。提供命令行与网页两种用法，零第三方依赖（纯标准库）。

## 运行

```bash
python3 server.py     # 网页：启动后自动打开 http://localhost:9000
python3 football.py   # 命令行：输入球队关键词 → 选比赛 → 输出预测
```

## 访问范围

| 场景 | 说明 |
|------|------|
| 本机 | `http://localhost:9000` |
| 局域网 | 服务绑定 `0.0.0.0`，手机/同网段设备用启动时打印的 `http://<局域网IP>:9000`（需连同一 Wi-Fi；开了代理/VPN 时认准 `192.168/10/172` 段那个） |
| 公网 | 见下，**务必先开鉴权** |

### 端口

默认 9000，可用环境变量改：`FOOTBALL_PORT=8000 python3 server.py`

## 公网访问

本服务**无默认鉴权**，且会主动抓取 500.com。公网暴露前**必须**启用 HTTP Basic 鉴权，
否则任何人都能打开并触发抓取，可能导致你的出口 IP 被 500.com 封禁。

### 1. 启用鉴权

单用户：

```bash
FOOTBALL_USER=yourname FOOTBALL_PASS=yourstrongpass python3 server.py
```

多用户（逗号分隔多组 `用户名:密码`，可与单用户变量并用）：

```bash
FOOTBALL_USERS="alice:pass1,bob:pass2,carol:pass3" python3 server.py
```

启用后浏览器首次访问会弹出用户名/密码框（前端无需改动，API 也一并受保护）。
任一组凭据均可登录；密码用 `hmac.compare_digest` 常量时间比较。

### 2. 暴露到公网（二选一）

**A. 隧道（推荐，免路由器配置，自带 HTTPS）**

```bash
# Cloudflare Tunnel（免费，临时公网 https 地址）
cloudflared tunnel --url http://localhost:9000

# 或 ngrok
ngrok http 9000
```

命令会打印一个公网地址（如 `https://xxxx.trycloudflare.com`），手机/外网直接访问。
隧道关闭地址即失效，适合临时分享。

**B. 路由器端口转发**

把路由器的某外部端口转发到「本机内网IP:9000」，再用「公网IP:外部端口」访问。
要求：宽带有公网 IP（很多家庭宽带是 CGNAT，没有公网 IP 则此法不可用），
并自行承担长期暴露的安全风险。建议同时启用上面的鉴权。

> 提示：隧道方案下服务仍只需监听本机，`cloudflared/ngrok` 负责把流量转进来；
> 端口转发方案才依赖 `0.0.0.0` 绑定。

## 测试

```bash
python3 tests/test_fetch_match_list.py   # 比赛列表解析
python3 tests/test_odds_direction.py     # 初/终盘方向
```

## 免责声明

分析仅为概率统计参考，不构成任何投注建议。请理性对待。
