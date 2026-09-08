"""Bounded read-only HTTP; untrusted pages cannot choose destinations or tools."""

from __future__ import annotations

import ipaddress
import json
import socket
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('redirects are disabled for evidence collection')


class ReadOnlyHTTP:
    def __init__(self, allowed_hosts, *, max_bytes=512_000, local_ollama=False):
        self.allowed_hosts = frozenset(str(host).lower() for host in allowed_hosts)
        self.max_bytes = min(max(int(max_bytes), 1024), 2_000_000)
        self.local_ollama = local_ollama

    def validate_url(self, url):
        parts = urlsplit(url)
        if (parts.scheme not in ('http', 'https') or parts.hostname not in self.allowed_hosts
                or parts.username or parts.password or parts.fragment):
            raise ValueError('source URL is not in the configured allowlist')
        if self.local_ollama:
            # Local inference only; never attach a cloud endpoint or arbitrary API.
            if parts.hostname not in ('127.0.0.1', '::1', 'localhost') or parts.path != '/api/chat':
                raise ValueError('Ollama must use loopback /api/chat')
        else:
            if parts.scheme != 'https' or parts.port not in (None, 443):
                raise ValueError('evidence sources require HTTPS on port 443')
            addresses = socket.getaddrinfo(parts.hostname, 443, type=socket.SOCK_STREAM)
            if not addresses or any(not ipaddress.ip_address(row[4][0]).is_global for row in addresses):
                raise ValueError('private or non-public evidence destination')
        return parts

    def request(self, url, *, timeout, headers=None, payload=None):
        self.validate_url(url)
        if payload is not None and not self.local_ollama:
            raise ValueError('evidence tools only allow GET')
        deadline = time.monotonic() + max(0.01, timeout)
        request = Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                          headers={'User-Agent': 'FootballContextResearch/1.0',
                                   **({'Content-Type': 'application/json'} if payload is not None else {}),
                                   **(headers or {})}, method='POST' if payload is not None else 'GET')
        opener = build_opener(ProxyHandler({}), _NoRedirect())
        with opener.open(request, timeout=max(0.01, timeout)) as response:
            chunks, size = [], 0
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError('source response exceeded time budget')
                chunk = response.read(min(16_384, self.max_bytes + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > self.max_bytes:
                    raise ValueError('source response exceeded size budget')
                chunks.append(chunk)
            return b''.join(chunks).decode('utf-8', errors='replace')

    def get_json(self, url, *, timeout, headers=None):
        return json.loads(self.request(url, timeout=timeout, headers=headers))

