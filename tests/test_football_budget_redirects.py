"""urllib redirects share one match deadline; all transports here are stubs."""
from contextlib import ExitStack
from io import BytesIO
from unittest.mock import Mock, patch
import urllib.error
import urllib.request

import pytest

from src.football import analysis_budget as budget, fetching


def _request(method='GET', timeout=20):
    request = urllib.request.Request(
        'http://example.test/start',
        data=b'{"match":1}' if method == 'POST' else None,
        headers={'Content-Type': 'application/json', 'X-Test': 'preserved'},
    )
    request.timeout = timeout
    return request


def _handler():
    handler = fetching._BudgetHTTPRedirectHandler()
    parent = Mock()
    handler.add_parent(parent)
    return handler, parent


@pytest.mark.parametrize('code', [301, 302, 303, 307, 308])
def test_expired_budget_does_not_follow_redirect(code):
    handler, parent = _handler()
    response = BytesIO(b'redirect body')
    with patch.object(budget.time, 'monotonic', return_value=0) as clock:
        with budget.limit(5):
            clock.return_value = 5
            with pytest.raises(budget.AnalysisTimeout):
                getattr(handler, f'http_error_{code}')(
                    _request(), response, code, 'Redirect', {'location': '/next'},
                )
    parent.open.assert_not_called()
    assert response.closed


@pytest.mark.parametrize('expiry_phase', ['read', 'close'])
def test_budget_is_checked_again_after_redirect_body_and_close(expiry_phase):
    handler, parent = _handler()
    with patch.object(budget.time, 'monotonic', return_value=0) as clock:
        class Response(BytesIO):
            def read1(self, size=-1):
                if expiry_phase == 'read':
                    clock.return_value = 5
                return super().read1(size)

            def close(self):
                if expiry_phase == 'close':
                    clock.return_value = 5
                return super().close()

        response = Response(b'body')
        with budget.limit(5), pytest.raises(budget.AnalysisTimeout):
            handler.http_error_302(
                _request(), response, 302, 'Redirect', {'location': '/next'},
            )
    parent.open.assert_not_called()
    assert response.closed


def test_each_hop_receives_only_the_remaining_timeout():
    handler, parent = _handler()
    redirects = []
    with patch.object(budget.time, 'monotonic', return_value=0) as clock:
        def follow(request, *, timeout):
            redirects.append((request.full_url, timeout))
            request.timeout = timeout  # OpenerDirector.open does this.
            if len(redirects) == 3:
                return 'finished'
            clock.return_value += 3
            return handler.http_error_302(
                request, BytesIO(b''), 302, 'Redirect',
                {'location': f'/hop-{len(redirects) + 1}'},
            )

        parent.open.side_effect = follow
        with budget.limit(20):
            clock.return_value = 3
            result = handler.http_error_302(
                _request(), BytesIO(b''), 302, 'Redirect', {'location': '/hop-1'},
            )
    assert result == 'finished'
    assert redirects == [
        ('http://example.test/hop-1', 17),
        ('http://example.test/hop-2', 14),
        ('http://example.test/hop-3', 11),
    ]


def test_redirect_does_not_extend_a_shorter_socket_timeout():
    handler, parent = _handler()
    with budget.limit(20):
        handler.http_error_302(
            _request(timeout=2), BytesIO(b''), 302, 'Redirect', {'location': '/next'},
        )
    assert parent.open.call_args.kwargs['timeout'] == 2


@pytest.mark.parametrize('method,code', [
    ('GET', 301), ('GET', 302), ('GET', 303), ('GET', 307), ('GET', 308),
    ('POST', 301), ('POST', 302), ('POST', 303),
])
def test_standard_redirect_method_and_header_rules_are_preserved(method, code):
    handler, parent = _handler()
    with budget.limit(20):
        getattr(handler, f'http_error_{code}')(
            _request(method), BytesIO(b''), code, 'Redirect',
            {'location': '/a path'},
        )
    request = parent.open.call_args.args[0]
    assert request.full_url == 'http://example.test/a%20path'
    assert request.get_method() == 'GET'
    assert request.data is None
    assert request.get_header('Content-type') is None
    assert request.get_header('X-test') == 'preserved'


@pytest.mark.parametrize('code', [307, 308])
def test_post_redirects_rejected_by_urllib_are_still_rejected(code):
    handler, parent = _handler()
    with budget.limit(20), pytest.raises(urllib.error.HTTPError):
        getattr(handler, f'http_error_{code}')(
            _request('POST'), BytesIO(b''), code, 'Redirect', {'location': '/next'},
        )
    parent.open.assert_not_called()


@pytest.mark.parametrize('case', ['unsafe_scheme', 'loop'])
def test_redirect_address_and_loop_protections_remain_in_effect(case):
    handler, parent = _handler()
    request = _request()
    location = '/next'
    if case == 'unsafe_scheme':
        location = 'file:///private.txt'
    else:
        request.redirect_dict = {'http://example.test/next': handler.max_repeats}
    with budget.limit(20), pytest.raises(urllib.error.HTTPError):
        handler.http_error_302(
            request, BytesIO(b''), 302, 'Redirect', {'location': location},
        )
    parent.open.assert_not_called()


def _disable_cache_and_throttle(stack):
    stack.enter_context(patch.object(fetching, '_fetch_cache_get', return_value=None))
    stack.enter_context(patch.object(fetching, '_fetch_cache_set'))
    stack.enter_context(patch.object(fetching, '_await_rate_slot'))
    stack.enter_context(patch.object(fetching, '_await_fetch_throttle'))


@pytest.mark.parametrize('method', ['GET', 'POST'])
def test_budget_get_and_post_install_the_redirect_handler(method):
    with ExitStack() as stack:
        _disable_cache_and_throttle(stack)
        open_url = stack.enter_context(patch.object(fetching.urllib.request, 'urlopen'))
        build = stack.enter_context(patch.object(fetching.urllib.request, 'build_opener'))
        build.return_value.open.return_value = BytesIO(b'{"ok":true}')
        with budget.limit(5):
            if method == 'GET':
                assert fetching._fetch_once('http://example.test', 'utf-8') == '{"ok":true}'
            else:
                assert fetching.fetch_json_post('http://example.test', {}) == {'ok': True}
        open_url.assert_not_called()
        assert len(build.call_args.args) == 1
        assert isinstance(build.call_args.args[0], fetching._BudgetHTTPRedirectHandler)
        request = build.return_value.open.call_args.args[0]
        assert request.get_method() == method
        assert 0 < build.return_value.open.call_args.kwargs['timeout'] <= 5


def test_cookie_retry_also_installs_budget_redirect_handler():
    first, cookie = Mock(), Mock()
    with patch.object(budget.time, 'monotonic', return_value=0) as clock, \
            patch.object(fetching.urllib.request, 'urlopen') as open_url, \
            patch.object(fetching.urllib.request, 'build_opener', side_effect=[first, cookie]) as build:
        def fail(*args, **kwargs):
            clock.return_value = 2
            raise urllib.error.HTTPError('http://example.test', 403, 'Cookie needed', {}, None)

        first.open.side_effect = fail
        cookie.open.return_value = BytesIO(b'cookie page')
        with budget.limit(5):
            assert fetching._fetch_once('http://example.test', 'utf-8') == 'cookie page'
        open_url.assert_not_called()
        assert isinstance(build.call_args_list[0].args[0], fetching._BudgetHTTPRedirectHandler)
        assert isinstance(build.call_args_list[1].args[0], fetching._BudgetHTTPRedirectHandler)
        assert isinstance(build.call_args_list[1].args[1], urllib.request.HTTPCookieProcessor)
        assert first.open.call_args.kwargs['timeout'] == 5
        assert cookie.open.call_args.kwargs['timeout'] == 3


@pytest.mark.parametrize('method', ['GET', 'POST'])
def test_no_budget_keeps_the_original_urlopen_transport(method):
    with ExitStack() as stack:
        _disable_cache_and_throttle(stack)
        open_url = stack.enter_context(patch.object(
            fetching.urllib.request, 'urlopen', return_value=BytesIO(b'{"ok":true}'),
        ))
        build = stack.enter_context(patch.object(fetching.urllib.request, 'build_opener'))
        if method == 'GET':
            assert fetching._fetch_once('http://example.test', 'utf-8') == '{"ok":true}'
        else:
            assert fetching.fetch_json_post('http://example.test', {}) == {'ok': True}
        build.assert_not_called()
        assert open_url.call_args.args[0].get_method() == method
        assert open_url.call_args.kwargs['timeout'] == 20


def test_no_budget_cookie_retry_keeps_original_cookie_opener():
    with patch.object(fetching.urllib.request, 'urlopen', side_effect=urllib.error.HTTPError(
            'http://example.test', 403, 'Cookie needed', {}, None,
    )) as open_url, patch.object(fetching.urllib.request, 'build_opener') as build:
        build.return_value.open.return_value = BytesIO(b'cookie page')
        assert fetching._fetch_once('http://example.test', 'utf-8') == 'cookie page'
        open_url.assert_called_once()
        assert len(build.call_args.args) == 1
        assert isinstance(build.call_args.args[0], urllib.request.HTTPCookieProcessor)
        assert build.return_value.open.call_args.kwargs['timeout'] == 20


def test_handler_without_budget_delegates_unchanged_to_urllib():
    handler, parent = _handler()
    with patch.object(fetching, '_read_response') as read_body:
        handler.http_error_302(
            _request(timeout=7), BytesIO(b''), 302, 'Redirect', {'location': '/next'},
        )
    read_body.assert_not_called()
    assert parent.open.call_args.kwargs['timeout'] == 7
