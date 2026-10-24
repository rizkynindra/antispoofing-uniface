"""Tests for request identifier helpers used in error mapping headers."""

from unittest.mock import MagicMock

from app.request_context import (
    DEFAULT_CLIENT_IP,
    get_client_ip,
    get_or_generate_trace_id,
    get_prev_transaction_id,
    resolve_channel_id,
    trace_response_headers,
)


def _request(**headers):
    request = MagicMock()
    request.headers.get.side_effect = lambda key, default=None: headers.get(key, default)
    request.client.host = "10.0.0.8"
    return request


def test_get_client_ip_prefers_forwarded_for() -> None:
    request = _request(**{"X-Forwarded-For": "1.1.1.1, 2.2.2.2"})
    assert get_client_ip(request) == "1.1.1.1"


def test_get_client_ip_falls_back_to_peer() -> None:
    request = _request()
    assert get_client_ip(request) == "10.0.0.8"


def test_trace_id_adds_prefix() -> None:
    request = _request(**{"X-TRACE-ID": "abc"})
    assert get_or_generate_trace_id(request) == "TC-abc"


def test_prev_transaction_empty_without_header() -> None:
    request = _request()
    assert get_prev_transaction_id(request) == ""


def test_resolve_channel_id_from_header() -> None:
    request = _request(**{"X-ChId": "JMO"})
    assert resolve_channel_id(None, request) == "JMO"


def test_trace_response_headers() -> None:
    headers = trace_response_headers("TC-1", "TR-2")
    assert headers["X-TRACE-ID"] == "TC-1"
    assert headers["X-TRANSACTION-ID"] == "TR-2"


def test_default_ip_constant() -> None:
    assert DEFAULT_CLIENT_IP == "127.0.0.1"
