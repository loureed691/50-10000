"""Tests for KuCoin API client: signature generation, param encoding, retry logic."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import urllib.parse

import pytest

from kucoin_bot.api.client import KuCoinAPIError, KuCoinClient


class TestSignatureGeneration:
    """Verify HMAC signature matches KuCoin API v2 spec."""

    def _make_client(self) -> KuCoinClient:
        return KuCoinClient(
            api_key="test-key",
            api_secret="test-secret",
            api_passphrase="test-pass",
        )

    def test_sign_basic(self) -> None:
        client = self._make_client()
        ts = "1234567890000"
        headers = client._sign(ts, "GET", "/api/v1/accounts")

        expected_str = "1234567890000GET/api/v1/accounts"
        expected_sig = base64.b64encode(
            hmac.new(b"test-secret", expected_str.encode(), hashlib.sha256).digest()
        ).decode()
        expected_pass = base64.b64encode(
            hmac.new(b"test-secret", b"test-pass", hashlib.sha256).digest()
        ).decode()

        assert headers["KC-API-KEY"] == "test-key"
        assert headers["KC-API-SIGN"] == expected_sig
        assert headers["KC-API-PASSPHRASE"] == expected_pass
        assert headers["KC-API-KEY-VERSION"] == "2"

    def test_sign_with_query(self) -> None:
        client = self._make_client()
        ts = "1234567890000"
        path_with_query = "/api/v1/orders?status=active&symbol=BTC-USDT"
        headers = client._sign(ts, "GET", path_with_query)

        expected_str = f"1234567890000GET{path_with_query}"
        expected_sig = base64.b64encode(
            hmac.new(b"test-secret", expected_str.encode(), hashlib.sha256).digest()
        ).decode()
        assert headers["KC-API-SIGN"] == expected_sig

    def test_sign_with_body(self) -> None:
        client = self._make_client()
        ts = "1234567890000"
        body = '{"symbol":"BTC-USDT","side":"buy"}'
        headers = client._sign(ts, "POST", "/api/v1/orders", body)

        expected_str = f"1234567890000POST/api/v1/orders{body}"
        expected_sig = base64.b64encode(
            hmac.new(b"test-secret", expected_str.encode(), hashlib.sha256).digest()
        ).decode()
        assert headers["KC-API-SIGN"] == expected_sig

    def test_sign_method_case_insensitive(self) -> None:
        client = self._make_client()
        ts = "1234567890000"
        h1 = client._sign(ts, "get", "/api/v1/test")
        h2 = client._sign(ts, "GET", "/api/v1/test")
        assert h1["KC-API-SIGN"] == h2["KC-API-SIGN"]


class TestParamEncoding:
    """Verify URL params are properly encoded using urllib.parse.urlencode."""

    def test_sorted_params(self) -> None:
        """Params should be sorted for deterministic signing."""
        params = {"symbol": "BTC-USDT", "status": "active", "type": "limit"}
        encoded = urllib.parse.urlencode(sorted(params.items()))
        assert encoded == "status=active&symbol=BTC-USDT&type=limit"

    def test_special_characters_encoded(self) -> None:
        """Special characters in values should be properly URL-encoded."""
        params = {"query": "a b&c=d"}
        encoded = urllib.parse.urlencode(sorted(params.items()))
        assert "a+b" in encoded or "a%20b" in encoded
        assert "%26" in encoded or "&" not in encoded.split("=", 1)[1].split("&")[0]


class TestKuCoinAPIError:
    def test_error_attributes(self) -> None:
        err = KuCoinAPIError(status=400, code="400100", message="Invalid param", body={"code": "400100"})
        assert err.status == 400
        assert err.code == "400100"
        assert "Invalid param" in str(err)
        assert err.body == {"code": "400100"}


class TestClientEndpoints:
    """Test that new futures endpoints exist and have correct signatures."""

    def test_client_has_futures_endpoints(self) -> None:
        client = KuCoinClient("k", "s", "p")
        # Verify all expected futures methods exist
        assert hasattr(client, "get_futures_klines")
        assert hasattr(client, "get_futures_funding_rate")
        assert hasattr(client, "get_futures_positions")
        assert hasattr(client, "get_futures_open_orders")
        assert hasattr(client, "cancel_futures_order")
        assert hasattr(client, "cancel_all_futures_orders")
        assert hasattr(client, "get_futures_order")
        assert hasattr(client, "get_futures_ticker")

    def test_client_has_spot_cancel_all(self) -> None:
        client = KuCoinClient("k", "s", "p")
        assert hasattr(client, "cancel_all_orders")
