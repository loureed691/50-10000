"""KuCoin REST API client with HMAC authentication and rate-limit handling."""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import time
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# KuCoin rate-limit: 30 requests / 3 seconds for private endpoints
_RATE_LIMIT_WINDOW = 3.0
_RATE_LIMIT_MAX = 30


class KuCoinClient:
    """Async REST client for KuCoin spot and futures APIs."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        rest_url: str = "https://api.kucoin.com",
        futures_rest_url: str = "https://api-futures.kucoin.com",
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._api_passphrase = api_passphrase
        self._rest_url = rest_url.rstrip("/")
        self._futures_rest_url = futures_rest_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None
        self._request_times: list[float] = []
        self._time_offset_ms: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        await self._sync_time()

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Time sync
    # ------------------------------------------------------------------

    async def _sync_time(self) -> None:
        """Synchronize with KuCoin server clock."""
        try:
            data = await self._public_get("/api/v1/timestamp")
            server_ts = int(data.get("data", 0))
            local_ts = int(time.time() * 1000)
            self._time_offset_ms = server_ts - local_ts
            logger.info("Time offset: %d ms", self._time_offset_ms)
        except Exception:
            logger.warning("Time sync failed, using local clock", exc_info=True)

    def _timestamp_ms(self) -> int:
        return int(time.time() * 1000) + self._time_offset_ms

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> Dict[str, str]:
        str_to_sign = f"{timestamp}{method.upper()}{path}{body}"
        signature = base64.b64encode(
            hmac.new(
                self._api_secret.encode(),
                str_to_sign.encode(),
                hashlib.sha256,
            ).digest()
        ).decode()
        passphrase = base64.b64encode(
            hmac.new(
                self._api_secret.encode(),
                self._api_passphrase.encode(),
                hashlib.sha256,
            ).digest()
        ).decode()
        return {
            "KC-API-KEY": self._api_key,
            "KC-API-SIGN": signature,
            "KC-API-TIMESTAMP": timestamp,
            "KC-API-PASSPHRASE": passphrase,
            "KC-API-KEY-VERSION": "2",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    async def _throttle(self) -> None:
        now = time.monotonic()
        self._request_times = [t for t in self._request_times if now - t < _RATE_LIMIT_WINDOW]
        if len(self._request_times) >= _RATE_LIMIT_MAX:
            wait = _RATE_LIMIT_WINDOW - (now - self._request_times[0])
            if wait > 0:
                logger.debug("Rate limit reached, sleeping %.2fs", wait)
                await _async_sleep(wait)
        self._request_times.append(time.monotonic())

    # ------------------------------------------------------------------
    # HTTP methods
    # ------------------------------------------------------------------

    async def _public_get(self, path: str, params: Optional[dict] = None) -> dict:
        return await self._request("GET", path, params=params, signed=False)

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        body: Optional[dict] = None,
        signed: bool = True,
        base_url: Optional[str] = None,
    ) -> dict:
        assert self._session is not None, "Client not started"
        await self._throttle()

        url = (base_url or self._rest_url) + path
        headers: Dict[str, str] = {}
        body_str = ""

        if body:
            import json
            body_str = json.dumps(body)

        if signed:
            ts = str(self._timestamp_ms())
            query = ""
            if params:
                query = "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            headers = self._sign(ts, method, path + query, body_str)

        for attempt in range(3):
            try:
                async with self._session.request(
                    method, url, params=params, data=body_str if body_str else None, headers=headers
                ) as resp:
                    data = await resp.json()
                    if resp.status == 429:
                        wait = 2 ** attempt
                        logger.warning("Rate-limited (429), backing off %ds", wait)
                        await _async_sleep(wait)
                        continue
                    if resp.status >= 400:
                        logger.error("HTTP %d: %s", resp.status, data)
                    return data
            except (aiohttp.ClientError, ValueError) as exc:
                logger.warning("Request error (attempt %d): %s", attempt, exc)
                await _async_sleep(1)
        return {"code": "error", "msg": "max retries exceeded"}

    # ------------------------------------------------------------------
    # Spot market endpoints
    # ------------------------------------------------------------------

    async def get_symbols(self) -> List[dict]:
        """Fetch all trading symbols."""
        resp = await self._public_get("/api/v2/symbols")
        return resp.get("data", [])

    async def get_ticker(self, symbol: str) -> dict:
        resp = await self._public_get("/api/v1/market/orderbook/level1", {"symbol": symbol})
        return resp.get("data", {})

    async def get_klines(self, symbol: str, kline_type: str = "1hour", start: Optional[int] = None, end: Optional[int] = None) -> List[list]:
        params: dict = {"symbol": symbol, "type": kline_type}
        if start:
            params["startAt"] = start
        if end:
            params["endAt"] = end
        resp = await self._public_get("/api/v1/market/candles", params)
        return resp.get("data", [])

    async def get_orderbook(self, symbol: str, depth: int = 20) -> dict:
        resp = await self._public_get(f"/api/v1/market/orderbook/level2_{depth}", {"symbol": symbol})
        return resp.get("data", {})

    # ------------------------------------------------------------------
    # Account / balance endpoints
    # ------------------------------------------------------------------

    async def get_accounts(self, account_type: Optional[str] = None) -> List[dict]:
        params = {}
        if account_type:
            params["type"] = account_type
        resp = await self._request("GET", "/api/v1/accounts", params=params)
        return resp.get("data", [])

    async def get_account_balance(self, currency: str = "USDT") -> float:
        accounts = await self.get_accounts("trade")
        for acc in accounts:
            if acc.get("currency") == currency:
                return float(acc.get("available", 0))
        return 0.0

    # ------------------------------------------------------------------
    # Order endpoints
    # ------------------------------------------------------------------

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str = "limit",
        size: Optional[float] = None,
        price: Optional[float] = None,
        client_oid: Optional[str] = None,
        post_only: bool = False,
    ) -> dict:
        body: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
        }
        if client_oid:
            body["clientOid"] = client_oid
        else:
            import uuid
            body["clientOid"] = str(uuid.uuid4())
        if size is not None:
            body["size"] = str(size)
        if price is not None:
            body["price"] = str(price)
        if post_only:
            body["postOnly"] = True
        return await self._request("POST", "/api/v1/orders", body=body)

    async def cancel_order(self, order_id: str) -> dict:
        return await self._request("DELETE", f"/api/v1/orders/{order_id}")

    async def get_order(self, order_id: str) -> dict:
        resp = await self._request("GET", f"/api/v1/orders/{order_id}")
        return resp.get("data", {})

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[dict]:
        params = {}
        if symbol:
            params["symbol"] = symbol
        resp = await self._request("GET", "/api/v1/orders", params={**params, "status": "active"})
        return resp.get("data", {}).get("items", [])

    # ------------------------------------------------------------------
    # Internal transfer
    # ------------------------------------------------------------------

    async def inner_transfer(
        self,
        currency: str,
        from_account: str,
        to_account: str,
        amount: float,
        client_oid: Optional[str] = None,
    ) -> dict:
        import uuid
        body = {
            "clientOid": client_oid or str(uuid.uuid4()),
            "currency": currency,
            "from": from_account,
            "to": to_account,
            "amount": str(amount),
        }
        return await self._request("POST", "/api/v2/accounts/inner-transfer", body=body)

    # ------------------------------------------------------------------
    # Futures endpoints
    # ------------------------------------------------------------------

    async def get_futures_contracts(self) -> List[dict]:
        resp = await self._request(
            "GET", "/api/v1/contracts/active", signed=False, base_url=self._futures_rest_url
        )
        return resp.get("data", [])

    async def get_futures_position(self, symbol: str) -> dict:
        resp = await self._request(
            "GET", "/api/v1/position", params={"symbol": symbol}, base_url=self._futures_rest_url
        )
        return resp.get("data", {})

    async def place_futures_order(
        self,
        symbol: str,
        side: str,
        size: int,
        leverage: float = 1.0,
        order_type: str = "limit",
        price: Optional[float] = None,
        client_oid: Optional[str] = None,
        reduce_only: bool = False,
    ) -> dict:
        import uuid
        body: Dict[str, Any] = {
            "clientOid": client_oid or str(uuid.uuid4()),
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "size": size,
            "leverage": str(leverage),
        }
        if price is not None:
            body["price"] = str(price)
        if reduce_only:
            body["reduceOnly"] = True
        return await self._request("POST", "/api/v1/orders", body=body, base_url=self._futures_rest_url)


async def _async_sleep(seconds: float) -> None:
    import asyncio
    await asyncio.sleep(seconds)
