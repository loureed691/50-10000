"""KuCoin WebSocket client for real-time market data."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)


class KuCoinWebSocket:
    """Manages a WebSocket connection to KuCoin for ticker/orderbook/trade feeds."""

    def __init__(self, rest_url: str = "https://api.kucoin.com") -> None:
        self._rest_url = rest_url.rstrip("/")
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._callbacks: Dict[str, Callable] = {}
        self._running = False
        self._ping_interval: int = 30
        self._connect_id: int = 0

    async def start(self) -> None:
        """Obtain a WS token and connect."""
        token_data = await self._get_public_token()
        if not token_data:
            logger.error("Failed to get WS token")
            return
        self._session = aiohttp.ClientSession()
        endpoint = token_data["instanceServers"][0]["endpoint"]
        token = token_data["token"]
        self._ping_interval = token_data["instanceServers"][0].get("pingInterval", 30000) // 1000
        url = f"{endpoint}?token={token}&connectId={self._connect_id}"
        self._ws = await self._session.ws_connect(url)
        self._running = True
        logger.info("WebSocket connected")

    async def _get_public_token(self) -> Optional[dict]:
        try:
            if self._session is None:
                self._session = aiohttp.ClientSession()
            async with self._session.post(f"{self._rest_url}/api/v1/bullet-public") as resp:
                data = await resp.json()
                return dict(data.get("data")) if data.get("data") else None
        except Exception:
            logger.error("WS token request failed", exc_info=True)
            return None

    async def subscribe(self, topic: str, callback: Callable[[dict], Any]) -> None:
        """Subscribe to a topic (e.g., /market/ticker:BTC-USDT)."""
        self._callbacks[topic] = callback
        if self._ws:
            self._connect_id += 1
            msg = {
                "id": self._connect_id,
                "type": "subscribe",
                "topic": topic,
                "privateChannel": False,
                "response": True,
            }
            await self._ws.send_json(msg)
            logger.info("Subscribed to %s", topic)

    async def listen(self) -> None:
        """Main read loop – dispatch messages to callbacks."""
        if not self._ws:
            return
        try:
            while self._running:
                msg = await asyncio.wait_for(self._ws.receive(), timeout=self._ping_interval + 10)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    msg_type = data.get("type")
                    if msg_type == "message":
                        topic = data.get("topic", "")
                        cb = self._callbacks.get(topic)
                        if cb:
                            try:
                                result = cb(data.get("data", {}))
                                if asyncio.iscoroutine(result):
                                    await result
                            except Exception:
                                logger.error("Callback error for %s", topic, exc_info=True)
                    elif msg_type == "welcome":
                        logger.debug("WS welcome received")
                    elif msg_type == "pong":
                        pass
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.warning("WebSocket closed/error, reconnecting...")
                    break
        except asyncio.TimeoutError:
            logger.warning("WS receive timeout, sending ping")
            if self._ws:
                await self._ws.send_json({"id": str(time.time()), "type": "ping"})
        except Exception:
            logger.error("WS listen error", exc_info=True)

    async def close(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()
