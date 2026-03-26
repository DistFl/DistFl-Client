"""WebSocket connection manager with automatic reconnect and heartbeat.

Implements the server protocol:
  - Connect: GET /ws/:room_id?client_id=xxx
  - Messages: Binary (gzip-compressed JSON) or Text (plain JSON)
  - Server sends sync payload on connect automatically
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional

import websockets
import websockets.exceptions

from fl_client.communication.compressor import compress, decompress

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages the WebSocket connection to the FL server.

    Features:
        - Automatic reconnect with exponential backoff
        - Heartbeat via WebSocket ping/pong
        - Message sending with gzip compression
        - Async message receiving loop

    Usage::

        conn = ConnectionManager(
            server_url="ws://localhost:8080",
            room_id="R123",
            client_id="client-abc",
        )
        await conn.connect()
        await conn.send({"type": "model_update", "payload": {...}})
        message = await conn.receive()
    """

    def __init__(
        self,
        server_url: str,
        room_id: str,
        client_id: str,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        max_retries: int = 0,
        heartbeat_interval: float = 30.0,
    ) -> None:
        self._server_url = server_url.rstrip("/")
        self._room_id = room_id
        self._client_id = client_id
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._max_retries = max_retries
        self._heartbeat_interval = heartbeat_interval

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._reconnect_attempt = 0
        self._should_run = False
        self._heartbeat_task: Optional[asyncio.Task] = None

    @property
    def ws_url(self) -> str:
        """Full WebSocket URL including room_id and client_id."""
        return f"{self._server_url}/ws/{self._room_id}?client_id={self._client_id}"

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    # ── Connection lifecycle ──────────────────────────────────────────────

    async def connect(self) -> None:
        """Establish the WebSocket connection.

        On success, the server automatically sends a sync payload with
        the latest global model and current round.
        """
        self._should_run = True
        await self._do_connect()

    async def _do_connect(self) -> None:
        """Internal connection logic with error handling."""
        try:
            logger.info("Connecting to %s", self.ws_url)
            self._ws = await websockets.connect(
                self.ws_url,
                ping_interval=self._heartbeat_interval,
                ping_timeout=self._heartbeat_interval * 2,
                max_size=50 * 1024 * 1024,  # 50 MB max message size
                close_timeout=10,
            )
            self._connected = True
            self._reconnect_attempt = 0
            logger.info("Connected successfully to %s", self.ws_url)
        except Exception as e:
            self._connected = False
            logger.error("Connection failed: %s", e)
            raise

    async def disconnect(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._should_run = False
        self._connected = False

        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        logger.info("Disconnected from server")

    # ── Reconnect ─────────────────────────────────────────────────────────

    async def reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff.

        Backoff: base_delay * 2^attempt, capped at max_delay.
        On reconnect, the server automatically sends a sync payload.
        """
        while self._should_run:
            if self._max_retries > 0 and self._reconnect_attempt >= self._max_retries:
                logger.error("Max reconnect retries (%d) reached. Aborting connection.", self._max_retries)
                self._should_run = False
                raise ConnectionError("Max reconnect retries reached")

            delay = min(
                self._base_delay * (2 ** self._reconnect_attempt),
                self._max_delay,
            )
            self._reconnect_attempt += 1

            logger.info(
                "Reconnecting in %.1fs (attempt %d)",
                delay,
                self._reconnect_attempt,
            )
            await asyncio.sleep(delay)

            try:
                await self._do_connect()
                logger.info("Reconnected successfully")
                return
            except Exception as e:
                logger.warning("Reconnect attempt %d failed: %s", self._reconnect_attempt, e)

        logger.info("Reconnect loop stopped (should_run=False)")

    # ── Messaging ─────────────────────────────────────────────────────────

    async def send(self, message: Dict[str, Any]) -> None:
        """Send a message to the server as gzip-compressed binary.

        Args:
            message: Dictionary to JSON-encode, compress, and send.

        Raises:
            ConnectionError: If not connected.
        """
        if not self.is_connected or self._ws is None:
            raise ConnectionError("Not connected to server")

        try:
            json_bytes = json.dumps(message).encode("utf-8")
            compressed = compress(json_bytes)
            await self._ws.send(compressed)
            logger.debug(
                "Sent message type=%s (%d bytes compressed)",
                message.get("type", "unknown"),
                len(compressed),
            )
        except websockets.exceptions.ConnectionClosed:
            self._connected = False
            raise ConnectionError("Connection closed while sending")

    async def receive(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Receive and decode one message from the server.

        The server sends binary (gzip-compressed JSON) or text (plain JSON).

        Args:
            timeout: Max seconds to wait for a message. None = wait forever.

        Returns:
            Parsed message dictionary, or None if connection closed or timeout.
        """
        if not self.is_connected or self._ws is None:
            return None

        try:
            if timeout is not None:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
            else:
                raw = await self._ws.recv()

            if isinstance(raw, bytes):
                # Binary message → decompress
                try:
                    decompressed = decompress(raw)
                    data = json.loads(decompressed)
                except RuntimeError:
                    # Fallback: maybe it's uncompressed binary JSON
                    data = json.loads(raw)
            else:
                # Text message → parse directly
                data = json.loads(raw)

            logger.debug("Received message type=%s", data.get("type", "unknown"))
            return data

        except asyncio.TimeoutError:
            return None
        except websockets.exceptions.ConnectionClosed:
            self._connected = False
            logger.warning("Connection closed during receive")
            return None
        except json.JSONDecodeError as e:
            logger.warning("Failed to decode message: %s", e)
            return None

    # ── Message loop ──────────────────────────────────────────────────────

    async def listen(
        self,
        handler: Callable[[Dict[str, Any]], Any],
    ) -> None:
        """Run the main message-receiving loop with auto-reconnect.

        Args:
            handler: Async callable invoked with each received message dict.
        """
        while self._should_run:
            if not self.is_connected:
                try:
                    await self.reconnect()
                except Exception:
                    continue

            message = await self.receive()
            if message is None:
                # Connection lost
                if self._should_run:
                    logger.warning("Connection lost, initiating reconnect")
                    self._connected = False
                    continue
                break

            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error("Message handler error: %s", e, exc_info=True)
