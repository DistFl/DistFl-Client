"""Unit tests for WebSocket connection manager (mocked)."""

import asyncio
import gzip
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fl_client.core.connection import ConnectionManager


def _make_mock_ws():
    """Create a mock websocket that works with both await and context manager."""
    mock_ws = AsyncMock()
    mock_ws.close = AsyncMock()
    mock_ws.send = AsyncMock()
    mock_ws.recv = AsyncMock()
    return mock_ws


def _patch_connect(mock_ws):
    """Patch websockets.connect to return the mock_ws as an awaitable."""
    async def fake_connect(*args, **kwargs):
        return mock_ws
    return patch("fl_client.core.connection.websockets.connect", side_effect=fake_connect)


class TestConnectionManager:
    """Test the connection manager with mocked WebSocket."""

    def test_ws_url_construction(self):
        """WebSocket URL should include room_id and client_id."""
        conn = ConnectionManager(
            server_url="ws://localhost:8080",
            room_id="R123",
            client_id="client-abc",
        )
        assert conn.ws_url == "ws://localhost:8080/ws/R123?client_id=client-abc"

    def test_ws_url_strips_trailing_slash(self):
        conn = ConnectionManager(
            server_url="ws://localhost:8080/",
            room_id="R1",
            client_id="c1",
        )
        assert conn.ws_url == "ws://localhost:8080/ws/R1?client_id=c1"

    def test_initial_state(self):
        conn = ConnectionManager(
            server_url="ws://localhost:8080",
            room_id="R1",
            client_id="c1",
        )
        assert not conn.is_connected

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Successful connection should set is_connected=True."""
        conn = ConnectionManager(
            server_url="ws://localhost:8080",
            room_id="R1",
            client_id="c1",
        )

        mock_ws = _make_mock_ws()
        with _patch_connect(mock_ws):
            await conn.connect()

        assert conn.is_connected

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Disconnect should set is_connected=False."""
        conn = ConnectionManager(
            server_url="ws://localhost:8080",
            room_id="R1",
            client_id="c1",
        )

        mock_ws = _make_mock_ws()
        with _patch_connect(mock_ws):
            await conn.connect()

        await conn.disconnect()
        assert not conn.is_connected

    @pytest.mark.asyncio
    async def test_send_compresses_message(self):
        """Sent messages should be gzip-compressed binary."""
        conn = ConnectionManager(
            server_url="ws://localhost:8080",
            room_id="R1",
            client_id="c1",
        )

        mock_ws = _make_mock_ws()
        with _patch_connect(mock_ws):
            await conn.connect()

        message = {"type": "model_update", "payload": {"round": 1}}
        await conn.send(message)

        # Verify send was called with bytes (compressed)
        mock_ws.send.assert_called_once()
        sent_data = mock_ws.send.call_args[0][0]
        assert isinstance(sent_data, bytes)

        # Verify we can decompress and get original message
        decompressed = gzip.decompress(sent_data)
        restored = json.loads(decompressed)
        assert restored == message

    @pytest.mark.asyncio
    async def test_send_not_connected_raises(self):
        """Sending without connection should raise ConnectionError."""
        conn = ConnectionManager(
            server_url="ws://localhost:8080",
            room_id="R1",
            client_id="c1",
        )
        with pytest.raises(ConnectionError):
            await conn.send({"type": "test"})

    @pytest.mark.asyncio
    async def test_receive_binary_message(self):
        """Binary (gzip) messages should be decompressed and parsed."""
        conn = ConnectionManager(
            server_url="ws://localhost:8080",
            room_id="R1",
            client_id="c1",
        )

        payload = {"type": "global_model", "round": 1, "weights": [[[1.0]]]}
        compressed = gzip.compress(json.dumps(payload).encode())

        mock_ws = _make_mock_ws()
        mock_ws.recv = AsyncMock(return_value=compressed)

        with _patch_connect(mock_ws):
            await conn.connect()

        result = await conn.receive()
        assert result == payload

    @pytest.mark.asyncio
    async def test_receive_text_message(self):
        """Text JSON messages should be parsed directly."""
        conn = ConnectionManager(
            server_url="ws://localhost:8080",
            room_id="R1",
            client_id="c1",
        )

        payload = {"type": "error", "payload": {"error": "test"}}

        mock_ws = _make_mock_ws()
        mock_ws.recv = AsyncMock(return_value=json.dumps(payload))

        with _patch_connect(mock_ws):
            await conn.connect()

        result = await conn.receive()
        assert result == payload
