"""RoomSession — high-level lobby-style API for federated learning rooms."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class RoomSession:
    """Represents an active session within a federated learning room.

    Returned by ``FLClient.create_room()`` (role=creator) or
    ``FLClient.join()`` (role=participant).

    Usage (Creator)::

        room = client.create_room(model, data_path="data.csv", ...)
        room.validate("data.csv")
        room.ready()
        room.start_training()         # Only creator can call this
        room.wait_for_training()

    Usage (Participant)::

        room = client.join("ROOM_ID")
        room.validate("data.csv")
        room.ready()
        room.wait_for_training()      # Blocks until training completes
    """

    def __init__(
        self,
        client: Any,  # FLClient — forward ref to avoid circular import
        room_id: str,
        role: str,  # "creator" | "participant"
        model: Any = None,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._client = client
        self.room_id = room_id
        self.role = role
        self._model = model
        self._training_config = training_config or {}
        self._validated = False
        self._ready = False

    @property
    def is_creator(self) -> bool:
        """True if this client created the room."""
        return self.role == "creator"

    # ── Data Validation ───────────────────────────────────────────────────

    def validate(self, data_path: str, strict: bool = False) -> Dict[str, Any]:
        """Validate the local dataset against the room's schema.

        Args:
            data_path: Path to the CSV dataset.
            strict: If True, raise on schema mismatch (default: warn).

        Returns:
            Metadata dict with num_samples, num_features, etc.
        """
        # Ensure client is initialized for this room
        if not self._client._initialized:
            self._client._config.room_id = self.room_id
            self._client.initialize(model=self._model)

        metadata = self._client.validate(data_path)
        self._validated = True
        logger.info("✅ Dataset validated — %d samples", metadata.get("num_samples", 0))
        return metadata

    # ── Ready Signal ──────────────────────────────────────────────────────

    def ready(self, device_info: Optional[Dict[str, Any]] = None) -> None:
        """Signal readiness to the server.

        Args:
            device_info: Optional hardware info (gpu, ram_gb, cpu_cores).
        """
        if not self._validated:
            raise RuntimeError("Call validate() before ready()")

        asyncio.run(self._do_ready(device_info))
        self._ready = True
        logger.info("✅ Client marked as ready")

    async def _do_ready(self, device_info: Optional[Dict[str, Any]]) -> None:
        url = f"{self._client._config.server_http_url}/client_ready"
        payload = {
            "room_id": self.room_id,
            "client_id": self._client._config.client_id,
            "device_info": device_info or {},
        }

        async with httpx.AsyncClient(timeout=15.0) as http:
            try:
                resp = await http.post(url, json=payload)
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"Ready signal failed: {e.response.text}") from e
            except httpx.RequestError as e:
                raise RuntimeError(f"Cannot reach server: {e}") from e

    # ── Start Training (Creator Only) ─────────────────────────────────────

    def start_training(self) -> None:
        """Trigger training for all ready participants.

        Only the room creator can call this.

        Raises:
            RuntimeError: If caller is not the creator or server rejects.
        """
        if not self.is_creator:
            raise RuntimeError("Only the room creator can call start_training()")

        asyncio.run(self._do_start_training())
        logger.info("✅ Training started by creator")

    async def _do_start_training(self) -> None:
        url = f"{self._client._config.server_http_url}/start_training"
        payload = {
            "room_id": self.room_id,
            "client_id": self._client._config.client_id,
        }

        async with httpx.AsyncClient(timeout=15.0) as http:
            try:
                resp = await http.post(url, json=payload)
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"Start training failed: {e.response.text}") from e
            except httpx.RequestError as e:
                raise RuntimeError(f"Cannot reach server: {e}") from e

    # ── Wait for Training ─────────────────────────────────────────────────

    def wait_for_training(self, max_rounds: int = 0) -> None:
        """Connect via WebSocket and run the training event loop.

        Blocks until training completes or max_rounds is reached.

        Args:
            max_rounds: Stop after N rounds (0 = train until server stops).
        """
        if not self._validated:
            raise RuntimeError("Call validate() before wait_for_training()")

        self._client.start(max_rounds=max_rounds)

    # ── Status ────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Query the server for the current room status."""
        return self._client.room_status()

    def __repr__(self) -> str:
        return f"RoomSession(room_id={self.room_id!r}, role={self.role!r})"
