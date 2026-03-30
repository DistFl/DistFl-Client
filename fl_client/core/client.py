"""Main FL Client — the orchestrator that ties all modules together.

Client-centric lifecycle API:

  Creator flow::

    client = FLClient(server_url="wss://fedlearn-server.onrender.com")
    room = client.create_room(model=model, data_path="data.csv", target="label", ...)
    client.wait_for_clients(min_clients=2)
    client.start_training()

  Joiner flow::

    client = FLClient(server_url="wss://fedlearn-server.onrender.com")
    client.join(room_id, invite_code="abc123")
    client.validate(data_path="data.csv")
    client.ready()
    client.start()
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import signal
import sys
import time as _time
import threading
from typing import Any, Dict, List, Optional

import httpx
import numpy as np

from fl_client.communication.compressor import compress, decompress
from fl_client.communication.serializer import (
    deserialize_weights,
    serialize_weights,
)
from fl_client.config.config import FLConfig
from fl_client.core.connection import ConnectionManager
from fl_client.core.state_manager import StateManager
from fl_client.model.wrapper import wrap_model, FLModelWrapper
from fl_client.training.dataset import (
    DatasetValidationError,
    load_data,
)
from fl_client.training.trainer import Trainer
from fl_client.validation.checks import validate_loss, validate_weights

logger = logging.getLogger(__name__)


class FLClient:
    """Production-grade Federated Learning Client.

    Creator flow::

        client = FLClient(server_url="wss://fedlearn-server.onrender.com")
        room = client.create_room(model=model, data_path="data.csv", target="label")
        client.wait_for_clients(min_clients=2)
        client.start_training()

    Joiner flow::

        client = FLClient(server_url="wss://fedlearn-server.onrender.com")
        client.join(room_id, invite_code="abc123")
        client.validate("data.csv")
        client.ready()
        client.start()
    """

    def __init__(
        self,
        server_url: str = "wss://fedlearn-server.onrender.com",
        room_id: str = "",
        invite_code: str = "",
        client_id: Optional[str] = None,
        db_path: str = "fl_client_state.db",
        log_level: str = "INFO",
        dashboard_port: int = 5050,
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        self._config = FLConfig(
            server_url=server_url,
            room_id=room_id,
            invite_code=invite_code,
            db_path=db_path,
            log_level=log_level,
            dashboard_port=dashboard_port,
        )
        if client_id:
            self._config.client_id = client_id

        logging.basicConfig(
            level=getattr(logging, self._config.log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # State
        self._state_manager: Optional[StateManager] = None
        self._connection: Optional[ConnectionManager] = None
        self._trainer: Optional[Trainer] = None
        self._model: Optional[FLModelWrapper] = wrap_model(model) if model is not None else None
        self._dataloader = None
        self._dataset_metadata: Optional[Dict] = None
        self._model_config: Optional[Dict] = None
        self._data_schema: Optional[Dict] = None
        self._training_config: Optional[Dict] = None
        self._data_path: Optional[str] = None
        self._initialized = False
        self._validated = False
        self._running = False
        self._training_in_progress = False
        self._is_creator = False

        # Round limit (0 = unlimited)
        self._max_rounds = 0
        self._completed_rounds = 0

        # Metrics tracking
        self._metrics_history: List[Dict[str, Any]] = []
        self._dashboard = None

    @classmethod
    def from_config(cls, config: FLConfig) -> "FLClient":
        """Create an FLClient from an FLConfig instance."""
        client = cls(
            server_url=config.server_url,
            room_id=config.room_id,
            invite_code=config.invite_code,
            client_id=config.client_id,
            db_path=config.db_path,
            log_level=config.log_level,
            dashboard_port=config.dashboard_port,
        )
        client._config = config
        if config.data_path:
            client._data_path = config.data_path
        return client

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC API — 3-step lifecycle
    # ═══════════════════════════════════════════════════════════════════════

    def initialize(self, model: Any = None) -> None:
        """Step 1: Fetch model_config, schema, weights from server.

        Connects to GET /room_info to retrieve the room configuration.
        Must be called before validate().

        Raises:
            RuntimeError: If room_id is not set or server is unreachable.
        """
        if not self._config.room_id:
            raise RuntimeError("room_id is required. Set it or call create_room() first.")

        if model is not None:
            self._model = wrap_model(model)

        asyncio.run(self._do_initialize())
        self._initialized = True
        logger.info("✅ Initialization complete")

    async def _do_initialize(self) -> None:
        """Async initialization — fetch room info from server."""
        url = f"{self._config.server_http_url}/room_info?room_id={self._config.room_id}"
        logger.info("Fetching room info from %s", url)

        async with httpx.AsyncClient(timeout=30.0) as http:
            try:
                resp = await http.get(url)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"Failed to fetch room info: {e.response.text}") from e
            except httpx.RequestError as e:
                raise RuntimeError(f"Cannot reach server: {e}") from e

        # Store room config with fallback against null API responses
        self._model_config = data.get("model_config") or {}
        self._data_schema = data.get("data_schema") or {}
        self._training_config = data.get("training_config") or {}

        # Apply training config to local config
        if self._training_config:
            tc = self._training_config
            self._config.local_epochs = tc.get("local_epochs", self._config.local_epochs)
            self._config.batch_size = tc.get("batch_size", self._config.batch_size)
            self._config.learning_rate = tc.get("learning_rate", self._config.learning_rate)

        # Apply label column from data_schema
        if self._data_schema and self._data_schema.get("target_column"):
            self._config.label_column = self._data_schema["target_column"]

        # Initialize model architecture info if provided by server
        if self._model_config and self._model_config.get("model_type"):
            logger.info("Room requires model_type: %s", self._model_config.get("model_type"))

        # Load initial weights if available
        weights_3d = data.get("weights")
        if weights_3d and self._model is not None:
            assert self._model is not None
            try:
                self._model.set_weights(weights_3d)
                logger.info("Initial weights loaded into model")
            except Exception as e:
                logger.warning("Could not load initial weights: %s", e)

        # Store in config for persistence
        self._config.data_schema = self._data_schema
        self._config.training_config = self._training_config
        self._config.model_config = self._model_config

        columns = self._data_schema.get("columns") or []
        target = self._data_schema.get("target_column") or "unknown"

        logger.info(
            "Room info — model_type=%s schema_columns=%d target=%s",
            self._model_config.get("model_type", "unknown"),
            len(columns),
            target,
        )

    def validate(self, data: Any) -> Dict[str, Any]:
        """Step 2: Validate dataset against the room's schema optionally.

        Args:
            data: The dataset. Can be a filepath, a PyTorch DataLoader, or a tuple (X,y).

        Returns:
            Metadata dict with num_samples, label_distribution, num_features.

        Raises:
            RuntimeError: If initialize() hasn't been called.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before validate()")

        self._data_path = str(data) if type(data).__name__ in ("str", "PosixPath", "WindowsPath") else "custom_data_object"
        self._config.data_path = self._data_path

        logger.info("Validating and parsing provided raw data structure...")
        adapter = load_data(data, schema=self._data_schema)
        self._train_data = adapter.train_data

        metadata = {
            "num_samples": adapter.num_samples,
            "label_distribution": adapter.label_distribution,
            "num_features": adapter.num_features,
            "num_classes": adapter.num_classes,
        }

        self._dataset_metadata = metadata
        self._validated = True

        if self._model is not None:
            logger.info("Running dummy forward pass to validate model inputs...")
            try:
                if hasattr(self._train_data, "__iter__") and not isinstance(self._train_data, tuple):
                    batch = next(iter(self._train_data))
                    batch_features, batch_labels = batch[0], batch[1]
                elif isinstance(self._train_data, tuple) and len(self._train_data) == 2:
                    batch_features = self._train_data[0][:2]
                    batch_labels = self._train_data[1][:2]
                else:
                    logger.warning("Unrecognized train_data format. Bypassing dummy pass.")
                    batch_features, batch_labels = None, None
                    
                if batch_features is not None:
                    self._model.validate_dummy_pass(batch_features, batch_labels)
                    logger.info("✅ Dummy forward pass succeeded")
            except StopIteration:
                logger.warning("Dataset is empty, skipping dummy forward pass")
            except Exception as e:
                self._validated = False
                raise RuntimeError(f"Dummy model pass failed: {e}") from e

        logger.info(
            "✅ Dataset validation passed — samples=%d features=%d classes=%d",
            metadata["num_samples"], metadata.get("num_features", 0), metadata.get("num_classes", 0)
        )

        return metadata

    def start(self, max_rounds: int = 0) -> None:
        """Step 3: Begin federated training.

        Joins the room, connects via WebSocket, and enters the training event loop.

        Args:
            max_rounds: Stop after this many rounds (0 = train forever).

        Raises:
            RuntimeError: If validate() hasn't been called.
        """
        if not self._validated:
            raise RuntimeError("Call validate() before start()")

        self._max_rounds = max_rounds
        self._completed_rounds = 0

        try:
            asyncio.run(self._run_async())
        except KeyboardInterrupt:
            logger.info("Client interrupted by user")

    # ═══════════════════════════════════════════════════════════════════════
    # ROOM STATUS / MEMBER COUNT
    # ═══════════════════════════════════════════════════════════════════════

    def room_status(self) -> Dict[str, Any]:
        """Query the server for the current room status.

        Returns a dict with keys like:
            - ``num_clients``: number of members in the room
            - ``state``: room state (waiting, ready, running, ...)
            - ``current_round``: current training round
            - ``clients``: list of client details
            - ``name``, ``invite_code``, etc.

        Raises:
            RuntimeError: If room_id is not set or server is unreachable.

        Example::

            cl = FLClient(server_url="wss://fedlearn-server.onrender.com", room_id="R123")
            cl.initialize()
            status = cl.room_status()
            print(f"Members in room: {status['num_clients']}")
        """
        if not self._config.room_id:
            raise RuntimeError("room_id is required. Set it or call create_room() first.")

        return asyncio.run(self._do_room_status())

    async def _do_room_status(self) -> Dict[str, Any]:
        url = f"{self._config.server_http_url}/room_status?room_id={self._config.room_id}"
        async with httpx.AsyncClient(timeout=15.0) as http:
            try:
                resp = await http.get(url)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"Failed to fetch room status: {e.response.text}") from e
            except httpx.RequestError as e:
                raise RuntimeError(f"Cannot reach server: {e}") from e

    # ═══════════════════════════════════════════════════════════════════════
    # ROOM CREATION
    # ═══════════════════════════════════════════════════════════════════════

    def create_room(
        self,
        model: Any,
        data_path: str = "",
        target: str = "label",
        training_config: Optional[Dict[str, Any]] = None,
        room_name: str = "fl-room",
        model_config: Optional[Dict[str, Any]] = None,
        data_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new federated learning room.

        Args:
            model: The instantiated PyTorch (nn.Module) or Scikit-Learn model.
            data_path: Path to a CSV file to infer schema from.
            target: Name of the target/label column.
            training_config: Training params (local_epochs, batch_size, learning_rate).
            room_name: Human-readable room name.
            model_config: Optional metadata about the model architecture.
            data_schema: Explicit schema dict. If omitted, inferred from data_path.

        Returns:
            Dict with room metadata including ``id`` and ``invite_code``.
        """
        if training_config is None:
            training_config = {
                "local_epochs": 2,
                "batch_size": 32,
                "learning_rate": 0.001,
            }

        # Infer schema from CSV if data_path is provided and no explicit schema
        if data_schema is None and data_path:
            import pandas as pd
            df = pd.read_csv(data_path, nrows=5)
            columns = [c for c in df.columns if c != target]
            data_schema = {
                "columns": [{"name": c, "type": "float"} for c in columns],
                "target_column": target,
            }
            logger.info("Inferred schema from %s — %d features, target=%s", data_path, len(columns), target)
        elif data_schema is None:
            data_schema = {"columns": [], "target_column": target}

        # Build wrapper locally and generate initial weights
        logger.info("Wrapping model and extracting initial weights...")
        self._model = wrap_model(model)

        try:
            initial_weights = self._model.get_weights()
        except RuntimeError:
            logger.info("Model cannot provide initial weights before being fitted. Providing empty initial weights.")
            initial_weights = []

        # Store locally
        self._model_config = model_config or {"model_type": self._model.model_type}
        self._data_schema = data_schema
        self._training_config = training_config
        self._is_creator = True

        # Apply to config
        self._config.label_column = data_schema.get("target_column", "label")
        self._config.local_epochs = training_config.get("local_epochs", 2)
        self._config.batch_size = training_config.get("batch_size", 32)
        self._config.learning_rate = training_config.get("learning_rate", 0.001)
        self._config.data_schema = data_schema
        self._config.training_config = training_config
        self._config.model_config = model_config

        result = asyncio.run(
            self._do_create_room(room_name, self._model_config, self._data_schema, self._training_config, initial_weights)
        )

        # Set room_id from server response
        room = result.get("room", {})
        self._config.room_id = room.get("id", "")
        self._initialized = True

        # Build a simple return dict
        room_info = {
            "id": room.get("id", ""),
            "invite_code": room.get("invite_code", ""),
            "name": room.get("name", room_name),
            "state": room.get("state", "waiting"),
        }

        logger.info(
            "✅ Room created — id=%s invite_code=%s",
            room_info["id"],
            room_info["invite_code"],
        )

        return room_info

    async def _do_create_room(
        self, name, model_config, data_schema, training_config, initial_weights
    ) -> Dict[str, Any]:
        payload = {
            "name": name,
            "creator_id": self._config.client_id,
            "model_config": model_config,
            "data_schema": data_schema,
            "training_config": training_config,
            "initial_weights": initial_weights,
        }

        url = f"{self._config.server_http_url}/create_room"
        logger.info("Creating room via %s", url)

        async with httpx.AsyncClient(timeout=30.0) as http:
            try:
                resp = await http.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                error_body = e.response.text if hasattr(e.response, "text") else "No response body"
                raise RuntimeError(f"Create room failed: {error_body}") from e
            except httpx.RequestError as e:
                raise RuntimeError(f"Cannot reach server: {e}") from e

    def join(
        self,
        room_id: str,
        invite_code: str = "",
        model: Any = None,
    ) -> None:
        """Join an existing room.

        Registers this client with the room and fetches room config.

        Args:
            room_id: The room to join.
            invite_code: Invite code for the room.
            model: The local model instance (PyTorch or Scikit-Learn).
        """
        self._config.room_id = room_id
        if invite_code:
            self._config.invite_code = invite_code
        self._is_creator = False

        # Initialize (fetches room config from server)
        self.initialize(model=model)
        logger.info("✅ Joined room %s", room_id)

    # ═══════════════════════════════════════════════════════════════════════
    # LOBBY — ready / wait_for_clients / start_training
    # ═══════════════════════════════════════════════════════════════════════

    def ready(self, device_info: Optional[Dict[str, Any]] = None) -> None:
        """Signal readiness to the server.

        Call after validate(). Tells the server this client is prepared
        to begin training.

        Args:
            device_info: Optional hardware info (gpu, ram_gb, etc.).
        """
        if not self._validated:
            raise RuntimeError("Call validate() before ready()")

        asyncio.run(self._do_ready(device_info))
        logger.info("✅ Client marked as ready")

    async def _do_ready(self, device_info: Optional[Dict[str, Any]]) -> None:
        if not getattr(self, "_has_joined", False):
            await self._join_room()
            self._has_joined = True

        url = f"{self._config.server_http_url}/client_ready"
        payload = {
            "room_id": self._config.room_id,
            "client_id": self._config.client_id,
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

    def wait_for_clients(self, min_clients: int = 1, timeout: int = 300) -> None:
        """Block until the room has at least ``min_clients`` participants.

        Polls ``GET /room_status`` every 3 seconds.

        Args:
            min_clients: Minimum number of clients required.
            timeout: Maximum seconds to wait before raising.

        Raises:
            RuntimeError: If room_id is not set.
            TimeoutError: If timeout is exceeded.
        """
        if not self._config.room_id:
            raise RuntimeError("room_id is required. Call create_room() first.")

        logger.info("⏳ Waiting for %d client(s) (timeout=%ds)...", min_clients, timeout)
        start = _time.time()

        while True:
            status = self.room_status()
            num = status.get("num_clients", 0)
            if num >= min_clients:
                logger.info("✅ %d client(s) connected", num)
                return
            elapsed = _time.time() - start
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Timed out waiting for clients: {num}/{min_clients} after {int(elapsed)}s"
                )
            _time.sleep(3)

    def start_training(self) -> None:
        """Trigger training for all participants (creator only).

        Sends ``POST /start_training`` to the server, which broadcasts
        ``start_training`` + global model to all connected clients.

        Raises:
            RuntimeError: If server rejects the request.
        """
        asyncio.run(self._do_start_training())
        logger.info("✅ Training started")

    async def _do_start_training(self) -> None:
        url = f"{self._config.server_http_url}/start_training"
        payload = {
            "room_id": self._config.room_id,
            "client_id": self._config.client_id,
        }
        async with httpx.AsyncClient(timeout=15.0) as http:
            try:
                resp = await http.post(url, json=payload)
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"Start training failed: {e.response.text}") from e
            except httpx.RequestError as e:
                raise RuntimeError(f"Cannot reach server: {e}") from e

    # ═══════════════════════════════════════════════════════════════════════
    # BACKWARD-COMPATIBLE run() METHOD
    # ═══════════════════════════════════════════════════════════════════════

    def run(self) -> None:
        """Run the FL client (blocking). Legacy compatibility method.

        For new code, use initialize() → validate() → start() instead.
        """
        try:
            asyncio.run(self._run_async())
        except KeyboardInterrupt:
            logger.info("Client interrupted by user")

    # ═══════════════════════════════════════════════════════════════════════
    # INTERNAL — async lifecycle
    # ═══════════════════════════════════════════════════════════════════════

    async def _run_async(self) -> None:
        self._running = True

        # Signal handlers can only be set in the main thread
        if threading.current_thread() is threading.main_thread():
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self._shutdown()))

        try:
            await self._setup_components()
            if not getattr(self, "_has_joined", False):
                await self._join_room()
                self._has_joined = True
            await self._connect_websocket()

            # Start dashboard
            self._start_dashboard()

            await self._event_loop()
        except Exception as e:
            logger.error("Fatal error: %s", e, exc_info=True)
        finally:
            await self._cleanup()

    async def _shutdown(self) -> None:
        logger.info("Shutting down...")
        self._running = False

    async def _setup_components(self) -> None:
        """Initialize state manager, dataset, and trainer."""
        # State manager
        self._state_manager = StateManager(
            db_path=self._config.db_path,
            client_id=self._config.client_id,
            room_id=self._config.room_id,
        )
        self._state_manager.restore_or_init()

        # Persist schema & model_config
        if self._data_schema:
            self._state_manager.save_schema(self._data_schema)
        if self._model_config:
            self._state_manager.save_model_config(self._model_config)

        if not hasattr(self, "_train_data") or self._train_data is None:
            raise RuntimeError("Data was never explicitly validated/loaded via validate().")

        assert self._state_manager is not None, (
            "StateManager failed to initialize. Check db_path and permissions."
        )

        self._state_manager.set_dataset_metadata(
            num_samples=self._dataset_metadata.get("num_samples", 0) if self._dataset_metadata else 0,
            label_distribution=self._dataset_metadata.get("label_distribution", {}) if self._dataset_metadata else {},
        )
        dataset_samples = self._dataset_metadata.get("num_samples", 0) if self._dataset_metadata else 0

        # Trainer
        self._trainer = Trainer(
            local_epochs=self._config.local_epochs,
            learning_rate=self._config.learning_rate,
        )

        logger.info(
            "Components ready — client=%s room=%s samples=%d",
            self._config.client_id, self._config.room_id, dataset_samples,
        )

    async def _join_room(self) -> None:
        """Join the FL room via HTTP POST /join_room."""
        meta = getattr(self, "_dataset_metadata", {})

        url = f"{self._config.server_http_url}/join_room"
        payload = {
            "room_id": self._config.room_id,
            "client_id": self._config.client_id,
            "num_samples": meta.get("num_samples", 0),
            "label_distribution": meta.get("label_distribution", {}),
        }

        logger.info("Joining room via %s", url)

        async with httpx.AsyncClient(timeout=30.0) as http:
            try:
                resp = await http.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                logger.info("Joined room — %s", data.get("message", "ok"))
            except httpx.HTTPStatusError as e:
                err = e.response.json() if e.response.content else {}
                raise RuntimeError(f"Join failed: {err.get('error', str(e))}") from e
            except httpx.RequestError as e:
                raise RuntimeError(f"Cannot reach server: {e}") from e

    async def _connect_websocket(self) -> None:
        """Establish WebSocket connection."""
        self._connection = ConnectionManager(
            server_url=self._config.server_url,
            room_id=self._config.room_id,
            client_id=self._config.client_id,
            base_delay=self._config.reconnect_base_delay,
            max_delay=self._config.reconnect_max_delay,
            max_retries=self._config.reconnect_max_retries,
            heartbeat_interval=self._config.heartbeat_interval,
        )
        await self._connection.connect()

    async def _event_loop(self) -> None:
        """Main message loop. Checks _running flag between messages."""
        assert self._connection is not None
        logger.info("Entering event loop")

        while self._running:
            if not self._connection.is_connected:
                if not self._running:
                    break
                try:
                    await self._connection.reconnect()
                except Exception:
                    continue

            # Use timeout so we can periodically check _running flag
            message = await self._connection.receive(timeout=5.0)
            if message is None:
                # Timeout (normal) or connection lost
                if not self._connection.is_connected and self._running:
                    logger.warning("Connection lost, initiating reconnect")
                    self._connection._connected = False
                continue

            try:
                await self._handle_message(message)
            except Exception as e:
                logger.error("Message handler error: %s", e, exc_info=True)

        logger.info("Event loop exited (running=%s)", self._running)

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Dispatch incoming server messages."""
        msg_type = message.get("type", "")

        if msg_type in ("global_model", "start_round", "new_global_model", "sync"):
            await self._handle_global_model(message)
        elif msg_type == "start_training":
            await self._handle_start_training(message)
        elif msg_type == "room_init":
            await self._handle_room_init(message)
        elif msg_type == "update_result":
            self._handle_update_result(message)
        elif msg_type == "error":
            self._handle_error(message)
        else:
            logger.warning("Unknown message type: %s", msg_type)

    # ── Message handlers ──────────────────────────────────────────────────

    async def _handle_start_training(self, message: Dict[str, Any]) -> None:
        """Handle start_training message — server signals that training has begun."""
        round_number = message.get("round", 1)
        weights_3d = message.get("weights")

        logger.info("🚀 start_training received — round=%d", round_number)

        # Treat as a global_model trigger to begin the first training round
        await self._handle_global_model({
            "round": round_number,
            "weights": weights_3d,
        })

    async def _handle_room_init(self, message: Dict[str, Any]) -> None:
        """Handle room_init message (sent on WS connect)."""
        round_number = message.get("round", 0)
        weights_3d = message.get("weights")
        model_cfg = message.get("model_config")
        data_schema = message.get("data_schema")
        training_cfg = message.get("training_config")

        logger.info("Room init received — round=%d", round_number)

        # Update local config
        if model_cfg:
            self._model_config = model_cfg
        if data_schema:
            self._data_schema = data_schema
        if training_cfg:
            self._training_config = training_cfg

        # We no longer auto-build the model using the registry here.
        # It is expected that the client has provided the model locally via initialize().
        if self._model is None:
            logger.warning("No model found. Received model_config but cannot build it automatically in BYOM mode.")

        # Load weights
        if weights_3d and self._model is not None:
            try:
                self._model.set_weights(weights_3d)
                logger.info("Weights loaded from room_init")
            except Exception as e:
                logger.error("Failed to load init weights: %s", e)

        if self._state_manager:
            self._state_manager.update_round(round_number)

        # Bootstrap first round: treat room_init with weights as training trigger
        # This breaks the chicken-and-egg problem where server waits for updates
        # but clients wait for global_model broadcast
        if weights_3d and self._model is not None and self._trainer is not None:
            logger.info("Bootstrapping initial training round...")
            await self._handle_global_model({
                "round": round_number,
                "weights": weights_3d,
            })

    async def _handle_global_model(self, message: Dict[str, Any]) -> None:
        """Handle global_model — load weights, train, validate, send update."""
        assert self._state_manager is not None
        assert self._trainer is not None

        round_number = message.get("round", 0)
        weights_3d = message.get("weights")

        logger.info("Global model received — round=%d", round_number)
        self._state_manager.update_round(round_number)

        # Duplicate check
        if self._state_manager.has_submitted_round(round_number):
            logger.warning("Already submitted round %d, skipping", round_number)
            return

        if self._training_in_progress:
            logger.warning("Training in progress, skipping round %d", round_number)
            return

        # Load weights
        if weights_3d and self._model is not None:
            try:
                self._model.set_weights(weights_3d)
                self._state_manager.save_weights(weights_3d)
            except Exception as e:
                logger.error("Failed to load weights: %s", e)
                return
        elif self._model is None:
            logger.error("No model available. Make sure to pass a model to initialize().")
            return

        # Train
        self._training_in_progress = True
        try:
            result = self._trainer.train(self._model, self._train_data)
        except Exception as e:
            logger.error("Training failed round %d: %s", round_number, e)
            self._state_manager.log_round(round_number, 0.0, 0, 0.0, status="failed")
            self._training_in_progress = False
            return
        finally:
            self._training_in_progress = False

        # Serialize & validate
        assert self._model is not None
        updated_weights = self._model.get_weights()

        w_valid, w_reason = validate_weights(updated_weights)
        if not w_valid:
            logger.error("Weight validation failed: %s", w_reason)
            self._state_manager.log_round(
                round_number, result.loss, result.num_samples,
                result.training_time, status="validation_failed",
            )
            return

        l_valid, l_reason = validate_loss(result.loss)
        if not l_valid:
            logger.error("Loss validation failed: %s", l_reason)
            self._state_manager.log_round(
                round_number, result.loss, result.num_samples,
                result.training_time, status="validation_failed",
            )
            return

        # Compute ΔW (weight update magnitude)
        delta_w = self._compute_delta_w(weights_3d, updated_weights)

        # Send
        update_msg = {
            "type": "model_update",
            "payload": {
                "round": round_number,
                "weights": updated_weights,
                "loss": result.loss,
                "metrics": result.metrics,
            },
        }

        try:
            assert self._connection is not None
            await self._connection.send(update_msg)
            self._state_manager.mark_round_submitted(round_number)
            self._state_manager.log_round(
                round_number, result.loss, result.num_samples,
                result.training_time, status="completed", delta_w=delta_w,
            )

            # Track metrics for dashboard
            self._track_metrics(round_number, result.loss, result.training_time,
                                result.num_samples, delta_w)

            logger.info(
                "Update sent — round=%d loss=%.4f samples=%d time=%.2fs ΔW=%.4f",
                round_number, result.loss, result.num_samples,
                result.training_time, delta_w,
            )

            # Check round limit
            self._completed_rounds += 1
            if self._max_rounds > 0 and self._completed_rounds >= self._max_rounds:
                logger.info("Reached max_rounds=%d, stopping training", self._max_rounds)
                self._running = False
        except ConnectionError as e:
            logger.error("Failed to send round %d: %s", round_number, e)
            self._state_manager.log_round(
                round_number, result.loss, result.num_samples,
                result.training_time, status="send_failed",
            )

    def _handle_update_result(self, message: Dict[str, Any]) -> None:
        payload = message.get("payload", {})
        if payload.get("valid", False):
            logger.info("Server accepted update")
        else:
            logger.warning("Server rejected update: %s", payload.get("reason", ""))

    def _handle_error(self, message: Dict[str, Any]) -> None:
        payload = message.get("payload", {})
        logger.error("Server error: %s", payload.get("error", "unknown"))

    # ── Model Persistence ─────────────────────────────────────────────────

    def save_model(self, path: str = "fl_model.pt", fmt: str = "pt") -> str:
        """Save the current trained model to a local file.

        Args:
            path: Output file path. Defaults to ``fl_model.pt``.
            fmt:  Format — ``'pt'`` for PyTorch state_dict (default),
                  ``'full'`` for full model + metadata,
                  ``'onnx'`` for ONNX export.

        Returns:
            Absolute path to the saved file.

        Raises:
            RuntimeError: If no model has been trained yet.

        Example::

            cl = FLClient(server_url="ws://localhost:8080", room_id="R123")
            cl.initialize()
            cl.validate("./data.csv")
            cl.start()
            saved = cl.save_model("my_model.pt")
            print(f"Model saved to {saved}")
        """
        import torch

        if self._model is None:
            raise RuntimeError("No model available. Run training first.")

        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)

        if fmt == "pt":
            torch.save(self._model.state_dict(), abs_path)
            logger.info("✅ Model state_dict saved to %s", abs_path)

        elif fmt == "full":
            checkpoint = {
                "state_dict": self._model.state_dict(),
                "model_config": self._model_config,
                "training_config": self._training_config,
                "data_schema": self._data_schema,
                "round": self._state_manager.current_round if self._state_manager else 0,
                "client_id": self._config.client_id,
                "room_id": self._config.room_id,
                "metrics_history": self._metrics_history,
                "timestamp": _time.time(),
            }
            torch.save(checkpoint, abs_path)
            logger.info("✅ Full checkpoint saved to %s", abs_path)

        elif fmt == "onnx":
            try:
                input_size = self._model_config.get("input_size", 1) if self._model_config else 1
                dummy = torch.randn(1, input_size)
                torch.onnx.export(self._model, dummy, abs_path,
                                  input_names=["input"], output_names=["output"])
                logger.info("✅ ONNX model exported to %s", abs_path)
            except Exception as e:
                raise RuntimeError(f"ONNX export failed: {e}") from e
        else:
            raise ValueError(f"Unknown format '{fmt}'. Use 'pt', 'full', or 'onnx'.")

        return abs_path

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(
        self,
        data: Any,
        return_probs: bool = True,
    ) -> Dict[str, Any]:
        """Run inference on the trained model.

        Accepts multiple input formats and returns class predictions with
        optional probabilities.

        Args:
            data: Input data — one of:
                - ``torch.Tensor``: shape ``(N, input_size)`` or ``(input_size,)``
                - ``numpy.ndarray``: same shapes as above
                - ``list[list[float]]``: batch of feature vectors
                - ``list[float]``: single feature vector
                - ``str``: path to a CSV file (features only, no label column)
            return_probs: If True, include class probabilities in the output.

        Returns:
            Dict with keys:
                - ``predictions``: list of predicted class indices
                - ``probabilities`` *(optional)*: list of per-class probability vectors
                - ``logits``: raw model output tensor as nested list
                - ``num_samples``: number of input samples

        Raises:
            RuntimeError: If no model is available.
            ValueError: If input data format is unrecognized.

        Example::

            cl = FLClient(server_url="ws://localhost:8080", room_id="R123")
            cl.initialize()
            cl.validate("./data.csv")
            cl.start(max_rounds=3)

            # Predict from a list of features
            result = cl.predict([[0.5, -0.3, 1.2, ...]])
            print(result["predictions"])   # [2]
            print(result["probabilities"]) # [[0.05, 0.10, 0.85, ...]]

            # Predict from a CSV file
            result = cl.predict("./test_data.csv")
            print(result["predictions"])   # [2, 0, 1, 7, ...]

            # Predict from a saved model (static method)
            result = FLClient.predict_from_file(
                model_path="my_model.pt",
                model_config={...},
                data=[[0.5, -0.3, 1.2, ...]],
            )
        """
        import torch

        if self._model is None:
            raise RuntimeError(
                "No model available. Run training first, or load a model "
                "with load_model()."
            )

        tensor = self._prepare_input(data)
        return self._run_inference(self._model, tensor, return_probs)

    @staticmethod
    def predict_from_file(
        model_path: str,
        model_config: Dict[str, Any],
        data: Any,
        return_probs: bool = True,
    ) -> Dict[str, Any]:
        """Run inference using a saved model file (no server connection needed).

        Args:
            model_path: Path to a saved ``.pt`` model file (state_dict or full
                checkpoint).
            model_config: Model architecture config dict (same as used for
                room creation).
            data: Input data (same formats as :meth:`predict`).
            return_probs: If True, include class probabilities.

        Returns:
            Same dict as :meth:`predict`.

        Example::

            result = FLClient.predict_from_file(
                model_path="saved_models/client1_model.pt",
                model_config={
                    "model_type": "mlp",
                    "input_size": 128,
                    "hidden_size": 8000,
                    "output_size": 10,
                },
                data=[[0.5, -0.3, 1.2, ...]],
            )
            print(result["predictions"])
        """
        import torch
        from fl_client.model.builder import build_model

        model = build_model(model_config)
        saved = torch.load(model_path, map_location="cpu", weights_only=False)

        # Handle both state_dict and full checkpoint formats
        if isinstance(saved, dict) and "state_dict" in saved:
            model.load_state_dict(saved["state_dict"])
        else:
            model.load_state_dict(saved)

        tensor = FLClient._prepare_input(data)
        return FLClient._run_inference(model, tensor, return_probs)

    @staticmethod
    def _prepare_input(data: Any) -> "torch.Tensor":
        """Convert various input formats to a 2D torch.Tensor."""
        import torch

        if isinstance(data, torch.Tensor):
            tensor = data.float()
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float()
        elif isinstance(data, str):
            # CSV file path
            tensor = FLClient._load_csv_features(data)
        elif isinstance(data, list):
            if len(data) == 0:
                raise ValueError("Input data list is empty")
            # Single sample: [f1, f2, ...] → [[f1, f2, ...]]
            if not isinstance(data[0], (list, tuple)):
                data = [data]
            tensor = torch.tensor(data, dtype=torch.float32)
        else:
            raise ValueError(
                f"Unsupported input type: {type(data).__name__}. "
                "Use torch.Tensor, numpy.ndarray, list, or a CSV file path."
            )

        # Ensure 2D: (N, features)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        return tensor

    @staticmethod
    def _load_csv_features(csv_path: str) -> "torch.Tensor":
        """Load features from a CSV file (all columns treated as float features)."""
        import torch
        import csv as csv_mod

        rows = []
        with open(csv_path, "r") as f:
            reader = csv_mod.reader(f)
            header = next(reader)  # skip header
            for row in reader:
                rows.append([float(v) for v in row])

        if not rows:
            raise ValueError(f"CSV file is empty: {csv_path}")

        return torch.tensor(rows, dtype=torch.float32)

    @staticmethod
    def _run_inference(
        model: "torch.nn.Module",
        tensor: "torch.Tensor",
        return_probs: bool,
    ) -> Dict[str, Any]:
        """Run forward pass and format results."""
        import torch
        import torch.nn.functional as F

        model.eval()
        with torch.no_grad():
            logits = model(tensor)

        predictions = torch.argmax(logits, dim=1).tolist()

        result: Dict[str, Any] = {
            "predictions": predictions,
            "logits": logits.tolist(),
            "num_samples": tensor.shape[0],
        }

        if return_probs:
            probs = F.softmax(logits, dim=1)
            result["probabilities"] = probs.tolist()

        return result

    def _compute_delta_w(
        self,
        old_weights: Optional[list],
        new_weights: list,
    ) -> float:
        """Compute L2 norm of weight change (ΔW)."""
        if not old_weights:
            return 0.0
        try:
            flat_old = []
            flat_new = []
            for layer in old_weights:
                for row in layer:
                    flat_old.extend(row)
            for layer in new_weights:
                for row in layer:
                    flat_new.extend(row)
            diff = [a - b for a, b in zip(flat_new, flat_old)]
            return math.sqrt(sum(d * d for d in diff))
        except Exception:
            return 0.0

    def _track_metrics(
        self, round_num: int, loss: float, training_time: float,
        num_samples: int, delta_w: float,
    ) -> None:
        """Track per-round metrics for dashboard."""
        entry = {
            "round": round_num,
            "loss": loss,
            "training_time": training_time,
            "num_samples": num_samples,
            "delta_w": delta_w,
            "participated": True,
            "timestamp": _time.time(),
        }
        self._metrics_history.append(entry)

        # Update dashboard if running
        if self._dashboard:
            try:
                self._dashboard.update(self._metrics_history)
            except Exception as e:
                logger.debug("Dashboard update failed: %s", e)

    def _start_dashboard(self) -> None:
        """Start the metrics dashboard in a background thread."""
        if self._config.dashboard_port <= 0:
            return
        try:
            from fl_client.dashboard.dashboard import MetricsDashboard
            self._dashboard = MetricsDashboard(
                port=self._config.dashboard_port,
                client_id=self._config.client_id,
                room_id=self._config.room_id,
            )
            self._dashboard.start()
            logger.info("📊 Dashboard running at http://localhost:%d", self._config.dashboard_port)
        except Exception as e:
            logger.warning("Could not start dashboard: %s", e)

    async def _cleanup(self) -> None:
        if self._connection:
            await self._connection.disconnect()
        if self._state_manager:
            self._state_manager.close()
        if self._dashboard:
            self._dashboard.stop()
        logger.info("Client shutdown complete")
