"""Federated Learning client configuration."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class FLConfig:
    """Configuration for the FL client.

    Attributes:
        server_url: WebSocket URL of the FL server (e.g. ws://localhost:8080).
        server_http_url: HTTP URL derived from server_url for REST calls.
        room_id: The room to join.
        client_id: Unique client identifier (auto-generated if omitted).
        invite_code: Invite code for the room (optional).
        data_path: Path to the local CSV dataset.
        batch_size: Mini-batch size for local training.
        local_epochs: Number of local training epochs per round.
        learning_rate: Optimizer learning rate.
        db_path: Path to the SQLite state database.
        reconnect_max_delay: Maximum reconnect backoff in seconds.
        reconnect_base_delay: Initial reconnect delay in seconds.
        heartbeat_interval: WebSocket ping interval in seconds.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        label_column: Name of the label/target column in the CSV.
        feature_columns: List of feature column names (None = auto-detect).
        schema: Expected data schema string for server validation.
        data_schema: Structured data schema dict from server.
        training_config: Training config dict from server.
        model_config: Model configuration dict from server.
        dashboard_port: Port for the metrics dashboard (0 = disabled).
    """

    server_url: str = "ws://localhost:8080"
    room_id: str = ""
    client_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    invite_code: str = ""
    data_path: str = ""
    batch_size: int = 32
    local_epochs: int = 2
    learning_rate: float = 0.001
    db_path: str = "fl_client_state.db"
    reconnect_max_delay: float = 60.0
    reconnect_base_delay: float = 1.0
    heartbeat_interval: float = 30.0
    log_level: str = "INFO"
    label_column: str = "label"
    feature_columns: Optional[List[str]] = None
    schema: str = ""
    data_schema: Optional[Dict[str, Any]] = None
    training_config: Optional[Dict[str, Any]] = None
    model_config: Optional[Dict[str, Any]] = None
    dashboard_port: int = 5050

    @property
    def server_http_url(self) -> str:
        """Derive the HTTP URL from the WebSocket URL."""
        url = self.server_url
        if url.startswith("wss://"):
            return url.replace("wss://", "https://", 1)
        if url.startswith("ws://"):
            return url.replace("ws://", "http://", 1)
        return url

    @classmethod
    def from_yaml(cls, path: str) -> "FLConfig":
        """Load configuration from a YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path, "r") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "FLConfig":
        """Load configuration from environment variables (FL_ prefix)."""
        env_map = {
            "server_url": os.getenv("FL_SERVER_URL"),
            "room_id": os.getenv("FL_ROOM_ID"),
            "client_id": os.getenv("FL_CLIENT_ID"),
            "invite_code": os.getenv("FL_INVITE_CODE"),
            "data_path": os.getenv("FL_DATA_PATH"),
            "batch_size": os.getenv("FL_BATCH_SIZE"),
            "local_epochs": os.getenv("FL_LOCAL_EPOCHS"),
            "learning_rate": os.getenv("FL_LEARNING_RATE"),
            "db_path": os.getenv("FL_DB_PATH"),
            "log_level": os.getenv("FL_LOG_LEVEL"),
            "label_column": os.getenv("FL_LABEL_COLUMN"),
            "schema": os.getenv("FL_SCHEMA"),
            "dashboard_port": os.getenv("FL_DASHBOARD_PORT"),
        }

        kwargs: Dict[str, Any] = {}
        type_map = {
            "batch_size": int,
            "local_epochs": int,
            "learning_rate": float,
            "reconnect_max_delay": float,
            "reconnect_base_delay": float,
            "heartbeat_interval": float,
            "dashboard_port": int,
        }

        for key, val in env_map.items():
            if val is not None:
                if key in type_map:
                    kwargs[key] = type_map[key](val)
                else:
                    kwargs[key] = val

        return cls(**kwargs)

    def validate(self) -> None:
        """Validate that required configuration fields are present."""
        if not self.server_url:
            raise ValueError("server_url is required")
        if not self.room_id:
            raise ValueError("room_id is required")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.local_epochs <= 0:
            raise ValueError("local_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def validate_for_training(self) -> None:
        """Validate config is ready for training (data_path required)."""
        self.validate()
        if not self.data_path:
            raise ValueError("data_path is required for training")
