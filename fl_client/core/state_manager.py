"""State manager — wraps SQLite persistence with round-tracking logic."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from fl_client.storage.db import ClientState, StateDB

logger = logging.getLogger(__name__)


class StateManager:
    """Manages in-memory client state with SQLite-backed persistence.

    Ensures:
        - Crash recovery (load from SQLite on startup)
        - No duplicate updates per round
        - Round and model version tracking
        - Schema and model_config persistence
    """

    def __init__(
        self,
        db_path: str,
        client_id: str,
        room_id: str,
    ) -> None:
        self._db = StateDB(db_path)
        self._state = ClientState(client_id=client_id, room_id=room_id)

    @property
    def client_id(self) -> str:
        return self._state.client_id

    @property
    def room_id(self) -> str:
        return self._state.room_id

    @property
    def current_round(self) -> int:
        return self._state.current_round

    @property
    def model_version(self) -> int:
        return self._state.model_version

    @property
    def num_samples(self) -> int:
        return self._state.num_samples

    @property
    def label_distribution(self) -> Optional[Dict[str, float]]:
        return self._state.label_distribution

    @property
    def last_weights(self) -> Optional[List]:
        return self._state.last_weights

    # ── Recovery ──────────────────────────────────────────────────────────

    def restore_or_init(self) -> bool:
        """Restore state from SQLite or initialize fresh.

        Returns:
            True if state was restored from DB, False if fresh init.
        """
        existing = self._db.load_state(self._state.client_id)
        if existing is not None:
            self._state = existing
            logger.info(
                "State restored — round=%d model_version=%d",
                self._state.current_round,
                self._state.model_version,
            )
            return True

        logger.info("No previous state found, starting fresh")
        self._persist()
        return False

    # ── Round management ──────────────────────────────────────────────────

    def update_round(self, round_number: int) -> None:
        self._state.current_round = round_number
        self._persist()

    def mark_round_submitted(self, round_number: int) -> None:
        self._state.submitted_rounds.add(round_number)
        self._persist()

    def has_submitted_round(self, round_number: int) -> bool:
        return round_number in self._state.submitted_rounds

    # ── Weight management ─────────────────────────────────────────────────

    def save_weights(self, weights: List) -> None:
        self._state.last_weights = weights
        self._state.model_version += 1
        self._persist()

    # ── Metadata ──────────────────────────────────────────────────────────

    def set_dataset_metadata(
        self,
        num_samples: int,
        label_distribution: Dict[str, float],
    ) -> None:
        self._state.num_samples = num_samples
        self._state.label_distribution = label_distribution
        self._persist()

    # ── Schema & config persistence ───────────────────────────────────────

    def save_schema(self, schema: Dict[str, Any]) -> None:
        """Persist the data schema from the room."""
        self._state.schema_json = schema
        self._persist()

    def save_model_config(self, model_config: Dict[str, Any]) -> None:
        """Persist the model config from the room."""
        self._state.model_config_json = model_config
        self._persist()

    def save_training_config(self, training_config: Dict[str, Any]) -> None:
        """Persist the training config from the room."""
        self._state.training_config_json = training_config
        self._persist()

    # ── Round logging ─────────────────────────────────────────────────────

    def log_round(
        self,
        round_number: int,
        loss: float,
        num_samples: int,
        training_time: float,
        status: str = "completed",
        delta_w: float = 0.0,
    ) -> None:
        self._db.log_round(
            self._state.client_id,
            round_number,
            loss,
            num_samples,
            training_time,
            status,
            delta_w,
        )

    def get_round_history(self) -> List[Dict[str, Any]]:
        """Get all round logs for this client."""
        return self._db.get_round_history(self._state.client_id)

    # ── Cleanup ───────────────────────────────────────────────────────────

    def clear(self) -> None:
        self._db.clear_state(self._state.client_id)
        self._state = ClientState(
            client_id=self._state.client_id,
            room_id=self._state.room_id,
        )

    def close(self) -> None:
        self._db.close()

    def _persist(self) -> None:
        self._db.save_state(self._state)
