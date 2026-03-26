"""SQLite-based persistent state storage for the FL client."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS client_state (
    client_id       TEXT PRIMARY KEY,
    room_id         TEXT NOT NULL,
    current_round   INTEGER NOT NULL DEFAULT 0,
    model_version   INTEGER NOT NULL DEFAULT 0,
    last_weights    TEXT,
    label_distribution TEXT,
    num_samples     INTEGER NOT NULL DEFAULT 0,
    submitted_rounds TEXT NOT NULL DEFAULT '[]',
    schema_json     TEXT,
    model_config_json TEXT,
    training_config_json TEXT,
    last_updated    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS round_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id       TEXT NOT NULL,
    round_number    INTEGER NOT NULL,
    loss            REAL,
    num_samples     INTEGER,
    training_time   REAL,
    delta_w         REAL,
    status          TEXT NOT NULL DEFAULT 'completed',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(client_id, round_number)
);
"""

# Migration to add new columns to existing tables
_MIGRATION_SQL = """
-- Add schema_json column if not exists
CREATE TABLE IF NOT EXISTS _migration_check (id INTEGER);
DROP TABLE _migration_check;
"""


@dataclass
class ClientState:
    """In-memory representation of the persisted client state."""

    client_id: str = ""
    room_id: str = ""
    current_round: int = 0
    model_version: int = 0
    last_weights: Optional[List] = None
    label_distribution: Optional[Dict[str, float]] = None
    num_samples: int = 0
    submitted_rounds: Set[int] = field(default_factory=set)
    schema_json: Optional[Dict[str, Any]] = None
    model_config_json: Optional[Dict[str, Any]] = None
    training_config_json: Optional[Dict[str, Any]] = None


class StateDB:
    """Thread-safe SQLite state database for crash recovery and persistence."""

    def __init__(self, db_path: str = "fl_client_state.db") -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
                timeout=10.0,
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
        return self._local.conn

    def _init_schema(self) -> None:
        """Create tables if they do not exist."""
        conn = self._get_conn()
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

        # Migrate existing tables (add columns that may not exist)
        self._migrate(conn)

        logger.debug("SQLite schema initialized at %s", self._db_path)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add new columns to existing tables (safe ALTERs)."""
        migrations = [
            ("client_state", "schema_json", "TEXT"),
            ("client_state", "model_config_json", "TEXT"),
            ("client_state", "training_config_json", "TEXT"),
            ("round_log", "delta_w", "REAL"),
        ]
        for table, column, col_type in migrations:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                conn.commit()
            except sqlite3.OperationalError:
                pass  # Column already exists

    # ── State persistence ─────────────────────────────────────────────────

    def save_state(self, state: ClientState) -> None:
        """Persist the client state to SQLite (upsert)."""
        conn = self._get_conn()
        weights_json = json.dumps(state.last_weights) if state.last_weights else None
        label_json = json.dumps(state.label_distribution) if state.label_distribution else None
        submitted_json = json.dumps(sorted(state.submitted_rounds))
        schema_json = json.dumps(state.schema_json) if state.schema_json else None
        model_config = json.dumps(state.model_config_json) if state.model_config_json else None
        training_config = json.dumps(state.training_config_json) if state.training_config_json else None

        conn.execute(
            """
            INSERT INTO client_state
                (client_id, room_id, current_round, model_version,
                 last_weights, label_distribution, num_samples,
                 submitted_rounds, schema_json, model_config_json,
                 training_config_json, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(client_id) DO UPDATE SET
                room_id = excluded.room_id,
                current_round = excluded.current_round,
                model_version = excluded.model_version,
                last_weights = excluded.last_weights,
                label_distribution = excluded.label_distribution,
                num_samples = excluded.num_samples,
                submitted_rounds = excluded.submitted_rounds,
                schema_json = excluded.schema_json,
                model_config_json = excluded.model_config_json,
                training_config_json = excluded.training_config_json,
                last_updated = CURRENT_TIMESTAMP
            """,
            (
                state.client_id,
                state.room_id,
                state.current_round,
                state.model_version,
                weights_json,
                label_json,
                state.num_samples,
                submitted_json,
                schema_json,
                model_config,
                training_config,
            ),
        )
        conn.commit()

    def load_state(self, client_id: str) -> Optional[ClientState]:
        """Load a previously persisted client state."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM client_state WHERE client_id = ?", (client_id,)
        ).fetchone()

        if row is None:
            return None

        weights = json.loads(row["last_weights"]) if row["last_weights"] else None
        labels = json.loads(row["label_distribution"]) if row["label_distribution"] else None
        submitted = set(json.loads(row["submitted_rounds"])) if row["submitted_rounds"] else set()

        # Handle new columns that might not exist in old DBs
        schema = None
        model_cfg = None
        training_cfg = None
        try:
            schema = json.loads(row["schema_json"]) if row["schema_json"] else None
            model_cfg = json.loads(row["model_config_json"]) if row["model_config_json"] else None
            training_cfg = json.loads(row["training_config_json"]) if row["training_config_json"] else None
        except (IndexError, KeyError):
            pass

        return ClientState(
            client_id=row["client_id"],
            room_id=row["room_id"],
            current_round=row["current_round"],
            model_version=row["model_version"],
            last_weights=weights,
            label_distribution=labels,
            num_samples=row["num_samples"],
            submitted_rounds=submitted,
            schema_json=schema,
            model_config_json=model_cfg,
            training_config_json=training_cfg,
        )

    def clear_state(self, client_id: str) -> None:
        """Remove persisted state for a client."""
        conn = self._get_conn()
        conn.execute("DELETE FROM client_state WHERE client_id = ?", (client_id,))
        conn.commit()

    # ── Round logging ─────────────────────────────────────────────────────

    def log_round(
        self,
        client_id: str,
        round_number: int,
        loss: float,
        num_samples: int,
        training_time: float,
        status: str = "completed",
        delta_w: float = 0.0,
    ) -> None:
        """Log a completed training round."""
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO round_log
                    (client_id, round_number, loss, num_samples, training_time, delta_w, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (client_id, round_number, loss, num_samples, training_time, delta_w, status),
            )
            conn.commit()
        except sqlite3.Error as e:
            logger.warning("Failed to log round %d: %s", round_number, e)

    def get_round_history(self, client_id: str) -> List[Dict[str, Any]]:
        """Get all round logs for a client, ordered by round number."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT round_number, loss, num_samples, training_time,
                      delta_w, status, created_at
               FROM round_log WHERE client_id = ? ORDER BY round_number""",
            (client_id,),
        ).fetchall()

        return [
            {
                "round": r["round_number"],
                "loss": r["loss"],
                "num_samples": r["num_samples"],
                "training_time": r["training_time"],
                "delta_w": r["delta_w"],
                "status": r["status"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
