"""Unit tests for SQLite state persistence."""

import os
import tempfile

import pytest

from fl_client.storage.db import ClientState, StateDB


class TestStateDB:
    """Test SQLite state database operations."""

    def setup_method(self):
        """Create a temporary database for each test."""
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._db_path = self._tmpfile.name
        self._tmpfile.close()
        self.db = StateDB(self._db_path)

    def teardown_method(self):
        """Clean up temp database."""
        self.db.close()
        if os.path.exists(self._db_path):
            os.unlink(self._db_path)

    def test_save_and_load(self):
        """Save then load should return equivalent state."""
        state = ClientState(
            client_id="test-client",
            room_id="R123",
            current_round=3,
            model_version=2,
            num_samples=1000,
            label_distribution={"0": 0.6, "1": 0.4},
            submitted_rounds={1, 2, 3},
        )
        self.db.save_state(state)
        loaded = self.db.load_state("test-client")

        assert loaded is not None
        assert loaded.client_id == "test-client"
        assert loaded.room_id == "R123"
        assert loaded.current_round == 3
        assert loaded.model_version == 2
        assert loaded.num_samples == 1000
        assert loaded.label_distribution == {"0": 0.6, "1": 0.4}
        assert loaded.submitted_rounds == {1, 2, 3}

    def test_load_nonexistent(self):
        """Loading a non-existent client should return None."""
        result = self.db.load_state("does-not-exist")
        assert result is None

    def test_save_with_weights(self):
        """Weights should survive JSON serialization in SQLite."""
        weights = [[[1.0, 2.0], [3.0, 4.0]], [[5.0]]]
        state = ClientState(
            client_id="w-client",
            room_id="R1",
            last_weights=weights,
        )
        self.db.save_state(state)
        loaded = self.db.load_state("w-client")

        assert loaded is not None
        assert loaded.last_weights == weights

    def test_upsert(self):
        """Saving the same client_id twice should update, not duplicate."""
        state1 = ClientState(client_id="up-client", room_id="R1", current_round=1)
        self.db.save_state(state1)

        state2 = ClientState(client_id="up-client", room_id="R1", current_round=5)
        self.db.save_state(state2)

        loaded = self.db.load_state("up-client")
        assert loaded is not None
        assert loaded.current_round == 5

    def test_clear_state(self):
        """Clearing state should make it unloadable."""
        state = ClientState(client_id="clear-me", room_id="R1")
        self.db.save_state(state)
        self.db.clear_state("clear-me")
        loaded = self.db.load_state("clear-me")
        assert loaded is None

    def test_log_round(self):
        """Round logging should not raise."""
        self.db.log_round(
            client_id="log-client",
            round_number=1,
            loss=0.5,
            num_samples=100,
            training_time=2.5,
            status="completed",
        )
        # No assertion — just ensure no exception

    def test_log_round_duplicate(self):
        """Logging the same round twice should not raise (REPLACE)."""
        self.db.log_round("dup-client", 1, 0.5, 100, 2.0)
        self.db.log_round("dup-client", 1, 0.3, 100, 1.5)
        # No exception = pass
