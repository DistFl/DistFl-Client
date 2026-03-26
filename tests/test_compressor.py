"""Unit tests for gzip compression and decompression."""

import json
import pytest

from fl_client.communication.compressor import compress, decompress


class TestCompress:
    """Test gzip compression."""

    def test_compress_returns_bytes(self):
        data = b"hello world"
        result = compress(data)
        assert isinstance(result, bytes)

    def test_compress_reduces_size_for_large_data(self):
        """Gzip should reduce size for sufficiently large data."""
        data = b"a" * 10000
        result = compress(data)
        assert len(result) < len(data)

    def test_compress_empty_bytes(self):
        result = compress(b"")
        assert isinstance(result, bytes)


class TestDecompress:
    """Test gzip decompression."""

    def test_decompress_returns_original(self):
        original = b"test data for compression"
        compressed = compress(original)
        result = decompress(compressed)
        assert result == original

    def test_round_trip_json(self):
        """JSON payload should survive compress/decompress unchanged."""
        payload = {"type": "model_update", "round": 5, "weights": [[[1.0, 2.0]]]}
        json_bytes = json.dumps(payload).encode("utf-8")
        compressed = compress(json_bytes)
        decompressed = decompress(compressed)
        restored = json.loads(decompressed)
        assert restored == payload

    def test_round_trip_large_weights(self):
        """Large weight payloads should round-trip correctly."""
        import random
        weights = [[[random.random() for _ in range(100)] for _ in range(50)]]
        data = json.dumps(weights).encode("utf-8")
        compressed = compress(data)
        decompressed = decompress(compressed)
        restored = json.loads(decompressed)
        assert len(restored[0]) == 50
        assert len(restored[0][0]) == 100

    def test_decompress_invalid_data_raises(self):
        """Decompressing non-gzip data should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="decompression failed"):
            decompress(b"not gzip data")
