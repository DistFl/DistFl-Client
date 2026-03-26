"""Gzip compression and decompression utilities.

Matches the Go server's utils.CompressGzip / DecompressGzip functions.
"""

from __future__ import annotations

import gzip
import logging

logger = logging.getLogger(__name__)


def compress(data: bytes) -> bytes:
    """Compress data using gzip.

    Args:
        data: Raw bytes to compress.

    Returns:
        Gzip-compressed bytes.

    Raises:
        RuntimeError: If compression fails.
    """
    try:
        compressed = gzip.compress(data, compresslevel=6)
        ratio = len(compressed) / len(data) * 100 if data else 0
        logger.debug(
            "Compressed %d → %d bytes (%.1f%%)", len(data), len(compressed), ratio
        )
        return compressed
    except Exception as e:
        raise RuntimeError(f"Gzip compression failed: {e}") from e


def decompress(data: bytes) -> bytes:
    """Decompress gzip data.

    Args:
        data: Gzip-compressed bytes.

    Returns:
        Decompressed raw bytes.

    Raises:
        RuntimeError: If decompression fails (corrupted payload).
    """
    try:
        decompressed = gzip.decompress(data)
        logger.debug(
            "Decompressed %d → %d bytes", len(data), len(decompressed)
        )
        return decompressed
    except Exception as e:
        raise RuntimeError(f"Gzip decompression failed: {e}") from e
