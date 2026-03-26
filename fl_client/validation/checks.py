"""Pre-send validation checks for weight updates.

Mirrors the server-side validation in validation/validator.go to catch
obvious issues client-side before wasting bandwidth.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def validate_weights(
    weights: List[List[List[float]]],
    reference_shapes: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[bool, str]:
    """Validate serialized weights before sending to server.

    Checks:
        1. Non-empty weights
        2. Shape consistency (if reference provided)
        3. No NaN or Inf values
        4. Values within reasonable range

    Args:
        weights: 3D nested list (layers × rows × cols).
        reference_shapes: Optional list of (rows, cols) tuples for each layer.

    Returns:
        Tuple of (is_valid, reason). reason is empty string if valid.
    """
    if not weights:
        return False, "weights are empty"

    for layer_idx, layer in enumerate(weights):
        if not layer:
            return False, f"layer {layer_idx} is empty"

        # Check each row in the layer
        expected_cols = len(layer[0]) if layer else 0
        for row_idx, row in enumerate(layer):
            if not isinstance(row, list):
                return False, f"layer {layer_idx} row {row_idx} is not a list"

            if len(row) != expected_cols:
                return (
                    False,
                    f"layer {layer_idx} has inconsistent column count: "
                    f"row 0 has {expected_cols}, row {row_idx} has {len(row)}",
                )

            for col_idx, val in enumerate(row):
                if math.isnan(val):
                    return (
                        False,
                        f"NaN found at layer={layer_idx} row={row_idx} col={col_idx}",
                    )
                if math.isinf(val):
                    return (
                        False,
                        f"Inf found at layer={layer_idx} row={row_idx} col={col_idx}",
                    )

    # Check shapes against reference
    if reference_shapes is not None:
        if len(weights) != len(reference_shapes):
            return (
                False,
                f"layer count mismatch: got {len(weights)}, "
                f"expected {len(reference_shapes)}",
            )
        for i, (layer, (exp_rows, exp_cols)) in enumerate(
            zip(weights, reference_shapes)
        ):
            actual_rows = len(layer)
            actual_cols = len(layer[0]) if layer else 0
            if actual_rows != exp_rows or actual_cols != exp_cols:
                return (
                    False,
                    f"shape mismatch at layer {i}: "
                    f"got ({actual_rows}, {actual_cols}), "
                    f"expected ({exp_rows}, {exp_cols})",
                )

    logger.debug("Weight validation passed — %d layers", len(weights))
    return True, ""


def validate_loss(loss: float) -> Tuple[bool, str]:
    """Validate the training loss before sending to server.

    The server (validator.go) rejects NaN, Inf, and negative loss values.

    Args:
        loss: The computed training loss.

    Returns:
        Tuple of (is_valid, reason).
    """
    if math.isnan(loss):
        return False, "loss is NaN"
    if math.isinf(loss):
        return False, "loss is Inf"
    if loss < 0:
        return False, f"loss is negative: {loss}"
    return True, ""


def get_weight_shapes(
    weights: List[List[List[float]]],
) -> List[Tuple[int, int]]:
    """Extract (rows, cols) shapes from 3D weights.

    Args:
        weights: 3D nested list.

    Returns:
        List of (rows, cols) tuples.
    """
    shapes = []
    for layer in weights:
        rows = len(layer)
        cols = len(layer[0]) if layer else 0
        shapes.append((rows, cols))
    return shapes
