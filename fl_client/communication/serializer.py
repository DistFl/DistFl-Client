"""Weight serialization between PyTorch tensors and the server's JSON format.

Server weight format: 3D nested list [[[float32]]] — layers × rows × cols.

Serialization pipeline:  tensor → numpy → float32 → nested list → JSON
Deserialization pipeline: JSON → nested list → numpy float32 → tensor
"""

from __future__ import annotations

import json
import logging
from typing import Any, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


def serialize_weights(state_dict: dict[str, torch.Tensor]) -> List[List[List[float]]]:
    """Convert a PyTorch state_dict to the server's 3D weight format.

    Each parameter tensor is reshaped to 2D (rows × cols) and kept as
    float32 for full precision. The result is a list of 2D layers.

    Args:
        state_dict: PyTorch model state dictionary.

    Returns:
        3D nested list: layers × rows × cols (float32 values as Python floats).
    """
    weights_3d: List[List[List[float]]] = []

    for name, param in state_dict.items():
        arr = param.detach().cpu().numpy()

        # Keep float32 for full precision
        arr_f32 = arr.astype(np.float32)

        # Reshape to 2D: (rows, cols)
        if arr_f32.ndim == 0:
            # Scalar → 1×1
            layer = [[float(arr_f32)]]
        elif arr_f32.ndim == 1:
            # 1D (bias) → 1×N
            layer = [arr_f32.tolist()]
        else:
            # 2D+ → reshape to 2D
            rows = arr_f32.shape[0]
            cols = int(np.prod(arr_f32.shape[1:])) if arr_f32.ndim > 1 else 1
            reshaped = arr_f32.reshape(rows, cols)
            layer = reshaped.tolist()

        weights_3d.append(layer)

    logger.debug("Serialized %d parameter tensors to 3D weight format", len(weights_3d))
    return weights_3d


def deserialize_weights(
    weights_3d: List[List[List[float]]], state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Convert the server's 3D weight format back into a PyTorch state_dict.

    Uses the reference state_dict to determine the original shapes of each
    parameter tensor.

    Args:
        weights_3d: 3D nested list from the server (layers × rows × cols).
        state_dict: Reference state_dict for shape information.

    Returns:
        A new state_dict with tensors populated from the deserialized weights.

    Raises:
        ValueError: If the number of layers doesn't match state_dict params.
    """
    param_names = list(state_dict.keys())

    if len(weights_3d) != len(param_names):
        raise ValueError(
            f"Weight layer count mismatch: got {len(weights_3d)}, "
            f"expected {len(param_names)} parameters"
        )

    new_state_dict: dict[str, torch.Tensor] = {}

    for i, name in enumerate(param_names):
        original_shape = state_dict[name].shape
        layer_data = weights_3d[i]

        # Convert nested list → numpy → float32
        arr = np.array(layer_data, dtype=np.float32)

        # Reshape to original tensor shape
        try:
            arr = arr.reshape(original_shape)
        except ValueError as e:
            raise ValueError(
                f"Shape mismatch for parameter '{name}': "
                f"received {arr.shape}, expected {original_shape}"
            ) from e

        new_state_dict[name] = torch.from_numpy(arr)

    logger.debug("Deserialized %d parameter tensors from 3D weight format", len(new_state_dict))
    return new_state_dict


def weights_to_json(weights_3d: List[List[List[float]]]) -> str:
    """Encode 3D weights as a JSON string.

    Args:
        weights_3d: 3D nested list of weight values.

    Returns:
        JSON string representation.
    """
    return json.dumps(weights_3d)


def weights_from_json(json_str: str) -> List[List[List[float]]]:
    """Decode 3D weights from a JSON string.

    Args:
        json_str: JSON-encoded 3D weight list.

    Returns:
        3D nested list of weight values.
    """
    return json.loads(json_str)
