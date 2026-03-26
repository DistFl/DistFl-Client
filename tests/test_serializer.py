"""Unit tests for weight serialization round-trip."""

import torch
import torch.nn as nn
import numpy as np
import pytest

from fl_client.communication.serializer import (
    deserialize_weights,
    serialize_weights,
    weights_from_json,
    weights_to_json,
)


class TestSerializeWeights:
    """Test tensor → 3D list serialization."""

    def test_linear_layer(self):
        """Test serialization of a simple linear model."""
        model = nn.Linear(10, 5)
        state_dict = model.state_dict()
        weights = serialize_weights(state_dict)

        # Should have 2 layers (weight + bias)
        assert len(weights) == 2
        # Weight: 5 rows × 10 cols
        assert len(weights[0]) == 5
        assert len(weights[0][0]) == 10
        # Bias: 1 row × 5 cols
        assert len(weights[1]) == 1
        assert len(weights[1][0]) == 5

    def test_mlp_model(self):
        """Test serialization of a multi-layer model."""
        model = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 3),
        )
        state_dict = model.state_dict()
        weights = serialize_weights(state_dict)

        # 4 parameter tensors (2 Linear layers × 2 each)
        assert len(weights) == 4

    def test_no_nan_or_inf(self):
        """Serialized weights should not contain NaN or Inf."""
        model = nn.Linear(5, 3)
        weights = serialize_weights(model.state_dict())
        for layer in weights:
            for row in layer:
                for val in row:
                    assert not np.isnan(val), "NaN found in serialized weights"
                    assert not np.isinf(val), "Inf found in serialized weights"

    def test_float16_precision(self):
        """Float16 conversion should preserve values within precision."""
        model = nn.Linear(5, 3)
        original_weight = model.weight.data.clone()
        weights = serialize_weights(model.state_dict())

        # Reconstruct and check approximate equality
        restored = deserialize_weights(weights, model.state_dict())
        np.testing.assert_allclose(
            restored["weight"].numpy(),
            original_weight.numpy(),
            atol=0.01,  # float16 has ~3 decimal digits of precision
        )


class TestDeserializeWeights:
    """Test 3D list → tensor deserialization."""

    def test_round_trip(self):
        """Serialize then deserialize should produce approximately equal weights."""
        model = nn.Linear(10, 5)
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}

        weights_3d = serialize_weights(model.state_dict())
        restored_sd = deserialize_weights(weights_3d, model.state_dict())

        for key in original_sd:
            np.testing.assert_allclose(
                restored_sd[key].numpy(),
                original_sd[key].numpy(),
                atol=0.01,
            )

    def test_shape_preservation(self):
        """Deserialized tensors should match original shapes."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5),
        )
        original_sd = model.state_dict()
        weights_3d = serialize_weights(original_sd)
        restored_sd = deserialize_weights(weights_3d, original_sd)

        for key in original_sd:
            assert restored_sd[key].shape == original_sd[key].shape, (
                f"Shape mismatch for {key}: "
                f"{restored_sd[key].shape} != {original_sd[key].shape}"
            )

    def test_layer_count_mismatch_raises(self):
        """Mismatched layer counts should raise ValueError."""
        model = nn.Linear(5, 3)
        weights_3d = [[[1.0, 2.0]]]  # Only 1 layer, model has 2
        with pytest.raises(ValueError, match="layer count mismatch"):
            deserialize_weights(weights_3d, model.state_dict())


class TestJsonConversion:
    """Test JSON encoding/decoding of weights."""

    def test_json_round_trip(self):
        """JSON encode then decode should preserve values exactly."""
        weights = [[[1.0, 2.0], [3.0, 4.0]], [[5.0]]]
        json_str = weights_to_json(weights)
        restored = weights_from_json(json_str)
        assert restored == weights

    def test_json_is_valid(self):
        """Output should be valid JSON string."""
        import json

        weights = [[[1.5, -2.5]]]
        json_str = weights_to_json(weights)
        parsed = json.loads(json_str)
        assert parsed == weights
