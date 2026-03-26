"""Unit tests for pre-send validation checks."""

import math
import pytest

from fl_client.validation.checks import (
    get_weight_shapes,
    validate_loss,
    validate_weights,
)


class TestValidateWeights:
    """Test weight validation."""

    def test_valid_weights(self):
        weights = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0]]]
        valid, reason = validate_weights(weights)
        assert valid
        assert reason == ""

    def test_empty_weights(self):
        valid, reason = validate_weights([])
        assert not valid
        assert "empty" in reason.lower()

    def test_nan_in_weights(self):
        weights = [[[1.0, float("nan")]]]
        valid, reason = validate_weights(weights)
        assert not valid
        assert "NaN" in reason

    def test_inf_in_weights(self):
        weights = [[[float("inf"), 1.0]]]
        valid, reason = validate_weights(weights)
        assert not valid
        assert "Inf" in reason

    def test_negative_inf_in_weights(self):
        weights = [[[float("-inf"), 1.0]]]
        valid, reason = validate_weights(weights)
        assert not valid
        assert "Inf" in reason

    def test_shape_reference_match(self):
        weights = [[[1.0, 2.0], [3.0, 4.0]]]
        ref_shapes = [(2, 2)]
        valid, reason = validate_weights(weights, ref_shapes)
        assert valid

    def test_shape_reference_mismatch(self):
        weights = [[[1.0, 2.0], [3.0, 4.0]]]
        ref_shapes = [(3, 2)]  # Wrong row count
        valid, reason = validate_weights(weights, ref_shapes)
        assert not valid
        assert "shape mismatch" in reason.lower()

    def test_layer_count_mismatch(self):
        weights = [[[1.0]], [[2.0]]]
        ref_shapes = [(1, 1)]  # Only 1 ref shape, but 2 layers
        valid, reason = validate_weights(weights, ref_shapes)
        assert not valid
        assert "layer count" in reason.lower()

    def test_inconsistent_columns(self):
        weights = [[[1.0, 2.0], [3.0]]]  # Row 1 has 1 col, row 0 has 2
        valid, reason = validate_weights(weights)
        assert not valid
        assert "inconsistent" in reason.lower()


class TestValidateLoss:
    """Test loss validation."""

    def test_valid_loss(self):
        valid, reason = validate_loss(0.42)
        assert valid
        assert reason == ""

    def test_zero_loss(self):
        valid, reason = validate_loss(0.0)
        assert valid

    def test_nan_loss(self):
        valid, reason = validate_loss(float("nan"))
        assert not valid
        assert "NaN" in reason

    def test_inf_loss(self):
        valid, reason = validate_loss(float("inf"))
        assert not valid
        assert "Inf" in reason

    def test_negative_loss(self):
        valid, reason = validate_loss(-0.5)
        assert not valid
        assert "negative" in reason


class TestGetWeightShapes:
    """Test shape extraction."""

    def test_shapes(self):
        weights = [[[1.0, 2.0], [3.0, 4.0]], [[5.0]]]
        shapes = get_weight_shapes(weights)
        assert shapes == [(2, 2), (1, 1)]

    def test_empty_layer(self):
        weights = [[]]
        shapes = get_weight_shapes(weights)
        assert shapes == [(0, 0)]
