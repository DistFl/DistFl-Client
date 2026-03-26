"""Unit tests for the model registry and builder."""

import torch
import torch.nn as nn
import pytest

from fl_client.model.builder import build_model
from fl_client.model.registry import ModelRegistry


class TestModelRegistry:
    """Test the model registry pattern."""

    def test_list_builtin_models(self):
        """Built-in models should be available."""
        models = ModelRegistry.list_models()
        assert "mlp" in models
        assert "lstm" in models
        assert "cnn" in models

    def test_get_registered_model(self):
        """Getting a registered builder should return a callable."""
        builder = ModelRegistry.get("mlp")
        assert callable(builder)

    def test_get_unknown_raises(self):
        """Getting an unregistered model should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown model type"):
            ModelRegistry.get("nonexistent_model_type")

    def test_custom_registration(self):
        """Custom models should be registerable."""

        @ModelRegistry.register("test_custom")
        def build_custom(config: dict) -> nn.Module:
            return nn.Linear(config["input_size"], config["output_size"])

        assert "test_custom" in ModelRegistry.list_models()
        model = ModelRegistry.get("test_custom")(
            {"input_size": 5, "output_size": 2}
        )
        assert isinstance(model, nn.Module)


class TestBuildModel:
    """Test the safe model builder."""

    def test_build_mlp(self):
        """Building an MLP should return a valid module."""
        model = build_model({
            "model_type": "mlp",
            "input_size": 10,
            "hidden_size": 32,
            "output_size": 5,
        })
        assert isinstance(model, nn.Module)
        # Test forward pass
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 5)

    def test_build_lstm(self):
        """Building an LSTM should return a valid module."""
        model = build_model({
            "model_type": "lstm",
            "input_size": 10,
            "hidden_size": 32,
            "output_size": 2,
        })
        assert isinstance(model, nn.Module)
        # Test forward: (batch, seq_len, input_size)
        x = torch.randn(4, 5, 10)
        out = model(x)
        assert out.shape == (4, 2)

    def test_build_cnn(self):
        """Building a CNN should return a valid module."""
        model = build_model({
            "model_type": "cnn",
            "input_channels": 1,
            "input_size": 28,
            "hidden_size": 16,
            "output_size": 10,
        })
        assert isinstance(model, nn.Module)
        # Test forward: (batch, channels, H, W)
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)

    def test_build_missing_type_raises(self):
        """Missing model_type should raise ValueError."""
        with pytest.raises(ValueError, match="model_type"):
            build_model({"input_size": 10})

    def test_build_unknown_type_raises(self):
        """Unknown model_type should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown model type"):
            build_model({"model_type": "transformer_9000"})
