"""Unit tests for the local training engine (BYOM adapter pattern)."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl_client.training.trainer import Trainer, TrainingResult
from fl_client.model.wrapper import wrap_model


class TestTrainer:
    """Test the epoch-based trainer with the PyTorch adapter pattern."""

    def _make_dataloader(self, n_samples=100, n_features=10, n_classes=3):
        """Create a simple dataloader for testing."""
        X = torch.randn(n_samples, n_features)
        y = torch.randint(0, n_classes, (n_samples,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=16, shuffle=True)

    def _make_model(self, n_features=10, n_classes=3):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def _make_wrapped_model(self, n_features=10, n_classes=3):
        """Create a wrapped model with criterion for training."""
        model = self._make_model(n_features, n_classes)
        return wrap_model(model)

    def test_train_returns_result(self):
        """Training should return a TrainingResult."""
        wrapped = self._make_wrapped_model()
        loader = self._make_dataloader()
        trainer = Trainer(local_epochs=1, learning_rate=0.01)

        result = trainer.train(wrapped, loader)

        assert isinstance(result, TrainingResult)
        assert result.loss >= 0
        assert result.num_samples > 0
        assert result.training_time > 0

    def test_train_loss_is_finite(self):
        """Loss should be a finite number."""
        import math

        wrapped = self._make_wrapped_model()
        loader = self._make_dataloader()
        trainer = Trainer(local_epochs=1)
        result = trainer.train(wrapped, loader)

        assert math.isfinite(result.loss)

    def test_train_metrics_contain_accuracy(self):
        """Metrics should include accuracy."""
        wrapped = self._make_wrapped_model()
        loader = self._make_dataloader()
        trainer = Trainer(local_epochs=1)
        result = trainer.train(wrapped, loader)

        assert "accuracy" in result.metrics
        assert 0.0 <= result.metrics["accuracy"] <= 1.0

    def test_train_multiple_epochs(self):
        """Multi-epoch training should complete successfully."""
        wrapped = self._make_wrapped_model()
        loader = self._make_dataloader()
        trainer = Trainer(local_epochs=5, learning_rate=0.01)
        result = trainer.train(wrapped, loader)

        assert result.loss >= 0
        assert result.training_time > 0
