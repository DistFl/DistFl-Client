"""Generic, extensible local training engine for federated learning.

Provides an Adapter/Strategy architecture to decouple the training
loop from framework-specific implementation details. Supports PyTorch,
Scikit-Learn, and custom legacy wrappers seamlessly.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Framework-agnostic result of a local training round.

    Attributes:
        model_state: Extracted model weights/state after training.
        loss: Final average loss of the training round.
        metrics: Additional validation or training metrics (e.g. accuracy, score).
        training_time: Wall-clock training execution time in seconds.
        num_samples: Total number of samples trained on in the dataset.
    """

    model_state: Any
    loss: float
    metrics: Dict[str, float]
    training_time: float
    num_samples: int


class BaseModelAdapter(ABC):
    """Abstract interface for model training adapters."""

    @abstractmethod
    def train(
        self,
        train_data: Any,
        local_epochs: int,
        learning_rate: float,
        criterion: Optional[Any] = None,
    ) -> TrainingResult:
        """Execute local training logic and return uniform results.
        
        Args:
            train_data: The framework-specific representation of the dataset.
            local_epochs: The number of loops/epochs to execute locally.
            learning_rate: Update magnitude for iterators.
            criterion: The loss function to minimize, if applicable.
            
        Returns:
            A populated TrainingResult dataclass instance.
        """
        pass

    @abstractmethod
    def get_weights(self) -> Any:
        """Extract model weights in a federated-learning-friendly format."""
        pass

    @abstractmethod
    def set_weights(self, weights: Any) -> None:
        """Inject model weights received from the federated server."""
        pass


class PyTorchAdapter(BaseModelAdapter):
    """Adapter for raw PyTorch `nn.Module` object training."""

    def __init__(self, model: "nn.Module"):
        if not HAS_TORCH:
            raise RuntimeError("PyTorchAdapter cannot be used: PyTorch is not installed.")
        self.model = model
        self.device = self._resolve_device()
        self.model.to(self.device)

    def _resolve_device(self) -> "torch.device":
        """Safely determine the most optimal PyTorch hardware device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def train(
        self,
        train_data: "DataLoader",
        local_epochs: int,
        learning_rate: float,
        criterion: Optional[Any] = None,
    ) -> TrainingResult:
        """Trains the PyTorch model over the DataLoader batches."""
        if criterion is None:
            raise ValueError("PyTorchAdapter requires a `criterion` (loss function).")

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()

        start_time = time.time()
        final_loss = 0.0
        final_accuracy = 0.0
        total_samples = 0

        for epoch in range(local_epochs):
            epoch_loss = 0.0
            correct = 0
            samples_in_epoch = 0

            for inputs, targets in train_data:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                batch_size = inputs.size(0)
                epoch_loss += loss.item() * batch_size
                samples_in_epoch += batch_size

                # Optional heuristics: accuracy metrics for classification
                if outputs.dim() == 2 and targets.dtype in (torch.long, torch.int):
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == targets).sum().item()

            final_loss = epoch_loss / max(1, samples_in_epoch)
            final_accuracy = correct / max(1, samples_in_epoch)
            total_samples = samples_in_epoch

            logger.info(
                "PyTorch Epoch %d/%d — loss=%.4f accuracy=%.4f samples=%d",
                epoch + 1, local_epochs, final_loss, final_accuracy, total_samples
            )

        training_time = time.time() - start_time
        metrics = {
            "accuracy": final_accuracy,
            "final_epoch_loss": final_loss,
        }

        return TrainingResult(
            model_state=self.get_weights(),
            loss=float(final_loss),
            metrics=metrics,
            training_time=float(training_time),
            num_samples=int(total_samples),
        )

    def get_weights(self) -> Any:
        """Extract the model's state_dict pushed comfortably to CPU."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights: Any) -> None:
        """Load external parameters immediately onto the underlying device."""
        if isinstance(weights, dict):
            # Send loaded weights directly mapping to the target hardware device
            self.model.load_state_dict(weights)
        else:
            logger.warning("PyTorchAdapter received non-dict weights. Ignoring.")


class SklearnAdapter(BaseModelAdapter):
    """Adapter for Scikit-Learn `BaseEstimator` classes."""

    def __init__(self, model: Any):
        self.model = model

    def train(
        self,
        train_data: Tuple[Any, Any],
        local_epochs: int,
        learning_rate: float,
        criterion: Optional[Any] = None,
    ) -> TrainingResult:
        """Trains the estimator utilizing either .fit() or .partial_fit()."""
        start_time = time.time()
        X, y = train_data
        
        num_samples = len(X) if hasattr(X, "__len__") else 0
        logger.info("Training Sklearn model on %d samples.", num_samples)

        # Iterative models natively support partial_fit per epoch
        if hasattr(self.model, "partial_fit") and local_epochs > 1:
            for epoch in range(local_epochs):
                self.model.partial_fit(X, y)
        else:
            self.model.fit(X, y)

        score = 0.0
        if hasattr(self.model, "score"):
            try:
                score = float(self.model.score(X, y))
            except Exception as e:
                logger.debug("Failed to compute sklearn score metric: %s", e)

        metrics = {"score": score}
        training_time = time.time() - start_time
        
        # In sklearn, loss computation varies widely, often proxying zero when untracked
        return TrainingResult(
            model_state=self.get_weights(),
            loss=0.0,
            metrics=metrics,
            training_time=float(training_time),
            num_samples=int(num_samples),
        )

    def get_weights(self) -> Any:
        """Export exposed coefficient and intercept vectors."""
        weights = {}
        if hasattr(self.model, "coef_"):
            weights["coef_"] = self.model.coef_
        if hasattr(self.model, "intercept_"):
            weights["intercept_"] = self.model.intercept_
        return weights

    def set_weights(self, weights: Any) -> None:
        """Apply parameter vectors strictly if they exist."""
        if isinstance(weights, dict):
            if "coef_" in weights:
                self.model.coef_ = weights["coef_"]
            if "intercept_" in weights:
                self.model.intercept_ = weights["intercept_"]


class WrapperAdapter(BaseModelAdapter):
    """Adapter preserving backwards compatibility with legacy `FLModelWrapper`."""

    def __init__(self, model: Any):
        self.model = model

    def train(
        self,
        train_data: Any,
        local_epochs: int,
        learning_rate: float,
        criterion: Optional[Any] = None,
    ) -> TrainingResult:
        """Delegates completely to the explicitly defined wrapper method."""
        start_time = time.time()
        final_loss = 0.0
        final_accuracy = 0.0
        total_samples = 0

        # Existing wrapper handles device auto-assignment internally
        device = None 

        for epoch in range(local_epochs):
            avg_loss, accuracy, samples = self.model.train_epoch(
                dataloader=train_data,
                criterion=criterion,
                learning_rate=learning_rate,
                device=device,
            )
            final_loss = avg_loss
            final_accuracy = accuracy
            total_samples = samples

            logger.info(
                "Wrapper Epoch %d/%d — loss=%.4f accuracy=%.4f samples=%d",
                epoch + 1, local_epochs, final_loss, final_accuracy, total_samples
            )

        training_time = time.time() - start_time
        metrics = {"accuracy": float(final_accuracy), "final_epoch_loss": float(final_loss)}

        return TrainingResult(
            model_state=self.get_weights(),
            loss=float(final_loss),
            metrics=metrics,
            training_time=float(training_time),
            num_samples=int(total_samples),
        )

    def get_weights(self) -> Any:
        """Safely call inner logic if exported."""
        if hasattr(self.model, "get_weights"):
            return self.model.get_weights()
        return {}

    def set_weights(self, weights: Any) -> None:
        """Safely overwrite inner target weights if logic corresponds."""
        if hasattr(self.model, "set_weights"):
            self.model.set_weights(weights)


class Trainer:
    """Generic, extensible local trainer for federated learning.
    
    Independent of framework-specific training details. Auto-selects the
    correct adapter/strategy corresponding to the generated model provided.
    
    Usage::
    
        trainer = Trainer(local_epochs=2, learning_rate=0.001)
        result = trainer.train(model, train_data, criterion=criterion)
    """

    def __init__(
        self,
        local_epochs: int = 1,
        learning_rate: float = 0.001,
        criterion: Optional[Any] = None,
    ) -> None:
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.criterion = criterion

    def _select_adapter(self, model: Any) -> BaseModelAdapter:
        """Factory method to resolve the adapter instance for arbitrary architectures."""
        # Check priority: explicitly defined legacy wrappers
        if hasattr(model, "train_epoch"):
            logger.debug("Auto-selected WrapperAdapter for model.")
            return WrapperAdapter(model)

        # Check secondary: Scikit-learn estimator signatures
        model_module = type(model).__module__
        if model_module.startswith("sklearn") or hasattr(model, "fit"):
            logger.debug("Auto-selected SklearnAdapter for model.")
            return SklearnAdapter(model)

        # Check underlying: Native PyTorch objects
        if HAS_TORCH and isinstance(model, nn.Module):
            logger.debug("Auto-selected PyTorchAdapter for model.")
            return PyTorchAdapter(model)

        raise TypeError(
            f"Unable to auto-select a training adapter for model of type {type(model)}. "
            "Model is not recognized as a registered Wrapper, Sklearn estimator, or PyTorch nn.Module."
        )

    def train(
        self,
        model: Any,
        train_data: Any,
        adapter: Optional[BaseModelAdapter] = None,
    ) -> TrainingResult:
        """Run local training on the abstract model via an adapter layer.
        
        Args:
            model: The machine learning model memory object to train.
            train_data: The dataset structure (DataLoader for PyTorch, (X,y) tuple for Sklearn).
            adapter: Manual bypass strategy. If None, calculates dynamically from model parameter.
            
        Returns:
            A framework-agnostic `TrainingResult` populated securely.
        """
        active_adapter = adapter if adapter is not None else self._select_adapter(model)

        logger.info(
            "Trainer starting — adapter=%s epochs=%d lr=%f",
            type(active_adapter).__name__,
            self.local_epochs,
            self.learning_rate,
        )

        return active_adapter.train(
            train_data=train_data,
            local_epochs=self.local_epochs,
            learning_rate=self.learning_rate,
            criterion=self.criterion,
        )


# =====================================================================
# EXAMPLE USAGE
# =====================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("--- 1. PyTorch Example ---")
    if HAS_TORCH:
        X_pt = torch.randn(10, 5)
        y_pt = torch.randint(0, 2, (10,))
        dataset = torch.utils.data.TensorDataset(X_pt, y_pt)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        model_pt = nn.Sequential(nn.Linear(5, 2))
        criterion_pt = nn.CrossEntropyLoss()
        
        trainer_pt = Trainer(local_epochs=2, learning_rate=0.01, criterion=criterion_pt)
        res_pt = trainer_pt.train(model_pt, dataloader)
        print("PyTorch Result metrics:", res_pt.metrics)
        print("PyTorch model state exists:", bool(res_pt.model_state))
    else:
        print("PyTorch not installed, skipping.")

    print("\n--- 2. Sklearn Example ---")
    try:
        from sklearn.linear_model import SGDClassifier
        import numpy as np
        
        X_sk = np.random.rand(10, 5)
        y_sk = np.random.randint(0, 2, 10)
        
        model_sk = SGDClassifier(loss='log_loss', max_iter=10)
        trainer_sk = Trainer(local_epochs=1, learning_rate=0.01)
        res_sk = trainer_sk.train(model_sk, (X_sk, y_sk))
        print("Sklearn Result metrics:", res_sk.metrics)
        print("Sklearn model state exists:", bool(res_sk.model_state))
    except ImportError:
        print("Scikit-Learn not installed, skipping.")

    print("\n--- 3. Legacy Wrapper Example ---")
    class DummyWrapper:
        def train_epoch(self, dataloader, criterion, learning_rate, device):
            return 0.5, 0.95, 10  # loss, accuracy, samples
        def get_weights(self):
            return [1.0, 2.0]
            
    model_wrap = DummyWrapper()
    trainer_wrap = Trainer(local_epochs=2)
    res_wrap = trainer_wrap.train(model_wrap, "dummy_dataloader")
    print("Wrapper Result metrics:", res_wrap.metrics)
    print("Wrapper model state exists:", bool(res_wrap.model_state))
