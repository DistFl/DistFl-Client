"""Unified model wrapper for PyTorch and Scikit-Learn.

Allows developers to bring their own models and hides the differences
between PyTorch (nn.Module) and Scikit-Learn (SGDClassifier) from the rest
of the SDK.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.base import BaseEstimator
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from fl_client.communication.serializer import deserialize_weights, serialize_weights

logger = logging.getLogger(__name__)


# ── Generic Model Exception ──────────────────────────────────────────────────

class ModelFormatError(Exception):
    """Raised when an unsupported model format is provided."""
    pass


# ── Base Wrapper Interface ───────────────────────────────────────────────────

class FLModelWrapper(ABC):
    """Abstract base class unifying FL model APIs."""

    @property
    @abstractmethod
    def model_type(self) -> str:
        """String identifier of the underlying framework (e.g., 'pytorch')."""
        pass

    @abstractmethod
    def get_weights(self) -> List[List[List[float]]]:
        """Extract weights into the server's 3D float32 nested list format."""
        pass

    @abstractmethod
    def set_weights(self, weights_3d: List[List[List[float]]]) -> None:
        """Load weights from the server's 3D float32 nested list format."""
        pass

    @abstractmethod
    def validate_dummy_pass(self, batch_features: Any, batch_labels: Any) -> None:
        """Perform a single dummy forward pass to catch shape/configuration errors.
        
        Args:
            batch_features: A single batch of features from the DataLoader.
            batch_labels: A single batch of labels from the DataLoader.
            
        Raises:
            RuntimeError: If the forward pass fails (e.g., shape mismatch).
        """
        pass

    @abstractmethod
    def train_epoch(
        self,
        dataloader: Any,
        criterion: Any,
        learning_rate: float,
        device: Any
    ) -> Tuple[float, float, int]:
        """Train the model for a single epoch.

        Args:
            dataloader: DataLoader providing training batches.
            criterion: Loss function module.
            learning_rate: Optimizer learning rate.
            device: Target device (e.g., torch.device).

        Returns:
            Tuple of (average_loss, accuracy, num_samples_trained).
        """
        pass


# ── PyTorch Implementation ───────────────────────────────────────────────────

class PyTorchModelWrapper(FLModelWrapper):
    """Wrapper for torch.nn.Module models."""

    def __init__(self, model: "nn.Module"):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for PyTorch models.")
        if not isinstance(model, nn.Module):
            raise TypeError("PyTorchModelWrapper requires a torch.nn.Module instance.")
        self.model = model

    @property
    def model_type(self) -> str:
        return "pytorch"

    def get_weights(self) -> List[List[List[float]]]:
        """Serialize PyTorch state_dict into server format."""
        # Move temporarily to CPU for serialization
        orig_device = next(self.model.parameters()).device
        cpu_model = self.model.to("cpu")
        weights = serialize_weights(cpu_model.state_dict())
        self.model.to(orig_device)
        return weights

    def set_weights(self, weights_3d: List[List[List[float]]]) -> None:
        """Deserialize server format into PyTorch state_dict."""
        new_sd = deserialize_weights(weights_3d, self.model.state_dict())
        self.model.load_state_dict(new_sd)

    def validate_dummy_pass(self, batch_features: "torch.Tensor", batch_labels: "torch.Tensor") -> None:
        """Run a single forward pass without backprop to catch shape errors."""
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            try:
                # Assuming batch_features is a tensor
                if hasattr(batch_features, "to"):
                    features = batch_features.to(device)
                    labels = batch_labels.to(device)
                else:
                    features, labels = batch_features, batch_labels

                _ = self.model(features)
            except Exception as e:
                raise RuntimeError(
                    f"PyTorch dummy forward pass failed. Please ensure your model's "
                    f"input shape matches the dataset feature shape. Error: {e}"
                ) from e

    def train_epoch(
        self,
        dataloader: "torch.utils.data.DataLoader",
        criterion: "nn.Module",
        learning_rate: float,
        device: "torch.device"
    ) -> Tuple[float, float, int]:
        """Execute a single PyTorch training epoch."""
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.model = self.model.to(device)
        self.model.train()

        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=learning_rate, momentum=0.9
        )

        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = self.model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            # Tally metrics
            epoch_loss += loss.item() * batch_features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        avg_loss = epoch_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy, total


# ── Scikit-Learn Implementation ──────────────────────────────────────────────

class SklearnModelWrapper(FLModelWrapper):
    """Wrapper for scikit-learn estimators (e.g., SGDClassifier)."""

    def __init__(self, model: "BaseEstimator", classes: Optional[List[int]] = None):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for Sklearn models.")
        
        # We require partial_fit for iterative federated learning
        if not hasattr(model, "partial_fit"):
            raise TypeError(
                "Provided Scikit-Learn model does not support 'partial_fit'. "
                "Use models like SGDClassifier or SGDRegressor for Federated Learning."
            )
        self.model = model
        self.classes = classes

    @property
    def model_type(self) -> str:
        return "sklearn"

    def get_weights(self) -> List[List[List[float]]]:
        """Extract coef_ and intercept_ into server format."""
        if not hasattr(self.model, "coef_") or not hasattr(self.model, "intercept_"):
            # If the model hasn't been fitted once locally, it won't have coef_
            # Return empty/default or raise
            raise RuntimeError("Cannot extract weights: model has not been initialized. Call partial_fit first.")

        weights_3d = []
        
        # Scikit-learn shapes:
        # coef_: (n_classes, n_features) or (1, n_features)
        # intercept_: (n_classes,) or (1,)
        
        coef_f32 = self.model.coef_.astype(np.float32)
        intercept_f32 = self.model.intercept_.astype(np.float32)

        # Layers: [ [coef_layer], [intercept_layer] ]
        weights_3d.append(coef_f32.tolist())
        weights_3d.append([intercept_f32.tolist()])

        return weights_3d

    def set_weights(self, weights_3d: List[List[List[float]]]) -> None:
        """Inject coef_ and intercept_ from server format."""
        if len(weights_3d) != 2:
            logger.warning("Sklearn models expect exactly 2 weight matrices (coef, intercept)")
            return

        coef_list = weights_3d[0]
        intercept_list = weights_3d[1][0] if len(weights_3d[1]) > 0 else weights_3d[1]

        self.model.coef_ = np.array(coef_list, dtype=np.float64)
        self.model.intercept_ = np.array(intercept_list, dtype=np.float64)
        
        # Ensure classes_ exists (required for SGDClassifier internals if manually injected)
        if not hasattr(self.model, "classes_") and self.classes is not None:
            self.model.classes_ = np.array(self.classes)

    def validate_dummy_pass(self, batch_features: Any, batch_labels: Any) -> None:
        """Run a single partial_fit call to initialize shapes and catch errors."""
        # Convert PyTorch tensors to Numpy if necessary
        X = self._to_numpy(batch_features)
        y = self._to_numpy(batch_labels)
        
        try:
            # We must provide classes= on the first partial_fit for classifiers
            if hasattr(self.model, "classes_") or self.classes is None:
                self.model.partial_fit(X, y)
            else:
                self.model.partial_fit(X, y, classes=self.classes)
        except Exception as e:
            raise RuntimeError(
                f"Scikit-Learn dummy pass (partial_fit) failed. Please check your data shapes. "
                f"Error: {e}"
            ) from e

    def train_epoch(
        self,
        dataloader: Any,
        criterion: Any,
        learning_rate: float,
        device: Any
    ) -> Tuple[float, float, int]:
        """Execute a single scikit-learn training pass over the dataset."""
        total = 0
        correct = 0

        # Note: Scikit-learn doesn't typically output a per-batch scalar loss from partial_fit.
        
        # If the input is exactly a single unbatched (X, y) Numpy tuple 
        if isinstance(dataloader, tuple) and len(dataloader) == 2:
            X = self._to_numpy(dataloader[0])
            y = self._to_numpy(dataloader[1])
            
            total = len(y)
            
            if hasattr(self.model, "classes_") or self.classes is None:
                self.model.partial_fit(X, y)
            else:
                self.model.partial_fit(X, y, classes=self.classes)
                
            try:
                preds = self.model.predict(X)
                accuracy = np.sum(preds == y) / total if total > 0 else 0.0
            except:
                accuracy = 0.0
                
            return 0.0, float(accuracy), total

        # If it happens to be an iterable dataloader of smaller batches
        for batch_features, batch_labels in dataloader:
            X = self._to_numpy(batch_features)
            y = self._to_numpy(batch_labels)
            
            if hasattr(self.model, "classes_") or self.classes is None:
                self.model.partial_fit(X, y)
            else:
                self.model.partial_fit(X, y, classes=self.classes)

            total += len(y)
            try:
                preds = self.model.predict(X)
                correct += np.sum(preds == y)
            except:
                pass

        accuracy = correct / total if total > 0 else 0.0
        return 0.0, accuracy, total

    def _to_numpy(self, arr: Any) -> np.ndarray:
        if HAS_TORCH and isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        res = np.array(arr)
        if res.dtype == np.float32:
            res = res.astype(np.float64)
        return res


# ── Dependency Injection Factory ─────────────────────────────────────────────

def wrap_model(model: Any, **kwargs: Any) -> FLModelWrapper:
    """Auto-detect the model framework and wrap it in the appropriate FLModelWrapper."""
    if HAS_TORCH and isinstance(model, nn.Module):
        return PyTorchModelWrapper(model)
    elif HAS_SKLEARN and isinstance(model, BaseEstimator):
        classes = kwargs.get("classes")
        return SklearnModelWrapper(model, classes=classes)
    else:
        raise ModelFormatError(
            f"Unsupported model type: {type(model)}. "
            f"Must be a torch.nn.Module or sklearn Estimator."
        )
