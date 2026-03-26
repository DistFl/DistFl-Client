"""Dataset loading, validation, and metadata computation for federated learning.

Features a generic FlexibleDataAdapter to natively support PyTorch DataLoaders,
Scikit-Learn Numpy tuples, and legacy CSV string paths uniformly.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class DatasetValidationError(Exception):
    """Raised when dataset validation logic categorically fails."""
    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        super().__init__(f"Dataset validation failed: {'; '.join(errors)}")


class FlexibleDataAdapter:
    """A unified wrapper to extract metadata and standardize runtime iterations.
    
    Supports:
        - PyTorch DataLoader
        - Scikit-Learn `(X, y)` Numpy Tuples
        - Legacy Pandas CSV string paths
    """
    
    def __init__(self, data: Any, schema: Optional[Dict] = None):
        self._raw_data = data
        self.schema = schema or {}
        self.num_samples = 0
        self.num_features = 0
        self.num_classes = 0
        self.label_distribution: Dict[str, float] = {}
        
        # The uniform structure automatically ingested by Trainer Adapters
        self.train_data: Any = None
        
        self._detect_and_parse()
        
    def _detect_and_parse(self) -> None:
        """Determines object heritage and structures the adapter accordingly."""
        # 1. Native PyTorch Iterator
        if HAS_TORCH and isinstance(self._raw_data, DataLoader):
            self._parse_dataloader(self._raw_data)
        
        # 2. Native Numpy Data Matrix Sequence
        elif isinstance(self._raw_data, tuple) and len(self._raw_data) == 2:
            self._parse_numpy_tuple(self._raw_data)
            
        # 3. Legacy Tabular File Sequence
        elif isinstance(self._raw_data, (str, Path)):
            self._parse_csv_path(str(self._raw_data))
            
        # Extensible Blind Pass
        else:
            logger.warning(
                "Unrecognized custom data type %s. Metadata extraction bypassed.",
                type(self._raw_data)
            )
            self.train_data = self._raw_data
            
        logger.info(
            "FlexibleDataAdapter ready — samples=%d features=%d classes=%d",
            self.num_samples, self.num_features, self.num_classes
        )

    def _parse_dataloader(self, dataloader: "DataLoader") -> None:
        logger.info("Auto-detected PyTorch DataLoader.")
        self.train_data = dataloader
        
        dataset = dataloader.dataset
        self.num_samples = len(dataset) if hasattr(dataset, "__len__") else 0
        
        try:
            batch = next(iter(dataloader))
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                features = batch[0]
                self.num_features = features.shape[1] if features.dim() > 1 else 1
                
                # Distribution heuristic evaluation
                if hasattr(dataset, "labels"):
                    self._compute_distribution(dataset.labels.tolist())
                elif hasattr(dataset, "targets"):
                    if hasattr(dataset.targets, "tolist"):
                        self._compute_distribution(dataset.targets.tolist())
                    else:
                        self._compute_distribution(list(dataset.targets))
        except Exception as e:
            logger.warning("Could not automatically extract dimensional metadata from DataLoader: %s", e)

    def _parse_numpy_tuple(self, data_tuple: Tuple[Any, Any]) -> None:
        logger.info("Auto-detected (X, y) data tuple layout.")
        self.train_data = data_tuple
        X, y = data_tuple
        
        if hasattr(X, "shape"):
            self.num_samples = X.shape[0] if len(X.shape) > 0 else 0
            self.num_features = X.shape[1] if len(X.shape) > 1 else 1
        
        if hasattr(y, "__len__"):
            try:
                if hasattr(y, "tolist"):
                    self._compute_distribution(y.tolist())
                else:
                    self._compute_distribution(list(y))
            except Exception:
                pass

    def _parse_csv_path(self, path: str) -> None:
        if not HAS_PANDAS:
            raise RuntimeError("Pandas must be explicitly installed to natively load CSV file paths.")
            
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset block not found precisely at: {path}")

        logger.info("Auto-detected CSV path. Executing backwards compatible loader...")
        df = pd.read_csv(csv_path)
        if df.empty:
            raise DatasetValidationError(["The underlying tabular file is effectively empty."])
            
        target_column = self.schema.get("target_column", "label")
        if target_column not in df.columns:
            logger.warning("Target column '%s' missing. Replacing blindly with dummy logic.", target_column)
            features_df, labels = df, np.zeros(len(df))
        else:
            features_df = df.drop(columns=[target_column])
            labels = df[target_column].values
            
            # Compatibility schema verification purely if formally expressed within config bounds
            if self.schema.get("columns"):
                req_names = {c["name"] for c in self.schema["columns"] if c["name"] != target_column}
                missing = req_names - set(features_df.columns)
                if missing:
                    raise DatasetValidationError([f"Required strict validation missing columns explicitly: {missing}"])
                    
        X, y = features_df.values.astype(np.float32), labels
        
        # Nominal to continuous casting
        if y.dtype == object or y.dtype.kind in ("U", "S"):
            uniqs = sorted(set(y))
            lmap = {v: i for i, v in enumerate(uniqs)}
            y = np.array([lmap[v] for v in y], dtype=np.int64)
        else:
            y = y.astype(np.int64)
            
        self.train_data = (X, y)
        self.num_samples = len(X)
        self.num_features = X.shape[1] if X.ndim > 1 else 1
        self._compute_distribution(y.tolist())

        # Structural optimization if legacy environments leverage modern torch paradigms intrinsically 
        if HAS_TORCH:
            try:
                dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
                self.train_data = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
            except Exception as e:
                logger.warning("Optional explicit PyTorch CSV wrapping sequence suppressed: %s", e)


    def _compute_distribution(self, labels_list: List[Any]) -> None:
        try:
            counter = Counter(labels_list)
            total = sum(counter.values())
            self.label_distribution = {
                str(k): round(v / total, 6) for k, v in sorted(counter.items())
            }
            self.num_classes = len(self.label_distribution)
        except TypeError:
            self.label_distribution = {}
            self.num_classes = 0

def load_data(data: Any, schema: Optional[Dict[str, Any]] = None) -> FlexibleDataAdapter:
    """Entry point exposing generic Dataset/DataLoader abstraction interface cleanly."""
    return FlexibleDataAdapter(data, schema)
