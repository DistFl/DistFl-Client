"""Training module — dataset handling and training engine."""
from fl_client.training.dataset import FlexibleDataAdapter, load_data
from fl_client.training.trainer import Trainer

__all__ = ["FlexibleDataAdapter", "load_data", "Trainer"]
