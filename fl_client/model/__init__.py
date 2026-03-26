"""Model module — BYOM abstractions for PyTorch and Scikit-Learn."""
from fl_client.model.wrapper import wrap_model, FLModelWrapper, ModelFormatError

__all__ = ["wrap_model", "FLModelWrapper", "ModelFormatError"]
