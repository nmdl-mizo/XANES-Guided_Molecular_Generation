from __future__ import annotations

import pickle
import warnings
from typing import Any

import torch


def load_torch_file(path: str, map_location: str = "cpu") -> Any:
    """
    Prefer safe loading with weights_only=True.
    Falls back for older torch versions that do not support the argument.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError:
        # Some trusted files (or older formats) require full unpickling.
        warnings.warn(
            f"weights_only=True failed for '{path}'. Falling back to weights_only=False.",
            RuntimeWarning,
        )
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Backward compatibility for older PyTorch.
        return torch.load(path, map_location=map_location)
