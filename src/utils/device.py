"""
Device selection and seed management.
"""

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def resolve_device(requested_device: str = "auto") -> torch.device:
    req = requested_device.lower()
    if req == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif req == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    elif req == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("MPS requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("mps")
    elif req == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown device request: {requested_device}")

    logger.info("Using device: %s", device)
    return device


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    logger.debug("Random seed set to %d (deterministic=%s)", seed, deterministic)
