"""Atomic seeding for reproducibility across random/numpy/torch/env."""

from __future__ import annotations

import os
import random
from typing import Any


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def seed_env(env: Any, seed: int) -> Any:
    obs, info = env.reset(seed=seed)
    try:
        env.action_space.seed(seed)
    except AttributeError:
        pass
    return obs, info
