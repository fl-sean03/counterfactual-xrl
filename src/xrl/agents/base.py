"""Common agent interface used by the eval runner."""

from __future__ import annotations

from typing import Any, Protocol


class Agent(Protocol):
    """Minimum contract for any policy the eval/counterfactual code drives."""

    def predict(self, obs: Any, deterministic: bool = True) -> int:
        """Return an integer action given an observation."""


class RandomAgent:
    """Uniform-random policy over a discrete action space. Used as baseline."""

    def __init__(self, n_actions: int, rng_seed: int | None = None) -> None:
        import numpy as np

        self._rng = np.random.default_rng(rng_seed)
        self._n = n_actions

    def predict(self, obs: Any, deterministic: bool = True) -> int:
        return int(self._rng.integers(0, self._n))
