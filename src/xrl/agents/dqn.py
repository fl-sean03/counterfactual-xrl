"""Thin wrapper around Stable-Baselines3 DQN.

Exposed beyond SB3's API:
- ``q_values(obs)`` returns the raw per-action Q-vector (SB3 has no public
  accessor); needed by the explainer.

**This file is off-the-shelf in the report: SB3 does the learning.** Only
the ``q_values`` helper and save/load ergonomics are ours.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import DQN


class DQNAgent:
    def __init__(self, model: DQN) -> None:
        self.model = model

    @classmethod
    def new(
        cls,
        env,
        *,
        policy: str = "CnnPolicy",
        seed: int = 0,
        **dqn_kwargs: Any,
    ) -> DQNAgent:
        model = DQN(policy, env, seed=seed, verbose=0, **dqn_kwargs)
        return cls(model)

    @classmethod
    def load(cls, path: str | Path, env=None) -> DQNAgent:
        model = DQN.load(str(path), env=env)
        return cls(model)

    def save(self, path: str | Path) -> None:
        self.model.save(str(path))

    def learn(self, total_timesteps: int, **kwargs: Any) -> None:
        self.model.learn(total_timesteps=total_timesteps, **kwargs)

    def predict(self, obs: Any, deterministic: bool = True) -> int:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def q_values(self, obs: Any) -> np.ndarray:
        """Return the per-action Q-vector for a single observation.

        SB3's ``DQN.policy.q_net`` is the online Q-network. We push the obs
        through it exactly as ``_predict`` does internally.
        """
        obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            q = self.model.policy.q_net(obs_tensor)
        return q.cpu().numpy().squeeze(0)
