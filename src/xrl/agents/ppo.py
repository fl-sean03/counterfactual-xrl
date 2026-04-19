"""Thin wrapper around Stable-Baselines3 PPO.

PPO is consistently reported as the best off-the-shelf choice for
MiniGrid: its on-policy data collection and entropy bonus handle the
exploration demands of adversarial obstacle dynamics better than DQN's
epsilon-greedy. We keep the same ``predict`` / ``save`` / ``load`` API
as the DQN wrapper so the downstream pipeline is interchangeable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO


class PPOAgent:
    def __init__(self, model: PPO) -> None:
        self.model = model

    @classmethod
    def new(
        cls,
        env,
        *,
        policy: str = "MlpPolicy",
        seed: int = 0,
        **ppo_kwargs: Any,
    ) -> PPOAgent:
        model = PPO(policy, env, seed=seed, verbose=0, **ppo_kwargs)
        return cls(model)

    @classmethod
    def load(cls, path: str | Path, env=None) -> PPOAgent:
        model = PPO.load(str(path), env=env)
        return cls(model)

    def save(self, path: str | Path) -> None:
        self.model.save(str(path))

    def learn(self, total_timesteps: int, **kwargs: Any) -> None:
        self.model.learn(total_timesteps=total_timesteps, **kwargs)

    def predict(self, obs: Any, deterministic: bool = True) -> int:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def sample(self, obs: Any) -> int:
        """Stochastic sample from the policy distribution.

        Used for Monte-Carlo-style counterfactual rollouts: we want
        rollouts to reflect the agent's actual distribution over
        trajectories, not a single argmax trajectory.
        """
        action, _ = self.model.predict(obs, deterministic=False)
        return int(action)

    def action_probs(self, obs: Any) -> np.ndarray:
        """Return the per-action probability vector under the current policy.

        PPO does not have Q-values, so the explainer's ``agent_metadata``
        block uses these probabilities as the analogous per-action
        scalar.
        """
        obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_tensor)
        return dist.distribution.probs.cpu().numpy().squeeze(0)
