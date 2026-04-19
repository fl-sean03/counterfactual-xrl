"""Deep-copy simulator for MCTS.

MCTS requires the ability to "roll back" the environment to a known state
and branch forward along different action sequences. MiniGrid envs support
``copy.deepcopy`` (their RNG is a ``numpy.random.Generator`` stored on
``unwrapped.np_random``, which pickles cleanly). The only gotcha is that
pygame rendering contexts are not picklable, we therefore construct
simulator envs with ``render_mode=None`` (MiniGrid's default) and never
attach a renderer.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import gymnasium as gym

from xrl.envs.wrappers import ENV_ID


@dataclass
class StepResult:
    obs: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class Simulator:
    """Lightweight wrapper that lets MCTS branch off arbitrary copies.

    Usage:
        sim = Simulator.from_seed(seed=42)           # or from_env(env)
        # at each MCTS iteration:
        child = sim.clone()
        r = child.step(action)
        ...
    """

    def __init__(self, env: gym.Env) -> None:
        self._env = env

    @classmethod
    def from_seed(cls, seed: int) -> Simulator:
        env = gym.make(ENV_ID)
        env.reset(seed=seed)
        return cls(env)

    @classmethod
    def from_env(cls, env: gym.Env) -> Simulator:
        """Clone a live env's internal state into a fresh simulator."""
        return cls(copy.deepcopy(env.unwrapped if hasattr(env, "unwrapped") else env))

    def clone(self) -> Simulator:
        return Simulator(copy.deepcopy(self._env))

    def step(self, action: int) -> StepResult:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return StepResult(
            obs=obs,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )

    def legal_actions(self) -> list[int]:
        return list(range(self._env.action_space.n))

    @property
    def env(self) -> gym.Env:
        return self._env

    @property
    def state_fingerprint(self) -> tuple:
        """Cheap identity for state: agent pos/dir + obstacle positions.

        Not a hash of the full state (RNG state is excluded), but
        sufficient to detect duplicates in a search tree's key set.
        """
        u = self._env.unwrapped
        ob_pos = tuple(tuple(getattr(o, "cur_pos", (-1, -1))) for o in getattr(u, "obstacles", []))
        return (tuple(u.agent_pos), int(u.agent_dir), ob_pos)

    def close(self) -> None:
        self._env.close()
