"""Observation wrappers for MiniGrid-Dynamic-Obstacles-8x8.

Two modes exposed via `make_env`:

1. ``image``   — ``ImgObsWrapper`` from MiniGrid. The observation is the
   7x7x3 egocentric partial image. Consumed by ``CnnPolicy``. This is the
   baseline, partial-observability DQN.
2. ``symbolic`` — ``FlatSymbolicObsWrapper`` below. The observation is a
   flat vector of full ground-truth state: agent pos, agent dir,
   per-obstacle positions. Consumed by ``MlpPolicy``. This is the
   full-state control DQN used to isolate observability from
   training-paradigm effects (see docs/observation_vs_state.md).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation
from minigrid.wrappers import ImgObsWrapper

ENV_ID = "MiniGrid-Dynamic-Obstacles-8x8-v0"


class FlatSymbolicObsWrapper(gym.ObservationWrapper):
    """Expose the full ground-truth state as a fixed-length float vector.

    Layout (length = 2 + 4 + 2 * n_obstacles):
        [agent_x / width, agent_y / height,
         dir_onehot_0, dir_onehot_1, dir_onehot_2, dir_onehot_3,
         obs_0_x / width, obs_0_y / height, ..., obs_K_x, obs_K_y]

    Coordinates normalized to [0, 1].
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        u = env.unwrapped
        self._n_obstacles = u.n_obstacles
        self._w = u.width
        self._h = u.height
        dim = 2 + 4 + 2 * self._n_obstacles
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(dim,), dtype=np.float32)

    def observation(self, obs: dict) -> np.ndarray:
        u = self.env.unwrapped
        vec = np.zeros(self.observation_space.shape, dtype=np.float32)
        ax, ay = u.agent_pos
        vec[0] = ax / self._w
        vec[1] = ay / self._h
        d = int(u.agent_dir)
        vec[2 + d] = 1.0

        # Obstacles are stored on unwrapped.obstacles (list of WorldObj with .cur_pos).
        for i, ob in enumerate(u.obstacles[: self._n_obstacles]):
            pos = getattr(ob, "cur_pos", None)
            if pos is None:
                # Obstacle not yet placed for some reason — leave as 0.
                continue
            ox, oy = pos
            vec[6 + 2 * i] = ox / self._w
            vec[6 + 2 * i + 1] = oy / self._h
        return vec


def make_env(mode: str = "image", seed: int | None = None) -> gym.Env:
    """Construct a wrapped MiniGrid-Dynamic-Obstacles-8x8 env.

    Args:
        mode: one of ``"image"`` or ``"symbolic"``.
        seed: optional seed for the initial reset.

    Returns:
        A ``gymnasium.Env`` with the chosen observation wrapper.
    """
    env = gym.make(ENV_ID)
    if mode == "image":
        # 7x7x3 image, flattened to 147-d vector for MlpPolicy.
        # (Default SB3 NatureCNN requires 36x36 minimum; MiniGrid's 7x7
        # is too small. Flatten + MLP is simpler and adequate.)
        env = ImgObsWrapper(env)
        env = FlattenObservation(env)
    elif mode == "symbolic":
        env = FlatSymbolicObsWrapper(env)
    else:
        raise ValueError(f"Unknown obs mode: {mode!r}")
    if seed is not None:
        env.reset(seed=seed)
    return env
