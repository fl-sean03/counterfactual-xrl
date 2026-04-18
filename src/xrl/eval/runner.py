"""Policy-agnostic episode runner used by DQN, MCTS, and random baseline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class EpisodeResult:
    episode: int
    seed: int
    steps: int
    return_: float
    success: bool
    collision: bool


def _episode_is_success(reward: float, terminated: bool) -> bool:
    """MiniGrid: positive reward at termination means goal reached."""
    return terminated and reward > 0.0


def _episode_is_collision(reward: float, terminated: bool) -> bool:
    return terminated and reward < 0.0


def run_episodes(
    make_env: Callable[[int], Any],
    predict: Callable[[Any], int],
    n_episodes: int,
    base_seed: int = 1000,
) -> pd.DataFrame:
    """Run ``n_episodes``, seeding each env deterministically.

    Args:
        make_env: factory taking a seed and returning a fresh wrapped env.
        predict: callable mapping observation to integer action.
        n_episodes: number of episodes.
        base_seed: seeds used are ``base_seed + episode_idx``.
    """
    rows: list[EpisodeResult] = []
    for i in range(n_episodes):
        seed = base_seed + i
        env = make_env(seed)
        obs, _ = env.reset(seed=seed)
        total = 0.0
        steps = 0
        terminal_reward = 0.0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            terminal_reward = float(reward)
            steps += 1
        env.close()
        rows.append(
            EpisodeResult(
                episode=i,
                seed=seed,
                steps=steps,
                return_=total,
                success=_episode_is_success(terminal_reward, terminated),
                collision=_episode_is_collision(terminal_reward, terminated),
            )
        )
    return pd.DataFrame([vars(r) for r in rows])


def summarise(df: pd.DataFrame, n_bootstrap: int = 1000, seed: int = 0) -> dict:
    """Mean + 95% percentile-bootstrap CI on success, collision, return."""
    rng = np.random.default_rng(seed)
    out: dict = {}
    for col in ("success", "collision", "return_", "steps"):
        vals = df[col].to_numpy().astype(float)
        if len(vals) == 0:
            out[col] = {"mean": float("nan"), "ci": (float("nan"), float("nan"))}
            continue
        boots = rng.choice(vals, size=(n_bootstrap, len(vals)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[col] = {"mean": float(vals.mean()), "ci": (float(lo), float(hi))}
    return out
