"""Phase 0 smoke test: load env, verify core facts used by later phases."""

from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401  (registers MiniGrid envs)

ENV_ID = "MiniGrid-Dynamic-Obstacles-8x8-v0"


def main() -> None:
    env = gym.make(ENV_ID)
    obs, info = env.reset(seed=0)

    print(f"Env: {ENV_ID}")
    print(f"  action_space: {env.action_space}")
    print(f"  observation_space keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")
    if isinstance(obs, dict) and "image" in obs:
        print(f"  image shape: {obs['image'].shape}")
    print(f"  max_steps: {env.spec.max_episode_steps if env.spec else 'n/a'}")

    unwrapped = env.unwrapped
    n_obstacles = getattr(unwrapped, "n_obstacles", None)
    print(f"  n_obstacles: {n_obstacles}")

    total_reward = 0.0
    min_r, max_r = float("inf"), float("-inf")
    terminations = 0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        min_r = min(min_r, reward)
        max_r = max(max_r, reward)
        if terminated or truncated:
            terminations += 1
            obs, info = env.reset(seed=0)

    print(
        f"100-step random rollout: total_reward={total_reward:.3f}, "
        f"min_r={min_r:.3f}, max_r={max_r:.3f}, resets={terminations}"
    )

    # Assertions that later phases depend on.
    assert env.action_space.n == 3, (
        f"Expected Discrete(3), got {env.action_space.n}. "
        "Newer MiniGrid restricts Dynamic-Obstacles to {left, right, forward}."
    )
    assert n_obstacles == 4, f"Expected 4 obstacles, got {n_obstacles}"

    try:
        import torch

        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("torch not installed")

    print("SMOKE OK")


if __name__ == "__main__":
    main()
