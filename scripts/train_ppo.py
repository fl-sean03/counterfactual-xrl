"""Train a PPO agent on MiniGrid-Dynamic-Obstacles-8x8.

Usage:
    python scripts/train_ppo.py --config configs/ppo_baseline.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from xrl.agents.ppo import PPOAgent
from xrl.envs.wrappers import make_env
from xrl.utils.config import ensure_run_dir, load_config, write_meta
from xrl.utils.seeding import seed_everything


def _parse_schedule(value):
    """Allow ``"linear:start:end"`` strings in YAML to become callables.

    SB3 accepts a callable ``(progress_remaining: float in [0,1]) -> float``
    for both ``learning_rate`` and ``clip_range``. Linear-anneal is the
    standard PPO recipe.
    """
    if isinstance(value, str) and value.startswith("linear:"):
        _, start, end = value.split(":")
        start_f = float(start)
        end_f = float(end)

        def schedule(progress_remaining: float) -> float:
            return end_f + (start_f - end_f) * progress_remaining

        return schedule
    return value


SCHEDULE_KEYS = ("learning_rate", "clip_range")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=8, help="parallel envs for PPO")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    cfg["seed"] = seed
    seed_everything(seed)

    obs_mode = cfg["obs_mode"]
    shaping = bool(cfg.get("shaping", False))
    shaping_coef = float(cfg.get("shaping_coef", 0.02))
    step_penalty = float(cfg.get("step_penalty", 0.0))

    run_dir = ensure_run_dir(Path(cfg["output"]["run_dir"]) / f"seed{seed}")

    def make_single(i: int):
        def thunk():
            env = make_env(
                mode=obs_mode,
                shaping=shaping,
                shaping_coef=shaping_coef,
                step_penalty=step_penalty,
            )
            return Monitor(env, filename=str(run_dir / f"monitor_{i}.csv"))

        return thunk

    if args.n_envs > 1:
        venv = SubprocVecEnv([make_single(i) for i in range(args.n_envs)])
    else:
        venv = DummyVecEnv([make_single(0)])

    sb3_kwargs = dict(cfg["sb3_kwargs"])
    for k in SCHEDULE_KEYS:
        if k in sb3_kwargs:
            sb3_kwargs[k] = _parse_schedule(sb3_kwargs[k])

    agent = PPOAgent.new(
        venv,
        policy="MlpPolicy",
        seed=seed,
        tensorboard_log=str(run_dir / "tb"),
        **sb3_kwargs,
    )

    t0 = time.time()
    agent.learn(total_timesteps=int(cfg["total_timesteps"]))
    dt = time.time() - t0

    ckpt = run_dir / "model.zip"
    agent.save(ckpt)

    write_meta(
        run_dir,
        {
            "config_path": args.config,
            "config": cfg,
            "seed": seed,
            "obs_mode": obs_mode,
            "policy": "MlpPolicy",
            "total_timesteps": int(cfg["total_timesteps"]),
            "train_seconds": dt,
            "n_envs": args.n_envs,
            "step_penalty": step_penalty,
            "shaping": shaping,
            "checkpoint": str(ckpt),
        },
    )
    print(f"DONE. Trained PPO for {dt:.1f}s across {args.n_envs} envs. Saved {ckpt}")


if __name__ == "__main__":
    main()
