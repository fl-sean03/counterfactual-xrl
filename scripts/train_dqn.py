"""Train a DQN on MiniGrid-Dynamic-Obstacles-8x8 per a YAML config.

Usage:
    python scripts/train_dqn.py --config configs/dqn_baseline.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from xrl.agents.dqn import DQNAgent
from xrl.envs.wrappers import make_env
from xrl.utils.config import ensure_run_dir, load_config, write_meta
from xrl.utils.seeding import seed_everything


def build_policy_name(obs_mode: str) -> str:
    # Both modes use MlpPolicy after flattening — see envs/wrappers.py.
    return "MlpPolicy"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None, help="override config seed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    cfg["seed"] = seed
    seed_everything(seed)

    obs_mode = cfg["obs_mode"]
    run_dir = ensure_run_dir(Path(cfg["output"]["run_dir"]) / f"seed{seed}")

    def env_thunk():
        env = make_env(mode=obs_mode)
        return Monitor(env, filename=str(run_dir / "monitor.csv"))

    venv = DummyVecEnv([env_thunk])

    policy = build_policy_name(obs_mode)
    agent = DQNAgent.new(
        venv,
        policy=policy,
        seed=seed,
        tensorboard_log=str(run_dir / "tb"),
        **cfg["sb3_kwargs"],
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
            "policy": policy,
            "total_timesteps": int(cfg["total_timesteps"]),
            "train_seconds": dt,
            "checkpoint": str(ckpt),
        },
    )
    print(f"DONE. Trained for {dt:.1f}s. Saved {ckpt}")


if __name__ == "__main__":
    main()
