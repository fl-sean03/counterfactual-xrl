"""Evaluate a trained policy (or random baseline) over N episodes.

Usage:
    python scripts/eval.py --run-dir results/dqn/baseline/seed0
    python scripts/eval.py --random   # baseline over N episodes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from xrl.agents.base import RandomAgent
from xrl.agents.dqn import DQNAgent
from xrl.envs.wrappers import make_env
from xrl.eval.runner import run_episodes, summarise
from xrl.utils.seeding import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", help="DQN run dir containing model.zip + meta.json")
    ap.add_argument("--random", action="store_true", help="evaluate a random baseline")
    ap.add_argument("--obs-mode", default="image", help="for --random only")
    ap.add_argument("--n-episodes", type=int, default=300)
    ap.add_argument("--base-seed", type=int, default=10000)
    ap.add_argument("--out", default=None, help="output dir; defaults to run-dir or results/random")
    args = ap.parse_args()

    seed_everything(0)

    if args.random:
        obs_mode = args.obs_mode
        agent = RandomAgent(n_actions=3, rng_seed=0)

        def make(s):
            return make_env(mode=obs_mode, seed=s)

        def predict(obs):
            return agent.predict(obs)

        out_dir = Path(args.out or "results/random")
    else:
        assert args.run_dir, "--run-dir required unless --random"
        run_dir = Path(args.run_dir)
        with open(run_dir / "meta.json") as f:
            meta = json.load(f)
        obs_mode = meta["obs_mode"]
        dqn = DQNAgent.load(run_dir / "model.zip")

        def make(s):
            return make_env(mode=obs_mode, seed=s)

        def predict(obs):
            return dqn.predict(obs)

        out_dir = Path(args.out or run_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    df = run_episodes(
        make_env=make, predict=predict, n_episodes=args.n_episodes, base_seed=args.base_seed
    )
    df.to_csv(out_dir / "eval.csv", index=False)
    summ = summarise(df)
    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(summ, f, indent=2)

    print(f"n={len(df)}  obs_mode={obs_mode}")
    for k, v in summ.items():
        lo, hi = v["ci"]
        print(f"  {k:>10s}: {v['mean']:.3f}   95%CI=({lo:.3f}, {hi:.3f})")


if __name__ == "__main__":
    main()
