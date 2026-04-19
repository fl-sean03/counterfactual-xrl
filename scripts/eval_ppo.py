"""Evaluate a trained PPO policy (analogue to scripts/eval.py for DQN).

Usage:
    python scripts/eval_ppo.py --run-dir results/ppo/baseline/seed0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from xrl.agents.ppo import PPOAgent
from xrl.envs.wrappers import make_env
from xrl.eval.runner import run_episodes, summarise
from xrl.utils.seeding import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-episodes", type=int, default=300)
    ap.add_argument("--base-seed", type=int, default=10000)
    args = ap.parse_args()

    seed_everything(0)
    run_dir = Path(args.run_dir)
    with open(run_dir / "meta.json") as f:
        meta = json.load(f)
    obs_mode = meta["obs_mode"]
    step_penalty = float(meta.get("step_penalty", 0.0))
    shaping = bool(meta.get("shaping", False))

    ppo = PPOAgent.load(run_dir / "model.zip")

    def make(s):
        # Evaluation uses the *base* reward (no shaping/penalty) to measure
        # task performance apples-to-apples with other agents.
        return make_env(mode=obs_mode, seed=s)

    def predict(obs):
        return ppo.predict(obs)

    df = run_episodes(
        make_env=make, predict=predict, n_episodes=args.n_episodes, base_seed=args.base_seed
    )
    df.to_csv(run_dir / "eval.csv", index=False)
    summ = summarise(df)
    with open(run_dir / "eval_summary.json", "w") as f:
        json.dump(summ, f, indent=2)

    print(f"n={len(df)}  obs_mode={obs_mode}  shaped={shaping}  step_penalty={step_penalty}")
    for k, v in summ.items():
        lo, hi = v["ci"]
        print(f"  {k:>10s}: {v['mean']:.3f}   95%CI=({lo:.3f}, {hi:.3f})")


if __name__ == "__main__":
    main()
