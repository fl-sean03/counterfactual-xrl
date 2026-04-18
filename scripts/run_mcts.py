"""Evaluate MCTS agent over N episodes.

MCTS acts on live env state at each step by cloning the simulator and
searching. No training phase — this is planning-from-scratch.

Usage:
    python scripts/run_mcts.py --config configs/mcts_baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import pandas as pd

from xrl.agents.mcts import MCTS, MCTSConfig, root_to_dict
from xrl.envs.simulator import Simulator
from xrl.envs.wrappers import ENV_ID
from xrl.eval.runner import summarise
from xrl.utils.config import ensure_run_dir, load_config, write_meta
from xrl.utils.seeding import seed_everything


def run_one_episode(seed: int, mcts: MCTS, save_tree_dir: Path | None = None) -> dict:
    env = gym.make(ENV_ID)
    env.reset(seed=seed)

    steps = 0
    total = 0.0
    terminated = False
    truncated = False
    tree_dumps: list[dict] = []

    while not (terminated or truncated):
        sim = Simulator.from_env(env)
        action, root = mcts.plan(sim)
        sim.close()

        if save_tree_dir is not None:
            tree_dumps.append(
                {
                    "step": steps,
                    "action": action,
                    "tree": root_to_dict(root, legal_actions=[0, 1, 2]),
                }
            )

        _, reward, terminated, truncated, _ = env.step(int(action))
        total += float(reward)
        steps += 1

    success = terminated and total > 0
    collision = terminated and reward < 0
    env.close()

    out = {
        "seed": seed,
        "steps": steps,
        "return_": total,
        "success": bool(success),
        "collision": bool(collision),
    }
    if save_tree_dir is not None:
        save_tree_dir.mkdir(parents=True, exist_ok=True)
        with open(save_tree_dir / f"trees_seed{seed}.json", "w") as f:
            json.dump(tree_dumps, f, indent=2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-episodes", type=int, default=None, help="override config")
    ap.add_argument("--save-trees", action="store_true", help="dump tree JSONs per step")
    ap.add_argument("--tree-every", type=int, default=5, help="save trees every K episodes")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    run_dir = ensure_run_dir(Path(cfg["output"]["run_dir"]) / f"seed{cfg['seed']}")

    mcts = MCTS(MCTSConfig(**cfg["mcts"]))

    n_ep = args.n_episodes or int(cfg["eval"]["n_episodes"])
    base = int(cfg["eval"]["base_seed"])

    rows: list[dict] = []
    t0 = time.time()
    for i in range(n_ep):
        seed = base + i
        save_trees = args.save_trees and (i % args.tree_every == 0)
        tree_dir = run_dir / "trees" if save_trees else None
        r = run_one_episode(seed, mcts, save_tree_dir=tree_dir)
        rows.append(r)
        if (i + 1) % 5 == 0 or i == n_ep - 1:
            succ_so_far = sum(row["success"] for row in rows) / len(rows)
            print(f"  [{i+1}/{n_ep}] success-rate so far = {succ_so_far:.3f}")

    dt = time.time() - t0
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "eval.csv", index=False)
    summ = summarise(df)
    with open(run_dir / "eval_summary.json", "w") as f:
        json.dump(summ, f, indent=2)

    write_meta(
        run_dir,
        {
            "config_path": args.config,
            "config": cfg,
            "n_episodes": n_ep,
            "seconds": dt,
        },
    )
    print(f"DONE. {n_ep} episodes in {dt:.1f}s")
    for k, v in summ.items():
        lo, hi = v["ci"]
        print(f"  {k:>10s}: {v['mean']:.3f}   95%CI=({lo:.3f}, {hi:.3f})")


if __name__ == "__main__":
    main()
