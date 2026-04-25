"""Generate DecisionRecord JSONs for every step of a trajectory, for both agents.

Usage:
    python scripts/build_decision_records.py \
        --dqn-run-dir results/dqn/symbolic/seed0 \
        --mcts-config configs/mcts_baseline.yaml \
        --seeds 10000 10001 10002 10003 10004 \
        --n-per-action 100 \
        --out-dir results/decision_records
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401

from xrl.agents.dqn import DQNAgent
from xrl.agents.mcts import MCTS, MCTSConfig
from xrl.agents.ppo import PPOAgent
from xrl.analysis.counterfactual import counterfactual_rollouts
from xrl.analysis.records import DecisionRecord, save_record, validate_record_dict
from xrl.analysis.tree_stats import mcts_root_to_action_stats
from xrl.envs.simulator import Simulator
from xrl.envs.wrappers import ENV_ID, FlatSymbolicObsWrapper, make_env
from xrl.utils.config import load_config


def obstacle_positions(env: gym.Env) -> list[tuple[int, int]]:
    u = env.unwrapped
    return [tuple(getattr(o, "cur_pos", (-1, -1))) for o in getattr(u, "obstacles", [])]


def obs_from_sim_factory(obs_mode: str):
    """Return a callable ``obs_fn(sim) -> observation`` matching make_env's output."""
    if obs_mode == "symbolic":
        # Build one wrapper instance; reuse it by recomputing from current state.
        _w_cache: dict[int, Any] = {}

        def obs_fn(sim: Simulator) -> Any:
            # Create a shim wrapper tied to the sim env the first time we see it.
            key = id(sim.env)
            w = _w_cache.get(key)
            if w is None:
                w = FlatSymbolicObsWrapper(sim.env)
                _w_cache[key] = w
            return w.observation({})

        return obs_fn
    if obs_mode == "image":
        import numpy as np

        def obs_fn(sim: Simulator) -> Any:
            # gen_obs returns a dict with keys image/direction/mission.
            # Our make_env('image') flattens the 7x7x3 image to a 147-d float vector.
            d = sim.env.unwrapped.gen_obs()
            img = d["image"].astype(np.float32).reshape(-1)
            return img

        return obs_fn
    raise ValueError(f"Unknown obs_mode: {obs_mode}")


def build_dqn_records(
    agent_id: str,
    run_dir: Path,
    seeds: list[int],
    n_per_action: int,
    out_dir: Path,
    max_steps_per_traj: int = 20,
    rollout_cap: int = 50,
) -> int:
    with open(run_dir / "meta.json") as f:
        meta = json.load(f)
    obs_mode = meta["obs_mode"]
    dqn = DQNAgent.load(run_dir / "model.zip")
    obs_fn = obs_from_sim_factory(obs_mode)

    def policy_predict(obs):
        return dqn.predict(obs)

    written = 0
    for seed in seeds:
        env = make_env(mode=obs_mode, seed=seed)
        obs, _ = env.reset(seed=seed)
        step = 0
        terminated = False
        truncated = False
        while not (terminated or truncated) and step < max_steps_per_traj:
            sim = Simulator.from_env(env)
            stats = counterfactual_rollouts(
                sim,
                policy_predict,
                obs_fn,
                n_per_action=n_per_action,
                seed=seed * 1000 + step,
                max_steps=rollout_cap,
            )
            sim.close()
            action = dqn.predict(obs)
            u = env.unwrapped
            rec = DecisionRecord(
                source="dqn_rollout",
                agent_id=agent_id,
                state_id=f"{seed}:{step}",
                step=step,
                agent_pos=tuple(int(x) for x in u.agent_pos),
                agent_dir=int(u.agent_dir),
                obstacle_positions=[tuple(int(x) for x in p) for p in obstacle_positions(env)],
                chosen_action=int(action),
                per_action_stats=stats,
                agent_metadata={"q_values": dqn.q_values(obs).tolist()},
            )
            # Validate before writing.
            d = rec.to_dict()
            validate_record_dict(d)
            save_record(rec, out_dir / agent_id / f"seed{seed}_step{step:03d}.json")
            written += 1

            obs, _, terminated, truncated, _ = env.step(int(action))
            step += 1
        env.close()
    return written


def build_ppo_records(
    agent_id: str,
    run_dir: Path,
    seeds: list[int],
    n_per_action: int,
    out_dir: Path,
    max_steps_per_traj: int = 20,
    rollout_cap: int = 50,
    stochastic_rollouts: bool = True,
) -> int:
    with open(run_dir / "meta.json") as f:
        meta = json.load(f)
    obs_mode = meta["obs_mode"]
    ppo = PPOAgent.load(run_dir / "model.zip")
    obs_fn = obs_from_sim_factory(obs_mode)

    # Rollouts draw from PPO's action distribution (stochastic) so the
    # Monte Carlo evidence reflects the agent's policy distribution
    # rather than collapsing to a single argmax trajectory.
    policy_predict = ppo.sample if stochastic_rollouts else ppo.predict

    written = 0
    for seed in seeds:
        # For evaluation fairness, record-building uses the *base* reward env
        # (no shaping / step penalty). The policy still acts the same; only
        # the recorded returns reflect the clean task reward.
        env = make_env(mode=obs_mode, seed=seed)
        obs, _ = env.reset(seed=seed)
        step = 0
        terminated = False
        truncated = False
        while not (terminated or truncated) and step < max_steps_per_traj:
            sim = Simulator.from_env(env)
            stats = counterfactual_rollouts(
                sim,
                policy_predict,
                obs_fn,
                n_per_action=n_per_action,
                seed=seed * 1000 + step,
                max_steps=rollout_cap,
            )
            sim.close()
            action = ppo.predict(obs)
            u = env.unwrapped
            rec = DecisionRecord(
                source="policy_rollout",
                agent_id=agent_id,
                state_id=f"{seed}:{step}",
                step=step,
                agent_pos=tuple(int(x) for x in u.agent_pos),
                agent_dir=int(u.agent_dir),
                obstacle_positions=[tuple(int(x) for x in p) for p in obstacle_positions(env)],
                chosen_action=int(action),
                per_action_stats=stats,
                agent_metadata={
                    "action_probs": ppo.action_probs(obs).tolist(),
                    "policy": "PPO",
                    "rollouts_stochastic": bool(stochastic_rollouts),
                    "n_per_action": int(n_per_action),
                },
            )
            d = rec.to_dict()
            validate_record_dict(d)
            save_record(rec, out_dir / agent_id / f"seed{seed}_step{step:03d}.json")
            written += 1
            obs, _, terminated, truncated, _ = env.step(int(action))
            step += 1
        env.close()
    return written


def build_mcts_records(
    agent_id: str,
    mcts_cfg_path: str,
    seeds: list[int],
    out_dir: Path,
    max_steps_per_traj: int = 20,
) -> int:
    cfg = load_config(mcts_cfg_path)
    mcts = MCTS(MCTSConfig(**cfg["mcts"]))

    written = 0
    for seed in seeds:
        env = gym.make(ENV_ID)
        env.reset(seed=seed)
        step = 0
        terminated = False
        truncated = False
        while not (terminated or truncated) and step < max_steps_per_traj:
            sim = Simulator.from_env(env)
            action, root = mcts.plan(sim)
            sim.close()
            u = env.unwrapped
            stats = mcts_root_to_action_stats(root, legal_actions=[0, 1, 2])
            rec = DecisionRecord(
                source="mcts_tree",
                agent_id=agent_id,
                state_id=f"{seed}:{step}",
                step=step,
                agent_pos=tuple(int(x) for x in u.agent_pos),
                agent_dir=int(u.agent_dir),
                obstacle_positions=[tuple(int(x) for x in p) for p in obstacle_positions(env)],
                chosen_action=int(action),
                per_action_stats=stats,
                agent_metadata={
                    "root_visits": root.visits,
                    "root_value": root.mean_value,
                    "budget": cfg["mcts"]["sims_per_decision"],
                    "rollout_policy": cfg["mcts"]["rollout_policy"],
                },
            )
            d = rec.to_dict()
            validate_record_dict(d)
            save_record(rec, out_dir / agent_id / f"seed{seed}_step{step:03d}.json")
            written += 1

            _, _, terminated, truncated, _ = env.step(int(action))
            step += 1
        env.close()
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dqn-run-dir", help="DQN run directory with model.zip + meta.json")
    ap.add_argument("--dqn-agent-id", default=None)
    ap.add_argument("--ppo-run-dir", help="PPO run directory with model.zip + meta.json")
    ap.add_argument("--ppo-agent-id", default=None)
    ap.add_argument("--mcts-config", help="Path to MCTS config")
    ap.add_argument("--mcts-agent-id", default="mcts_baseline")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--n-per-action", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=20, help="cap steps per trajectory")
    ap.add_argument("--rollout-cap", type=int, default=50)
    ap.add_argument(
        "--ppo-deterministic-rollouts",
        action="store_true",
        help="use PPO argmax (old behaviour). Default is stochastic sampling.",
    )
    ap.add_argument("--out-dir", default="results/decision_records")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    if args.dqn_run_dir:
        aid = (
            args.dqn_agent_id
            or Path(args.dqn_run_dir).parent.name + "_" + Path(args.dqn_run_dir).name
        )
        n = build_dqn_records(
            aid,
            Path(args.dqn_run_dir),
            args.seeds,
            args.n_per_action,
            out_dir,
            max_steps_per_traj=args.max_steps,
            rollout_cap=args.rollout_cap,
        )
        print(f"DQN: wrote {n} records to {out_dir / aid}")

    if args.ppo_run_dir:
        aid = args.ppo_agent_id or "ppo_baseline"
        n = build_ppo_records(
            aid,
            Path(args.ppo_run_dir),
            args.seeds,
            args.n_per_action,
            out_dir,
            max_steps_per_traj=args.max_steps,
            rollout_cap=args.rollout_cap,
            stochastic_rollouts=not args.ppo_deterministic_rollouts,
        )
        print(
            f"PPO: wrote {n} records to {out_dir / aid} "
            f"(n_per_action={args.n_per_action}, stochastic="
            f"{not args.ppo_deterministic_rollouts})"
        )

    if args.mcts_config:
        n = build_mcts_records(
            args.mcts_agent_id,
            args.mcts_config,
            args.seeds,
            out_dir,
            max_steps_per_traj=args.max_steps,
        )
        print(f"MCTS: wrote {n} records to {out_dir / args.mcts_agent_id}")


if __name__ == "__main__":
    main()
