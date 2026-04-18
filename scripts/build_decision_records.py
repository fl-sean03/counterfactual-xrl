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
    """Return a callable ``obs_fn(sim) -> observation`` that re-applies the wrapper."""
    if obs_mode == "symbolic":
        # Re-wrap the simulator's env to get the flat symbolic obs.
        def obs_fn(sim: Simulator) -> Any:
            w = FlatSymbolicObsWrapper(sim.env)
            return w.observation({})

        return obs_fn
    if obs_mode == "image":
        from gymnasium.wrappers import FlattenObservation
        from minigrid.wrappers import ImgObsWrapper

        def obs_fn(sim: Simulator) -> Any:
            e = ImgObsWrapper(sim.env)
            e = FlattenObservation(e)
            return e.observation(sim.env.unwrapped.gen_obs())

        return obs_fn
    raise ValueError(f"Unknown obs_mode: {obs_mode}")


def build_dqn_records(
    agent_id: str,
    run_dir: Path,
    seeds: list[int],
    n_per_action: int,
    out_dir: Path,
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
        while not (terminated or truncated):
            sim = Simulator.from_env(env)
            stats = counterfactual_rollouts(
                sim, policy_predict, obs_fn, n_per_action=n_per_action, seed=seed * 1000 + step
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


def build_mcts_records(
    agent_id: str,
    mcts_cfg_path: str,
    seeds: list[int],
    out_dir: Path,
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
        while not (terminated or truncated):
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
    ap.add_argument("--mcts-config", help="Path to MCTS config")
    ap.add_argument("--mcts-agent-id", default="mcts_baseline")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--n-per-action", type=int, default=100)
    ap.add_argument("--out-dir", default="results/decision_records")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    if args.dqn_run_dir:
        aid = (
            args.dqn_agent_id
            or Path(args.dqn_run_dir).parent.name + "_" + Path(args.dqn_run_dir).name
        )
        n = build_dqn_records(aid, Path(args.dqn_run_dir), args.seeds, args.n_per_action, out_dir)
        print(f"DQN: wrote {n} records to {out_dir / aid}")

    if args.mcts_config:
        n = build_mcts_records(args.mcts_agent_id, args.mcts_config, args.seeds, out_dir)
        print(f"MCTS: wrote {n} records to {out_dir / args.mcts_agent_id}")


if __name__ == "__main__":
    main()
