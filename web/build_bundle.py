"""Bundle decision records + explanations into a single data.json for the SPA.

Reads:
    results/decision_records/{ppo_tuned,mcts_baseline}/seed{N}_step{K}.json
    results/explanations/{ppo_tuned,mcts_baseline}/seed{N}_step{K}.json
    results/{ppo,mcts}/...eval_summary.json
    results/random/eval_summary.json
    results/metrics/summary.json

Writes:
    web/site/data.json

Output schema (1 file, ~1 MB):

    {
      "task_summary":      {"random": {...}, "ppo_tuned": {...}, "mcts_baseline": {...}},
      "metrics_summary":   {"policy_rollout": {...}, "mcts_tree": {...}},
      "agents": {
        "ppo_tuned":     {"label": "PPO (tuned)",     "trajectories": {"10000": [step_obj, ...], ...}},
        "mcts_baseline": {"label": "MCTS (500 sims)", "trajectories": {"10000": [...], ...}}
      }
    }

Each step_obj is the per-step DecisionRecord union the per-step Explanation,
flattened into one object so the frontend doesn't have to join.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORDS = ROOT / "results" / "decision_records"
EXPLANATIONS = ROOT / "results" / "explanations"
SITE = ROOT / "web" / "site"

AGENTS = {
    "ppo_tuned": "PPO (tuned)",
    "mcts_baseline": "MCTS (500 sims)",
}


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _collect_agent(agent_id: str) -> dict:
    rec_dir = RECORDS / agent_id
    exp_dir = EXPLANATIONS / agent_id
    trajectories: dict[str, list[dict]] = {}
    for rec_path in sorted(rec_dir.glob("*.json")):
        seed_step = rec_path.stem  # "seed10000_step000"
        seed = seed_step.split("_")[0][len("seed"):]
        rec = _read_json(rec_path)
        exp_path = exp_dir / rec_path.name
        explanation = _read_json(exp_path) if exp_path.exists() else None
        step_obj = {
            "step": rec["step"],
            "agent_pos": rec["agent_pos"],
            "agent_dir": rec["agent_dir"],
            "obstacle_positions": rec["obstacle_positions"],
            "chosen_action": rec["chosen_action"],
            "per_action_stats": rec["per_action_stats"],
            "agent_metadata": rec.get("agent_metadata", {}),
            "explanation": explanation,
        }
        trajectories.setdefault(seed, []).append(step_obj)
    # Each trajectory list is already in step order because of glob sort
    # (seed10000_step000.json < step001.json ...). Verify and sort just in case.
    for seed, lst in trajectories.items():
        lst.sort(key=lambda s: s["step"])
    return {"label": AGENTS[agent_id], "trajectories": trajectories}


def _collect_task_summary() -> dict:
    out: dict[str, dict] = {}
    candidates = {
        "random": ROOT / "results/random/eval_summary.json",
        "ppo_tuned": ROOT / "results/ppo/tuned/seed0/eval_summary.json",
        "mcts_baseline": ROOT / "results/mcts/baseline/seed0/eval_summary.json",
    }
    for k, p in candidates.items():
        if p.exists():
            out[k] = _read_json(p)
    return out


def _collect_metrics_summary() -> dict | None:
    p = ROOT / "results/metrics/summary.json"
    return _read_json(p) if p.exists() else None


def main() -> None:
    SITE.mkdir(parents=True, exist_ok=True)
    bundle = {
        "task_summary": _collect_task_summary(),
        "metrics_summary": _collect_metrics_summary(),
        "agents": {aid: _collect_agent(aid) for aid in AGENTS},
        "schema_version": 1,
    }
    out = SITE / "data.json"
    with open(out, "w") as f:
        json.dump(bundle, f, separators=(",", ":"))  # minified
    n_total = sum(
        sum(len(t) for t in agent_data["trajectories"].values())
        for agent_data in bundle["agents"].values()
    )
    size_kb = out.stat().st_size / 1024
    print(f"Wrote {out} ({size_kb:.1f} KB, {n_total} step records across "
          f"{len(bundle['agents'])} agents)")


if __name__ == "__main__":
    main()
