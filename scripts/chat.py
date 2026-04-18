"""Interactive chatbot (Phase 7 stretch).

Steps a user through the trajectory of a saved agent, decision by
decision, letting them ask counterfactual questions. The chatbot has
access to the `DecisionRecord` for the current step and calls Claude
with a multi-turn conversation.

Usage:
    python scripts/chat.py --records-dir results/decision_records/mcts_baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from xrl.analysis.records import load_record
from xrl.explainer.client import make_client

CHAT_SYSTEM = """\
You are an interactive assistant helping a user understand an RL agent's
decisions in the MiniGrid-Dynamic-Obstacles-8x8 environment.

You have access to one DecisionRecord at a time — the structured
evidence (counterfactual rollout stats or MCTS tree stats, depending on
the agent) for the current decision in the trajectory.

Respond naturally to the user's questions. Cite specific numbers from
the DecisionRecord when relevant. If the user asks about a past or
future step that isn't in the current record, say so politely.

Always stay faithful to the evidence: do not invent numbers.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-dir", required=True)
    ap.add_argument("--seed", type=int, default=None, help="only use records for this seed")
    ap.add_argument("--force-mock", action="store_true")
    args = ap.parse_args()

    rec_paths = sorted(Path(args.records_dir).rglob("*.json"))
    if args.seed is not None:
        rec_paths = [p for p in rec_paths if f"seed{args.seed}" in p.name]
    if not rec_paths:
        print("No records found.")
        return

    client = make_client(mock=True if args.force_mock else None)

    print(f"Loaded {len(rec_paths)} decisions from {args.records_dir}.")
    print("Commands: next / prev / step <n> / ask <question> / show / quit")
    idx = 0
    history: list[dict] = []

    while True:
        rec = load_record(rec_paths[idx])
        user_in = input(f"[step {rec.step} / idx {idx}] > ").strip()
        if not user_in:
            continue
        if user_in in ("quit", "exit", "q"):
            break
        if user_in == "next":
            idx = min(idx + 1, len(rec_paths) - 1)
            history = []
            continue
        if user_in == "prev":
            idx = max(idx - 1, 0)
            history = []
            continue
        if user_in.startswith("step "):
            try:
                idx = max(0, min(len(rec_paths) - 1, int(user_in.split(None, 1)[1])))
                history = []
            except ValueError:
                print("usage: step <int>")
            continue
        if user_in == "show":
            print(json.dumps(rec.to_dict(), indent=2))
            continue
        if user_in.startswith("ask "):
            question = user_in[4:]
            evidence_block = json.dumps(rec.to_dict(), indent=2)
            history.append(
                {"role": "user", "content": f"Evidence:\n{evidence_block}\n\n{question}"}
            )
            # Concatenate history into a simple user string for the client.
            user_payload = "\n\n".join(m["content"] for m in history if m["role"] == "user")
            result = client.call(system=CHAT_SYSTEM, user=user_payload, cache_system=True)
            print("\n" + result.text + "\n")
            history.append({"role": "assistant", "content": result.text})
            continue
        print("unrecognized. Commands: next / prev / step <n> / ask <q> / show / quit")


if __name__ == "__main__":
    main()
