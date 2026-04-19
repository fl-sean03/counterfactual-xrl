"""Prompt templates for the explainer.

Design principle: the non-evidence parts of every prompt are BYTE-
IDENTICAL across the DQN and MCTS variants. Only the EVIDENCE section
differs. This is enforced by a test and is load-bearing for the fairness
of the DQN-vs-MCTS explanation-quality comparison.
"""

from __future__ import annotations

# Prompt format version. Bump when editing; never edit in place after a
# real run has been logged against the old version.
PROMPT_VERSION = 1


SHARED_SYSTEM_HEADER = """\
You are analyzing an RL agent's decision in the MiniGrid-Dynamic-Obstacles-8x8
environment, an 8x8 gridworld where an agent must reach a goal while
avoiding 4 moving obstacles.

Action legend:
  0 = turn_left, 1 = turn_right, 2 = move_forward
Direction legend:
  0 = east (+x), 1 = south (+y), 2 = west (-x), 3 = north (-y)
Reward structure:
  +(1 - 0.9 * step_count/256) on reaching the goal at (6,6),
  -1 on collision, 0 otherwise.

You will be given structured JSON evidence about one decision. Your job:
explain why the chosen action is preferable to the alternatives, using
ONLY values present in the evidence. Do not invent numbers.

Output a single JSON object with exactly these keys:
  rationale (string): 2-3 sentence explanation of why the chosen action
    is best, citing specific numbers from the evidence.
  counterfactual (string): 1-2 sentences describing what would happen
    if the agent had taken the best alternative action, grounded in the
    alternative's evidence.
  confidence (number, [0,1]): how confident you are in the explanation
    based on evidence strength (sample size, CI width, contrast between
    actions).
  claims (array of objects): every verifiable numerical claim in your
    explanation. Each object: {"text": <string>, "type": "rate" |
    "value" | "comparison", "action": <int>, "metric": <string>,
    "value": <number>}.

Return ONLY the JSON object, no preamble.
"""


DQN_EVIDENCE_HEADER = """\
EVIDENCE SOURCE: counterfactual Monte Carlo rollouts.
For each legal action, we forced the agent to take that action once and
then followed the trained DQN policy to termination. Rollouts were run
N times per action and aggregated into means, rates, and bootstrap CIs.
"""

MCTS_EVIDENCE_HEADER = """\
EVIDENCE SOURCE: MCTS search tree (UCT).
At the decision state, MCTS ran B simulations. Each simulation selected
actions with UCB1, expanded leaves, and ran a rollout policy to a
terminal or a depth cap. The per-action numbers are the visit counts,
mean values, and empirical success/collision rates accumulated at the
root's children during search.
"""


def build_system_prompt(source: str) -> str:
    """Build the full system prompt for a given evidence source."""
    if source == "dqn_rollout":
        evidence_header = DQN_EVIDENCE_HEADER
    elif source == "mcts_tree":
        evidence_header = MCTS_EVIDENCE_HEADER
    else:
        raise ValueError(f"Unknown source {source!r}")
    return SHARED_SYSTEM_HEADER + "\n" + evidence_header


def build_user_prompt(record_dict: dict) -> str:
    """The user prompt is the DecisionRecord JSON, verbatim."""
    import json as _json

    return _json.dumps(record_dict, indent=2)


def shared_nonevidence_portion(system_prompt: str) -> str:
    """Return the part of the system prompt that must match across sources.

    Used by the parity test, the evidence header is sliced off and the
    rest must match byte-for-byte between DQN and MCTS variants.
    """
    return system_prompt.split("EVIDENCE SOURCE:", 1)[0]
