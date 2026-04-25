"""Explanation-quality metrics.

Three metrics, all defined on a (DecisionRecord, Explanation) pair and
aggregated with bootstrap CIs:

1. **Fidelity.** For every numerical claim in an explanation (each entry
   in ``Explanation.claims``), verify that the cited value is within
   tolerance of the corresponding value in the evidence record. Score =
   fraction of claims within tolerance.

2. **Soundness.** Use an LLM judge (a different model from the
   generator) to rate whether each reason in the rationale is supported
   by the evidence. Maps to {2: supported, 1: partial, 0: unsupported}.
   Score = mean / 2 ∈ [0, 1].

3. **Post-hoc inferability.** Hide the chosen action from the
   explanation and ask a second LLM call "which action was chosen?".
   Score = accuracy over the dataset.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from xrl.analysis.records import DecisionRecord
from xrl.explainer.client import AnthropicClient, MockClient
from xrl.explainer.pipeline import Explanation


@dataclass
class MetricScores:
    fidelity: float
    soundness: float
    inferability: bool  # per-decision: did the judge correctly infer the action?


def _find_stat(record: DecisionRecord, action: int) -> dict | None:
    for s in record.per_action_stats:
        if s.action == action:
            return {
                "mean_return": s.mean_return,
                "success_rate": s.success_rate,
                "collision_rate": s.collision_rate,
                "std_return": s.std_return,
                "mean_steps_to_end": s.mean_steps_to_end,
            }
    return None


def fidelity_score(record: DecisionRecord, explanation: Explanation, tol: float = 0.1) -> float:
    """Fraction of numerical claims that match the record within tol."""
    if not explanation.claims:
        return 0.0
    hits = 0
    for claim in explanation.claims:
        action = claim.get("action")
        metric = claim.get("metric")
        value = claim.get("value")
        if action is None or metric is None or value is None:
            continue
        stat = _find_stat(record, int(action))
        if stat is None or metric not in stat or stat[metric] is None:
            continue
        if abs(stat[metric] - float(value)) <= tol:
            hits += 1
    return hits / max(1, len(explanation.claims))


SOUNDNESS_SYSTEM = """\
You are an impartial judge evaluating one sentence of an AI-generated
explanation about an RL agent's action choice.

You will be given:
1. The structured evidence (decision record) the agent had.
2. ONE claim/reason extracted from the explanation.

Rate how well the evidence supports the claim:
  2 = fully supported: the numbers cited match the evidence, and the
      causal/reason framing is consistent with what the evidence shows.
  1 = partially supported: the direction is right but some specific
      numbers are off, or the reason is plausible but not directly
      shown by the evidence.
  0 = unsupported: claim is contradicted by the evidence or invents
      numbers not present.

Reply with ONLY a single integer 0, 1, or 2. No other text.
"""


INFERABILITY_SYSTEM = """\
You will be given (a) a brief description of a state and (b) an
explanation of why the agent chose some action. The explanation has been
scrubbed of any direct mention of the chosen action number.

The legal actions are:
  0 = turn_left, 1 = turn_right, 2 = move_forward

Based on the explanation, infer which action was chosen.
Reply with ONLY a single digit 0, 1, or 2. No other text.
"""


def _scrub_action_mentions(text: str, action: int) -> str:
    """Remove ``action {action}`` patterns so the judge can't cheat."""
    pattern = re.compile(rf"action[\s]*{action}\b", flags=re.IGNORECASE)
    return pattern.sub("[CHOSEN]", text)


def soundness_score(
    record: DecisionRecord,
    explanation: Explanation,
    judge: AnthropicClient | MockClient,
) -> float:
    """Average support score over reasons in the rationale."""
    # Split rationale into sentences (rough) + include counterfactual.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", explanation.rationale) if s.strip()]
    if explanation.counterfactual:
        sentences.append(explanation.counterfactual.strip())
    if not sentences:
        return 0.0
    evidence = json.dumps(record.to_dict(), indent=2)
    totals = 0
    n = 0
    for sent in sentences:
        user = f"EVIDENCE:\n{evidence}\n\nCLAIM:\n{sent}"
        result = judge.call(system=SOUNDNESS_SYSTEM, user=user, cache_system=True)
        m = re.search(r"[0-2]", result.text.strip())
        if m is None:
            continue
        totals += int(m.group(0))
        n += 1
    if n == 0:
        return 0.0
    return totals / (2.0 * n)


def inferability(
    record: DecisionRecord,
    explanation: Explanation,
    judge: AnthropicClient | MockClient,
) -> bool:
    """Does a held-out judge recover the chosen action from the explanation alone?"""
    scrubbed_rationale = _scrub_action_mentions(explanation.rationale, explanation.chosen_action)
    scrubbed_cf = _scrub_action_mentions(explanation.counterfactual, explanation.chosen_action)
    state_desc = (
        f"agent at {record.agent_pos} facing dir {record.agent_dir}, "
        f"obstacles at {record.obstacle_positions}, step {record.step}."
    )
    user = (
        f"STATE:\n{state_desc}\n\n"
        f"EXPLANATION (rationale):\n{scrubbed_rationale}\n\n"
        f"COUNTERFACTUAL:\n{scrubbed_cf}\n"
    )
    result = judge.call(system=INFERABILITY_SYSTEM, user=user, cache_system=True)
    m = re.search(r"[0-2]", result.text.strip())
    if m is None:
        return False
    return int(m.group(0)) == explanation.chosen_action


def score_pair(
    record: DecisionRecord,
    explanation: Explanation,
    judge: AnthropicClient | MockClient,
    fidelity_tol: float = 0.1,
) -> MetricScores:
    return MetricScores(
        fidelity=fidelity_score(record, explanation, tol=fidelity_tol),
        soundness=soundness_score(record, explanation, judge),
        inferability=inferability(record, explanation, judge),
    )


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 1000, seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.percentile(boots, [2.5, 97.5]))


def aggregate(scores: list[MetricScores]) -> dict:
    if not scores:
        return {
            "n": 0,
            "fidelity": {"mean": float("nan"), "ci": (float("nan"), float("nan"))},
            "soundness": {"mean": float("nan"), "ci": (float("nan"), float("nan"))},
            "inferability": {"mean": float("nan"), "ci": (float("nan"), float("nan"))},
        }
    fid = np.array([s.fidelity for s in scores])
    snd = np.array([s.soundness for s in scores])
    inf = np.array([float(s.inferability) for s in scores])
    return {
        "n": len(scores),
        "fidelity": {"mean": float(fid.mean()), "ci": bootstrap_ci(fid)},
        "soundness": {"mean": float(snd.mean()), "ci": bootstrap_ci(snd)},
        "inferability": {"mean": float(inf.mean()), "ci": bootstrap_ci(inf)},
    }
