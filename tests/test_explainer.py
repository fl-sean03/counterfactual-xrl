"""Tests for the LLM explainer pipeline (mock-only, no API spend)."""

from __future__ import annotations

import pytest

pytest.importorskip("gymnasium")

from xrl.analysis.records import ActionStats, DecisionRecord  # noqa: E402
from xrl.explainer.client import MockClient  # noqa: E402
from xrl.explainer.pipeline import explain  # noqa: E402
from xrl.explainer.prompts import (  # noqa: E402
    build_system_prompt,
    shared_nonevidence_portion,
)


def _make_record(source: str) -> DecisionRecord:
    stats = [
        ActionStats(
            action=a,
            mean_return=0.8 if a == 2 else 0.1,
            std_return=0.2,
            success_rate=0.75 if a == 2 else 0.1,
            collision_rate=0.05 if a == 2 else 0.3,
            mean_steps_to_end=25.0,
            success_ci=(0.7, 0.8) if a == 2 else (0.05, 0.15),
            collision_ci=(0.02, 0.08) if a == 2 else (0.25, 0.35),
            n_rollouts=100,
        )
        for a in (0, 1, 2)
    ]
    return DecisionRecord(
        source=source,
        agent_id="test_agent",
        state_id="test:0",
        step=0,
        agent_pos=(1, 1),
        agent_dir=0,
        obstacle_positions=[(2, 2)],
        chosen_action=2,
        per_action_stats=stats,
    )


def test_prompt_parity_between_sources() -> None:
    """The non-evidence parts of the system prompt must be byte-identical."""
    policy_sys = build_system_prompt("policy_rollout")
    mcts_sys = build_system_prompt("mcts_tree")
    assert shared_nonevidence_portion(policy_sys) == shared_nonevidence_portion(mcts_sys)
    # Legacy alias still produces the same shared portion.
    legacy_sys = build_system_prompt("dqn_rollout")
    assert shared_nonevidence_portion(legacy_sys) == shared_nonevidence_portion(mcts_sys)


def test_explain_end_to_end_with_mock() -> None:
    rec = _make_record("policy_rollout")
    client = MockClient()
    exp = explain(rec, client)
    assert exp.chosen_action == 2
    assert "2" in exp.rationale
    assert exp.confidence == pytest.approx(0.75)
    assert any(c["metric"] == "success_rate" for c in exp.claims)


def test_cache_hit_avoids_second_call(tmp_path) -> None:
    rec = _make_record("mcts_tree")
    client = MockClient()
    exp1 = explain(rec, client, cache_dir=tmp_path)
    calls_after_first = client.calls
    exp2 = explain(rec, client, cache_dir=tmp_path)
    assert client.calls == calls_after_first, "Second call should hit cache"
    assert exp1.rationale == exp2.rationale
