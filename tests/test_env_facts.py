"""Assert the environment numeric facts we rely on in problem_formulation.md.

If any of these assertions fail after a MiniGrid upgrade, update both this
test and docs/problem_formulation.md atomically.
"""

from __future__ import annotations

import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("minigrid")

import minigrid  # noqa: E402, F401  (registers envs)

ENV_ID = "MiniGrid-Dynamic-Obstacles-8x8-v0"


@pytest.fixture
def env():
    e = gym.make(ENV_ID)
    e.reset(seed=0)
    yield e
    e.close()


def test_action_space_is_discrete_3(env) -> None:
    assert env.action_space.n == 3, (
        f"Expected Discrete(3) = {{left, right, forward}}, got {env.action_space.n}. "
        "MiniGrid version may have changed the Dynamic-Obstacles action restriction."
    )


def test_obstacle_count_is_4(env) -> None:
    n = getattr(env.unwrapped, "n_obstacles", None)
    assert n == 4, f"Expected 4 obstacles, got {n}"


def test_grid_is_8x8(env) -> None:
    assert env.unwrapped.width == 8
    assert env.unwrapped.height == 8


def test_max_steps_is_256(env) -> None:
    # MiniGrid 2.3 sets max_steps on the unwrapped env, not env.spec.
    ms = env.unwrapped.max_steps
    assert ms == 256, f"Expected max_steps=256 (4*8^2), got {ms}"


def test_observation_has_image_and_direction(env) -> None:
    obs, _ = env.reset(seed=0)
    assert isinstance(obs, dict)
    assert "image" in obs
    assert obs["image"].shape == (7, 7, 3), f"Unexpected image shape: {obs['image'].shape}"
    assert "direction" in obs


def test_goal_reward_on_success_bounds(env) -> None:
    """The success reward formula is +(1 - 0.9 * step/max_steps), so the
    maximum possible step reward is in (0.1, 1.0]."""
    # We can't force a successful episode in a unit test without a policy,
    # but we can at least assert the formula shape by inspecting the env's
    # reward function if exposed.
    e = env.unwrapped
    assert hasattr(
        e, "_reward"
    ), "Expected unwrapped env to expose _reward(); MiniGrid internal may have changed."
    # Call at step 0 → reward ≈ 1.0
    r0 = e._reward()
    assert 0.99 < r0 <= 1.0, f"_reward() at step 0 expected ~1.0, got {r0}"


def test_deepcopy_preserves_state():
    """Simulator clone relies on deepcopy preserving obstacle RNG state."""
    import copy

    e = gym.make(ENV_ID)
    e.reset(seed=123)
    # advance a few steps to populate RNG state
    for _ in range(5):
        e.step(e.action_space.sample())
    clone = copy.deepcopy(e)

    # Take the same action from both; they should produce identical outcomes.
    act = 0  # deterministic action (turn left), no agent-movement noise
    out_a = e.step(act)
    out_b = clone.step(act)
    # Compare images (as proxy for full state)
    import numpy as np

    obs_a, obs_b = out_a[0], out_b[0]
    assert np.array_equal(obs_a["image"], obs_b["image"]), (
        "deepcopy did not preserve stochastic obstacle state; "
        "simulator clone will not be deterministic."
    )
    e.close()
    clone.close()
