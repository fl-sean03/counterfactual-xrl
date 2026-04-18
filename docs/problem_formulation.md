# Problem Formulation — MiniGrid-Dynamic-Obstacles-8x8

Phase 1 deliverable. This is the authoritative MDP specification; Section 3
of the final report is derived from it. Every numeric claim is verified by
`tests/test_env_facts.py`.

## Environment

- **ID:** `MiniGrid-Dynamic-Obstacles-8x8-v0` (Farama Foundation MiniGrid).
- **Grid:** 8×8 tiles with a 1-tile wall border, so the agent navigable
  region is 6×6 (positions `[1..6] × [1..6]`).
- **Agent start:** upper-left interior cell `(1,1)`, facing a random
  cardinal direction at reset.
- **Goal:** bottom-right interior cell `(6,6)`, visually marked.
- **Obstacles:** 4 moving "ball" entities (confirmed in smoke test). At
  each agent step, every obstacle attempts a random walk to one of its
  4-neighbors; if the chosen cell is occupied by another obstacle, wall,
  goal, or the agent, the obstacle stays.

## Formal MDP

We formulate the problem as an MDP on the fully observable state.
MiniGrid's default observation is partial (egocentric 7×7), making the
problem technically a POMDP. For tree search to be tractable, MCTS is
given full-state access through the simulator. We acknowledge this
simplification and discuss its implications in `observation_vs_state.md`.

### State S

A state is the tuple
`s = (agent_pos, agent_dir, {obstacle_pos_k}_{k=1..4}, step_count)`

- `agent_pos ∈ {1,…,6}² = 36 cells`
- `agent_dir ∈ {0 = east, 1 = south, 2 = west, 3 = north}`
- `obstacle_pos_k ∈ {1,…,6}²` (multiset, obstacles are interchangeable in
  state identity for reward purposes but distinguishable by position)
- `step_count ∈ {0, 1, …, max_steps}` — used only for reward discounting
  and truncation; transitions and actions do not condition on it.

The state space is combinatorially large but finite. Ignoring agent
placement on the goal cell and collisions, a crude upper bound is
`36 × 4 × C(36, 4) × 257 ≈ 2.2 × 10⁸` states — prohibitive for exact
value iteration, motivating the learning/planning approach.

### Action space A

`A = {0 = turn_left, 1 = turn_right, 2 = move_forward}`, `|A| = 3`.
MiniGrid's full action enumeration has 7 entries; the Dynamic-Obstacles
variant restricts to these 3 via `spaces.Discrete(forward + 1)`. This is
verified in the smoke test.

### Transition T(s' | s, a)

Transitions factor into a deterministic agent step followed by a joint
stochastic obstacle step:

1. **Agent step (deterministic).**
   - `turn_left / turn_right`: `agent_dir ← (agent_dir ± 1) mod 4`, no
     position change.
   - `move_forward`: if the cell in front is empty (no wall, no obstacle,
     no goal), `agent_pos ← agent_pos + dir_vec(agent_dir)`. If it is the
     goal, agent moves onto the goal and the episode terminates
     (success). If it is an obstacle, the agent moves onto the obstacle
     and the episode terminates (collision, reward −1). Into a wall: no
     movement.
2. **Obstacle step (stochastic, joint).** Processed sequentially over
   obstacles in their internal order: each obstacle samples a uniform
   move from its 4 neighbors; if the chosen cell is blocked (other
   obstacle, wall, goal, agent), it stays. A moving obstacle stepping
   onto the agent ends the episode (collision).

Because the obstacle update happens *after* the agent step, stepping
forward into an obstacle is decided before the obstacles move (the agent
can only collide with an obstacle by stepping into it); the stochastic
component affects the agent only on the *next* decision — the chance
that an obstacle moves into the cell the agent is currently in. This
asymmetry is load-bearing for MCTS: from the agent's perspective at time
`t`, the outcome of action `a_t` depends on the *post-step* obstacle
distribution at time `t+1`.

### Reward R(s, a, s')

- `+ (1 − 0.9 · step_count / max_steps)` on reaching the goal
  (terminal).
- `− 1` on collision with any obstacle (terminal, by either the agent
  stepping onto an obstacle or an obstacle stepping onto the agent).
- `0` otherwise.

`max_steps = 4 × grid_size² = 4 × 64 = 256` per episode (verified in
smoke test; if this changes in a MiniGrid release, update here).

### Discount γ

`γ = 1.0` (undiscounted, finite-horizon). The step-count term inside the
success reward provides the implicit time pressure.

### Initial-state distribution

Agent at `(1,1)`, direction uniform over `{0,1,2,3}`, obstacles placed
uniformly at random in free interior cells at reset.

### Termination

- Success: agent reaches goal.
- Collision: agent and obstacle occupy the same cell.
- Truncation: `step_count ≥ max_steps` (treated as non-terminal for
  value-function purposes but ends the episode in practice).

## Observability Decision

DQN uses the default partial egocentric observation
(`image` 7×7×3 + `direction` scalar). MCTS plans on the full state via a
deep-copied simulator. This is asymmetric and must be justified in the
report. Two mitigations, ranked by cost:

1. **Cheap:** compare DQN (partial obs) vs MCTS (full state) and
   explicitly discuss the observability gap as a confounder in the
   results section.
2. **Expensive:** train a second DQN variant on flat symbolic full-state
   observations (`FlatSymbolicObsWrapper`) and report both DQN variants
   against MCTS. Controls for observability.

The plan calls for option 2 as mitigation in Phase 2 risks.

## Citations Needed (for report §3)

- Farama Foundation MiniGrid (Chevalier-Boisvert et al.): environment spec.
- Kocsis & Szepesvári 2006: UCT formal setup on MDPs.
- Mnih et al. 2015: DQN on the same class of discrete-action MDPs.
