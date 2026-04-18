# Engineering Log — Counterfactual XRL

Append-only, chronological log of decisions, experiments, results, and
rationale. The final report's Results and Discussion sections will be
distilled from this file. Every entry should have: date, what, why.

Format:

```
## YYYY-MM-DD HH:MM — <short title>
**What:** <what was done>
**Why:** <rationale or constraint>
**Result:** <numerical / qualitative outcome>
**Next:** <implied next action>
```

---

## 2026-04-17 — Project kickoff and scaffolding

**What:** Proposal approved by Prof. Sunberg. Wrote EXECUTION_PLAN.md.
Scaffolded repo (env, CI, tests, directory layout). Froze MDP formulation
(S, A, T, R) in `docs/problem_formulation.md`.
**Why:** No definitive spec existed; needed a single source of truth so
later phases don't drift.
**Result:** 9/9 initial tests green. Env facts verified (Discrete(3),
4 obstacles, 8×8 grid, max_steps=256, reward formula verified).
**Next:** Build DQN and MCTS.

## 2026-04-17 — DQN image baseline trained

**What:** SB3 DQN with MlpPolicy on flattened 7×7×3 image obs, 300k
timesteps, SB3 defaults + `net_arch=[256,256]`, `exp_fraction=0.3`.
**Why:** Baseline deep-RL agent for the comparison.
**Result:** **1.0% success, 0% collision, 254 avg steps.** Agent
converged to a "stall" policy — never moves forward, avoids collision
by timing out.
**Diagnosis:** Classic DQN failure on MiniGrid-Dynamic-Obstacles:
collision penalty dominates early exploration, agent learns risk
aversion before it ever sees a successful trajectory.
**Next:** Try symbolic obs (full state); if that fails, add reward
shaping or try PPO.

## 2026-04-17 — DQN symbolic attempt

**What:** 200k-step DQN on flat symbolic full-state obs (14-d vector:
agent xy/dir one-hot + 4 obstacle xys).
**Why:** Hypothesis was that richer observation might break the stall.
**Result:** **0% success, 20.7% collision, 232 avg steps.** Worse than
image baseline — now it sometimes collides but never succeeds.
**Diagnosis:** Same fundamental issue; richer obs didn't help.
**Next:** Try DQN with distance-based reward shaping (+ε for getting
closer to goal) to provide a dense gradient.

## 2026-04-17 — MCTS from scratch works

**What:** UCT with UCB1 (c=√2), full-width expansion at root (|A|=3),
greedy-toward-goal rollout policy, 500 sims/decision, 60-step rollout cap.
**Why:** From-scratch planner per assignment requirement and the core
comparison.
**Result:** On 5-episode smoke test: **5/5 success, mean return 0.918,
avg 23 steps.** Substantially outperforms a random agent (0.3%) without
training.
**Next:** Full 100-episode eval; produce decision records for explainer.

## 2026-04-17 — Counterfactual framework complete

**What:** `DecisionRecord` dataclass + JSON schema unifying DQN rollout
evidence and MCTS tree evidence. `counterfactual_rollouts()` forces each
legal action at a decision state then follows the policy; returns
per-action stats with 95% bootstrap CIs. `mcts_root_to_action_stats()`
translates tree root into the same schema (Wilson score CIs).
**Why:** Explainer input must be identical in shape across agents so the
only variable is the evidence content.
**Result:** 17/17 tests green, schema round-trips, lint clean.
**Next:** Run on held-out seeds for real records.

## 2026-04-17 — Key scope decisions (autonomous)

Sean delegated to push through. Decisions:

1. **DQN path.** Try distance-based reward shaping once; if still <0.4
   success, accept suboptimal DQN per Sunberg's RQ framing ("is
   suboptimal online planning more explainable than suboptimal deep
   RL"). Document the suboptimality as a threat to validity.
2. **Phase 7 human pilot — CUT.** Requires live participants; note as
   future work in report's Conclusion.
3. **Phase 7 chatbot — KEEP** as a scaffolded stretch; demo-only.
4. **LLM explainer.** If `ANTHROPIC_API_KEY` is present, run real API
   calls with a **$10 hard cost cap** and prompt caching on. If not,
   build a mockable pipeline and produce a smaller artifact set.
   Generator: Claude Sonnet 4.6. Judge (Phase 6): Claude Haiku 4.5 +
   spot-check.
5. **Report prose.** AI policy says prose/math must be authored by Sean
   and Andrew. Deliverable from me: a fully-fleshed IEEE draft **clearly
   labeled "[DRAFT — rewrite in your voice]"** at every section, plus
   every figure/table regenerable from `scripts/`. Sean+Andrew do the
   final pass.
6. **Andrew coordination.** Keep pushing; leave a CONTRIBUTIONS.md stub
   that Andrew can fill in; `AGENTS.md` and this log carry context for
   his review.

**Why these choices:** best expected deliverable per remaining time
budget, honouring syllabus AI policy and the prof's stated question.
