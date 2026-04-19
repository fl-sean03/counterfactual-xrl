# EXECUTION PLAN, Counterfactual XRL Final Project

**Last updated:** 2026-04-17
**Owners:** Sean Florez, Andrew Wernersbach
**Deliverable:** 4–8 page IEEE report + public GitHub repo
**Weight:** 30% of ASEN 5264 grade

This document is the authoritative step-by-step guide for taking the project
from its current proposal-only state to a submitted final report. Every phase
has explicit artifacts and exit gates. Do not start a phase until its
prerequisites are green; do not call a phase done until its gates are green.

---

## 1. Research Questions (what the report must answer)

**RQ1.** At matched task performance, do MCTS search-tree explanations exhibit
higher *fidelity* to the agent's actual decision process than DQN post-hoc
counterfactual-rollout explanations?

**RQ2.** At matched task performance, are MCTS-derived explanations more
*sound* (their stated causal claims hold up under verification) than DQN
counterfactual explanations?

**RQ3 (stretch).** Do human participants rate MCTS-tree explanations as more
*plausible / useful / clear* than DQN-rollout explanations for the same
decision?

Every experiment we run must move one of these three needles.

---

## 2. Report Rubric (what graders score against)

Pulled verbatim from `CU-DMU-Materials/project/report_assignment.tex` and
syllabus. Treat this as the acceptance test for the whole project.

- **Format:** 4–8 pages, IEEE template, single-spaced.
- **Outline:** Introduction → Background & Related Work → Problem Formulation
  (S, A, T, R clearly defined) → Solution Approach → Results (with uncertainty
  quantification) → Conclusion → Contributions & Release.
- **≥5 citations** of related published work with brief relation-to-project
  blurbs.
- **Per-member contributions** paragraph.
- **Release statement** (verbatim one of):
  - *"The authors grant permission for this report to be posted publicly."*
  - *"The authors do NOT grant permission for this report to be posted publicly."*
- **Clearly label from-scratch vs off-the-shelf** algorithms.
- **AI policy:** prose/math written by us; code written or closely directed by
  us. Any AI-assisted code must be understood and reviewed.

---

## 3. Prerequisites & Assumptions

- Python 3.11+ on WSL2 Ubuntu with CUDA 12.8 available (RTX 5080 Laptop GPU,
  16 GB). DQN training will use GPU; MCTS is CPU.
- Conda env `xrl` pinned via `environment.yml`.
- Anthropic API key in `~/.config/anthropic/key` (ignored by git).
- Andrew has equal commit access. Coordination via GitHub PRs or direct
  pushes to `main`, decide in Phase 0.
- Commits on this repo are attributed to Sean only (no AI co-authors).

---

## 4. Timeline

Absolute dates will be filled once the Canvas calendar is checked. Use
relative anchors below; adjust once the final report due date is confirmed.

| Anchor | Phase |
|---|---|
| T-10 weeks | Phase 0–1 complete |
| T-8 weeks | Phase 2 (DQN) complete |
| T-6 weeks | Phase 3 (MCTS) complete |
| T-5 weeks | Phase 4 (counterfactual framework) complete |
| T-4 weeks | Phase 5 (LLM explainer) complete |
| T-3 weeks | Phase 6 (evaluation) complete; report draft started |
| T-2 weeks | Phase 7 (stretch) complete or cut |
| T-1 week  | Report v1 done; internal review |
| T-0       | Submit to Gradescope |

---

## 5. Phased Plan

### Phase 0, Environment & Repo Scaffold

**Goal.** A clean, reproducible dev environment and repo layout so all later
work is friction-free.

**Work items.**

- [ ] **Kickoff sync with Andrew**, ratify the Phase 0/1 ownership split
  (Section 6.5), agree on PR-vs-direct-push workflow, exchange GitHub
  handles and Anthropic API arrangement.
- [ ] **Confirm final report due date** via Canvas; update Section 4
  timeline with absolute dates. Treat this as a blocker, no point
  running a timeline against unknown deadlines.
- [ ] Create `environment.yml` pinning: `python=3.11`, `gymnasium`,
  `minigrid`, `stable-baselines3`, `torch` (CUDA build), `numpy`,
  `matplotlib`, `pandas`, `pyyaml`, `anthropic`, `pytest`, `pytest-cov`,
  `ruff`, `black`, `mypy`, `tqdm`, `seaborn`.
- [ ] Create directory layout:
  ```
  src/xrl/
    agents/         # dqn.py, mcts.py, base.py
    envs/           # wrappers.py, simulator.py
    analysis/       # counterfactual.py, tree_stats.py
    explainer/      # prompts.py, client.py, pipeline.py
    eval/           # metrics.py, runner.py
    utils/          # seeding.py, io.py, config.py
  tests/
  configs/          # YAML configs per experiment
  scripts/          # train_dqn.py, run_mcts.py, eval.py, explain.py
  results/          # gitignored, per-run artifacts
  report/           # IEEE tex sources
  docs/             # design notes
  ```
- [ ] `pyproject.toml` with `ruff`, `black`, `mypy`, `pytest` config.
- [ ] `Makefile` with targets: `install`, `lint`, `test`, `train`, `eval`,
  `explain`, `report`.
- [ ] Pre-commit hook: `ruff`, `black`, `pytest` on staged files.
- [ ] Seeding utility (`utils/seeding.py`): sets `random`, `numpy`, `torch`,
  `env.reset(seed=...)` atomically.
- [ ] Smoke-test script `scripts/smoke_env.py` that resets
  `MiniGrid-Dynamic-Obstacles-8x8-v0`, rolls out 100 random steps, prints
  observation shape, action space, reward range. **Explicitly assert**:
  action space size (expect 3, older MiniGrid versions expose 7), number
  of obstacles (expect 4), max episode steps, reward-on-success formula
 , these are all facts later phases depend on.
- [ ] README update with setup instructions.
- [ ] CI workflow (`.github/workflows/ci.yml`) running lint + tests on push.

**Artifacts.** `environment.yml`, `pyproject.toml`, `Makefile`, empty
package skeleton with `__init__.py`s, CI config, smoke-test script.

**Exit gates.**

1. `make install` succeeds from a clean conda env.
2. `make test` runs (may be empty) and passes.
3. `make lint` passes with zero errors.
4. `python scripts/smoke_env.py` prints env summary and does not error.
5. `import torch; torch.cuda.is_available()` returns `True`.
6. CI workflow green on main.

**Risks.** Gymnasium vs old Gym API mismatch with SB3 version. Pin both
explicitly and verify `DQN("MlpPolicy", env).learn(total_timesteps=100)`
runs during Phase 0 smoke test.

---

### Phase 1, Problem Formulation Freeze

**Goal.** A precise, written-down MDP specification that Section 3 of the
report will be copied from. Resolves any ambiguity about observability,
action-space restriction, and reward structure before we write agent code.

**Work items.**

- [ ] Write `docs/problem_formulation.md` defining:
  - **State S:** full ground-truth state = (agent_pos ∈ [1..6]², agent_dir
    ∈ {0,1,2,3}, obstacle_positions ∈ [1..6]^(2K), goal_pos = (6,6), step_count).
    Document *K* (number of obstacles), confirm from env source.
  - **Observation O:** 7×7×3 partial egocentric image plus direction scalar.
    Note whether we treat this as MDP (full state access, justified by
    tree search operating on the simulator) or POMDP (obs-only). **Decision:
    MDP for tractability + alignment with proposal; note the simplification
    and discuss in report.**
  - **Action A:** Discrete(3) = {left, right, forward}. Confirm
    `MiniGrid`'s 7-action space is safely restricted (the other actions are
    no-ops or invalid here).
  - **Transition T:** deterministic agent motion; stochastic obstacle
    motion (random walk into free 4-neighbors, else stay). Write the
    explicit factorization.
  - **Reward R:** +(1 − 0.9 · step/max_steps) on goal, −1 on obstacle
    collision, 0 otherwise. Episode terminates on either. Confirm
    `max_steps` value from env.
  - **Discount γ:** 1.0 (episodic), confirm or justify.
- [ ] Write `docs/observation_vs_state.md`, explicit note that DQN sees
  observation, MCTS uses full state (access via simulator copy); this is a
  design decision that affects fairness of comparison and must be addressed
  in the report.
- [ ] Write `tests/test_env_facts.py`, asserts each numeric fact above
  against the live env. Re-run whenever MiniGrid is upgraded.

**Artifacts.** `docs/problem_formulation.md`, `docs/observation_vs_state.md`,
`tests/test_env_facts.py`.

**Exit gates.**

1. `pytest tests/test_env_facts.py` green, every formulation claim verified.
2. Review pass with Andrew: both team members sign off on formulation.
3. No TODOs remain in `problem_formulation.md`.

**Risks.** Observation/state mismatch making the "is MCTS more explainable"
comparison unfair. Mitigation: run a second DQN variant with full-state obs
(flattened symbolic obs) so the observation-space variable is controlled;
report both.

---

### Phase 2, DQN Agent

**Goal.** A trained DQN baseline with a rigorous eval harness, ready to
feed into the counterfactual framework.

**Work items.**

- [ ] `src/xrl/envs/wrappers.py`: `FlatSymbolicObsWrapper` (full state
  flattened) + `ImgObsWrapper` (use existing MiniGrid wrapper). Both
  available via config.
- [ ] `src/xrl/agents/dqn.py`: thin SB3 wrapper exposing `predict(obs)`,
  `q_values(obs)` (implemented by calling `model.policy.q_net(obs_tensor)`
  and returning the per-action Q-vector, SB3 has no public Q-value
  accessor, so we add one), `save`, `load`. Off-the-shelf = SB3; **our
  wrapper is thin and must be labeled off-the-shelf in the report.**
- [ ] `configs/dqn_baseline.yaml`: hyperparameters (learning rate, buffer
  size, exploration schedule, total_timesteps, network arch). Start from
  SB3 DQN defaults; tune only if needed to clear the eval threshold.
- [ ] `scripts/train_dqn.py`: reads config, trains, saves checkpoints +
  TensorBoard logs to `results/dqn/<run_id>/`.
- [ ] `src/xrl/eval/runner.py`: runs N episodes with a given agent, logs
  per-episode return, success (reached goal), collision, steps. Returns a
  pandas DataFrame.
- [ ] `scripts/eval.py`: evaluates a saved policy over ≥300 episodes across
  ≥3 seeds; emits `results/dqn/<run_id>/eval.csv` and a learning-curve
  plot.
- [ ] **Matched-performance gate:** tune training until mean success rate
  ≥ 0.70 over the 300-episode eval (literature suggests 0.6–0.8 for
  vanilla DQN on this env; raise the bar if we clear it comfortably).
- [ ] **Random-agent baseline.** Run the same eval harness on a uniform-
  random policy and report its success rate. Protects against "the env is
  trivial" failure mode and gives a lower-bound reference in the report.
- [ ] Save ≥20 **interesting** held-out trajectory seeds for "decision
  point" use in Phase 4. "Interesting" = trajectory contains at least one
  step where the agent is within Manhattan distance 2 of a moving
  obstacle. Uninteresting trajectories don't exercise the explainer.

**Artifacts.** Trained DQN checkpoint(s), eval CSV, learning curves,
held-out seeds list.

**Exit gates.**

1. Training reproduces from config + seed (≤1% variance in final success
   rate across re-runs).
2. Success rate ≥ 0.70 over 300 episodes, averaged over ≥3 training seeds.
3. 95% bootstrap CI of success rate reported and ≤ ±0.05 wide.
4. Per-step Q-values extractable (needed for explainer).
5. Unit test: saved-then-loaded policy matches original on 100 obs.
6. Random-agent baseline reported; trained DQN beats it by ≥0.30 absolute
   success rate.

**Risks.** SB3 DQN underperforming on dynamic-obstacles (known to be
harder than static grids). Mitigations, in order: tune exploration
schedule; switch to `DoubleDQN` or `DuelingDQN` via SB3-contrib; as last
resort, add reward shaping (clearly flagged in report).

---

### Phase 3, MCTS Agent (from scratch)

**Goal.** A from-scratch MCTS planner with a per-decision tree that we can
dump and reason about, performance-matched with DQN.

**Work items.**

- [ ] `src/xrl/envs/simulator.py`: `Simulator` class wrapping a
  deep-copyable env. API: `clone()`, `step(action) -> (next_state, reward,
  done)`, `legal_actions()`, `seed(rng_state)`. Verify `deepcopy(env)`
  preserves obstacle RNG state (`np_random` Generator); write
  `test_simulator_determinism.py`. **Gotcha:** call `env.close()` on any
  render-mode env before deepcopy (pygame/window handles aren't
  picklable); construct sim envs with `render_mode=None`.
- [ ] `src/xrl/agents/mcts.py`: UCT with UCB1 selection, random rollout
  policy, reward backup. Configurable budget (sims per decision). **From
  scratch, clearly labeled in report.**
- [ ] Handle stochastic transitions: use standard sampled-transition UCT
  (each expand/simulate call samples obstacle transition fresh; visit
  counts accumulate enough for convergence in expectation). Document this
  choice in `docs/mcts_design.md`.
- [ ] Tree dump: at each real decision, serialize root + children to JSON
 , action, visit count, mean value, UCB score, per-child collision/success
  counts from rollouts. Schema pinned in `docs/tree_schema.md`.
- [ ] `configs/mcts_baseline.yaml`: sims per decision (start 500), UCB1 `c`
  (start √2), max rollout depth (start 50), rollout policy (random /
  greedy-toward-goal).
- [ ] `scripts/run_mcts.py`: uses the same `eval/runner.py` harness as DQN.
- [ ] **Matched-performance gate:** tune sims/`c`/rollout policy until mean
  success rate is within ±0.05 of DQN. If MCTS is too weak, try a
  goal-heuristic rollout; if too slow, cache rollouts.

**Artifacts.** MCTS implementation, tree-dump JSONs for every decision in
the held-out trajectories, eval CSV, design doc.

**Exit gates.**

1. Unit tests: UCB1 formula, backup correctness, visit-count monotonicity,
   tree integrity (children sum matches parent visits − 1).
2. Simulator determinism test: `clone + step(seq) == step(seq)` on a fixed
   seed sequence.
3. Success rate within ±0.05 of DQN over 300 episodes × ≥3 seeds.
4. Tree dump loads back into memory, stats validate against schema.
5. Tree-stats sanity: empirical visit-count ratios ≈ child-value ranking
   within sampling noise on 20 random decisions.

**Risks.**

- **Stochastic-transition variance** inflates sims required. Mitigation:
  start with 500 sims, measure variance across repeat root searches, bump
  budget until top-action agreement ≥ 0.9.
- **Tree-dump size** bloats for deep trees. Mitigation: depth cap +
  top-K children per node in the dump; preserve full tree only for
  highlighted decisions.
- **Deepcopy performance.** If `deepcopy(env)` dominates runtime, implement
  a lightweight state record/restore using raw grid + RNG state.

---

### Phase 4, Counterfactual Rollout Framework

**Goal.** A policy-agnostic module that, given a decision point, produces
structured per-action statistics suitable for feeding to an LLM. Shared
between DQN and MCTS paths so the only variable in explanation quality is
the data source.

**Work items.**

- [ ] `src/xrl/analysis/counterfactual.py`:
  - `counterfactual_rollouts(state, policy, n_per_action) -> Dict[action, Stats]`
  - For each legal action, force that action at step 0, then follow
    `policy` for the rest of the episode. Run `n_per_action` rollouts
    (start 100).
  - `Stats` fields: mean return, std, success rate, collision rate, mean
    steps-to-terminal, most-common-terminal-reason, 95% bootstrap CIs
    (≥1000 resamples, percentile method).
- [ ] `src/xrl/analysis/tree_stats.py`: extract equivalent stats from an
  MCTS tree (visit-count-weighted values, empirical success/collision rates
  from rollouts accumulated during search).
- [ ] Unified `DecisionRecord` dataclass with fields: `state_id`,
  `chosen_action`, `per_action_stats`, `source` ∈ {`dqn_rollout`,
  `mcts_tree`, `mcts_tree+rollout`}, `agent_metadata` (Q-values or UCB
  scores).
- [ ] `scripts/build_decision_records.py`: for each agent × each held-out
  trajectory (≥20), emit a `DecisionRecord` per step.

**Artifacts.** `decision_records/dqn/*.json`, `decision_records/mcts/*.json`,
schema doc.

**Exit gates.**

1. Schema validation via `jsonschema` on all emitted records.
2. MCTS rollout stats and tree stats agree within sampling noise on the
   same decisions (spot check ≥20 decisions).
3. Counterfactual rollouts are reproducible from seed.
4. Unit tests for `Stats` aggregation (including bootstrap CI computation).

**Risks.** Counterfactual rollouts for DQN are slow if the policy isn't
vectorized, benchmark early, optimize with SB3's `vec_env` if needed.

---

### Phase 5, LLM Explanation Pipeline

**Goal.** Convert a `DecisionRecord` into a natural-language explanation
via Claude, using prompt templates that are **structurally identical**
across agents so the only input that varies is the evidence source.

**Work items.**

- [ ] `src/xrl/explainer/client.py`: thin Anthropic SDK wrapper with
  prompt caching on the system prompt (use `cache_control` block on the
  system message; note the minimum-token threshold, keep the cached
  system prompt ≥1024 tokens or caching is a no-op), retry with backoff,
  request logging (prompt, response, tokens, cost) to
  `results/explanations/<run_id>/`.
- [ ] `src/xrl/explainer/prompts.py`: three prompt templates:
  - `EXPLAIN_DQN`, system prompt + (decision record with rollout stats)
    → "Why did the agent choose X?"
  - `EXPLAIN_MCTS`, same structure, tree stats replace rollout stats.
  - `EXPLAIN_HYBRID`, MCTS with both tree and auxiliary rollouts, for
    stretch analysis.
  - All three share identical instruction section, schema description,
    and output-format request (JSON with `rationale`, `counterfactual`,
    `confidence`, `claims` list).
- [ ] `src/xrl/explainer/pipeline.py`: given a `DecisionRecord`, selects
  template, calls Claude, parses JSON, returns an `Explanation` dataclass.
- [ ] `scripts/explain.py`: generates explanations for all decisions in
  `decision_records/`.
- [ ] Prompt versioning: every prompt change is a new `prompts_vN.py`; old
  results are never overwritten.
- [ ] Cost guardrail: hard cap on total API spend per run
  (`configs/explainer.yaml::max_usd`).

**Artifacts.** Explanations JSON per decision, prompt-version audit log,
cost report.

**Exit gates.**

1. ≥20 decisions × 2 agents = ≥40 explanations generated end-to-end
   without errors.
2. Output JSON validates against `Explanation` schema 100% of the time
   (prompt iterated until schema compliance is reliable).
3. Prompts byte-identical in the non-evidence sections between `EXPLAIN_DQN`
   and `EXPLAIN_MCTS` (diff asserted in a test).
4. Cost stays under budget cap per run.
5. Caching working: re-running the same decision hits cache and skips API.

**Risks.** LLM hallucinating statistics not present in the record.
Mitigation: prompt explicitly says "cite only values from the provided
JSON"; Phase 6's fidelity metric catches violations.

---

### Phase 6, Evaluation Metrics

**Goal.** Quantitative answers to RQ1 and RQ2 with confidence intervals.

**Work items.**

- [ ] **Verify Belouadah et al. 2025 citation** (exact title/venue/DOI).
  If the paper does not exist or is not applicable, replace with an
  alternative XRL-evaluation source (Milani 2024 survey section on
  metrics; Hoffman et al. 2018 "Metrics for Explainable AI"). Document
  the chosen source in `docs/metrics_spec.md`.
- [ ] Extract fidelity/soundness/plausibility metric definitions from
  the chosen source into `docs/metrics_spec.md`. If a definition is
  under-specified, pin our operationalization and note it.
- [ ] `src/xrl/eval/metrics.py`:
  - **Fidelity.** Parse claims from the explanation (the `claims` JSON
    field enforced in Phase 5). For each claim of the form "action X leads
    to outcome Y with probability p", verify against held-out rollouts
    from the same state. Score = fraction of claims within tolerance
    (± ε, start ε = 0.1).
  - **Soundness.** For each "reason" in the explanation (e.g., "because
    forward risks collision with the obstacle moving toward us"), check
    whether the reason is supported by the evidence in the decision
    record: does the cited action actually have the claimed outcome
    distribution, and is the referenced spatial fact (obstacle position,
    distance, trajectory) actually present in the state? Use an LLM
    judge (different model; see below) scoring each reason on a
    three-point scale {supported, partially supported, unsupported},
    with a human spot-check on ≥30 reasons.
  - **Post-hoc inferability.** Given the explanation (without revealing
    which action was chosen), can a second LLM call predict the chosen
    action? Accuracy across decisions. (This measures "does the
    explanation contain enough signal to recover the decision", a
    complement to fidelity, not a substitute.)
- [ ] Bootstrap CIs over decisions for all metrics.
- [ ] `scripts/evaluate.py`: produces a results table + plots: metric ×
  agent with CIs.
- [ ] **Guard against LLM-judge bias:** judge calls must use a different
  model from the generator. Concrete recommendation: generator =
  `claude-opus-4-7`, judge = `claude-haiku-4-5` (different size) **and**
  a GPT-class model (different family) as a cross-check on a subsample.
  Report human-judge agreement on the same subsample.

**Artifacts.** `results/metrics/<run>/table.csv`, comparison plots,
spot-check annotations.

**Exit gates.**

1. All three metrics computed for both agents with 95% bootstrap CIs
   (≥1000 resamples).
2. Human spot-check inter-rater agreement (Cohen's κ) computed and
   reported. κ < 0.6 is not a failure, it is a finding that gets a
   threat-to-validity paragraph.
3. Reproducible: `scripts/evaluate.py` regenerates the results table
   byte-identically given the same explanations + seed.
4. Differences between agents are either statistically significant or
   explicitly reported as null.

**Risks.** Metrics may all come out null, the hypothesis might be wrong.
That is a fine outcome if reported honestly; the report should present the
null result with a discussion of why.

---

### Phase 7, Stretch Goals

Only attempt if Phases 0–6 are green with ≥2 weeks to deadline.

#### 7a. Chatbot

- [ ] `scripts/chat.py`: CLI (or Streamlit) that lets a user walk a saved
  trajectory frame-by-frame and ask questions. Backed by the Phase 5
  pipeline plus a conversational wrapper that maintains the `DecisionRecord`
  context across turns.
- [ ] Support 3+ follow-up turns without losing context.

**Exit gate.** Demo video (≤2 min) of a sample conversation.

#### 7b. Human Pilot Study

- [ ] Recruit ≥5 participants unfamiliar with MiniGrid.
- [ ] Show each participant the same 10 decisions, each with DQN and MCTS
  explanations (randomized/blinded order). Collect Likert ratings
  (clarity, plausibility, helpfulness) + free-form notes.
- [ ] Analyze with paired Wilcoxon signed-rank per rating dimension.
- [ ] Note this is informal (no IRB), report explicitly frames it as a
  pilot.

**Exit gates.**

1. ≥5 participants × ≥10 decisions each completed.
2. Results table with significance tests.
3. Consent note shown to participants and retained.

**Risks.** Recruitment takes time. Schedule pilot before locking the
report draft so results can be included.

---

### Phase 8, Report Writing

**Goal.** A 4–8 page IEEE paper that hits every rubric item.

**Work items.**

- [ ] Download IEEE conference template into `report/`.
- [ ] Draft in this order (write alongside experiments, never wait for
  all results to start):
  1. Problem Formulation (from Phase 1 doc)
  2. Background & Related Work (5+ citations: Milani 2024, Amitai 2024,
     Baier & Kaisers 2021, Belouadah 2025, Gajcin & Dusparic 2024; add
     MiniGrid/SB3/UCT as context cites)
  3. Solution Approach (DQN, MCTS, counterfactual framework, LLM pipeline,
     metrics)
  4. Results (tables + plots from Phase 6, uncertainty quantified)
  5. Introduction (written last, motivated by actual findings)
  6. Conclusion + Future Work
  7. Contributions paragraph (per member)
  8. Release statement (verbatim)
- [ ] "Implementation disclosure" table: every component marked
  from-scratch / off-the-shelf / modified-off-the-shelf.
- [ ] **AI-assistance disclosure paragraph** in the report appendix or
  contributions section: state which AI tools were used, for what, and
  that all prose/math was written by the authors per syllabus policy.
  Does not count toward the 4–8 page limit if in an appendix.
- [ ] All figures regenerable from `scripts/report_figures.py`.
- [ ] Every numeric claim in the text traces to a specific artifact in
  `results/`, linked in a footnote comment for audit.
- [ ] Proofread, cut overly-long sections aggressively.

**Exit gates.**

1. PDF builds clean (no warnings other than IEEE boilerplate).
2. Page count 4–8 not including references.
3. ≥5 citations present and blurbed in Related Work.
4. Contributions paragraph present; both authors sign off.
5. Release statement present verbatim.
6. From-scratch vs off-the-shelf table complete.
7. Every numeric claim links to an artifact.
8. One full read-through by each author after freeze.

**Risks.** Over-running page count, IEEE is dense, 8 pages goes fast with
plots. Plan figures tight; move supporting tables to appendix.

---

### Phase 9, Final Polish & Submission

- [ ] Reproducibility smoke test: fresh clone → `make install` →
  `make test` → `scripts/eval.py --smoke` all pass.
- [ ] Tag release `v1.0-final-report`.
- [ ] Upload PDF to Gradescope.
- [ ] Email the repo link to the prof if requested.
- [ ] Post-mortem note in `docs/postmortem.md`, what worked, what we'd do
  differently, for our own records.

**Exit gates.**

1. Gradescope submission confirmation saved.
2. Git tag pushed.
3. `git status` clean; `main` green on CI.

---

## 6. Cross-Cutting Concerns

### 6.1 Reproducibility

- Every script takes `--seed` and `--config`; both are logged to
  `results/<run>/meta.json`.
- Every stochastic component uses the seeding utility, never `random.seed`
  directly.
- `scripts/report_figures.py` regenerates every plot from `results/`.
- `environment.yml` pinned exactly (including build hashes where possible).

### 6.2 Testing Strategy

| Layer | Scope | Tool |
|---|---|---|
| Unit | UCB1, backup, Stats aggregation, schema parsing | pytest |
| Integration | Agent runs an episode; simulator determinism; explainer end-to-end | pytest |
| Smoke | Env loads, CUDA available, API key present | pytest + CI |
| Regression | DQN eval results don't drift after refactors | pytest (slow mark) |

Coverage target: ≥80% on `src/xrl/analysis`, `src/xrl/eval`, and
`src/xrl/explainer`. Agents are mostly library-bridging and can be lower.

### 6.3 Fair-Comparison Discipline

Three variables must be controlled for to trust RQ1/RQ2 answers:

1. **Task performance**, DQN and MCTS must be within ±0.05 success rate
   before explanations are compared. Enforced in Phases 2 & 3 gates.
2. **Decision points**, both agents must explain the *same* held-out
   states. Enforced in Phase 4 (`held_out_seeds`).
3. **Prompt structure**, same system prompt, same output format; only
   evidence section differs. Enforced in Phase 5 exit gate 3.

Any violation of these must be documented in the report's "Threats to
Validity" paragraph.

### 6.4 Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| DQN doesn't converge | Medium | High | SB3-contrib variants; reward shaping as last resort |
| MCTS too slow | Medium | High | Lightweight state snapshot; cap sims; cache rollouts |
| Stochastic transitions blow up MCTS variance | Medium | Medium | Bump sim budget; report budget-performance curve |
| LLM hallucinates stats | High | High | Prompt constraints + fidelity metric catches it |
| Metrics all null | Low | Low | Report honestly; null result is a finding |
| Human pilot under-recruits | Medium | Low (stretch) | Cut cleanly; note in report |
| Page-count overrun | Medium | Medium | Write tight; move to appendix |
| API costs balloon | Low | Medium | Cost cap config; prompt caching |
| Andrew and Sean duplicate work | Medium | Medium | PR-based workflow; agree on ownership split in Phase 0 |

### 6.5 Division of Labor (to confirm with Andrew)

Proposed split (to be ratified before Phase 2 starts):

- **Sean:** Phase 0 scaffolding, Phase 3 MCTS from-scratch, Phase 5 LLM
  pipeline, Phase 6 metrics.
- **Andrew:** Phase 2 DQN, Phase 4 counterfactual framework, Phase 7
  pilot study (if attempted), co-author Phase 8.
- **Shared:** Phase 1 formulation, Phase 8 report, Phase 9 submission.

Every phase has a primary owner and a reviewer, the other team member.

---

## 7. Exit-Gate Summary (one-page status board)

Use this table at every sync. Green only when every gate listed in the
phase section is met.

| Phase | Status | Primary | Gate count | Notes |
|---|---|---|---|---|
| 0 Scaffold | ⬜ | Sean | 6 | |
| 1 Formulation | ⬜ | Shared | 3 | |
| 2 DQN | ⬜ | Andrew | 5 | |
| 3 MCTS | ⬜ | Sean | 5 | |
| 4 Counterfactual framework | ⬜ | Andrew | 4 | |
| 5 LLM pipeline | ⬜ | Sean | 5 | |
| 6 Metrics | ⬜ | Sean | 4 | |
| 7 Stretch | ⬜ | Andrew (if attempted) | 2+3 | |
| 8 Report | ⬜ | Shared | 8 | |
| 9 Submission | ⬜ | Shared | 3 | |

---

## 8. Living-Document Policy

This plan is living. Any change, scope cut, phase reorder, gate relaxation
, must be committed with a message prefixed `plan:` so the history is
easy to audit. Do not silently move the goalposts.
