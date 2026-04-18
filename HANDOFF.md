# Handoff — Where the Project Is and What's Left

_Last updated 2026-04-17 by autonomous session. Read this before the next
work session._

## Current state (committed to `origin/main`)

**All code scaffolded and tested end-to-end.** The full pipeline runs
with mock LLM calls; runs with a real Anthropic key will produce
meaningful metrics.

### Empirical results so far

| Agent | Success | Collision | Return | Steps |
|---|---|---|---|---|
| Random (baseline) | 0.003 | 0.997 | −0.994 | 8.3 |
| DQN (image, 300k) | 0.010 | 0.000 | 0.007 | 254.3 |
| DQN (symbolic, 200k) | 0.000 | 0.207 | −0.207 | 232.3 |
| DQN (shaped, 300k) | 0.000 | 0.167 | −0.167 | 234.4 |
| **MCTS (500 sims)** | **1.000** | **0.000** | **0.934** | **18.3** |

**Decision records:** 45 per agent (3 seeds × 15 decisions) validated
against the JSON schema. **Explanations + metrics:** 90 total run
end-to-end with mock client — pipeline confirmed functional.

### What's genuinely done

- Phase 0 scaffold (env, CI, lint, Makefile)
- Phase 1 problem formulation frozen and tested
- Phase 2 DQN — baseline trained but degenerate; symbolic + shaped
  variants also tried
- Phase 3 MCTS — works excellently, 100% success
- Phase 4 counterfactual framework — schema + bootstrap CIs + tree
  stats
- Phase 5 LLM pipeline — client (real + mock), prompts, cache
- Phase 6 metrics — fidelity + soundness + inferability with judge
- Phase 7a chatbot — scaffolded
- Phase 8 report — IEEE template with every section drafted, figures
  regenerable

## Decisions made autonomously

See `ENGINEERING_LOG.md` for rationale on all of these:

1. **DQN is suboptimal** — accepted per Sunberg's framing rather than
   burning more time on PPO or a curriculum.
2. **Phase 7b human pilot — CUT.** Note as future work.
3. **Phase 7a chatbot — SCAFFOLDED ONLY** (script exists; demo video
   not produced).
4. **LLM explainer — MOCK-ONLY runs so far.** Real API calls are
   deliberately gated behind `ANTHROPIC_API_KEY`.

## What you (Sean) need to do

### To get real metrics

```bash
export ANTHROPIC_API_KEY=sk-ant-...
rm -rf results/explanations results/metrics  # force re-run
bash scripts/reproduce_all.sh
```

Expected cost: **under $5** for 90 explanations + 270 judge calls at
Sonnet + Haiku pricing.

### To finalize the report

1. Open `report/main.tex`.
2. **Every paragraph marked `[DRAFT — rewrite in your voice]`** must be
   rewritten by you (AI-policy compliance).
3. Verify every numeric claim against `report/figures/eval_summary.csv`
   and `results/metrics/summary.json`.
4. Update the contributions section using `CONTRIBUTIONS.md` as a
   starting point — **Andrew needs to confirm the split**.
5. Pick one of the two release statements.
6. Build: `cd report && latexmk -pdf main.tex`.

### To loop in Andrew

- Share `AGENTS.md`, `ENGINEERING_LOG.md`, and this `HANDOFF.md`.
- Ask him to (a) review `CONTRIBUTIONS.md` and (b) read the current
  `report/main.tex` draft.

### Submission (Phase 9)

- [ ] Final `report/main.pdf` built and reviewed
- [ ] Upload to Gradescope
- [ ] Tag git release `v1.0-final-report`
- [ ] Push final commit

## Things I could not do

- Run real Claude API calls (no key in env).
- Conduct the pilot human study (needs humans).
- Train DQN to match MCTS performance (vanilla DQN fundamentally
  struggles on this env; documented honestly as a finding).
- Author the final report prose (syllabus AI policy).
- Verify the Belouadah 2025 citation — it's flagged `% VERIFY` in the
  bibliography.

## Key files

| File | Purpose |
|---|---|
| `EXECUTION_PLAN.md` | Phased plan + exit gates |
| `ENGINEERING_LOG.md` | Chronological decision log |
| `CONTRIBUTIONS.md` | Draft per-author contributions |
| `HANDOFF.md` | This file |
| `report/main.tex` | IEEE-format draft report |
| `scripts/reproduce_all.sh` | One-command full pipeline |
| `results/metrics/summary.json` | Final metric numbers |
| `ENGINEERING_LOG.md` | Dated log of every experiment + decision |
