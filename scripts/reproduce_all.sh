#!/usr/bin/env bash
# Reproduce every artifact from scratch. Run from repo root.
#
# Takes ~30-45 minutes total on an RTX 5080 laptop. Skips steps whose
# output already exists (idempotent); delete `results/` to force re-run.
#
# Prereq: `make install` has been run and `ANTHROPIC_API_KEY` is set
# (or you accept mock-only explanations).

set -euo pipefail

ENV="conda run -n xrl --no-capture-output"

echo "=== 1. Smoke test ==="
$ENV python scripts/smoke_env.py

echo "=== 2. Random baseline eval ==="
[ -f results/random/eval_summary.json ] || \
    $ENV python scripts/eval.py --random --n-episodes 300

echo "=== 3. DQN image training (300k steps) ==="
[ -f results/dqn/baseline/seed0/model.zip ] || \
    $ENV python scripts/train_dqn.py --config configs/dqn_baseline.yaml

echo "=== 4. DQN image eval ==="
[ -f results/dqn/baseline/seed0/eval_summary.json ] || \
    $ENV python scripts/eval.py --run-dir results/dqn/baseline/seed0 --n-episodes 300

echo "=== 5. MCTS full eval (50 episodes) ==="
[ -f results/mcts/baseline/seed0/eval_summary.json ] || \
    $ENV python scripts/run_mcts.py --config configs/mcts_baseline.yaml --n-episodes 50

echo "=== 6. Build DecisionRecords (3 seeds, 15 steps each) ==="
[ -d results/decision_records/mcts_baseline ] || \
    $ENV python scripts/build_decision_records.py \
        --dqn-run-dir results/dqn/baseline/seed0 \
        --dqn-agent-id dqn_baseline \
        --mcts-config configs/mcts_baseline.yaml \
        --mcts-agent-id mcts_baseline \
        --seeds 10000 10001 10002 \
        --n-per-action 20 --max-steps 15 --rollout-cap 40 \
        --out-dir results/decision_records

echo "=== 7. Generate explanations (uses API if key set, else mock) ==="
[ -d results/explanations/dqn_baseline ] || \
    $ENV python scripts/explain.py --records-dir results/decision_records --out-dir results/explanations

echo "=== 8. Score explanations ==="
[ -f results/metrics/summary.json ] || \
    $ENV python scripts/evaluate_explanations.py \
        --records-dir results/decision_records \
        --explanations-dir results/explanations \
        --out-dir results/metrics

echo "=== 9. Regenerate report figures ==="
$ENV python scripts/report_figures.py

echo "=== 10. Test suite ==="
$ENV python -m pytest -q

echo
echo "DONE. Artifacts:"
echo "  task performance     → report/figures/task_performance.pdf"
echo "  explanation metrics  → report/figures/metric_comparison.pdf"
echo "  learning curves      → report/figures/learning_curves.pdf"
echo "  eval summary table   → report/figures/eval_summary.csv"
echo "  engineering log      → ENGINEERING_LOG.md"
