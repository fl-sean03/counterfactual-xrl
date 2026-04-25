#!/usr/bin/env bash
# Reproduce the full pipeline from scratch. Run from repo root.
#
# Expected wall-clock (RTX 5080 laptop):
#   PPO training (symbolic, 16 env, 2M steps):~30-45 min
#   MCTS 50-episode eval:                       ~12 min
#   DecisionRecord build:                       ~6 min
#   Explainer (270 calls, gpt-4o):              ~20 min   (requires OPENAI_API_KEY)
#   Metrics (1300+ judge calls):                 ~8 min
#   Total: ~75 min + ~$2 API
#
# Idempotent: skips steps whose output already exists. Delete results/
# to force re-run.
#
# DQN ablation (kept for the "why DQN was cut" report paragraph) is not
# part of the main path. To regenerate the ablation, run
# ``scripts/run_dqn_ablation.sh``.
#
# Prereq: `make install`, `export OPENAI_API_KEY=...` (or ANTHROPIC_API_KEY).

set -euo pipefail

PY="${PY:-$(dirname "$0")/../.venv/bin/python}"
ENV="$PY"

echo "=== 1. Smoke test ==="
$ENV scripts/smoke_env.py

echo "=== 2. Random baseline eval (sanity: env is non-trivial) ==="
[ -f results/random/eval_summary.json ] || \
    $ENV scripts/eval.py --random --n-episodes 300

echo "=== 3. PPO training - 2M steps, 16 parallel envs ==="
[ -f results/ppo/tuned/seed0/model.zip ] || \
    $ENV scripts/train_ppo.py --config configs/ppo_tuned.yaml --n-envs 16

echo "=== 4. PPO eval ==="
[ -f results/ppo/tuned/seed0/eval_summary.json ] || \
    $ENV scripts/eval_ppo.py --run-dir results/ppo/tuned/seed0 --n-episodes 300

echo "=== 5. MCTS 50-episode eval ==="
[ -f results/mcts/baseline/seed0/eval_summary.json ] || \
    $ENV scripts/run_mcts.py --config configs/mcts_baseline.yaml --n-episodes 50

echo "=== 6. Build DecisionRecords (PPO + MCTS, 10 seeds, N=100 per action) ==="
if [ ! -d results/decision_records/ppo_tuned ] || [ ! -d results/decision_records/mcts_baseline ]; then
    $ENV scripts/build_decision_records.py \
        --ppo-run-dir results/ppo/tuned/seed0 \
        --ppo-agent-id ppo_tuned \
        --mcts-config configs/mcts_baseline.yaml \
        --mcts-agent-id mcts_baseline \
        --seeds 10000 10001 10002 10003 10004 10005 10006 10007 10008 10009 \
        --n-per-action 100 \
        --max-steps 15 \
        --rollout-cap 40 \
        --out-dir results/decision_records
fi

echo "=== 7. Generate explanations (gpt-4o; requires OPENAI_API_KEY) ==="
[ -d results/explanations/ppo_tuned ] || \
    $ENV scripts/explain.py --records-dir results/decision_records --out-dir results/explanations

echo "=== 8. Score explanations (gpt-4o-mini judge) ==="
[ -f results/metrics/summary.json ] || \
    $ENV scripts/evaluate_explanations.py \
        --records-dir results/decision_records \
        --explanations-dir results/explanations \
        --out-dir results/metrics

echo "=== 9. Regenerate report figures ==="
$ENV scripts/report_figures.py

echo "=== 10. Test suite ==="
$ENV -m pytest -q

echo
echo "DONE. Primary artifacts:"
echo "  report/main.pdf                  - final report"
echo "  results/metrics/summary.json      - headline explanation-quality numbers"
echo "  report/figures/eval_summary.csv   - task-performance table"
echo "  report/figures/metric_comparison.pdf - fidelity/soundness/inferability bar chart"
echo "  ENGINEERING_LOG.md                - decisions and raw findings"
