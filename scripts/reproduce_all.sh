#!/usr/bin/env bash
# Reproduce the full pipeline from scratch. Run from repo root.
#
# Expected wall-clock (RTX 5080 laptop):
#   DQN training (image):          ~5 min
#   PPO training (symbolic, 8 env):~8 min
#   MCTS 50-episode eval:         ~12 min
#   DecisionRecord build:          ~6 min
#   Explainer (~260 calls, gpt-4o): ~30 min (requires OPENAI_API_KEY)
#   Metrics (1300+ judge calls):    ~8 min
#   Total: ~60 min + ~$3 API
#
# Idempotent: skips steps whose output already exists. Delete results/ or
# the affected subdirectory to force a re-run after code changes.
#
# Prereq: `make install`, `export OPENAI_API_KEY=...` (or ANTHROPIC_API_KEY).

set -euo pipefail

ENV="conda run -n xrl --no-capture-output"

echo "=== 1. Smoke test ==="
$ENV python scripts/smoke_env.py

echo "=== 2. Random baseline eval (sanity: env is non-trivial) ==="
[ -f results/random/eval_summary.json ] || \
    $ENV python scripts/eval.py --random --n-episodes 300

echo "=== 3. DQN (image) training — 300k steps ==="
[ -f results/dqn/baseline/seed0/model.zip ] || \
    $ENV python scripts/train_dqn.py --config configs/dqn_baseline.yaml

echo "=== 4. DQN eval ==="
[ -f results/dqn/baseline/seed0/eval_summary.json ] || \
    $ENV python scripts/eval.py --run-dir results/dqn/baseline/seed0 --n-episodes 300

echo "=== 5. PPO training — 500k steps (primary RL agent) ==="
[ -f results/ppo/baseline/seed0/model.zip ] || \
    $ENV python scripts/train_ppo.py --config configs/ppo_baseline.yaml --n-envs 8

echo "=== 6. PPO eval ==="
[ -f results/ppo/baseline/seed0/eval_summary.json ] || \
    $ENV python scripts/eval_ppo.py --run-dir results/ppo/baseline/seed0 --n-episodes 300

echo "=== 7. MCTS 50-episode eval ==="
[ -f results/mcts/baseline/seed0/eval_summary.json ] || \
    $ENV python scripts/run_mcts.py --config configs/mcts_baseline.yaml --n-episodes 50

echo "=== 8. Build DecisionRecords (PPO + MCTS, 10 seeds, N=100 per action) ==="
if [ ! -d results/decision_records/ppo_baseline ] || [ ! -d results/decision_records/mcts_baseline ]; then
    $ENV python scripts/build_decision_records.py \
        --ppo-run-dir results/ppo/baseline/seed0 \
        --ppo-agent-id ppo_baseline \
        --mcts-config configs/mcts_baseline.yaml \
        --mcts-agent-id mcts_baseline \
        --seeds 10000 10001 10002 10003 10004 10005 10006 10007 10008 10009 \
        --n-per-action 100 \
        --max-steps 15 \
        --rollout-cap 40 \
        --out-dir results/decision_records
fi

echo "=== 9. Generate explanations (gpt-4o; requires OPENAI_API_KEY) ==="
[ -d results/explanations/ppo_baseline ] || \
    $ENV python scripts/explain.py --records-dir results/decision_records --out-dir results/explanations

echo "=== 10. Score explanations (gpt-4o-mini judge) ==="
[ -f results/metrics/summary.json ] || \
    $ENV python scripts/evaluate_explanations.py \
        --records-dir results/decision_records \
        --explanations-dir results/explanations \
        --out-dir results/metrics

echo "=== 11. Regenerate report figures ==="
$ENV python scripts/report_figures.py

echo "=== 12. Test suite ==="
$ENV python -m pytest -q

echo
echo "DONE. Primary artifacts:"
echo "  report/main.pdf                  — final report"
echo "  results/metrics/summary.json      — headline explanation-quality numbers"
echo "  report/figures/eval_summary.csv   — task-performance table"
echo "  report/figures/metric_comparison.pdf — fidelity/soundness/inferability bar chart"
echo "  ENGINEERING_LOG.md                — decisions and raw findings"
