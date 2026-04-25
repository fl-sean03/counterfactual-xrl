#!/usr/bin/env bash
# Reproduce only the DQN ablation referenced in the report's "Why DQN was
# cut" paragraph. DQN is no longer part of the main pipeline; this script
# exists so the ablation numbers in the report can still be regenerated.
#
# Usage: bash scripts/run_dqn_ablation.sh

set -euo pipefail
PY="${PY:-$(dirname "$0")/../.venv/bin/python}"
ENV="$PY"

echo "=== DQN (image, partial-obs CNN-flat) ==="
[ -f results/dqn/baseline/seed0/model.zip ] || \
    $ENV scripts/train_dqn.py --config configs/dqn_baseline.yaml
[ -f results/dqn/baseline/seed0/eval_summary.json ] || \
    $ENV scripts/eval.py --run-dir results/dqn/baseline/seed0 --n-episodes 300

echo "=== DQN (symbolic, full-state) ==="
[ -f results/dqn/symbolic/seed0/model.zip ] || \
    $ENV scripts/train_dqn.py --config configs/dqn_symbolic.yaml
[ -f results/dqn/symbolic/seed0/eval_summary.json ] || \
    $ENV scripts/eval.py --run-dir results/dqn/symbolic/seed0 --n-episodes 300

echo "=== DQN (shaped, distance-shaping) ==="
[ -f results/dqn/shaped/seed0/model.zip ] || \
    $ENV scripts/train_dqn.py --config configs/dqn_shaped.yaml
[ -f results/dqn/shaped/seed0/eval_summary.json ] || \
    $ENV scripts/eval.py --run-dir results/dqn/shaped/seed0 --n-episodes 300

echo "DQN ablation complete. All three variants stall under raw MiniGrid reward."
