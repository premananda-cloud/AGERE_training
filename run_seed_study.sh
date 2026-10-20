#!/usr/bin/env bash
# Multi-seed training study for the hover 3x5 disturbance policy.
# Run from the AGERE repo root:   bash run_seed_study.sh
#
# Recipe (frozen; every seed identical):
#   from scratch (no --init-from), --stage disturbance_3x5,
#   500k steps, checkpoint every 50k, n_envs=6, ent_coef=0.003, gamma=0.995
#
# Pilot first (one throwaway seed, NOT part of the reported set):
#   SEEDS="100" bash run_seed_study.sh
# Then the real study:
#   bash run_seed_study.sh
#
# TAG is deliberately NOT "disturbance_3x5": that tag already holds the old
# seed-0 chain checkpoints, and reusing it would overwrite them.
set -euo pipefail

TAG="${TAG:-seedstudy}"
TIMESTEPS="${TIMESTEPS:-500000}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"

mkdir -p logs

for SEED in $SEEDS; do
  FINAL="model/model_weights/hover_stabilize_ppo_seed${SEED}_${TAG}.zip"
  if [ -f "$FINAL" ]; then
    echo "[skip] seed ${SEED}: ${FINAL} already exists"
    continue
  fi
  echo "[run] seed ${SEED} -> ${FINAL}"
  python -m src.training.hover_train \
    --seed "$SEED" \
    --stage disturbance_3x5 \
    --tag "$TAG" \
    --timesteps "$TIMESTEPS" \
    --checkpoint-every 50000 \
    --n-envs 6 \
    --ent-coef 0.003 \
    --gamma 0.995 \
    2>&1 | tee "logs/${TAG}_seed${SEED}.log"
done

echo "Done. Next: sweep/screen each seed's checkpoints, then evaluate on held-out seeds."
