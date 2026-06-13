#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Progressive soccer training: two-stage pipeline for G1.
#
# Stage 1 (Mjlab-SoccerTracking-Terrain-G1):
#   gravel terrain, adaptive sampling, tracking rewards only.
#   Learns to reproduce kick motions without ball contact.
#
# Stage 2 (Mjlab-SoccerDestination-Flat-G1):
#   flat ground, uniform sampling, full kick rewards.
#   Fine-tunes Stage 1 checkpoint to actually kick the ball.
#
# Usage:
#   ./progressive_soccer_train.sh [RUN_NAME]
#   ./progressive_soccer_train.sh my_experiment
#
# Reference: arXiv-2602.05310v1 "Learning Soccer Skills for Humanoid Robots"
# Ported from HumanoidSoccer/shell/progressive_soccer_train_play.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

# Shared experiment directory (both stages use experiment_name=g1_soccer)
EXPERIMENT_DIR="${REPO_ROOT}/logs/rsl_rl/g1_soccer"

RUN_NAME="${1:-test}"

# ── motion files ──────────────────────────────────────────────

MOTION_DIR="${REPO_ROOT}/data/soccer-standard"
MOTION_FILES=(
    "${MOTION_DIR}/soccer-standard-001_right.npz"
    "${MOTION_DIR}/soccer-standard-002_left.npz"
    "${MOTION_DIR}/soccer-standard-003_left.npz"
    "${MOTION_DIR}/soccer-standard-004_right.npz"
    "${MOTION_DIR}/soccer-standard-005_right.npz"
    "${MOTION_DIR}/soccer-standard-006_right.npz"
    "${MOTION_DIR}/soccer-standard-007_left.npz"
    "${MOTION_DIR}/soccer-standard-008_left.npz"
    "${MOTION_DIR}/soccer-standard-009_right.npz"
    "${MOTION_DIR}/soccer-standard-010_right.npz"
)

# Build JSON array string for tyro's UsePythonSyntaxForLiteralCollections
_MOTION_JSON="["
for i in "${!MOTION_FILES[@]}"; do
    _MOTION_JSON+="\"${MOTION_FILES[$i]}\""
    if [[ $i -lt $((${#MOTION_FILES[@]} - 1)) ]]; then
        _MOTION_JSON+=", "
    fi
done
_MOTION_JSON+="]"

# ── train parameters (override via env vars) ───────────────────

NUM_ENVS="${NUM_ENVS:-4096}"
STAGE1_ITERS="${STAGE1_ITERS:-4000}"
STAGE2_ITERS="${STAGE2_ITERS:-30000}"
LOGGER="${LOGGER:-tensorboard}"  # tensorboard or wandb

cd "${REPO_ROOT}"

# ── Stage 1: motion-skill acquisition ─────────────────────────

echo "=============================================="
echo " Stage 1: Mjlab-SoccerTracking-Terrain-G1"
echo " run_name: ${RUN_NAME}"
echo " num_envs: ${NUM_ENVS}"
echo " max_iters: ${STAGE1_ITERS}"
echo "=============================================="

python -m mjlab.scripts.train Mjlab-SoccerTracking-Terrain-G1 \
    --env.commands.motion.motion_files "${_MOTION_JSON}" \
    --env.scene.num-envs "${NUM_ENVS}" \
    --agent.max_iterations "${STAGE1_ITERS}" \
    --agent.run_name "${RUN_NAME}" \
    --agent.logger "${LOGGER}"

# ── resolve Stage 1 run directory ─────────────────────────────

LOAD_RUN="$(find "${EXPERIMENT_DIR}" -maxdepth 1 -mindepth 1 -type d \
    -name "*_${RUN_NAME}" | sort | tail -n 1 | xargs -r basename)"

if [[ -z "${LOAD_RUN}" ]]; then
    echo "[ERROR] Failed to resolve Stage 1 run directory in ${EXPERIMENT_DIR}"
    exit 1
fi

echo ""
echo "[INFO] Stage 1 run directory: ${LOAD_RUN}"
echo ""

# ── Stage 2: kick-to-destination fine-tuning ──────────────────

echo "=============================================="
echo " Stage 2: Mjlab-SoccerDestination-Flat-G1"
echo " load_run: ${LOAD_RUN}"
echo " run_name: ${RUN_NAME}_resume"
echo " num_envs: ${NUM_ENVS}"
echo " max_iters: ${STAGE2_ITERS}"
echo "=============================================="

python -m mjlab.scripts.train Mjlab-SoccerDestination-Flat-G1 \
    --env.commands.motion.motion_files "${_MOTION_JSON}" \
    --env.scene.num-envs "${NUM_ENVS}" \
    --agent.max_iterations "${STAGE2_ITERS}" \
    --agent.run_name "${RUN_NAME}_resume" \
    --agent.resume true \
    --agent.load_run "${LOAD_RUN}" \
    --agent.load_checkpoint "model_.*.pt"

echo ""
echo "=============================================="
echo " Training complete."
echo " Stage 1 log: ${EXPERIMENT_DIR}/${LOAD_RUN}"
echo " Stage 2 log: ${EXPERIMENT_DIR}/$(ls -1t "${EXPERIMENT_DIR}" | head -1)"
echo "=============================================="
