#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
# Play a trained soccer policy (or watch with zero/random).
#
# Usage:
#   # Trained policy
#   ./soccer_play.sh Mjlab-SoccerTracking-G1 \
#       logs/rsl_rl/g1_soccer/my_run/model_4000.pt \
#       data/soccer-standard/soccer-standard-001_right.npz
#
#   # Zero actions (just watch the motion)
#   ./soccer_play.sh Mjlab-SoccerTracking-G1 --zero \
#       data/soccer-standard/soccer-standard-001_right.npz
#
#   # Random actions
#   ./soccer_play.sh Mjlab-SoccerTracking-G1 --random \
#       data/soccer-standard/soccer-standard-001_right.npz
# ──────────────────────────────────────────────────────────

set -euo pipefail

TASK="${1:?Usage: $0 <TaskID> <checkpoint|--zero|--random> <motion_file>}"
MODE="${2:?}"
MOTION_FILE="${3:?}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

case "${MODE}" in
    --zero)
        python -m mjlab.scripts.play "${TASK}" \
            --agent zero \
            --motion-file "${MOTION_FILE}"
        ;;
    --random)
        python -m mjlab.scripts.play "${TASK}" \
            --agent random \
            --motion-file "${MOTION_FILE}"
        ;;
    *)
        python -m mjlab.scripts.play "${TASK}" \
            --agent trained \
            --checkpoint-file "${MODE}" \
            --motion-file "${MOTION_FILE}"
        ;;
esac
