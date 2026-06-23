"""Convert IsaacLab-ordered G1 .npz files to MJCF/MJLab limb order.

Usage:
  uv run python scripts/soccer/g1_humanoidSoccer_to_mjlab.py \\
    --input data/HumanoidSoccer-IsaacLab/soccer-standard/soccer-standard-001_right.npz \\
    --output data/mjlab_playground-mjlab/soccer-standard/g1/soccer-standard-001_right.npz

  # Batch convert directory:
  uv run python scripts/soccer/g1_humanoidSoccer_to_mjlab.py \\
    --input-dir data/HumanoidSoccer-IsaacLab/soccer-standard/ \\
    --output-dir data/mjlab_playground-mjlab/soccer-standard/g1/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ── Joint-order remapping ──────────────────────────────────────────────
# IsaacLab groups joints by type (all hip_pitch, then all hip_roll…),
# MuJoCo/MJCF groups by limb (left leg, right leg, waist, left arm, right arm).
# Both have the same 29 joints, just in different orders.

_ISAACLAB_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)

_MJLAB_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

_isaac_to_idx = {n: i for i, n in enumerate(_ISAACLAB_JOINT_ORDER)}
_ISAACLAB_TO_MJCF = np.array([_isaac_to_idx[n] for n in _MJLAB_JOINT_ORDER], dtype=np.int64)

"""
# CRITICAL BUG FIX: G1 body order mismatch

## Phenomenon: G1 Stage 1 training failed

The robot's joints swung wildly and chaotically, even causing it to fly into the air, with almost no relation to soccer

## Root cause: G1 body order mismatch

The [g1_humanoidSoccer_to_mjlab.py](scripts/soccer/g1_humanoidSoccer_to_mjlab.py) conversion script only remapped **joint** data from IsaacLab order to MJLab/MuJoCo order, but left **body** data in IsaacLab's ordering. This meant:

- MJLab's `body_indexes` mapping looked up bodies by MuJoCo body index
- `torso_link` (MJCF body index 15) was reading the position of what was actually a **foot** in IsaacLab order (z=0.08m at ground level)
- `anchor_pos` termination compared reference torso at ground level against actual robot torso at standing height (z=0.87m) → terminated every step
- Constant resets kept `actions` observation always zero → 29/160 obs dims had zero std → policy corrupted

## Fix applied

Added body-order remapping to the conversion script with:
- `_MJLAB_BODY_ORDER` — 30 body names in MuJoCo/MJCF limb-grouped order
- `_ISAACLAB_BODY_ORDER` — 30 body names in IsaacLab type-grouped order (determined by FK matching)
- `_MJCF_TO_ISAAC_BODY` permutation to reorder `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`

All 10 G1 .npz files were regenerated. Verification confirms:
- `torso_link` at body index 15 now has z=0.866m (standing height)
- Environment runs 10+ steps without `anchor_pos` termination
- T1 .npz files were unaffected (already in correct order, explaining why T1 training worked)

The G1 Stage 1 training needs to be re-run with the fixed .npz files.

"""

# ── Body-order remapping ───────────────────────────────────────────────
# IsaacLab groups bodies by type (all left/right pairs together, similar to
# joint grouping), while MuJoCo/MJCF groups by limb in XML order.  The
# .npz body data must be reordered to match MJCF so that body_indexes
# lookups in MultiMotionLoader hit the right bodies.

_MJLAB_BODY_ORDER: tuple[str, ...] = (
    # MJCF body order excluding world (MuJoCo body indices 1-30).
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)

_ISAACLAB_BODY_ORDER: tuple[str, ...] = (
    # IsaacLab body order as it appears in the original .npz (determined by
    # comparing .npz positions against MuJoCo FK output on the first frame).
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "torso_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "waist_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

_isaac_body_to_idx = {n: i for i, n in enumerate(_ISAACLAB_BODY_ORDER)}
_MJCF_TO_ISAAC_BODY = np.array(
    [_isaac_body_to_idx[n] for n in _MJLAB_BODY_ORDER], dtype=np.int64
)


def convert_npz(input_path: str, output_path: str) -> None:
    """Load *input_path*, permute joint and body data, save to *output_path*."""
    src = dict(np.load(input_path, allow_pickle=True))

    for key in ("joint_pos", "joint_vel"):
        if key not in src:
            print(f"ERROR: input .npz missing key '{key}'", file=sys.stderr)
            sys.exit(1)

    n_joints = src["joint_pos"].shape[1]
    if n_joints != len(_ISAACLAB_JOINT_ORDER):
        print(
            f"ERROR: expected {len(_ISAACLAB_JOINT_ORDER)} joints, got {n_joints}",
            file=sys.stderr,
        )
        sys.exit(1)

    src["joint_pos"] = src["joint_pos"][:, _ISAACLAB_TO_MJCF]
    src["joint_vel"] = src["joint_vel"][:, _ISAACLAB_TO_MJCF]

    n_bodies = src["body_pos_w"].shape[1]
    if n_bodies != len(_MJCF_TO_ISAAC_BODY):
        print(
            f"ERROR: expected {len(_MJCF_TO_ISAAC_BODY)} bodies, got {n_bodies}",
            file=sys.stderr,
        )
        sys.exit(1)

    for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        if key in src:
            src[key] = src[key][:, _MJCF_TO_ISAAC_BODY]

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **src)
    print(f"Converted: {input_path} → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert IsaacLab-ordered G1 .npz to MJCF limb order"
    )
    parser.add_argument("--input", type=str, help="Single .npz file to convert")
    parser.add_argument("--output", type=str, help="Output path for single file")
    parser.add_argument("--input-dir", type=str, help="Directory of .npz files to batch convert")
    parser.add_argument("--output-dir", type=str, help="Output directory for batch conversion")
    args = parser.parse_args()

    single = bool(args.input or args.output)
    batch = bool(args.input_dir or args.output_dir)

    if single and batch:
        print("ERROR: use either --input/--output OR --input-dir/--output-dir, not both", file=sys.stderr)
        sys.exit(1)
    if not single and not batch:
        print("ERROR: specify --input/--output or --input-dir/--output-dir", file=sys.stderr)
        sys.exit(1)

    if single:
        if not args.input or not args.output:
            print("ERROR: both --input and --output are required", file=sys.stderr)
            sys.exit(1)
        convert_npz(args.input, args.output)
    else:
        if not args.input_dir or not args.output_dir:
            print("ERROR: both --input-dir and --output-dir are required", file=sys.stderr)
            sys.exit(1)
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(f"ERROR: input dir not found: {input_dir}", file=sys.stderr)
            sys.exit(1)
        npz_files = sorted(input_dir.glob("*.npz"))
        if not npz_files:
            print(f"ERROR: no .npz files found in {input_dir}", file=sys.stderr)
            sys.exit(1)
        for npz_file in npz_files:
            out_path = str(Path(args.output_dir) / npz_file.name)
            convert_npz(str(npz_file), out_path)


if __name__ == "__main__":
    main()
