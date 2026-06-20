"""Retarget a G1 motion .npz to the Booster T1 skeleton via MuJoCo FK.

Usage:
  uv run python scripts/soccer/g1_retarget_t1.py \\
    --input data/soccer-standard/soccer-standard-001_right.npz \\
    --output data/soccer-t1/soccer-t1-001_right.npz

  # With live viewer:
  uv run python scripts/soccer/g1_retarget_t1.py \\
    --input data/soccer-standard/soccer-standard-001_right.npz \\
    --output data/soccer-t1/soccer-t1-001_right.npz --view
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer  # noqa: F401 — lazy submodule, must be imported explicitly
import numpy as np

# ── Joint mapping: G1 (MJLab order) → T1 ──────────────────────────────

_G1_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

# Which T1 joint each G1 joint maps to (None = no counterpart).
_G1_TO_T1: dict[int, int | None] = {}

# T1 joint names in XML order — filled after model load.
_t1_joint_name_to_idx: dict[str, int] = {}

_G1_NAME_TO_T1_NAME: dict[str, str | None] = {
  # Legs — direct 1:1.
  "left_hip_pitch_joint": "Left_Hip_Pitch",
  "left_hip_roll_joint": "Left_Hip_Roll",
  "left_hip_yaw_joint": "Left_Hip_Yaw",
  "left_knee_joint": "Left_Knee_Pitch",
  "left_ankle_pitch_joint": "Left_Ankle_Pitch",
  "left_ankle_roll_joint": "Left_Ankle_Roll",
  "right_hip_pitch_joint": "Right_Hip_Pitch",
  "right_hip_roll_joint": "Right_Hip_Roll",
  "right_hip_yaw_joint": "Right_Hip_Yaw",
  "right_knee_joint": "Right_Knee_Pitch",
  "right_ankle_pitch_joint": "Right_Ankle_Pitch",
  "right_ankle_roll_joint": "Right_Ankle_Roll",
  # Waist — G1 has 3 DoF, T1 only 1 (z-rotation).
  "waist_yaw_joint": "Waist",
  "waist_roll_joint": None,
  "waist_pitch_joint": None,
  # Arms — partial overlap.
  "left_shoulder_pitch_joint": "Left_Shoulder_Pitch",
  "left_shoulder_roll_joint": "Left_Shoulder_Roll",
  "left_shoulder_yaw_joint": None,
  "left_elbow_joint": "Left_Elbow_Pitch",
  "left_wrist_roll_joint": None,
  "left_wrist_pitch_joint": None,
  "left_wrist_yaw_joint": None,
  "right_shoulder_pitch_joint": "Right_Shoulder_Pitch",
  "right_shoulder_roll_joint": "Right_Shoulder_Roll",
  "right_shoulder_yaw_joint": None,
  "right_elbow_joint": "Right_Elbow_Pitch",
  "right_wrist_roll_joint": None,
  "right_wrist_pitch_joint": None,
  "right_wrist_yaw_joint": None,
}


def _t1_xml_path() -> Path:
  return (
    Path(__file__).resolve().parents[2]
    / "src/mjlab_playground/asset_zoo/robots/booster_t1/xmls/t1.xml"
  )


def _resolve_indices(model: mujoco.MjModel) -> None:
  """Fill _t1_joint_name_to_idx and _G1_TO_T1 from the compiled model."""
  for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name and model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
      _t1_joint_name_to_idx[name] = i

  for g1_idx, g1_name in enumerate(_G1_JOINT_NAMES):
    t1_name = _G1_NAME_TO_T1_NAME.get(g1_name)
    if t1_name and t1_name in _t1_joint_name_to_idx:
      _G1_TO_T1[g1_idx] = _t1_joint_name_to_idx[t1_name]
    else:
      _G1_TO_T1[g1_idx] = None


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Retarget G1 motion .npz to Booster T1 skeleton"
  )
  parser.add_argument("--input", required=True, help="Path to G1 motion .npz")
  parser.add_argument("--output", required=True, help="Path for T1 motion .npz")
  parser.add_argument(
    "--t1-default-z",
    type=float,
    default=0.74,
    help="Default standing height for T1 root z (default: 0.74)",
  )
  parser.add_argument(
    "--view",
    action="store_true",
    help="Open a MuJoCo viewer to watch each frame live.",
  )
  parser.add_argument(
    "--speed",
    type=float,
    default=1.0,
    help="Playback speed multiplier (1.0 = real time, 0 = max speed).",
  )
  args = parser.parse_args()

  # ── Load G1 motion ─────────────────────────────────────────────────
  src = dict(np.load(args.input, allow_pickle=True))
  for key in (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  ):
    if key not in src:
      print(f"ERROR: input .npz missing key '{key}'", file=sys.stderr)
      sys.exit(1)

  frames = int(src["joint_pos"].shape[0])
  g1_joint_pos = src["joint_pos"].astype(np.float64)
  g1_pelvis_pos = src["body_pos_w"][:, 0, :].astype(np.float64)
  g1_pelvis_quat = src["body_quat_w"][:, 0, :].astype(np.float64)
  fps = float(np.asarray(src["fps"]).reshape(-1)[0])
  print(f"Loaded G1 motion: {frames} frames, {fps} fps")

  # ── Build T1 model ─────────────────────────────────────────────────
  spec = mujoco.MjSpec.from_file(str(_t1_xml_path()))
  model = spec.compile()
  data = mujoco.MjData(model)
  _resolve_indices(model)

  n_t1_joints = sum(
    1 for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
  )
  n_t1_bodies = model.nbody - 1  # exclude body 0 ('world')
  print(f"T1 model: {n_t1_joints} joints, {n_t1_bodies} bodies")

  _t1_joint_output_order = sorted(_t1_joint_name_to_idx.values())
  _t1_joint_out_idx = {
    mu_idx: out_idx for out_idx, mu_idx in enumerate(_t1_joint_output_order)
  }

  mapped = sum(1 for v in _G1_TO_T1.values() if v is not None)
  print(f"Joint mapping: {mapped}/{len(_G1_JOINT_NAMES)} G1 joints map to T1")

  # ── Viewer (optional) ──────────────────────────────────────────────
  viewer = None
  frame_dt = None
  if args.view:
    viewer = mujoco.viewer.launch_passive(model, data)
    frame_dt = 1.0 / fps if args.speed > 0 else 0.0
    print(f"Viewer opened (speed={args.speed}x, {frame_dt:.3f}s/frame)")

  # ── Retarget frame by frame ────────────────────────────────────────
  t1_joint_pos = np.zeros((frames, n_t1_joints), dtype=np.float32)
  t1_joint_vel = np.zeros((frames, n_t1_joints), dtype=np.float32)
  t1_body_pos_w = np.zeros((frames, n_t1_bodies, 3), dtype=np.float32)
  t1_body_quat_w = np.zeros((frames, n_t1_bodies, 4), dtype=np.float32)
  t1_body_lin_vel_w = np.zeros((frames, n_t1_bodies, 3), dtype=np.float32)
  t1_body_ang_vel_w = np.zeros((frames, n_t1_bodies, 3), dtype=np.float32)

  for frame in range(frames):
    # Root pose: copy G1 pelvis xy + orientation, use T1 default z.
    root_pos = g1_pelvis_pos[frame].copy()
    root_pos[2] = args.t1_default_z
    root_quat = g1_pelvis_quat[frame].copy()
    data.qpos[0:3] = root_pos
    data.qpos[3:7] = root_quat

    # Map joint angles.
    for g1_idx, t1_idx in _G1_TO_T1.items():
      if t1_idx is not None:
        qpos_adr = model.jnt_qposadr[t1_idx]
        data.qpos[qpos_adr] = g1_joint_pos[frame, g1_idx]

    mujoco.mj_forward(model, data)

    # Record joint positions and velocities after FK.
    for _t1_name, t1_mu_idx in _t1_joint_name_to_idx.items():
      out_idx = _t1_joint_out_idx[t1_mu_idx]
      t1_joint_pos[frame, out_idx] = float(data.qpos[model.jnt_qposadr[t1_mu_idx]])
      t1_joint_vel[frame, out_idx] = float(data.qvel[model.jnt_dofadr[t1_mu_idx]])

    # Body 0 is 'world' — skip it.
    for body_idx in range(1, model.nbody):
      out_body = body_idx - 1
      t1_body_pos_w[frame, out_body] = data.xpos[body_idx]
      t1_body_quat_w[frame, out_body] = data.xquat[body_idx]
      vel = np.zeros(6, dtype=np.float64)
      mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_idx, vel, 0)
      t1_body_lin_vel_w[frame, out_body] = vel[3:6]
      t1_body_ang_vel_w[frame, out_body] = vel[:3]

    if viewer is not None:
      viewer.sync()
      if frame_dt is not None and frame_dt > 0:
        time.sleep(frame_dt / args.speed)

  # ── Cleanup ─────────────────────────────────────────────────────────
  if viewer is not None:
    viewer.close()

  # ── Export ──────────────────────────────────────────────────────────
  out_dir = Path(args.output).parent
  out_dir.mkdir(parents=True, exist_ok=True)

  t1_body_names = np.array(
    [
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
      for i in range(1, model.nbody)
    ],
    dtype=object,
  )

  np.savez(
    args.output,
    fps=np.array([fps], dtype=np.float32),
    joint_pos=t1_joint_pos,
    joint_vel=t1_joint_vel,
    body_pos_w=t1_body_pos_w,
    body_quat_w=t1_body_quat_w,
    body_lin_vel_w=t1_body_lin_vel_w,
    body_ang_vel_w=t1_body_ang_vel_w,
    body_names=t1_body_names,
    kick_leg=src.get("kick_leg", "right"),
    source_npz=str(Path(args.input).resolve()),
  )
  print(f"Wrote T1 motion: {args.output}")
  print(f"  frames={frames}, fps={fps}, joints={n_t1_joints}, bodies={n_t1_bodies}")
  print(f"  body_names: {list(t1_body_names)}")


if __name__ == "__main__":
  main()
