"""Replay a G1 motion .npz on the G1 MJCF model via MuJoCo viewer.

Usage:
  uv run python scripts/soccer/g1_replay_npz.py \\
    --motion-path data/mjlab_playground-mjlab/soccer-standard/g1/soccer-standard-001_right.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer  # noqa: F401 — lazy submodule
import numpy as np

# G1 joint names in MJLab/IsaacLab order — matches the .npz joint_pos column layout.
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


def _g1_xml_path() -> Path:
  """Return path to the G1 MJCF XML shipped with mjlab."""
  import mjlab

  return (
    Path(mjlab.__file__).resolve().parent / "asset_zoo/robots/unitree_g1/xmls/g1.xml"
  )


def main() -> None:
  parser = argparse.ArgumentParser(description="Replay G1 motion .npz in MuJoCo viewer")
  parser.add_argument("--motion-path", required=True, help="Path to G1 motion .npz")
  parser.add_argument(
    "--speed",
    type=float,
    default=1.0,
    help="Playback speed multiplier (1.0 = real time, 0 = max speed).",
  )
  parser.add_argument(
    "--cam-distance",
    type=float,
    default=2.5,
    help="Camera distance from the robot (default: 2.5).",
  )
  parser.add_argument(
    "--cam-elevation",
    type=float,
    default=-15.0,
    help="Camera elevation angle in degrees (default: -15).",
  )
  parser.add_argument(
    "--cam-azimuth",
    type=float,
    default=90.0,
    help="Camera azimuth angle in degrees (default: 90).",
  )
  args = parser.parse_args()

  # ── Load G1 motion ─────────────────────────────────────────────────
  src = dict(np.load(args.motion_path, allow_pickle=True))
  for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "fps"):
    if key not in src:
      print(f"ERROR: input .npz missing key '{key}'", file=sys.stderr)
      sys.exit(1)

  frames = int(src["joint_pos"].shape[0])
  n_joints_src = src["joint_pos"].shape[1]
  if n_joints_src != len(_G1_JOINT_NAMES):
    print(
      f"ERROR: expected {len(_G1_JOINT_NAMES)} joints, got {n_joints_src}",
      file=sys.stderr,
    )
    sys.exit(1)

  fps = float(np.asarray(src["fps"]).reshape(-1)[0])
  g1_joint_pos = src["joint_pos"].astype(np.float64)
  g1_joint_vel = src["joint_vel"].astype(np.float64)
  g1_pelvis_pos = src["body_pos_w"][:, 0, :].astype(np.float64)
  g1_pelvis_quat = src["body_quat_w"][:, 0, :].astype(np.float64)
  print(f"Loaded G1 motion: {frames} frames, {fps} fps")

  # ── Build G1 model ─────────────────────────────────────────────────
  model = mujoco.MjModel.from_xml_path(str(_g1_xml_path()))
  data = mujoco.MjData(model)

  # Build joint name → MuJoCo joint index mapping.
  joint_name_to_id: dict[str, int] = {}
  for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name and model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
      joint_name_to_id[name] = i

  # Verify all G1 joints exist and record (qpos_adr, dof_adr) for each.
  npz_to_qposadr: list[int] = []
  npz_to_dofadr: list[int] = []
  for g1_name in _G1_JOINT_NAMES:
    if g1_name not in joint_name_to_id:
      print(f"ERROR: G1 joint '{g1_name}' not found in MJCF", file=sys.stderr)
      sys.exit(1)
    jid = joint_name_to_id[g1_name]
    npz_to_qposadr.append(model.jnt_qposadr[jid])
    npz_to_dofadr.append(model.jnt_dofadr[jid])

  print(f"G1 model: {model.nbody - 1} bodies, {len(npz_to_qposadr)} actuated joints")

  # ── Viewer ─────────────────────────────────────────────────────────
  viewer = mujoco.viewer.launch_passive(model, data)
  try:
    cam = viewer.cam
    cam.distance = args.cam_distance
    cam.elevation = args.cam_elevation
    cam.azimuth = args.cam_azimuth
    cam.lookat[:] = (0.0, 0.0, 0.74)
  except AttributeError:
    pass

  frame_dt = 1.0 / fps if args.speed > 0 else 0.0
  print(f"Viewer open (speed={args.speed}x, dt={frame_dt:.3f}s/frame)")

  # ── Replay loop ────────────────────────────────────────────────────
  frame = 0
  while viewer.is_running():
    # Root pose from .npz.
    data.qpos[0:3] = g1_pelvis_pos[frame]
    data.qpos[3:7] = g1_pelvis_quat[frame]

    # Joint positions from .npz.
    for j, qpos_adr in enumerate(npz_to_qposadr):
      data.qpos[qpos_adr] = g1_joint_pos[frame, j]

    # Joint velocities.
    for j, dof_adr in enumerate(npz_to_dofadr):
      if dof_adr >= 0:
        data.qvel[dof_adr] = g1_joint_vel[frame, j]

    mujoco.mj_forward(model, data)
    viewer.sync()

    if frame_dt > 0:
      time.sleep(frame_dt / args.speed)

    frame = (frame + 1) % frames

  viewer.close()
  print("Done.")


if __name__ == "__main__":
  main()
