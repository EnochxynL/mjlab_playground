"""Replay a T1 motion .npz on the Booster T1 MJCF model via MuJoCo viewer.

Usage:
  uv run python scripts/soccer/t1_replay_npz_mujoco.py \\
    --motion-path data/mjlab_playground-mjlab/soccer-standard/t1/soccer-standard-001_right.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer  # noqa: F401 — lazy submodule
import numpy as np


def _t1_xml_path() -> Path:
  return (
    Path(__file__).resolve().parents[2]
    / "src/mjlab_playground/asset_zoo/robots/booster_t1/xmls/t1.xml"
  )


def main() -> None:
  parser = argparse.ArgumentParser(description="Replay T1 motion .npz in MuJoCo viewer")
  parser.add_argument("--motion-path", required=True, help="Path to T1 motion .npz")
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

  # ── Load T1 motion ─────────────────────────────────────────────────
  src = dict(np.load(args.motion_path, allow_pickle=True))
  for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "fps"):
    if key not in src:
      print(f"ERROR: input .npz missing key '{key}'", file=sys.stderr)
      sys.exit(1)

  frames = int(src["joint_pos"].shape[0])
  n_joints_src = src["joint_pos"].shape[1]
  fps = float(np.asarray(src["fps"]).reshape(-1)[0])
  t1_joint_pos = src["joint_pos"].astype(np.float64)
  t1_joint_vel = src["joint_vel"].astype(np.float64)
  t1_pelvis_pos = src["body_pos_w"][:, 0, :].astype(np.float64)
  t1_pelvis_quat = src["body_quat_w"][:, 0, :].astype(np.float64)
  print(f"Loaded T1 motion: {frames} frames, {fps} fps, {n_joints_src} joints")

  # ── Build T1 model ─────────────────────────────────────────────────
  model = mujoco.MjModel.from_xml_path(str(_t1_xml_path()))
  data = mujoco.MjData(model)

  # T1 actuated joints in sorted MuJoCo index order (matches the npz column layout).
  joint_name_to_muidx: dict[str, int] = {}
  for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name and model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
      joint_name_to_muidx[name] = i

  t1_joint_order = sorted(joint_name_to_muidx.values())
  if len(t1_joint_order) != n_joints_src:
    print(
      f"ERROR: MJCF has {len(t1_joint_order)} actuated joints, npz has {n_joints_src}",
      file=sys.stderr,
    )
    sys.exit(1)

  npz_to_qposadr = [model.jnt_qposadr[mu_idx] for mu_idx in t1_joint_order]
  npz_to_dofadr = [model.jnt_dofadr[mu_idx] for mu_idx in t1_joint_order]
  print(f"T1 model: {model.nbody - 1} bodies, {len(t1_joint_order)} actuated joints")

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

  frame_dt = (1.0 / fps / args.speed) if args.speed > 0 else 0.0
  print(f"Viewer open (speed={args.speed}x, dt={frame_dt:.3f}s/frame)")

  # ── Replay loop ────────────────────────────────────────────────────
  frame = 0
  while viewer.is_running():
    frame_start = time.perf_counter()

    data.qpos[0:3] = t1_pelvis_pos[frame]
    data.qpos[3:7] = t1_pelvis_quat[frame]

    for j, qpos_adr in enumerate(npz_to_qposadr):
      data.qpos[qpos_adr] = t1_joint_pos[frame, j]

    for j, dof_adr in enumerate(npz_to_dofadr):
      if dof_adr >= 0:
        data.qvel[dof_adr] = t1_joint_vel[frame, j]

    mujoco.mj_forward(model, data)
    viewer.sync()

    elapsed = time.perf_counter() - frame_start
    if frame_dt > 0 and elapsed < frame_dt:
      time.sleep(frame_dt - elapsed)

    frame = (frame + 1) % frames

  viewer.close()
  print("Done.")


if __name__ == "__main__":
  main()
