"""Retarget a G1 motion .npz to the Booster T1 skeleton via inverse kinematics.

Uses MuJoCo's Levenberg-Marquardt optimizer to match G1 end-effector positions
(feet + hands) on the T1 skeleton. The FK-only mapped joint angles serve as the
initial guess.

Usage:
  uv run python scripts/soccer/g1_retarget_t1_ik.py \\
    --input data/soccer-standard/g1/soccer-standard-001_right.npz \\
    --output data/soccer-standard/t1/soccer-standard-001_right.npz

  # With live viewer:
  uv run python scripts/soccer/g1_retarget_t1_ik.py \\
    --input data/soccer-standard/g1/soccer-standard-001_right.npz \\
    --output data/soccer-standard/t1/soccer-standard-001_right.npz --view
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.minimize  # noqa: F401 — lazy submodule
import mujoco.viewer  # noqa: F401 — lazy submodule
import numpy as np

# ── Joint mapping: G1 (MJLab/limb order) → T1 ─────────────────────────

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

# Internal lookup tables — filled after T1 model load.
_G1_TO_T1: dict[int, int | None] = {}
_t1_joint_name_to_idx: dict[str, int] = {}

# ── End-effector mapping for IK ───────────────────────────────────────

_IK_TARGETS = ("left_foot", "right_foot", "left_hand", "right_hand")

_G1_EE_BODY_NAME: dict[str, str] = {
  "left_foot": "left_ankle_roll_link",
  "right_foot": "right_ankle_roll_link",
  "left_hand": "left_wrist_yaw_link",
  "right_hand": "right_wrist_yaw_link",
}

_T1_EE_BODY_NAME: dict[str, str] = {
  "left_foot": "left_foot_link",
  "right_foot": "right_foot_link",
  "left_hand": "left_hand_link",
  "right_hand": "right_hand_link",
}


def _t1_xml_path() -> Path:
  return (
    Path(__file__).resolve().parents[2]
    / "src/mjlab_playground/asset_zoo/robots/booster_t1/xmls/t1.xml"
  )


def _resolve_indices(model: mujoco.MjModel) -> None:
  """Fill _t1_joint_name_to_idx and _G1_TO_T1 from the compiled model."""
  _t1_joint_name_to_idx.clear()
  _G1_TO_T1.clear()
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


def _solve_ik(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_order: list[int],
  x0: np.ndarray,
  body_ids: list[int],
  targets: np.ndarray,
  weights: np.ndarray,
  lower: np.ndarray,
  upper: np.ndarray,
  max_iter: int,
) -> tuple[np.ndarray, object]:
  """Solve IK for one frame: match T1 body positions to *targets*."""

  n_joints = len(joint_order)
  n_ee = len(body_ids) * 3

  def residual(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).ravel()
    for out_idx, t1_mu_idx in enumerate(joint_order):
      data.qpos[model.jnt_qposadr[t1_mu_idx]] = float(x[out_idx])
    mujoco.mj_forward(model, data)
    res = np.zeros((n_ee, 1), dtype=np.float64)
    for i, bid in enumerate(body_ids):
      err = data.xpos[bid] - targets[i]
      w = weights[i]
      res[i * 3, 0] = err[0] * w
      res[i * 3 + 1, 0] = err[1] * w
      res[i * 3 + 2, 0] = err[2] * w
    return res

  def jacobian(x: np.ndarray, _r: np.ndarray) -> np.ndarray:
    jac = np.zeros((n_ee, n_joints), dtype=np.float64)
    for i, bid in enumerate(body_ids):
      jacp = np.zeros((3, model.nv), dtype=np.float64)
      mujoco.mj_jacBody(model, data, jacp, None, bid)
      w = weights[i]
      for out_idx, t1_mu_idx in enumerate(joint_order):
        dof_adr = model.jnt_dofadr[t1_mu_idx]
        jac[i * 3, out_idx] = jacp[0, dof_adr] * w
        jac[i * 3 + 1, out_idx] = jacp[1, dof_adr] * w
        jac[i * 3 + 2, out_idx] = jacp[2, dof_adr] * w
    return jac

  return mujoco.minimize.least_squares(
    x0,
    residual,
    bounds=(lower, upper),
    jacobian=jacobian,
    max_iter=max_iter,
    xtol=1e-6,
    gtol=1e-6,
    verbose=0,
  )


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Retarget G1 motion .npz to Booster T1 via IK"
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
  parser.add_argument(
    "--ik-foot-weight",
    type=float,
    default=10.0,
    help="Weight multiplier for foot vs hand position error (default: 10.0).",
  )
  parser.add_argument(
    "--ik-max-iter",
    type=int,
    default=100,
    help="Max Levenberg-Marquardt iterations per frame (default: 100).",
  )
  parser.add_argument(
    "--height-scale",
    type=float,
    default=0.95,
    help="Uniform scale applied to G1 body positions relative to pelvis (default: 0.95).",
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
  _ISAACLAB_TO_LIMB: list[int] = [
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
  ]
  g1_joint_pos = src["joint_pos"][:, _ISAACLAB_TO_LIMB].astype(np.float64)
  g1_pelvis_pos = src["body_pos_w"][:, 0, :].astype(np.float64)
  g1_pelvis_quat = src["body_quat_w"][:, 0, :].astype(np.float64)
  g1_body_pos_w = src["body_pos_w"].astype(np.float64)
  g1_body_names: list[str]
  if "body_names" in src:
    g1_body_names = [str(n) for n in src["body_names"]]
  elif g1_body_pos_w.shape[1] == 30:
    g1_body_names = [
      "pelvis",
      "left_hip_pitch_link",
      "right_hip_pitch_link",
      "waist_yaw_link",
      "left_hip_roll_link",
      "right_hip_roll_link",
      "waist_roll_link",
      "left_hip_yaw_link",
      "right_hip_yaw_link",
      "torso_link",
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
    ]
  else:
    g1_body_names = [f"body_{i}" for i in range(g1_body_pos_w.shape[1])]
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
  n_t1_bodies = model.nbody - 1
  print(f"T1 model: {n_t1_joints} joints, {n_t1_bodies} bodies")

  _t1_joint_output_order = sorted(_t1_joint_name_to_idx.values())
  _t1_joint_out_idx = {
    mu_idx: out_idx for out_idx, mu_idx in enumerate(_t1_joint_output_order)
  }

  mapped = sum(1 for v in _G1_TO_T1.values() if v is not None)
  print(f"Joint mapping: {mapped}/{len(_G1_JOINT_NAMES)} G1 joints map to T1")

  # ── Split joints: upper-body IK, lower-body FK ─────────────────────
  # Arm G1 indices (limb order) → T1, for IK.
  _ARM_G1_INDICES = tuple(range(15, 29))  # shoulders through wrists
  # IK joint group: only arm joints that map to T1.
  ik_joint_order: list[int] = []  # T1 mu_idx, in IK variable order
  ik_joint_out_idx: dict[int, int] = {}  # T1 mu_idx → IK variable index
  for g1_idx in _ARM_G1_INDICES:
    t1_idx = _G1_TO_T1.get(g1_idx)
    if t1_idx is not None:
      ik_joint_out_idx[t1_idx] = len(ik_joint_order)
      ik_joint_order.append(t1_idx)

  # FK joint group: all mapped non-arm joints (legs + waist).
  fk_joint_qposadr: dict[int, int] = {}  # T1 mu_idx → qpos_adr
  for g1_idx, t1_idx in _G1_TO_T1.items():
    if t1_idx is not None and g1_idx not in _ARM_G1_INDICES:
      fk_joint_qposadr[t1_idx] = model.jnt_qposadr[t1_idx]

  n_ik_joints = len(ik_joint_order)
  n_fk_joints = len(fk_joint_qposadr)
  print(f"IK joints (arms): {n_ik_joints}, FK joints (legs+waist): {n_fk_joints}")

  # IK targets: hands only (feet are FK, no need to constrain).
  _HAND_TARGETS = ("left_hand", "right_hand")
  g1_name_to_idx: dict[str, int] = {}
  for i, name in enumerate(g1_body_names):
    g1_name_to_idx[name] = i

  g1_ee_idx: dict[str, int] = {}
  t1_ee_id: dict[str, int] = {}
  for key in _HAND_TARGETS:
    g1_bname = _G1_EE_BODY_NAME[key]
    t1_bname = _T1_EE_BODY_NAME[key]
    if g1_bname not in g1_name_to_idx:
      print(
        f"ERROR: G1 body '{g1_bname}' not found in source npz body_names",
        file=sys.stderr,
      )
      sys.exit(1)
    g1_ee_idx[key] = g1_name_to_idx[g1_bname]
    t1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, t1_bname)
    if t1_id < 0:
      print(
        f"ERROR: T1 body '{t1_bname}' not found in T1 model",
        file=sys.stderr,
      )
      sys.exit(1)
    t1_ee_id[key] = t1_id
  print(f"IK targets (hands): {list(_HAND_TARGETS)}")
  print(f"  height-scale={args.height_scale}")

  # Hand IK: equal weights for both hands.
  ik_weights = np.ones(len(_HAND_TARGETS), dtype=np.float64)

  # Pre-compute arm joint bounds.
  ik_lower = np.zeros(n_ik_joints, dtype=np.float64)
  ik_upper = np.zeros(n_ik_joints, dtype=np.float64)
  for t1_mu_idx in ik_joint_order:
    oi = ik_joint_out_idx[t1_mu_idx]
    ik_lower[oi] = model.jnt_range[t1_mu_idx, 0]
    ik_upper[oi] = model.jnt_range[t1_mu_idx, 1]

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

  ik_failures = 0
  t1_hand_body_ids = [t1_ee_id[k] for k in _HAND_TARGETS]
  # Unmapped T1 joints: set to zero (no G1 counterpart).
  mapped_t1_mu_indices: set[int] = set()
  for t1_idx in _G1_TO_T1.values():
    if t1_idx is not None:
      mapped_t1_mu_indices.add(t1_idx)

  for frame in range(frames):
    # Root pose.
    root_pos = g1_pelvis_pos[frame].copy()
    root_pos[2] = args.t1_default_z
    root_quat = g1_pelvis_quat[frame].copy()
    data.qpos[0:3] = root_pos
    data.qpos[3:7] = root_quat

    # FK group (legs + waist): direct copy from G1 mapped angles.
    for g1_idx, t1_mu_idx in _G1_TO_T1.items():
      if t1_mu_idx is not None and g1_idx not in _ARM_G1_INDICES:
        data.qpos[model.jnt_qposadr[t1_mu_idx]] = g1_joint_pos[frame, g1_idx]

    # Unmapped T1 joints: zero / nominal.
    for t1_mu_idx in _t1_joint_name_to_idx.values():
      if t1_mu_idx not in mapped_t1_mu_indices:
        data.qpos[model.jnt_qposadr[t1_mu_idx]] = 0.0

    # Build IK targets for hands (world frame).
    targets_w = np.zeros((2, 3), dtype=np.float64)
    for ki, key in enumerate(_HAND_TARGETS):
      g1_pos = g1_body_pos_w[frame, g1_ee_idx[key]]
      offset = g1_pos - g1_pelvis_pos[frame]
      targets_w[ki] = root_pos + args.height_scale * offset

    # IK initial guess: FK-mapped arm angles from G1.
    x0 = np.zeros(n_ik_joints, dtype=np.float64)
    for g1_idx, t1_mu_idx in _G1_TO_T1.items():
      if t1_mu_idx is not None and g1_idx in _ARM_G1_INDICES:
        x0[ik_joint_out_idx[t1_mu_idx]] = g1_joint_pos[frame, g1_idx]
    x0 = np.clip(x0, ik_lower, ik_upper)

    result, _trace = _solve_ik(
      model,
      data,
      ik_joint_order,
      x0,
      t1_hand_body_ids,
      targets_w,
      ik_weights,
      ik_lower,
      ik_upper,
      args.ik_max_iter,
    )

    if _trace[-1].objective > 10.0:
      ik_failures += 1

    # Apply IK result (arm joints).
    for oi, t1_mu_idx in enumerate(ik_joint_order):
      data.qpos[model.jnt_qposadr[t1_mu_idx]] = float(result[oi])

    mujoco.mj_forward(model, data)

    # Record joint positions and velocities.
    for _t1_name, t1_mu_idx in _t1_joint_name_to_idx.items():
      out_idx = _t1_joint_out_idx[t1_mu_idx]
      t1_joint_pos[frame, out_idx] = float(data.qpos[model.jnt_qposadr[t1_mu_idx]])
      t1_joint_vel[frame, out_idx] = float(data.qvel[model.jnt_dofadr[t1_mu_idx]])

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

  if viewer is not None:
    viewer.close()

  if ik_failures > 0:
    print(f"IK: {ik_failures}/{frames} frames did not converge (using last iterate)")

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
