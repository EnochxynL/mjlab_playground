"""Play a soccer ONNX policy through the MJLab environment and viewer.

Loads an IsaacLab-exported ONNX policy (stage 2 kick-to-destination) and
runs it inside a MJLab ManagerBasedRlEnv.  No direct MuJoCo calls —
everything goes through MJLab's env + viewer.

The ONNX policy is recurrent (LSTM); this wrapper manages hidden state
internally and responds to viewer-initiated resets.

The actor observation structure (160 dims) has been aligned to match
the IsaacLab export: command, projected_gravity, motion_ref_ang_vel,
base_ang_vel, joint_pos, joint_vel, actions, target_point_pos,
target_destination_pos_local.

IsaacLab and MJLab order the G1's 29 joints differently — the motion
file is remapped at load time and observations/actions are permuted
inside OnnxPolicy.

Usage:
  python tests/play_onnx.py \\
    --policy /home/enoch/Public/HumanoidSoccer/ckp/policy_30000.onnx \\
    --motion data/soccer-standard/soccer-standard-001_right.npz

Dependencies: onnxruntime (pip install onnxruntime)
"""

from __future__ import annotations

import argparse
import os
import tempfile

# Trigger task registration.
import mjlab_playground  # noqa: F401
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab_playground.tasks.soccer.config.g1.env_cfgs import (
  g1_soccer_destination_env_cfg,
)

# ── Joint-order remapping ──────────────────────────────────────────────
#
# IsaacLab groups joints by type (all hip_pitch, then all hip_roll…),
# MJLab groups by limb (left leg, right leg, waist, left arm, right arm).
# Both have the same 29 joints, just in different orders.

_ISAACLAB_JOINT_ORDER: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
  "waist_yaw_joint",
  "left_hip_roll_joint",
  "right_hip_roll_joint",
  "waist_roll_joint",
  "left_hip_yaw_joint",
  "right_hip_yaw_joint",
  "waist_pitch_joint",
  "left_knee_joint",
  "right_knee_joint",
  "left_shoulder_pitch_joint",
  "right_shoulder_pitch_joint",
  "left_ankle_pitch_joint",
  "right_ankle_pitch_joint",
  "left_shoulder_roll_joint",
  "right_shoulder_roll_joint",
  "left_ankle_roll_joint",
  "right_ankle_roll_joint",
  "left_shoulder_yaw_joint",
  "right_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_elbow_joint",
  "left_wrist_roll_joint",
  "right_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "right_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_wrist_yaw_joint",
)

_MJLAB_JOINT_ORDER: tuple[str, ...] = (
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

_isaac_to_idx = {n: i for i, n in enumerate(_ISAACLAB_JOINT_ORDER)}
# Permutation: IsaacLab-ordered data → MJLab order.
# This is ISAACLAB_TO_MUJOCO_REINDEX from the exp constants.
_OBS_PERM = np.array([_isaac_to_idx[n] for n in _MJLAB_JOINT_ORDER], dtype=np.int64)
# Permutation: MJLab-ordered data → IsaacLab order.
# This is MUJOCO_TO_ISAACLAB_REINDEX from the exp constants.
_mjlab_to_idx = {n: i for i, n in enumerate(_MJLAB_JOINT_ORDER)}
_ACT_PERM = np.array([_mjlab_to_idx[n] for n in _ISAACLAB_JOINT_ORDER], dtype=np.int64)

# Within the 160-dim actor observation, joint-valued terms appear at:
#   command     [0:58]    (29 joint_pos_ref + 29 joint_vel_ref)
#   joint_pos   [67:96]   (29 dims)
#   joint_vel   [96:125]  (29 dims)
#   actions     [125:154] (29 dims)
_COMMAND_JPOS_SLICE = slice(0, 29)
_COMMAND_JVEL_SLICE = slice(29, 58)
_JOINT_POS_SLICE = slice(67, 96)
_JOINT_VEL_SLICE = slice(96, 125)
_ACTIONS_SLICE = slice(125, 154)


def _remap_motion_file(src_path: str) -> str:
  """Create a copy of *src_path* with joint data reordered to MJLab order."""
  data = dict(np.load(src_path, allow_pickle=True))
  data["joint_pos"] = data["joint_pos"][:, _OBS_PERM]
  data["joint_vel"] = data["joint_vel"][:, _OBS_PERM]
  fd, dst = tempfile.mkstemp(suffix=".npz", prefix="mjlab_motion_")
  os.close(fd)
  np.savez(dst, **data)
  return dst


# ── ONNX policy wrapper ────────────────────────────────────────────────


class OnnxPolicy:
  """Wraps an IsaacLab-exported ONNX policy for use with MJLab's viewer.

  The viewer calls ``policy(obs_dict)`` on every step and
  ``policy.reset()`` on episode boundaries.  This class manages the RNN
  hidden state and time-step counter internally.

  Joint-valued observation terms and policy actions are remapped
  between IsaacLab and MJLab joint orderings so the ONNX policy always
  sees the ordering it was trained on.
  """

  def __init__(self, path: str, device: str = "cpu"):
    try:
      import onnxruntime as ort
    except ImportError as exc:
      raise ImportError(
        "onnxruntime is required.  Install it with: pip install onnxruntime"
      ) from exc

    self.device = device
    self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    self.input_names = [i.name for i in self.session.get_inputs()]
    self.output_names = [o.name for o in self.session.get_outputs()]

    meta = dict(self.session.get_modelmeta().custom_metadata_map)
    self._obs_dim = self._resolve_obs_dim(meta)

    # RNN dimensions — hard-coded for the IsaacLab G1 soccer export.
    self.rnn_layers = 2
    self.rnn_hidden = 128
    self.h: np.ndarray | None = None
    self.c: np.ndarray | None = None
    self.step_counter = 0

    self.reset()

  # -- public interface expected by the viewer --------------------------

  def reset(self) -> None:
    """Called by the viewer on env reset (episode boundary)."""
    self.h = np.zeros((self.rnn_layers, 1, self.rnn_hidden), dtype=np.float32)
    self.c = np.zeros((self.rnn_layers, 1, self.rnn_hidden), dtype=np.float32)
    self.step_counter = 0

  def __call__(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Called by the viewer every physics step.

    Parameters
    ----------
    obs_dict:
        The dict returned by ``env.get_observations()``.
        Must contain an ``"actor"`` key whose value is a
        ``[B, obs_dim]`` tensor (B = num_envs).

    Returns
    -------
    actions:
        A ``[B, num_actions]`` tensor on the same device as the env.
    """
    actor_obs = obs_dict["actor"]

    if actor_obs.shape[0] != 1:
      raise ValueError(f"Expected num_envs=1 for ONNX play, got {actor_obs.shape[0]}")

    obs_np = actor_obs.cpu().numpy().astype(np.float32)

    # Remap joint-valued terms from MJLab → IsaacLab joint order.
    obs_np[:, _COMMAND_JPOS_SLICE] = obs_np[:, _COMMAND_JPOS_SLICE][:, _ACT_PERM]
    obs_np[:, _COMMAND_JVEL_SLICE] = obs_np[:, _COMMAND_JVEL_SLICE][:, _ACT_PERM]
    obs_np[:, _JOINT_POS_SLICE] = obs_np[:, _JOINT_POS_SLICE][:, _ACT_PERM]
    obs_np[:, _JOINT_VEL_SLICE] = obs_np[:, _JOINT_VEL_SLICE][:, _ACT_PERM]
    obs_np[:, _ACTIONS_SLICE] = obs_np[:, _ACTIONS_SLICE][:, _ACT_PERM]

    feeds: dict[str, np.ndarray] = {
      "obs": obs_np,
      "time_step": np.array([[self.step_counter]], dtype=np.float32),
      "h_in": self.h,
      "c_in": self.c,
    }

    outputs = self.session.run(None, feeds)
    out = dict(zip(self.output_names, outputs, strict=True))

    self.h = out["h_out"]
    self.c = out["c_out"]
    self.step_counter += 1

    # Remap actions from IsaacLab → MJLab joint order.
    actions_np = out["actions"]
    actions_np = actions_np[:, _OBS_PERM]

    return torch.from_numpy(actions_np).to(self.device)

  # -- helpers -----------------------------------------------------------

  @staticmethod
  def _resolve_obs_dim(meta: dict[str, str]) -> int | None:
    names = meta.get("observation_names", "")
    if names:
      return len(names.split(","))
    return None


# ── main ────────────────────────────────────────────────────────────────


def _resolve_viewer(viewer_arg: str):
  """Pick viewer backend, falling back to viser when there is no display."""
  if viewer_arg == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return "native" if has_display else "viser"
  return viewer_arg


def main() -> None:
  parser = argparse.ArgumentParser(description="Play soccer ONNX policy via MJLab")
  parser.add_argument("--policy", required=True, help="Path to .onnx policy file")
  parser.add_argument("--motion", required=True, help="Path to motion .npz file")
  parser.add_argument(
    "--viewer",
    choices=["native", "viser", "auto"],
    default="auto",
    help="Viewer backend (default: auto — native if DISPLAY, else viser)",
  )
  parser.add_argument(
    "--device",
    default="cpu",
    help="Device to run the environment on (default: cpu)",
  )
  args = parser.parse_args()

  # Remap IsaacLab motion file to MJLab joint order so the motion-
  # command system writes correct joint targets during reset/tracking.
  remapped_motion = _remap_motion_file(args.motion)

  # -- env ----------------------------------------------------------------
  cfg = g1_soccer_destination_env_cfg(play=True)
  cfg.scene.num_envs = 1
  cfg.commands["motion"].motion_files = [remapped_motion]

  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)

  # The MJLab viewer expects ``env.get_observations()`` and
  # ``env.reset()``, but ManagerBasedRlEnv only stores obs in
  # ``self.obs_buf`` (populated after step/reset) and returns a tuple
  # from reset().  MJLab's play.py uses RslRlVecEnvWrapper which
  # calls reset() on init and provides get_observations().  We mirror
  # that here without pulling in rsl_rl.
  def _get_observations():
    return env.observation_manager.compute()

  env.get_observations = _get_observations
  env.reset()

  # -- policy -------------------------------------------------------------
  policy = OnnxPolicy(args.policy, device=args.device)

  # -- viewer -------------------------------------------------------------
  viewer_backend = _resolve_viewer(args.viewer)
  if viewer_backend == "native":
    NativeMujocoViewer(env, policy).run()
  else:
    ViserPlayViewer(env, policy).run()

  env.close()


if __name__ == "__main__":
  main()
