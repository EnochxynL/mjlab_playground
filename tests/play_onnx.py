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

Usage:
  python tests/play_onnx.py \\
    --policy /home/enoch/Public/HumanoidSoccer/ckp/policy_30000.onnx \\
    --motion data/soccer-standard/soccer-standard-001_right.npz

Dependencies: onnxruntime (pip install onnxruntime)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

# Trigger task registration.
import mjlab_playground  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab_playground.tasks.soccer.config.g1.env_cfgs import (
    g1_soccer_destination_env_cfg,
)


class OnnxPolicy:
    """Wraps an IsaacLab-exported ONNX policy for use with MJLab's viewer.

    The viewer calls ``policy(obs_dict)`` on every step and
    ``policy.reset()`` on episode boundaries.  This class manages the RNN
    hidden state and time-step counter internally.
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
        self.h = np.zeros(
            (self.rnn_layers, 1, self.rnn_hidden), dtype=np.float32
        )
        self.c = np.zeros(
            (self.rnn_layers, 1, self.rnn_hidden), dtype=np.float32
        )
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

        # Viewer may step multiple envs; ONNX expects batch=1 per call.
        # Iterate over envs so RNN state stays per-env (simplified:
        # single-env play only).
        if actor_obs.shape[0] != 1:
            raise ValueError(
                f"Expected num_envs=1 for ONNX play, got {actor_obs.shape[0]}"
            )

        obs_np = actor_obs.cpu().numpy().astype(np.float32)

        feeds: dict[str, np.ndarray] = {
            "obs": obs_np,
            "time_step": np.array([[self.step_counter]], dtype=np.float32),
            "h_in": self.h,
            "c_in": self.c,
        }

        outputs = self.session.run(None, feeds)
        out = dict(zip(self.output_names, outputs))

        self.h = out["h_out"]
        self.c = out["c_out"]
        self.step_counter += 1

        return torch.from_numpy(out["actions"]).to(self.device)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _resolve_obs_dim(meta: dict[str, str]) -> int | None:
        names = meta.get("observation_names", "")
        if names:
            return len(names.split(","))
        return None


# ---------------------------------------------------------------------------


def _resolve_viewer(viewer_arg: str):
    """Pick viewer backend, falling back to viser when there is no display."""
    if viewer_arg == "auto":
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        return "native" if has_display else "viser"
    return viewer_arg


def main() -> None:
    parser = argparse.ArgumentParser(description="Play soccer ONNX policy via MJLab")
    parser.add_argument(
        "--policy", required=True, help="Path to .onnx policy file"
    )
    parser.add_argument(
        "--motion", required=True, help="Path to motion .npz file"
    )
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

    # -- env ----------------------------------------------------------------
    cfg = g1_soccer_destination_env_cfg(play=True)
    cfg.scene.num_envs = 1
    cfg.commands["motion"].motion_files = [args.motion]

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
