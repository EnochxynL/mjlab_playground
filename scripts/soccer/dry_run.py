"""Dry-run test for the soccer environment.

Verifies that the soccer tracking and destination env configs can be
instantiated and stepped without runtime errors.  Does not need a GPU
or a display — uses play mode (no terminations, no domain
randomisation, infinite episode length).

A motion file is required because the motion command loads animations
at construction time.  If you have a local copy of HumanoidSoccer the
script tries to pick up the first motion automatically; otherwise pass
one via ``--motion``.

Usage:
  python tests/dry_run.py
  python tests/dry_run.py --motion /path/to/motion.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Trigger task registration so factory functions are available.
import mjlab_playground  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab_playground.tasks.soccer.config.g1.env_cfgs import (
  g1_soccer_destination_env_cfg,
  g1_soccer_tracking_env_cfg,
)

_NUM_STEPS = 50

# Default motion location when HumanoidSoccer is checked out alongside.
_DEFAULT_MOTION_DIR = Path("/home/enoch/Public/mjlab_playground/data/soccer-standard")


def _find_default_motion() -> str | None:
  if _DEFAULT_MOTION_DIR.is_dir():
    npz_files = sorted(_DEFAULT_MOTION_DIR.glob("*.npz"))
    if npz_files:
      return str(npz_files[0])
  return None


def _build_and_step(
  name: str,
  env_cfg_fn,
  motion_file: str,
  *,
  num_steps: int = _NUM_STEPS,
) -> None:
  print(f"  Building {name} ...", flush=True)
  cfg = env_cfg_fn(play=True)
  cfg.scene.num_envs = 1
  cfg.commands["motion"].motion_files = [motion_file]

  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()

  actions = torch.zeros(env.action_space.shape, device="cpu")
  for _ in range(num_steps):
    env.step(actions)

  env.close()
  print(f"    ✓  {num_steps} steps completed, env closed")


def main() -> None:
  parser = argparse.ArgumentParser(description="Soccer env dry-run")
  parser.add_argument(
    "--motion",
    default=_find_default_motion(),
    help="Path to a motion .npz file (required by the motion command)",
  )
  args = parser.parse_args()

  if args.motion is None:
    print(
      "No motion file found.  Pass one via:\n"
      "  python tests/dry_run.py --motion /path/to/motion.npz"
    )
    return

  print(f"  Motion: {args.motion}")
  print("=== Soccer dry-run ===")
  _build_and_step("Stage 1 (tracking)", g1_soccer_tracking_env_cfg, args.motion)
  _build_and_step("Stage 2 (destination)", g1_soccer_destination_env_cfg, args.motion)
  print("All good.")


if __name__ == "__main__":
  main()
