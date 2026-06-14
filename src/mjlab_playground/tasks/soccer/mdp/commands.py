"""Soccer motion command with ball placement, destinations, and blind zones.

Extends the mjlab tracking ``MotionCommand`` with multi-motion loading,
soccer-ball position computation, target-point logic, and kick-leg labeling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.managers import CommandTermCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionLoader
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from .kick_detection import KickContactTracker

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


class MultiMotionLoader:
    """Load multiple .npz motion files with padding to a common length."""

    def __init__(
        self,
        motion_files: list[str],
        body_indexes: torch.Tensor,
        device: str = "cpu",
    ) -> None:
        assert len(motion_files) > 0, "motion_files must not be empty"
        self.num_files = len(motion_files)
        self._body_indexes = body_indexes
        self.device = device

        joint_pos_list: list[torch.Tensor] = []
        joint_vel_list: list[torch.Tensor] = []
        body_pos_w_list: list[torch.Tensor] = []
        body_quat_w_list: list[torch.Tensor] = []
        body_lin_vel_w_list: list[torch.Tensor] = []
        body_ang_vel_w_list: list[torch.Tensor] = []
        kick_leg_labels: list[str | None] = []
        self.motion_names: list[str] = []
        self.motion_lengths: list[int] = []

        max_T = 0
        for motion_file in motion_files:
            data = np.load(motion_file)
            self.motion_names.append(motion_file.rsplit("/", 1)[-1].rsplit(".", 1)[0])
            self.motion_lengths.append(int(data["joint_pos"].shape[0]))

            jp = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
            jv = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
            bp = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
            bq = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
            blv = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
            bav = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)

            joint_pos_list.append(jp)
            joint_vel_list.append(jv)
            body_pos_w_list.append(bp)
            body_quat_w_list.append(bq)
            body_lin_vel_w_list.append(blv)
            body_ang_vel_w_list.append(bav)

            label_value: str | None = None
            if "kick_leg" in data.files:
                raw_label = data["kick_leg"]
                try:
                    label_str = str(raw_label.item()).strip().lower()
                except Exception:
                    label_str = str(raw_label).strip().lower()
                if label_str in {"left", "right"}:
                    label_value = label_str
            kick_leg_labels.append(label_value)

            max_T = max(max_T, jp.shape[0])

        def _pad(tensor_list: list[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
            padded = []
            for t in tensor_list:
                T, *rest = t.shape
                pad_t = torch.cat(
                    [t, torch.full([max_T - T] + rest, pad_value, device=device)], dim=0
                )
                padded.append(pad_t)
            return torch.stack(padded, dim=0)

        self.joint_pos = _pad(joint_pos_list)
        self.joint_vel = _pad(joint_vel_list)
        self._body_pos_w = _pad(body_pos_w_list)
        self._body_quat_w = _pad(body_quat_w_list)
        self._body_lin_vel_w = _pad(body_lin_vel_w_list)
        self._body_ang_vel_w = _pad(body_ang_vel_w_list)

        self.time_step_total = max_T
        self.file_lengths = torch.tensor(
            [jp.shape[0] for jp in joint_pos_list], dtype=torch.long, device=device
        )
        self._kick_leg_labels = tuple(kick_leg_labels)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, :, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, :, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, :, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, :, self._body_indexes]

    @property
    def kick_leg_labels(self) -> tuple[str | None, ...]:
        return self._kick_leg_labels

    def get_first_frame_anchor_pos(
        self, motion_idx: int, anchor_body_idx: int
    ) -> torch.Tensor:
        return self._body_pos_w[motion_idx, 0, anchor_body_idx]

    def get_last_frame_anchor_pos(
        self, motion_idx: int, anchor_body_idx: int, motion_length: int
    ) -> torch.Tensor:
        last_frame = motion_length - 1
        return self._body_pos_w[motion_idx, last_frame, anchor_body_idx]


class SoccerMotionCommand(MotionCommand):
    """Motion command extended with soccer-ball logic.

    Inherits all core tracking functionality from the mjlab ``MotionCommand``
    and adds:
      - multi-motion loading with kick-leg labels
      - soccer-ball position computation (arc placement)
      - target-point / target-destination logic
      - blind-zone simulation
      - kick-contact tracking
    """

    cfg: SoccerMotionCommandCfg
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: SoccerMotionCommandCfg, env: ManagerBasedRlEnv):
        # Bypass MotionCommand.__init__ and go directly to CommandTerm.__init__
        from mjlab.managers.command_manager import CommandTerm
        CommandTerm.__init__(self, cfg, env)

        self.robot: Entity = env.scene[cfg.entity_name]
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        # Try to get soccer-ball entity
        self.soccer_ball: Entity | None = None
        if hasattr(env.scene, "entities") and "soccer_ball" in env.scene.entities:
            self.soccer_ball = env.scene["soccer_ball"]

        # Multi-motion loader — fall back to singular motion_file for
        # compatibility with play.py's --motion-file CLI.
        motion_files = list(cfg.motion_files) if cfg.motion_files else []
        if not motion_files and cfg.motion_file:
            motion_files = [cfg.motion_file]
        self.motion = MultiMotionLoader(motion_files, self.body_indexes, device=self.device)
        kick_leg_to_id = {"left": 0, "right": 1}
        self._kick_leg_id_to_name = {v: k for k, v in kick_leg_to_id.items()}
        self._kick_leg_id_to_name[-1] = "unknown"
        self.motion_kick_leg = torch.full(
            (self.motion.num_files,), -1, dtype=torch.int8, device=self.device
        )
        for idx, label in enumerate(self.motion.kick_leg_labels):
            normalized = label.lower() if isinstance(label, str) else None
            if normalized in kick_leg_to_id:
                self.motion_kick_leg[idx] = kick_leg_to_id[normalized]

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        if self.motion.num_files > 1:
            self.motion_idx = torch.randint(
                0, self.motion.num_files, (self.num_envs,), dtype=torch.long, device=self.device
            )
        self.motion_length[:] = self.motion.file_lengths[self.motion_idx]

        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device
        )
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device
        )
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Adaptive sampling
        self.bin_count = int(self.motion.time_step_total // (1 / env.step_dt)) + 1
        self.bin_failed_count = torch.zeros(
            (self.motion.num_files, self.bin_count), dtype=torch.float, device=self.device
        )
        self._current_bin_failed = torch.zeros_like(self.bin_failed_count)
        self.kernel = torch.tensor(
            [cfg.adaptive_lambda**i for i in range(cfg.adaptive_kernel_size)],
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()

        # Metrics
        for key in (
            "error_anchor_pos", "error_anchor_rot", "error_anchor_lin_vel",
            "error_anchor_ang_vel", "error_body_pos", "error_body_rot",
            "error_joint_pos", "error_joint_vel",
        ):
            self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

        # Soccer-specific state
        self.target_point_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.soccer_ball_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_destination_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.initial_target_point_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # Blind-zone
        self.blind_distance_min = torch.zeros(self.num_envs, device=self.device)
        self.blind_distance_max = torch.zeros(self.num_envs, device=self.device)
        self.last_visible_target_point_base = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_in_blind_zone = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Destination config
        self.destination_height = 0.11
        self.destination_center = torch.tensor([0.0, -5.0, 0.11], device=self.device)
        self.destination_length = 1.0
        self.destination_width = 0.5

        # Curve offset
        self.curve_radius_offset = torch.zeros(self.num_envs, device=self.device)
        self._radius_offset_min: float | None = None
        self._radius_offset_max: float | None = None
        curve_cfg = cfg.curve_offset_range or SoccerCurveOffsetCfg()
        if isinstance(curve_cfg.radius, (list, tuple)) and len(curve_cfg.radius) >= 2:
            self._radius_offset_min = float(curve_cfg.radius[0])
            self._radius_offset_max = float(curve_cfg.radius[1])
        elif curve_cfg.radius is not None:
            r = float(curve_cfg.radius)
            self._radius_offset_min = r
            self._radius_offset_max = r
        self._target_arc_angle = float(curve_cfg.arc_angle)
        self._target_height = float(curve_cfg.height)

        # Ghost model for visualization (created lazily on first use)
        self._ghost_model = None
        self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

        # Kick contact tracker
        self._state_prefix = "_motion"
        self.kick_contact_tracker = KickContactTracker(env, self._state_prefix)

        # Initial resample for all envs
        all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._sample_soccer_offset(all_env_ids)
        self._compute_soccer_ball_positions(all_env_ids)
        self._update_soccer_ball(all_env_ids)
        self._update_target_points(all_env_ids)
        self._update_destination_points(all_env_ids)

    # ── properties ─────────────────────────────────────────────────────

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.motion_idx, self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.motion_idx, self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.motion_idx, self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.motion_idx, self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.motion_idx, self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.motion_idx, self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body_pos_w[self.motion_idx, self.time_steps, self.motion_anchor_body_index]
            + self._env.scene.env_origins
        )

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.motion_idx, self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.motion_idx, self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.motion_idx, self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_pelvis_pos_w(self) -> torch.Tensor:
        pelvis_idx = self.robot.body_names.index("pelvis")
        return self.robot.data.body_link_pos_w[:, pelvis_idx]

    @property
    def robot_pelvis_quat_w(self) -> torch.Tensor:
        pelvis_idx = self.robot.body_names.index("pelvis")
        return self.robot.data.body_link_quat_w[:, pelvis_idx]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

    @property
    def kick_leg(self) -> torch.Tensor:
        return self.motion_kick_leg[self.motion_idx]

    @property
    def kick_leg_name(self) -> list[str]:
        ids = self.motion_kick_leg[self.motion_idx].tolist()
        return [self._kick_leg_id_to_name.get(i, "unknown") for i in ids]

    # ── soccer helpers ──────────────────────────────────────────────────

    def _to_env_id_tensor(self, env_ids) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(self.device, dtype=torch.long)
        return torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)

    def _sample_soccer_offset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        if self._radius_offset_min is not None and self._radius_offset_max is not None:
            span = self._radius_offset_max - self._radius_offset_min
            if abs(span) < 1e-6:
                self.curve_radius_offset[env_ids] = self._radius_offset_min
            else:
                rand = torch.rand(env_ids.numel(), device=self.device)
                self.curve_radius_offset[env_ids] = self._radius_offset_min + rand * span

    def _compute_soccer_ball_positions(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return

        arc_limit = float(self._target_arc_angle)
        base_height = float(self._target_height)

        for i in range(env_ids.numel()):
            env_id = int(env_ids[i].item())
            motion_idx = int(self.motion_idx[env_id].item())
            motion_len = max(1, int(self.motion_length[env_id].item()))

            first_anchor = self.motion.get_first_frame_anchor_pos(
                motion_idx, self.motion_anchor_body_index
            )
            last_anchor = self.motion.get_last_frame_anchor_pos(
                motion_idx, self.motion_anchor_body_index, motion_len
            )

            radius_vec = last_anchor[:2] - first_anchor[:2]
            radius_sq = torch.dot(radius_vec, radius_vec)

            radius = torch.sqrt(radius_sq) if float(radius_sq) > 1e-12 else torch.tensor(0.0, device=self.device)
            base_direction = (
                radius_vec / radius
                if float(radius_sq) > 1e-12
                else torch.tensor([1.0, 0.0], device=self.device)
            )

            if arc_limit > 0.0 and float(radius_sq) > 1e-12:
                base_angle = torch.atan2(radius_vec[1], radius_vec[0])
                angle_offset = sample_uniform(-arc_limit, arc_limit, (1,), device=self.device).squeeze(0)
                new_angle = base_angle + angle_offset
                direction = torch.stack((torch.cos(new_angle), torch.sin(new_angle)))
            else:
                direction = base_direction

            radius = torch.clamp(radius + self.curve_radius_offset[env_id], min=0.0)
            target_xy = first_anchor[:2] + radius * direction

            ball_pos = self.soccer_ball_pos.new_empty(3)
            ball_pos[:2] = target_xy
            ball_pos[2] = base_height
            self.soccer_ball_pos[env_id] = ball_pos

    def _update_target_points(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.target_point_pos[env_ids] = self.soccer_ball_pos[env_ids]
        self.initial_target_point_pos[env_ids] = self.soccer_ball_pos[env_ids].clone()

    def _update_target_points_from_sim(self) -> None:
        """Read soccer-ball position from simulation each step."""
        if self.soccer_ball is None:
            return
        env_origins = getattr(self._env.scene, "env_origins", None)
        if env_origins is None:
            return
        ball_world_pos = self.soccer_ball.data.root_link_pos_w
        self.soccer_ball_pos = ball_world_pos - env_origins
        self.target_point_pos = self.soccer_ball_pos.clone()

    def _update_destination_points(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        rand_x = (torch.rand(env_ids.numel(), device=self.device) - 0.5) * self.destination_length
        rand_y = (torch.rand(env_ids.numel(), device=self.device) - 0.5) * self.destination_width
        destination = self.destination_center.expand(env_ids.numel(), -1) + torch.stack(
            [rand_x, rand_y, torch.zeros_like(rand_x)], dim=1
        )
        self.target_destination_pos[env_ids] = destination

    def _update_soccer_ball(self, env_ids: torch.Tensor) -> None:
        if self.soccer_ball is None or env_ids.numel() == 0:
            return
        env_origins = getattr(self._env.scene, "env_origins", None)
        if env_origins is None:
            return

        ball_pos = self.soccer_ball_pos[env_ids] + env_origins[env_ids]
        ball_quat = ball_pos.new_zeros((env_ids.numel(), 4))
        ball_quat[:, 0] = 1.0

        if self.cfg.enable_soccer_ball_init_vel:
            lin_vel_cfg = self.cfg.soccer_ball_init_lin_vel_range or {}
            lin_vel_ranges = torch.tensor(
                [lin_vel_cfg.get(k, (0.0, 0.0)) for k in ["x", "y", "z"]],
                device=self.device,
            )
            ball_lin_vel = sample_uniform(
                lin_vel_ranges[:, 0], lin_vel_ranges[:, 1], (env_ids.numel(), 3), device=self.device
            )
        else:
            ball_lin_vel = ball_pos.new_zeros((env_ids.numel(), 3))

        ball_ang_vel = ball_pos.new_zeros((env_ids.numel(), 3))
        ball_state = torch.cat([ball_pos, ball_quat, ball_lin_vel, ball_ang_vel], dim=-1)
        self.soccer_ball.write_root_state_to_sim(ball_state, env_ids=env_ids)
        self.soccer_ball.reset(env_ids=env_ids)

    # ── sampling ────────────────────────────────────────────────────────

    def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
        motion_indices = torch.randint(
            0, self.motion.num_files, (len(env_ids),), device=self.device
        )
        self.motion_idx[env_ids] = motion_indices
        self.motion_length[env_ids] = self.motion.file_lengths[motion_indices]
        time_phase = torch.zeros(len(env_ids), device=self.device)
        self.time_steps[env_ids] = (time_phase * (self.motion_length[env_ids].float() - 1)).long()

    def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        episode_failed = self._env.termination_manager.terminated[env_ids]
        if isinstance(episode_failed, torch.Tensor):
            episode_failed = episode_failed.to(device=self.device, dtype=torch.bool)
        else:
            episode_failed = torch.tensor(episode_failed, dtype=torch.bool, device=self.device)

        self._current_bin_failed.zero_()

        if torch.any(episode_failed):
            failed_mask = episode_failed
            failed_motion_idx = self.motion_idx[env_ids][failed_mask]
            failed_lengths = self.motion_length[env_ids][failed_mask].clamp(min=1).float()
            failed_steps = self.time_steps[env_ids][failed_mask].float()
            failed_phase = failed_steps / (failed_lengths - 1.0 + 1e-6)
            failed_bins = torch.clamp((failed_phase * self.bin_count).long(), 0, self.bin_count - 1)
            flat_idx = failed_motion_idx * self.bin_count + failed_bins
            flat_size = int(self.motion.num_files * self.bin_count)

            flat_counts = torch.zeros(flat_size, dtype=self._current_bin_failed.dtype, device=self.device)
            if flat_idx.numel() > 0:
                flat_idx = flat_idx.to(self.device).long()
                ones = torch.ones_like(flat_idx, dtype=flat_counts.dtype, device=self.device)
                flat_counts = flat_counts.scatter_add(0, flat_idx, ones.float())

            self._current_bin_failed[:] = flat_counts.float().view(self.motion.num_files, self.bin_count)

        M = max(1, int(self.motion.num_files))
        B = max(1, int(self.bin_count))
        uniform_per_pair = self.cfg.adaptive_uniform_ratio / float(M * B)
        probs = self.bin_failed_count + self._current_bin_failed + uniform_per_pair
        probs = torch.nn.functional.pad(
            probs.unsqueeze(1), (0, self.cfg.adaptive_kernel_size - 1), mode="replicate"
        )
        probs = torch.nn.functional.conv1d(probs, self.kernel.view(1, 1, -1)).squeeze(1)

        probs = probs.view(-1)
        probs = probs / (probs.sum() + 1e-12)

        sampled_flat = torch.multinomial(probs, len(env_ids), replacement=True)
        sampled_motion = sampled_flat // self.bin_count
        sampled_bins = sampled_flat % self.bin_count

        self.motion_idx[env_ids] = sampled_motion
        self.motion_length[env_ids] = self.motion.file_lengths[self.motion_idx[env_ids]]
        rand_offset = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device).float()
        sampled_phase = (sampled_bins.float() + rand_offset) / float(self.bin_count)
        self.time_steps[env_ids] = (sampled_phase * (self.motion_length[env_ids].float() - 1)).long()

    # ── core overrides ──────────────────────────────────────────────────

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        env_ids = self._to_env_id_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        self._sample_soccer_offset(env_ids)

        sampling_strategy = str(self.cfg.sampling_strategy).lower()
        if sampling_strategy == "adaptive":
            self._adaptive_sampling(env_ids)
        elif sampling_strategy == "uniform":
            self._uniform_sampling(env_ids)
        else:
            raise ValueError(f"Unsupported sampling_strategy: {self.cfg.sampling_strategy}")

        self._compute_soccer_ball_positions(env_ids)
        self._update_soccer_ball(env_ids)
        self._update_target_points(env_ids)
        self._update_destination_points(env_ids)

        # Blind-zone resample
        bmin_low, bmin_high = self.cfg.blind_distance_min_range
        bmax_low, bmax_high = self.cfg.blind_distance_max_range
        n = env_ids.numel()
        self.blind_distance_min[env_ids] = bmin_low + torch.rand(n, device=self.device) * (bmin_high - bmin_low)
        self.blind_distance_max[env_ids] = bmax_low + torch.rand(n, device=self.device) * (bmax_high - bmax_low)
        self.is_in_blind_zone[env_ids] = False
        self.last_visible_target_point_base[env_ids] = 0.0

        # Root state sampling and sim write (from MotionCommand)
        root_pos = self.body_pos_w[env_ids, 0].clone()
        root_ori = self.body_quat_w[env_ids, 0].clone()
        root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
        root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

        range_list = [
            self.cfg.pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori = quat_mul(orientations_delta, root_ori)

        range_list = [
            self.cfg.velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]

        joint_pos = self.joint_pos[env_ids].clone()
        joint_vel = self.joint_vel[env_ids]

        joint_pos += sample_uniform(
            lower=self.cfg.joint_position_range[0],
            upper=self.cfg.joint_position_range[1],
            size=joint_pos.shape,
            device=joint_pos.device,
        )

        soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.reset(env_ids=env_ids)

        # Mark resample for kick tracker
        flag_name = f"{self._state_prefix}_motion_resampled"
        resample_flags = getattr(self._env, flag_name, None)
        if resample_flags is None or resample_flags.shape[0] != self.num_envs:
            resample_flags = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            resample_flags = resample_flags.to(device=self.device, dtype=torch.bool)
        resample_flags[env_ids] = True
        setattr(self._env, flag_name, resample_flags)

    def _update_metrics(self) -> None:
        self.metrics["error_anchor_pos"] = torch.norm(
            self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
        )
        self.metrics["error_anchor_rot"] = quat_error_magnitude(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        self.metrics["error_anchor_lin_vel"] = torch.norm(
            self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
        )
        self.metrics["error_anchor_ang_vel"] = torch.norm(
            self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
        )
        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)
        self.metrics["error_joint_pos"] = torch.norm(
            self.joint_pos - self.robot_joint_pos, dim=-1
        )
        self.metrics["error_joint_vel"] = torch.norm(
            self.joint_vel - self.robot_joint_vel, dim=-1
        )

    def update_relative_body_poses(self) -> None:
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat.clone()
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(
            delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
        )

    def _update_command(self) -> None:
        self.kick_contact_tracker.begin_step(self)

        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion_length)[0]
        if env_ids.numel() > 0:
            self._resample_command(env_ids)

        self._update_target_points_from_sim()

        # Keep initial target point frozen after contact
        if hasattr(self, "kick_contact_tracker"):
            contact_awarded = self.kick_contact_tracker.get_contact_awarded()
            no_contact_mask = ~contact_awarded
            if torch.any(no_contact_mask):
                self.initial_target_point_pos[no_contact_mask] = self.target_point_pos[no_contact_mask]

        self.update_relative_body_poses()

        if self.cfg.sampling_strategy == "adaptive":
            self.bin_failed_count = (
                self.cfg.adaptive_alpha * self._current_bin_failed
                + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
            )
            self._current_bin_failed.zero_()


@dataclass
class SoccerCurveOffsetCfg:
    """Parameters for placing the ball on an arc relative to the kick foot."""

    radius: tuple[float, float] = (-0.25, 0.25)
    arc_angle: float = math.pi / 18.0
    height: float = 0.11


@dataclass(kw_only=True)
class SoccerMotionCommandCfg(CommandTermCfg):
    """Configuration for the soccer motion command."""

    motion_files: list[str] = field(default_factory=list)
    motion_file: str = ""  # Singular alias for compatibility with play.py --motion-file
    anchor_body_name: str = ""
    body_names: tuple[str, ...] = ()
    entity_name: str = "robot"
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    sampling_strategy: Literal["adaptive", "uniform"] = "uniform"
    adaptive_kernel_size: int = 3
    adaptive_lambda: float = 0.1
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.4
    curve_offset_range: SoccerCurveOffsetCfg | None = None
    enable_soccer_ball_init_vel: bool = False
    soccer_ball_init_lin_vel_range: dict[str, tuple[float, float]] | None = None
    blind_distance_min_range: tuple[float, float] = (0.3, 0.5)
    blind_distance_max_range: tuple[float, float] = (1.5, 2.0)

    @dataclass
    class VizCfg:
        mode: Literal["ghost", "frames"] = "ghost"
        ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> SoccerMotionCommand:
        return SoccerMotionCommand(self, env)
