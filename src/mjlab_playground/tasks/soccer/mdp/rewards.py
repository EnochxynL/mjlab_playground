"""Soccer-specific reward functions ported from HumanoidSoccer (arXiv-2602.05310v1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_inv

from .commands import SoccerMotionCommand
from .observations import _get_target_point_world

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg


def _get_command(env: ManagerBasedRlEnv, command_name: str) -> SoccerMotionCommand:
    cmd = env.command_manager.get_term(command_name)
    if cmd is None:
        raise RuntimeError(f"command '{command_name}' not found")
    return cmd


def _resolve_body_ids(env: ManagerBasedRlEnv, foot_cfg: SceneEntityCfg) -> list[int]:
    ids = foot_cfg.body_ids
    if isinstance(ids, slice):
        robot = env.scene[foot_cfg.name]
        names = list(robot.body_names)
        return [names.index(n) for n in foot_cfg.body_names]
    return list(ids)


def _get_kick_tracker(command: SoccerMotionCommand):
    tracker = getattr(command, "kick_contact_tracker", None)
    if tracker is None:
        raise RuntimeError("MotionCommand missing kick_contact_tracker")
    return tracker


# ── soccer rewards ────────────────────────────────────────────────────────


def target_point_proximity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str = "motion",
) -> torch.Tensor:
    """Reward proximity to the target point (ball). Frozen at first kick contact."""
    command = _get_command(env, command_name)
    tracker = _get_kick_tracker(command)

    base_xy = command.robot_anchor_pos_w[..., :2]
    target = _get_target_point_world(env, command_name).to(device=base_xy.device, dtype=base_xy.dtype)
    diff_xy = base_xy - target[..., :2]
    error = torch.sum(diff_xy * diff_xy, dim=-1)
    proximity_reward = torch.exp(-error / std**2)

    contact_awarded = tracker.get_contact_awarded()
    frozen_reward = tracker.get_frozen_proximity_reward()

    new_kick_mask = contact_awarded & (frozen_reward == 0.0)
    if torch.any(new_kick_mask):
        new_kick_ids = torch.nonzero(new_kick_mask, as_tuple=False).squeeze(-1)
        tracker.freeze_proximity_reward(new_kick_ids, proximity_reward[new_kick_ids])
        frozen_reward = tracker.get_frozen_proximity_reward()

    return torch.where(contact_awarded, frozen_reward, proximity_reward)


def target_point_contact(
    env: ManagerBasedRlEnv,
    horizontal_force_threshold: float = 0.0,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    foot_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """One-shot reward for first valid kick contact with correct foot."""
    command = _get_command(env, command_name)
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if not torch.any(event.new_contact):
        return reward

    reward_scale = torch.zeros_like(reward)
    correct_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if foot_cfg is not None:
        foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
        if foot_info.env_ids.numel() > 0:
            valid_expectation = foot_info.expected >= 0
            correct = (foot_info.sides == foot_info.expected) & valid_expectation
            reward_scale[foot_info.env_ids] = correct.to(reward_scale.dtype)
            correct_mask[foot_info.env_ids] = correct

    tracker.record_expected_success(event.new_contact, correct_mask)
    return event.new_contact.to(reward.dtype) * reward_scale


def sideways_kick(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward foot lateral swing in the expected direction at contact moment.

    Left kick expects foot velocity along local -Y; right kick expects +Y.
    """
    if foot_cfg is None:
        raise ValueError("sideways_kick requires foot_cfg")

    command = _get_command(env, command_name)
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if not torch.any(event.new_contact):
        return reward

    foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
    if foot_info.env_ids.numel() == 0:
        return reward

    robot = env.scene[foot_cfg.name]
    foot_vel_w = robot.data.body_link_lin_vel_w[foot_info.env_ids, foot_info.body_indices]
    foot_quat_w = robot.data.body_link_quat_w[foot_info.env_ids, foot_info.body_indices]

    vel_local = quat_apply(quat_inv(foot_quat_w), foot_vel_w)
    vel_norm = torch.norm(vel_local, dim=-1)

    expected_leg = foot_info.expected.to(device=env.device, dtype=torch.int8)
    desired_sign = torch.where(expected_leg == 0, -1.0, 1.0)

    directional_component = vel_local[:, 1] * desired_sign
    axis_component = torch.clamp(directional_component, min=0.0)

    alignment = torch.where(vel_norm > 1e-6, axis_component / vel_norm, torch.zeros_like(vel_norm))
    reward[foot_info.env_ids] = alignment.to(reward.dtype)

    valid_expectation = expected_leg >= 0
    correct_foot = (foot_info.sides == foot_info.expected) & valid_expectation
    wrong_mask = ~correct_foot
    if torch.any(wrong_mask):
        reward[foot_info.env_ids[wrong_mask]] = 0.0

    return reward


def ball_velocity_direction_alignment(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    velocity_threshold: float = 0.1,
    horizontal_force_threshold: float = 0.0,
    ball_sensor_name: str = "soccer_ball_contact",
    foot_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward alignment between ball velocity and target-to-destination direction.

    Active only for a short window after expected-foot contact.
    """
    command = _get_command(env, command_name)
    soccer_ball = env.scene["soccer_ball"]
    vel = soccer_ball.data.root_link_lin_vel_w
    vel_xy = vel[:, :2]
    vel_xy_norm = torch.norm(vel_xy, dim=-1, keepdim=True)
    vel_norm = torch.norm(vel, dim=-1, keepdim=True)

    direction = command.target_destination_pos - command.initial_target_point_pos
    direction_xy = direction[:, :2]
    dir_norm = torch.norm(direction_xy, dim=-1, keepdim=True)

    valid_mask = (
        (vel_norm.squeeze(-1) > velocity_threshold)
        & (vel_xy_norm.squeeze(-1) > 1e-6)
        & (dir_norm.squeeze(-1) > 1e-6)
    )

    avg_angle = torch.tensor(0.0, device=env.device, dtype=torch.float32)
    if torch.any(valid_mask):
        dir_unit_valid = direction_xy[valid_mask] / dir_norm[valid_mask]
        vel_unit_valid = vel_xy[valid_mask] / vel_xy_norm[valid_mask]
        cos_theta_valid = torch.sum(vel_unit_valid * dir_unit_valid, dim=-1).clamp(-1.0, 1.0)
        theta_valid = torch.acos(cos_theta_valid)
        avg_angle = theta_valid.mean()
    if hasattr(command, "metrics"):
        command.metrics["ball_velocity_dir_alignment_angle"] = torch.full(
            (env.num_envs,), avg_angle.item(), device=env.device, dtype=torch.float32
        )

    timer_name = f"_{command_name}_dir_align_timer"
    timer = getattr(env, timer_name, None)
    if timer is None or timer.shape[0] != env.num_envs:
        timer = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    else:
        timer = timer.to(device=env.device, dtype=torch.int32)

    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

    if torch.any(event.new_contact) and foot_cfg is not None:
        foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
        if foot_info.env_ids.numel() > 0:
            valid_expectation = foot_info.expected >= 0
            correct_foot = (foot_info.sides == foot_info.expected) & valid_expectation
            correct_env_ids = foot_info.env_ids[correct_foot]
            if correct_env_ids.numel() > 0:
                timer[correct_env_ids] = 5

    speed_valid = (vel_xy_norm.squeeze(-1) > 1e-6) & (dir_norm.squeeze(-1) > 1e-6)
    active_mask = (timer > 0) & speed_valid

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if torch.any(active_mask):
        dir_unit = direction_xy[active_mask] / dir_norm[active_mask]
        vel_unit = vel_xy[active_mask] / vel_xy_norm[active_mask]
        cos_theta = torch.sum(vel_unit * dir_unit, dim=-1).clamp(-1.0, 1.0)
        error = torch.acos(cos_theta) ** 2
        reward[active_mask] = torch.exp(-error / (std**2))

    timer = torch.where(timer > 0, timer - 1, timer)
    setattr(env, timer_name, timer)
    return reward


def ball_speed_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    velocity_threshold: float = 0.1,
    horizontal_force_threshold: float = 0.0,
    ball_sensor_name: str = "soccer_ball_contact",
    foot_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward ball speed within a short window after expected-foot contact."""
    command = _get_command(env, command_name)
    soccer_ball = env.scene["soccer_ball"]
    vel = soccer_ball.data.root_link_lin_vel_w
    speed_xy = torch.norm(vel[:, :2], dim=-1)

    timer_name = f"_{command_name}_speed_timer"
    timer = getattr(env, timer_name, None)
    if timer is None or timer.shape[0] != env.num_envs:
        timer = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    else:
        timer = timer.to(device=env.device, dtype=torch.int32)

    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

    if torch.any(event.new_contact) and foot_cfg is not None:
        foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
        if foot_info.env_ids.numel() > 0:
            valid_expectation = foot_info.expected >= 0
            correct_foot = (foot_info.sides == foot_info.expected) & valid_expectation
            correct_env_ids = foot_info.env_ids[correct_foot]
            if correct_env_ids.numel() > 0:
                timer[correct_env_ids] = 5

    speed_valid = speed_xy > 1e-6
    active_mask = (timer > 0) & speed_valid

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if torch.any(active_mask):
        reward[active_mask] = 1.0 - torch.exp(-(speed_xy[active_mask] ** 2) / (std**2))

    timer = torch.where(timer > 0, timer - 1, timer)
    setattr(env, timer_name, timer)
    return reward


def ball_z_speed_penalty_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    velocity_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize excessive vertical ball speed in a short post-activation window."""
    soccer_ball = env.scene["soccer_ball"]
    vel = soccer_ball.data.root_link_lin_vel_w
    z_speed = vel[:, 2]
    speed = torch.norm(vel, dim=-1)

    valid_mask = speed > velocity_threshold

    timer_name = f"_{command_name}_z_speed_timer"
    prev_name = f"_{command_name}_z_speed_prev"

    timer = getattr(env, timer_name, None)
    if timer is None or timer.shape[0] != env.num_envs:
        timer = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    else:
        timer = timer.to(device=env.device, dtype=torch.int32)

    prev_valid = getattr(env, prev_name, None)
    if prev_valid is None or prev_valid.shape[0] != env.num_envs:
        prev_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        prev_valid = prev_valid.to(device=env.device, dtype=torch.bool)

    rising_mask = valid_mask & (~prev_valid)
    timer[rising_mask] = 5
    active_mask = timer > 0

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if torch.any(active_mask):
        scale = std if std > 0 else 1.0
        reward[active_mask] = torch.tanh(torch.abs(z_speed[active_mask]) / (scale + 1e-8))

    timer = torch.where(timer > 0, timer - 1, timer)
    setattr(env, timer_name, timer)
    setattr(env, prev_name, valid_mask.to(dtype=torch.bool))
    return reward


def pelvis_orientation(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Penalize pelvis pitch/roll tilt to keep the robot upright."""
    command = _get_command(env, command_name)
    robot = env.scene[command.cfg.entity_name]
    gravity_vec_w = robot.data.gravity_vec_w
    pelvis_proj_gravity = quat_apply_inverse(command.robot_pelvis_quat_w, gravity_vec_w)
    return torch.sum(torch.square(pelvis_proj_gravity[:, :2]), dim=1)


def foot_distance(
    env: ManagerBasedRlEnv,
    threshold: float,
    std: float,
    foot_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Encourage minimum separation between feet to avoid crossing."""
    if foot_cfg is None:
        raise ValueError("foot_distance requires foot_cfg")
    robot = env.scene[foot_cfg.name]
    body_ids = _resolve_body_ids(env, foot_cfg)
    left_pos = robot.data.body_link_pos_w[:, body_ids[0]]
    right_pos = robot.data.body_link_pos_w[:, body_ids[1]]
    distance = torch.norm(left_pos - right_pos, dim=1)
    return torch.where(
        distance >= threshold,
        torch.tensor(1.0, device=distance.device),
        torch.exp(-((distance / threshold - 1) ** 2) / (std**2)),
    )


# ── tracking rewards (extended from mjlab tracking) ─────────────────────


def motion_relative_foot_position_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    foot_body_names: list[str] | None = None,
) -> torch.Tensor:
    """Tracking error for foot body positions, relative to motion reference."""
    if foot_body_names is None:
        foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    command = _get_command(env, command_name)
    body_indexes = [
        i for i, name in enumerate(command.cfg.body_names) if name in foot_body_names
    ]
    error = torch.sum(
        torch.square(
            command.body_pos_relative_w[:, body_indexes]
            - command.robot_body_pos_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def waist_action_rate_l2_clip(
    env: ManagerBasedRlEnv,
    waist_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize action rate on waist joints only."""
    if waist_cfg is None:
        raise ValueError("waist_action_rate_l2_clip requires waist_cfg")
    robot = env.scene[waist_cfg.name]
    idx = torch.as_tensor(
        robot.find_joints(waist_cfg.joint_names, preserve_order=True)[0],
        device=env.device,
    )
    return torch.sum(
        torch.square(env.action_manager.action[:, idx] - env.action_manager.prev_action[:, idx]),
        dim=1,
    ).clamp(max=100.0)
