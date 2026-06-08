"""Soccer-specific observation functions for the soccer task.

Ported from HumanoidSoccer (arXiv-2602.05310v1) and adapted for mjlab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_apply, quat_inv

from .commands import SoccerMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _get_command(env: ManagerBasedRlEnv, command_name: str) -> SoccerMotionCommand:
    cmd = env.command_manager.get_term(command_name)
    if cmd is None:
        raise RuntimeError(f"command '{command_name}' not found")
    return cmd


def _get_target_point_world(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = _get_command(env, command_name)
    target_local = command.target_point_pos
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        return target_local + env_origins
    return target_local


def target_point_pos_local(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Target point (ball) position in robot pelvis frame."""
    command = _get_command(env, command_name)
    target_world = _get_target_point_world(env, command_name)
    delta = target_world - command.robot_pelvis_pos_w
    return quat_apply(quat_inv(command.robot_pelvis_quat_w), delta)


def target_point_pos_first_frame(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Target point in robot base frame, frozen at first frame of each episode."""
    cache_name = f"_{command_name}_target_point_cache"
    target_local = target_point_pos_local(env, command_name)

    cache = getattr(env, cache_name, None)
    if cache is None or cache.shape[0] != env.num_envs:
        cache = target_local.clone()
        setattr(env, cache_name, cache)

    step_buf = env.episode_length_buf
    first_step_mask = step_buf == 0
    if torch.any(first_step_mask):
        cache = getattr(env, cache_name)
        cache[first_step_mask] = target_local[first_step_mask]
        setattr(env, cache_name, cache)

    return getattr(env, cache_name)


def blind_zone_target_point_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Target point with blind-zone simulation.

    If robot-ball (x,y) distance is outside [blind_distance_min, blind_distance_max],
    returns the last visible position to emulate limited visibility.
    """
    command = _get_command(env, command_name)
    target_base = target_point_pos_local(env, command_name)

    target_world = _get_target_point_world(env, command_name)
    robot_pos = command.robot_pelvis_pos_w
    distance_xy = torch.norm(target_world[:, :2] - robot_pos[:, :2], dim=-1)

    in_visible_range = (distance_xy >= command.blind_distance_min) & (distance_xy <= command.blind_distance_max)

    if torch.any(in_visible_range):
        command.last_visible_target_point_base[in_visible_range] = target_base[in_visible_range]
        command.is_in_blind_zone[in_visible_range] = False

    command.is_in_blind_zone[~in_visible_range] = True

    return torch.where(
        command.is_in_blind_zone.unsqueeze(-1),
        command.last_visible_target_point_base,
        target_base,
    )


def target_destination_pos_local(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Target destination in robot pelvis frame."""
    command = _get_command(env, command_name)
    if not hasattr(command, "target_destination_pos"):
        return torch.zeros(env.num_envs, 3, device=env.device)

    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        target_world = command.target_destination_pos + env_origins
    else:
        target_world = command.target_destination_pos

    delta = target_world - command.robot_pelvis_pos_w
    return quat_apply(quat_inv(command.robot_pelvis_quat_w), delta)


def target_destination_pos_local_first_frame(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Target destination in base frame, frozen at first frame of each episode."""
    cache_name = f"_{command_name}_target_destination_local_cache"
    target_local = target_destination_pos_local(env, command_name)

    cache = getattr(env, cache_name, None)
    if cache is None or cache.shape[0] != env.num_envs:
        cache = target_local.clone()
        setattr(env, cache_name, cache)

    step_buf = env.episode_length_buf
    first_step_mask = step_buf == 0
    if torch.any(first_step_mask):
        cache = getattr(env, cache_name)
        cache[first_step_mask] = target_local[first_step_mask]
        setattr(env, cache_name, cache)

    return getattr(env, cache_name)
