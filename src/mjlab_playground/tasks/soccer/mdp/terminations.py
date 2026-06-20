from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from mjlab.utils.lab_api import math as math_utils  # MJLab: isaaclab.utils.math → mjlab.utils.lab_api.math

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv  # MJLab: ManagerBasedRLEnv → ManagerBasedRlEnv

from mjlab.entity import Entity  # MJLab: Articulation | RigidObject → Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg  # MJLab: isaaclab.managers → mjlab.managers

from .commands_multi_motion_soccer import MotionCommand  # MJLab: IsaacLab imports from commands_multi_motion_soccer

from .rewards import _get_body_indexes  # MJLab: module path differs from IsaacLab


def bad_anchor_pos(env: ManagerBasedRlEnv, command_name: str, threshold: float) -> torch.Tensor:  # MJLab: ManagerBasedRLEnv → ManagerBasedRlEnv
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRlEnv, command_name: str, threshold: float) -> torch.Tensor:  # MJLab: ManagerBasedRLEnv → ManagerBasedRlEnv
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float  # MJLab: ManagerBasedRLEnv → ManagerBasedRlEnv
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]  # MJLab: RigidObject | Articulation → Entity

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.gravity_vec_w)  # MJLab: GRAVITY_VEC_W → gravity_vec_w

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.gravity_vec_w)  # MJLab: GRAVITY_VEC_W → gravity_vec_w

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRlEnv, command_name: str, threshold: float, body_names: list[str] | None = None  # MJLab: ManagerBasedRLEnv → ManagerBasedRlEnv
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRlEnv, command_name: str, threshold: float, body_names: list[str] | None = None  # MJLab: ManagerBasedRLEnv → ManagerBasedRlEnv
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)


def motion_finished(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:  # MJLab: ManagerBasedRLEnv → ManagerBasedRlEnv
    command: MotionCommand = env.command_manager.get_term(command_name)
    last_step = (command.motion_length - 1).clamp(min=0)
    return command.time_steps >= last_step
