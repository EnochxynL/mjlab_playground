"""Booster T1 soccer environment factory functions.

Mirrors ``config/g1/env_cfgs.py`` but with T1 body/joint names.
"""

from __future__ import annotations

import math
from typing import Literal

from mjlab_playground.asset_zoo.robots import T1_ACTION_SCALE, get_t1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.observations import projected_gravity
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from ...mdp import commands_multi_motion_soccer as soccer_commands
from ...mdp import observations as soccer_obs
from ...mdp import rewards as soccer_rewards

from ...soccer_env_cfg import SOCCER_BALL_RADIUS, get_soccer_ball_cfg

# ── common setup shared by both stages ─────────────────────────────────────

# IsaacLab 的 `@configclass` 支持子类重定义父类字段 + mutable default，所以可以靠继承链自然叠加配置：
# ```
# G1FlatEnvCfg(TrackingEnvCfg)
# └─ G1FlatMotionEnvCfg(G1FlatEnvCfg)
#     └─ G1FlatProximityEnvCfg(G1FlatMotionEnvCfg)
#             └─ G1FlatKickEnvCfg(G1FlatProximityEnvCfg)
# ```
# 每一层在 `__post_init__` 里增量修改，继承链本身就是配置组合器。
# MJLab 的 `@dataclass` 不允许字段重定义，这个继承链断了。所以只能把同样的逻辑"展平"成一个工厂函数：调 `make_tracking_env_cfg()` 拿到裸配置，然后 _apply_common_soccer_config() 一次性把所有层级的修改叠上去，最后返回。
# 本质上 `_apply_common_soccer_config` 就是在模拟 `@configclass` 的继承叠加能力。

# ── T1 body names (G1 → T1 equivalents) ─────────────────────────────────────

# Full tracking body list (matches G1 set with T1 equivalents).

_T1_TRACKING_BODIES = (
    "Trunk",
    "Hip_Roll_Left",
    "Shank_Left",
    "left_foot_link",
    "Hip_Roll_Right",
    "Shank_Right",
    "right_foot_link",
    "Waist",
    "AL2",
    "AL3",
    "left_hand_link",
    "AR2",
    "AR3",
    "right_hand_link",
)

# Bodies without feet (used by motion_body_pos/ori rewards).
_T1_TRACKING_BODIES_NO_FEET = (
    "Trunk",
    "Hip_Roll_Left",
    "Shank_Left",
    "Hip_Roll_Right",
    "Shank_Right",
    "Waist",
    "AL2",
    "AL3",
    "left_hand_link",
    "AR2",
    "AR3",
    "right_hand_link",
)

_T1_FOOT_BODIES = ("left_foot_link", "right_foot_link")
_T1_EE_BODIES = (
    "left_foot_link",
    "right_foot_link",
    "left_hand_link",
    "right_hand_link",
)
_T1_WAIST_JOINTS = ("Waist",)


# ── common setup ────────────────────────────────────────────────────────────


def _apply_common_soccer_config(
    cfg: ManagerBasedRlEnvCfg,
    has_state_estimation: bool,
    play: bool,
    sampling_strategy: Literal["uniform", "adaptive"],
) -> tuple[SceneEntityCfg, SceneEntityCfg]:
    """Apply scene, physics, actions, commands, obs, terminations, events.

    Returns (foot_cfg, waist_cfg) for per-stage reward configuration.
    """

    # ── scene (robot + ball) ──────────────────────────────────────────

    cfg.scene.entities = {
        "robot": get_t1_robot_cfg(),
        "soccer_ball": get_soccer_ball_cfg(),
    }

    ball_contact = ContactSensorCfg(
        name="soccer_ball_contact",
        primary=ContactMatch(mode="body", pattern="soccer_ball", entity="soccer_ball"),
        secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
        history_length=4,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (ball_contact, self_collision_cfg)

    # ── physics ──────────────────────────────────────────────────────

    cfg.sim.mujoco.timestep = 0.005
    cfg.decimation = 4

    # ── actions ──────────────────────────────────────────────────────

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = T1_ACTION_SCALE

    # ── commands ─────────────────────────────────────────────────────

    motion_cfg = soccer_commands.MotionCommandCfg(  # MJLab: SoccerMotionCommandCfg → MotionCommandCfg
        entity_name="robot",
        anchor_body_name="Trunk",
        pelvis_body_name="Trunk",
        body_names=_T1_TRACKING_BODIES,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range={
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52),
            "pitch": (-0.52, 0.52),
            "yaw": (-0.78, 0.78),
        },
        joint_position_range=(-0.52, 0.52),
        sampling_strategy=sampling_strategy,
        # MJLab: SoccerCurveOffsetCfg → plain dict (IsaacLab uses dict-based curve_offset_range)
        curve_offset_range={  # MJLab: curve_offset_range is a dict in IsaacLab MotionCommandCfg
            "radius": (-0.25, 0.25),
            "arc_angle": math.pi / 9,
            "height": SOCCER_BALL_RADIUS,
        },
        blind_distance_min_range=(0.2, 0.8),
        blind_distance_max_range=(1.8, 2.5),
    )
    cfg.commands["motion"] = motion_cfg

    # ── observations ─────────────────────────────────────────────────
    #
    # IsaacLab actor obs: command, projected_gravity, motion_ref_ang_vel,
    #   base_ang_vel, joint_pos, joint_vel, actions,
    #   target_point_pos, target_destination_pos_local  → 160 dims.
    # MJLab tracking base uses motion_anchor_pos_b + motion_anchor_ori_b
    #   + base_lin_vel instead (166 dims).  We rebuild the actor terms
    #   here to match the IsaacLab structure so ONNX policies exported
    #   from IsaacLab can be played back.

    base_terms = cfg.observations["actor"].terms
    actor_terms = {
        "command": base_terms["command"],
        "projected_gravity": ObservationTermCfg(
            func=projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "motion_ref_ang_vel": ObservationTermCfg(
            func=soccer_obs.motion_anchor_ang_vel,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "base_ang_vel": base_terms["base_ang_vel"],
        "joint_pos": base_terms["joint_pos"],
        "joint_vel": base_terms["joint_vel"],
        "actions": base_terms["actions"],
        "target_point_pos": ObservationTermCfg(
            func=soccer_obs.constant_target_point_pos,
            params={"command_name": "motion"},
        ),
        "target_destination_pos_local": ObservationTermCfg(
            func=soccer_obs.target_destination_pos_local,
            params={"command_name": "motion"},
        ),
    }
    cfg.observations["actor"] = ObservationGroupCfg(
        terms=actor_terms,
        concatenate_terms=True,
        enable_corruption=not play,
    )

    # Critic keeps the MJLab base structure (no change needed — critic
    # is only used for training, not inference).
    critic_terms = dict(cfg.observations["critic"].terms)
    critic_terms["target_point_pos"] = ObservationTermCfg(
        func=soccer_obs.constant_target_point_pos,
        params={"command_name": "motion"},
    )
    critic_terms["target_destination_pos_local"] = ObservationTermCfg(
        func=soccer_obs.target_destination_pos_local,
        params={"command_name": "motion"},
    )
    cfg.observations["critic"] = ObservationGroupCfg(
        terms=critic_terms,
        concatenate_terms=True,
        enable_corruption=False,
    )

    # ── tracking reward weight adjustments (shared by both stages) ────

    cfg.rewards["motion_global_root_pos"].weight = 0.0
    cfg.rewards["motion_global_root_ori"].weight = 1.0
    cfg.rewards["motion_body_pos"].params["body_names"] = _T1_TRACKING_BODIES_NO_FEET
    cfg.rewards["motion_body_ori"].params["body_names"] = _T1_TRACKING_BODIES_NO_FEET

    # ── terminations ─────────────────────────────────────────────────

    cfg.terminations["anchor_pos"] = TerminationTermCfg(
        func=cfg.terminations["anchor_pos"].func,
        params={"command_name": "motion", "threshold": 0.25},
    )
    cfg.terminations["ee_body_pos"].params["body_names"] = _T1_EE_BODIES

    # ── events ───────────────────────────────────────────────────────

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        r"^(left|right)_foot\d+_collision$"
    )
    cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)

    # ── viewer ───────────────────────────────────────────────────────

    cfg.viewer.body_name = "Trunk"

    # ── play mode overrides ──────────────────────────────────────────

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.terminations = {}
        cfg.events.pop("push_robot", None)
        motion_cfg.pose_range = {}
        motion_cfg.velocity_range = {}
        motion_cfg.sampling_strategy = "uniform"

    # ── return SceneEntityCfg helpers for per-stage reward setup ─────

    foot_cfg = SceneEntityCfg("robot", body_names=_T1_FOOT_BODIES)
    waist_cfg = SceneEntityCfg("robot", joint_names=_T1_WAIST_JOINTS)
    return foot_cfg, waist_cfg


# ── Stage 2: kick-to-destination ─────────────────────────────────────────


def t1_soccer_destination_env_cfg(
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stage 2: flat ground, uniform sampling, tracking + kick rewards."""
    cfg = make_tracking_env_cfg()
    foot_cfg, waist_cfg = _apply_common_soccer_config(
        cfg, has_state_estimation, play, sampling_strategy="uniform",
    )

    # ── soccer-specific rewards ──────────────────────────────────────

    cfg.rewards["target_point_proximity"] = RewardTermCfg(
        func=soccer_rewards.target_point_proximity,
        weight=1.0,
        params={"std": 4.0, "command_name": "motion"},
    )
    cfg.rewards["target_point_contact"] = RewardTermCfg(
        func=soccer_rewards.target_point_contact,
        weight=50.0,
        params={
            "command_name": "motion",
            "ball_sensor_name": "soccer_ball_contact",
            "horizontal_force_threshold": 10,
            "foot_cfg": foot_cfg,
        },
    )
    cfg.rewards["sideways_kick"] = RewardTermCfg(
        func=soccer_rewards.sideways_kick,
        weight=50.0,
        params={
            "command_name": "motion",
            "ball_sensor_name": "soccer_ball_contact",
            "horizontal_force_threshold": 10,
            "foot_cfg": foot_cfg,
        },
    )
    cfg.rewards["ball_velocity_direction_alignment"] = RewardTermCfg(
        func=soccer_rewards.ball_velocity_direction_alignment,
        weight=30.0,
        params={
            "command_name": "motion",
            "std": 0.8,
            "velocity_threshold": 0.5,
            "ball_sensor_name": "soccer_ball_contact",
            "horizontal_force_threshold": 10,
            "foot_cfg": foot_cfg,
        },
    )
    cfg.rewards["ball_speed_reward"] = RewardTermCfg(
        func=soccer_rewards.ball_speed_reward,
        weight=10.0,
        params={
            "command_name": "motion",
            "std": 1.2,
            "velocity_threshold": 0.5,
            "ball_sensor_name": "soccer_ball_contact",
            "horizontal_force_threshold": 10,
            "foot_cfg": foot_cfg,
        },
    )
    cfg.rewards["ball_z_speed_penalty"] = RewardTermCfg(
        func=soccer_rewards.ball_z_speed_penalty_reward,
        weight=-0.0,
        params={"command_name": "motion", "std": 3, "velocity_threshold": 0.5},
    )
    cfg.rewards["pelvis_orientation"] = RewardTermCfg(
        func=soccer_rewards.pelvis_orientation,
        weight=-1.0,
        params={"command_name": "motion"},
    )
    cfg.rewards["foot_distance"] = RewardTermCfg(
        func=soccer_rewards.foot_distance,
        weight=0.2,
        params={"threshold": 0.24, "std": 0.5, "foot_cfg": foot_cfg},
    )
    cfg.rewards["motion_foot_pos"] = RewardTermCfg(
        func=soccer_rewards.motion_relative_foot_position_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 0.3,
            "foot_body_names": list(_T1_FOOT_BODIES),
        },
    )
    cfg.rewards["waist_action_rate_l2"] = RewardTermCfg(
        func=soccer_rewards.waist_action_rate_l2_clip,
        weight=-2.5e-1,
        params={"waist_cfg": waist_cfg},
    )

    return cfg


# ── Stage 1: motion-skill acquisition ─────────────────────────────────────


def t1_soccer_tracking_env_cfg(
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stage 1: flat ground, adaptive sampling, tracking rewards + soccer obs."""
    cfg = make_tracking_env_cfg()
    foot_cfg, waist_cfg = _apply_common_soccer_config(
        cfg, has_state_estimation, play, sampling_strategy="adaptive",
    )

    # Stage 1 uses tracking rewards only — no soccer-specific rewards added.
    # The tracking reward weights are already adjusted by the helper above.
    # Terrain stays as default flat plane (same memory footprint as Stage 2).

    return cfg
