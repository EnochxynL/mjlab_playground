"""Unitree G1 soccer environment configurations.

Inherits from the mjlab tracking task and adds soccer-specific entities,
observations, and rewards ported from HumanoidSoccer (arXiv-2602.05310v1).

Two stages:
  Stage 1 (Mjlab-SoccerTracking-G1):   adaptive sampling + tracking rewards only.
  Stage 2 (Mjlab-SoccerDestination-G1): uniform sampling + kick rewards.
"""

from dataclasses import dataclass  # MJLab: needed for @dataclass port of @configclass

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg

from ...mdp import commands_multi_motion_soccer as soccer_commands  # MJLab: IsaacLab imports commands_multi_motion_soccer
from ...tracking_env_config import TrackingEnvCfg  # MJLab: IsaacLab tracking_env_cfg TrackingEnvCfg → tracking_env_config


@dataclass(kw_only=True)  # MJLab: @configclass → @dataclass(kw_only=True)
class G1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = get_g1_robot_cfg()  # MJLab: G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") → get_g1_robot_cfg()
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.class_type = soccer_commands.MotionCommand  # MJLab: motion_cmds.MotionCommand → soccer_commands.MotionCommand
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


# G1FlatWoStateEstimationEnvCfg and G1FlatLowFreqEnvCfg classes are not necessarily needed for the soccer task, they are just used the experiments in HumanoidSoccer paper.