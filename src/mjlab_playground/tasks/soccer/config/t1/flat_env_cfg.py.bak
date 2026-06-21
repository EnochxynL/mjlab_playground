"""Booster T1 soccer environment configuration.

G1 → T1 body name mapping used throughout:

=========== ===================== ===================
Role        G1 body               T1 body
=========== ===================== ===================
root        pelvis                Trunk
torso       torso_link            Waist
L upper arm left_shoulder_roll    AL2
L forearm   left_elbow_link       AL3
L hand      left_wrist_yaw_link   left_hand_link
R upper arm right_shoulder_roll   AR2
R forearm   right_elbow_link      AR3
R hand      right_wrist_yaw_link  right_hand_link
L hip       left_hip_roll_link    Hip_Roll_Left
L knee      left_knee_link        Shank_Left
L foot      left_ankle_roll_link  left_foot_link
R hip       right_hip_roll_link   Hip_Roll_Right
R knee      right_knee_link       Shank_Right
R foot      right_ankle_roll_link right_foot_link
=========== ===================== ===================
"""

from __future__ import annotations

from dataclasses import dataclass

from mjlab_playground.asset_zoo.robots.booster_t1 import (
  T1_ACTION_SCALE,
  get_t1_robot_cfg,
)

from ...mdp import commands_multi_motion_soccer as soccer_commands
from ...tracking_env_cfg import TrackingEnvCfg

# T1 body names used for tracking (same structure as G1, different names).
_T1_TRACKING_BODY_NAMES = (
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


@dataclass(kw_only=True)
class T1FlatEnvCfg(TrackingEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.scene.robot = get_t1_robot_cfg()
    self.actions.joint_pos.scale = T1_ACTION_SCALE
    self.commands.motion.anchor_body_name = "Trunk"
    self.commands.motion.pelvis_body_name = "Trunk"
    self.commands.motion.class_type = soccer_commands.MotionCommand
    self.commands.motion.body_names = list(_T1_TRACKING_BODY_NAMES)
