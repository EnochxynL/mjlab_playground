"""AiNex robot constants (following mjlab T1 pattern).

Actuator parameters sourced from eliza_robot/bridge/isaaclab/ainex_cfg.py.
Home pose from STAND_JOINT_POSITIONS in the same source.
"""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

AINEX_XML: Path = Path(__file__).parent / "xmls" / "ainex.xml"
assert AINEX_XML.exists()


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(AINEX_XML))


##
# Actuator config.
# Parameters from eliza_robot/bridge/isaaclab/ainex_cfg.py.
# Actuator names match the <position name="..."> elements in ainex.xml.
##

AINEX_ACTUATOR_LEG = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "r_hip_yaw_act", "r_hip_roll_act", "r_hip_pitch_act",
        "r_knee_act", "r_ank_pitch_act", "r_ank_roll_act",
        "l_hip_yaw_act", "l_hip_roll_act", "l_hip_pitch_act",
        "l_knee_act", "l_ank_pitch_act", "l_ank_roll_act",
    ),
    stiffness=50.0,
    damping=5.0,
    effort_limit=6.0,
)

AINEX_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "r_sho_pitch_act", "r_sho_roll_act", "r_el_pitch_act",
        "r_el_yaw_act", "r_gripper_act",
        "l_sho_pitch_act", "l_sho_roll_act", "l_el_pitch_act",
        "l_el_yaw_act", "l_gripper_act",
    ),
    stiffness=10.0,
    damping=1.0,
    effort_limit=6.0,
)

AINEX_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
    target_names_expr=("head_pan_act", "head_tilt_act"),
    stiffness=10.0,
    damping=1.0,
    effort_limit=6.0,
)

##
# Keyframes.
# Home pose from eliza_robot/bridge/isaaclab/ainex_cfg.py STAND_JOINT_POSITIONS.
# All leg/head joints are zero; arms are tucked inward.
# Spawn height matches the "stand" keyframe in ainex.xml.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.25),
    joint_pos={
        "r_sho_roll": 1.403,
        "l_sho_roll": -1.403,
        "r_el_yaw": 1.226,
        "l_el_yaw": -1.226,
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
# Foot contact geoms are l_foot1, l_foot2, r_foot1, r_foot2 in ainex.xml.
##

_foot_regex = r"^[lr]_foot[12]$"

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(_foot_regex,),
    solref=(0.004, 1),
    condim=6,
    friction=(1.5, 0.5, 0.01),
    priority=1,
)

##
# Final config.
##

AINEX_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(AINEX_ACTUATOR_LEG, AINEX_ACTUATOR_ARM, AINEX_ACTUATOR_HEAD),
    soft_joint_pos_limit_factor=0.95,
)


def get_ainex_robot_cfg() -> EntityCfg:
    """Get a fresh AiNex robot configuration instance."""
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=AINEX_ARTICULATION,
    )


AINEX_ACTION_SCALE: dict[str, float] = {}
for a in AINEX_ARTICULATION.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg)
    e = a.effort_limit
    s = a.stiffness
    names = a.target_names_expr
    assert e is not None
    for n in names:
        AINEX_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_ainex_robot_cfg())
    viewer.launch(robot.spec.compile())
