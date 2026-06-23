"""AiNex velocity environment configurations."""

from mjlab_playground.asset_zoo.robots.ainex.ainex_constants import (
    AINEX_ACTION_SCALE,
    get_ainex_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab_playground.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def ainex_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create AiNex rough terrain velocity configuration."""
    cfg = make_velocity_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 70

    cfg.scene.entities = {"robot": get_ainex_robot_cfg()}

    # Set raycast sensor frame to AiNex torso.
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            assert isinstance(sensor.frame, ObjRef)
            sensor.frame.name = "body_link"

    site_names = ("left_foot", "right_foot")
    geom_names = ("l_foot1", "l_foot2", "r_foot1", "r_foot2")

    # Wire foot height scan to per-foot sites.
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=s, entity="robot") for s in site_names
            )
            sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(l_ank_roll_link|r_ank_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="body_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="body_link", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        self_collision_cfg,
    )

    if (
        cfg.scene.terrain is not None
        and cfg.scene.terrain.terrain_generator is not None
    ):
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = AINEX_ACTION_SCALE

    cfg.viewer.body_name = "body_link"

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 0.6

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
    cfg.events["base_com"].params["asset_cfg"].body_names = ("body_link",)

    # Pose std values for AiNex 24-DoF joint layout.
    # Leg joints use l_/r_ prefix; no waist joints; head + arm included.
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {
        # Head.
        r"head_pan": 0.1,
        r"head_tilt": 0.1,
        # Lower body.
        r".*hip_pitch.*": 0.3,
        r".*hip_roll.*": 0.15,
        r".*hip_yaw.*": 0.15,
        r".*knee.*": 0.35,
        r".*ank_pitch.*": 0.25,
        r".*ank_roll.*": 0.1,
        # Arms.
        r".*sho_pitch.*": 0.15,
        r".*sho_roll.*": 0.15,
        r".*el_pitch.*": 0.15,
        r".*el_yaw.*": 0.1,
        r".*gripper.*": 0.1,
    }
    cfg.rewards["pose"].params["std_running"] = {
        # Head.
        r"head_pan": 0.15,
        r"head_tilt": 0.15,
        # Lower body.
        r".*hip_pitch.*": 0.5,
        r".*hip_roll.*": 0.2,
        r".*hip_yaw.*": 0.2,
        r".*knee.*": 0.6,
        r".*ank_pitch.*": 0.35,
        r".*ank_roll.*": 0.15,
        # Arms.
        r".*sho_pitch.*": 0.5,
        r".*sho_roll.*": 0.2,
        r".*el_pitch.*": 0.35,
        r".*el_yaw.*": 0.15,
        r".*gripper.*": 0.15,
    }

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("body_link",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("body_link",)

    for reward_name in ["foot_clearance", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["air_time"].weight = 1.0

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
    )

    # Apply play mode overrides.
    if play:
        cfg.episode_length_s = int(1e9)

        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.terminations.pop("out_of_terrain_bounds", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )

        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


def ainex_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create AiNex flat terrain velocity configuration."""
    cfg = ainex_rough_env_cfg(play=play)

    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None

    # Switch to flat terrain.
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Remove raycast sensor and height scan (no terrain to scan).
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    cfg.terminations.pop("out_of_terrain_bounds", None)

    cfg.curriculum.pop("terrain_levels", None)

    if play:
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (0.2, 0.2)
        twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
        twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

    return cfg
