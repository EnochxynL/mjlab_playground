from __future__ import annotations

from dataclasses import MISSING, dataclass, field  # MJLab: @configclass → @dataclass; dataclasses needed for port

# MJLab: isaaclab.sim (sim_utils.RigidBodyMaterialCfg, DomeLightCfg, DistantLightCfg, MdlFileCfg) not available — MuJoCo uses XML
from mjlab.entity import EntityCfg  # MJLab: isaaclab.assets ArticulationCfg, AssetBaseCfg → mjlab.entity EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg  # MJLab: ManagerBasedRLEnvCfg → ManagerBasedRlEnvCfg (imported for type ref, not inheritance)
from mjlab.managers.event_manager import EventTermCfg as EventTerm  # MJLab: isaaclab.managers → mjlab.managers
from mjlab.managers.observation_manager import ObservationGroupCfg as ObsGroup  # MJLab: isaaclab.managers → mjlab.managers
from mjlab.managers.observation_manager import ObservationTermCfg as ObsTerm  # MJLab: isaaclab.managers → mjlab.managers
from mjlab.managers.reward_manager import RewardTermCfg as RewTerm  # MJLab: isaaclab.managers → mjlab.managers
from mjlab.managers.scene_entity_config import SceneEntityCfg  # MJLab: isaaclab.managers → mjlab.managers
from mjlab.managers.termination_manager import TerminationTermCfg as DoneTerm  # MJLab: isaaclab.managers → mjlab.managers
from mjlab.scene import SceneCfg  # MJLab: isaaclab.scene InteractiveSceneCfg → mjlab.scene SceneCfg
from mjlab.sensor import ContactSensorCfg  # MJLab: isaaclab.sensors → mjlab.sensor
from mjlab.terrains import TerrainEntityCfg  # MJLab: isaaclab.terrains TerrainImporterCfg → mjlab.terrains TerrainEntityCfg

##
# Pre-defined configs
##
from mjlab.utils.noise import UniformNoiseCfg as Unoise  # MJLab: isaaclab.utils.noise AdditiveUniformNoiseCfg → mjlab.utils.noise UniformNoiseCfg

from . import mdp as mdp  # MJLab: soccer.tasks.tracking.mdp → .mdp (combined MDP namespace)

##
# Scene definition
# MJLab: ISAAC_NUCLEUS_DIR / ISAACLAB_NUCLEUS_DIR not available in MJLab (MuJoCo/MJLab manages assets differently)
##

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


@dataclass(kw_only=True)  # MJLab: @configclass → @dataclass(kw_only=True)
class MySceneCfg:  # MJLab: InteractiveSceneCfg → standalone @dataclass (Python dataclass inheritance conflicts with field redefinition)
    """Configuration for the terrain scene with a legged robot."""

    # MJLab: InteractiveSceneCfg provides num_envs, env_spacing — added manually since we cannot inherit
    num_envs: int = 1  # MJLab: from InteractiveSceneCfg (default 1 in MJLab, 4096 in IsaacLab tasks)
    env_spacing: float = 2.0  # MJLab: from InteractiveSceneCfg (default 2.0 in MJLab, 2.5 in IsaacLab tasks)
    # ground terrain
    # MJLab: TerrainImporterCfg(prim_path=..., physics_material=..., visual_material=...) → TerrainEntityCfg(terrain_type="plane")
    # MJLab: sim_utils.RigidBodyMaterialCfg / MdlFileCfg not available (MuJoCo uses geom friction/size via XML)
    terrain: TerrainEntityCfg = field(default_factory=lambda: TerrainEntityCfg(  # MJLab: mutable default → field(default_factory=...)
        terrain_type="plane",  # MJLab: prim_path="/World/ground", collision_group=-1 → terrain_type only
    ))
    # robots
    robot: EntityCfg = MISSING  # MJLab: ArticulationCfg → EntityCfg
    # lights
    # MJLab: light = AssetBaseCfg(spawn=sim_utils.DistantLightCfg(...)) — not available (MuJoCo uses XML lighting)
    # MJLab: sky_light = AssetBaseCfg(spawn=sim_utils.DomeLightCfg(...)) — not available (MuJoCo uses XML lighting)
    # MJLab: contact_forces = ContactSensorCfg(prim_path=...) — arranged via scene.sensors ContactMatch in MJLab
    # MJLab: ISAAC_NUCLEUS_DIR texture file for sky light — not available


##
# MDP settings
##


@dataclass(kw_only=True)  # MJLab: @configclass → @dataclass(kw_only=True)
class CommandsCfg:
    """Command specifications for the MDP."""

    motion: mdp.MotionCommandCfg = field(  # MJLab: mutable default → field(default_factory=...); mdp.MotionCommandCfg from soccer MDP
        default_factory=lambda: mdp.MotionCommandCfg(
            entity_name="robot",  # MJLab: asset_name → entity_name
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
            pose_range={
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.01, 0.01),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
            velocity_range=VELOCITY_RANGE,
            joint_position_range=(-0.1, 0.1),
        )
    )


@dataclass(kw_only=True)  # MJLab: @configclass → @dataclass(kw_only=True)
class ActionsCfg:
    """Action specifications for the MDP."""

    # MJLab: JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)
    # MJLab: IsaacLab uses mdp.JointPositionActionCfg class import — MJLab imports from mjlab.envs.mdp.actions
    joint_pos: mdp.JointPositionActionCfg = field(  # MJLab: mutable default → field; configured via flat_env_config.py
        default_factory=lambda: mdp.JointPositionActionCfg(entity_name="robot", actuator_names=[".*"], use_default_offset=True)  # MJLab: asset_name → entity_name; joint_names → actuator_names
    )


@dataclass(kw_only=True)  # MJLab: @configclass → @dataclass(kw_only=True)
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @dataclass(kw_only=True)  # MJLab: @configclass → @dataclass(kw_only=True)
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)

        # obs v0: original obs 160 dims
        # command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        # motion_anchor_pos_b = ObsTerm(
        #     func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.25, n_max=0.25)
        # )
        # motion_anchor_ori_b = ObsTerm(
        #     func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        # )
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        # base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        # actions = ObsTerm(func=mdp.last_action)

        # obs v1: add_prog, 154 dims
        command: ObsTerm = field(default_factory=lambda: ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"}))  # MJLab: mutable default → field
        projected_gravity: ObsTerm = field(default_factory=lambda: ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)))  # MJLab: mutable default → field
        motion_ref_ang_vel: ObsTerm = field(default_factory=lambda: ObsTerm(func=mdp.motion_anchor_ang_vel, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)))  # MJLab: mutable default → field
        base_ang_vel: ObsTerm = field(default_factory=lambda: ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)))  # MJLab: mutable default → field
        joint_pos: ObsTerm = field(default_factory=lambda: ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)))  # MJLab: mutable default → field
        joint_vel: ObsTerm = field(default_factory=lambda: ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)))  # MJLab: mutable default → field
        actions: ObsTerm = field(default_factory=lambda: ObsTerm(func=mdp.last_action))  # MJLab: mutable default → field


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @dataclass(kw_only=True)  # MJLab: @configclass → @dataclass(kw_only=True)
    class PrivilegedCfg(ObsGroup):
        command: ObsTerm = field(default_factory=lambda: ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"}))  # MJLab: mutable default → field
        motion_anchor_pos_b: ObsTerm = field(default_factory=lambda: ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}))  # MJLab: mutable default → field
        motion_anchor_ori_b: ObsTerm = field(default_factory=lambda: ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}))  # MJLab: mutable default → field
        body_pos: ObsTerm = field(default_factory=lambda: ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"}))  # MJLab: mutable default → field
        body_ori: ObsTerm = field(default_factory=lambda: ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"}))  # MJLab: mutable default → field
        base_lin_vel: ObsTerm = field(default_factory=lambda: ObsTerm(func=mdp.base_lin_vel))  # MJLab: mutable default → field
        base_ang_vel: ObsTerm = field(default_factory=lambda: ObsTerm(func=mdp.base_ang_vel))  # MJLab: mutable default → field
        joint_pos: ObsTerm = field(default_factory=lambda: ObsTerm(func=mdp.joint_pos_rel))  # MJLab: mutable default → field
        joint_vel: ObsTerm = field(default_factory=lambda: ObsTerm(func=mdp.joint_vel_rel))  # MJLab: mutable default → field
        actions: ObsTerm = field(default_factory=lambda: ObsTerm(func=mdp.last_action))  # MJLab: mutable default → field

    # observation groups
    policy: PolicyCfg = field(default_factory=PolicyCfg)  # MJLab: mutable default → field
    critic: PrivilegedCfg = field(default_factory=PrivilegedCfg)  # MJLab: mutable default → field


@dataclass(kw_only=True)  # MJLab: @configclass → @dataclass(kw_only=True)
class EventCfg:
    """Configuration for events."""

    # startup
    # MJLab: physics_material randomize_rigid_body_material not available — MJLab uses dr.geom_friction for material randomization
    physics_material: EventTerm = field(  # MJLab: mutable default → field; func not available in MJLab
        default_factory=lambda: EventTerm(
            func=mdp.randomize_rigid_body_material,  # MJLab: not available — use dr.geom_friction in flat_env_config.py
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.3, 1.6),
                "dynamic_friction_range": (0.3, 1.2),
                "restitution_range": (0.0, 0.5),
                "num_buckets": 64,
            },
        )
    )

    add_joint_default_pos: EventTerm = field(
        default_factory=lambda: EventTerm(
            func=mdp.randomize_joint_default_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "pos_distribution_params": (-0.01, 0.01),
                "operation": "add",
            },
        )
    )

    base_com: EventTerm = field(
        default_factory=lambda: EventTerm(
            func=mdp.randomize_rigid_body_com,  # MJLab: ported from IsaacLab — uses model body_ipos instead of PhysX get_coms/set_coms
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            },
        )
    )

    # interval
    push_robot: EventTerm = field(
        default_factory=lambda: EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={"velocity_range": VELOCITY_RANGE},
        )
    )


@dataclass(kw_only=True)  # MJLab: @configclass → @dataclass(kw_only=True)
class RewardsCfg:
    """Reward terms for the MDP."""

    motion_global_anchor_pos: RewTerm = field(  # MJLab: mutable default → field
        default_factory=lambda: RewTerm(
            func=mdp.motion_global_anchor_position_error_exp,
            # weight=0.5,
            weight=1.0,
            params={"command_name": "motion", "std": 0.3},
        )
    )
    motion_global_anchor_ori: RewTerm = field(  # MJLab: mutable default → field
        default_factory=lambda: RewTerm(
            func=mdp.motion_global_anchor_orientation_error_exp,
            # weight=0.5,
            weight=1.0,
            params={"command_name": "motion", "std": 0.4},
        )
    )
    motion_body_pos: RewTerm = field(  # MJLab: mutable default → field
        default_factory=lambda: RewTerm(
            func=mdp.motion_relative_body_position_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.3},
        )
    )
    motion_body_ori: RewTerm = field(  # MJLab: mutable default → field
        default_factory=lambda: RewTerm(
            func=mdp.motion_relative_body_orientation_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.4},
        )
    )
    motion_body_lin_vel: RewTerm = field(  # MJLab: mutable default → field
        default_factory=lambda: RewTerm(
            func=mdp.motion_global_body_linear_velocity_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 1.0},
        )
    )
    motion_body_ang_vel: RewTerm = field(  # MJLab: mutable default → field
        default_factory=lambda: RewTerm(
            func=mdp.motion_global_body_angular_velocity_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 3.14},
        )
    )
    action_rate_l2: RewTerm = field(  # MJLab: mutable default → field
        default_factory=lambda: RewTerm(func=mdp.action_rate_l2_clip, weight=-1e-1)
    )
    joint_limit: RewTerm = field(  # MJLab: mutable default → field
        default_factory=lambda: RewTerm(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )
    )
    undesired_contacts: RewTerm = field(  # MJLab: mutable default → field
        default_factory=lambda: RewTerm(
            func=mdp.undesired_contacts,  # MJLab: ported from isaaclab.envs.mdp.rewards to soccer mdp/rewards.py
            weight=-0.1,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                    ],
                ),
                "threshold": 1.0,
            },
        )
    )


@dataclass(kw_only=True)  # MJLab: @configclass → @dataclass(kw_only=True)
class TerminationsCfg:
    """Termination terms for the MDP."""

    # motion_finished = DoneTerm(func=mdp.motion_finished, params={"command_name": "motion"}, time_out=True)
    time_out: DoneTerm = field(  # MJLab: mutable default → field
        default_factory=lambda: DoneTerm(func=mdp.time_out, time_out=True)
    )
    # anchor_pos = DoneTerm(
    #     func=mdp.bad_anchor_pos,
    #     params={"command_name": "motion", "threshold": 0.50},
    # )
    anchor_pos_z: DoneTerm = field(  # MJLab: mutable default → field
        default_factory=lambda: DoneTerm(
            func=mdp.bad_anchor_pos_z_only,
            params={"command_name": "motion", "threshold": 0.25},
        )
    )
    anchor_ori: DoneTerm = field(  # MJLab: mutable default → field
        default_factory=lambda: DoneTerm(
            func=mdp.bad_anchor_ori,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
        )
    )
    ee_body_pos: DoneTerm = field(  # MJLab: mutable default → field
        default_factory=lambda: DoneTerm(
            func=mdp.bad_motion_body_pos_z_only,
            params={
                "command_name": "motion",
                "threshold": 0.25,
                "body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
            },
        )
    )


@dataclass(kw_only=True)  # MJLab: @configclass → @dataclass(kw_only=True)
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@dataclass(kw_only=True)  # MJLab: @configclass → @dataclass(kw_only=True)
class TrackingEnvCfg:  # MJLab: ManagerBasedRLEnvCfg inheritance replaced — @dataclass cannot redefine parent fields like @configclass
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = field(default_factory=lambda: MySceneCfg(num_envs=4096, env_spacing=2.5))  # MJLab: mutable default → field
    # Basic settings
    observations: ObservationsCfg = field(default_factory=ObservationsCfg)  # MJLab: mutable default → field
    actions: ActionsCfg = field(default_factory=ActionsCfg)  # MJLab: mutable default → field
    commands: CommandsCfg = field(default_factory=CommandsCfg)  # MJLab: mutable default → field
    # MDP settings
    rewards: RewardsCfg = field(default_factory=RewardsCfg)  # MJLab: mutable default → field
    terminations: TerminationsCfg = field(default_factory=TerminationsCfg)  # MJLab: mutable default → field
    events: EventCfg = field(default_factory=EventCfg)  # MJLab: mutable default → field
    curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)  # MJLab: mutable default → field

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        # MJLab: self.sim.dt = 0.005 → set via SimulationCfg(MujocoCfg(timestep=0.005))
        # MJLab: self.sim.render_interval = self.decimation — not available (MJLab viewer is separate)
        # MJLab: self.sim.physics_material = self.scene.terrain.physics_material — not available (MuJoCo uses geom friction)
        # MJLab: self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15 — PhysX-specific, not available
        # viewer settings
        # MJLab: self.viewer.eye = (1.5, 1.5, 1.5) → ViewerConfig(azimuth, elevation, distance)
        # MJLab: self.viewer.origin_type = "asset_root" → ViewerConfig.OriginType.ASSET_BODY
        # MJLab: self.viewer.asset_name = "robot" → ViewerConfig(entity_name="robot")
        pass  # MJLab: configuration done via flat_env_config.py function composition
