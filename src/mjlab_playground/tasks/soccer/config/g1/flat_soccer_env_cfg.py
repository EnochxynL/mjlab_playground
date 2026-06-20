import math
import mujoco
from dataclasses import dataclass, field  # MJLab: needed for @dataclass port of @configclass
from pathlib import Path

from mjlab.entity import EntityCfg

from mjlab.managers.observation_manager import ObservationTermCfg as ObsTerm  # MJLab: isaaclab.managers → mjlab.managers
from mjlab.managers.reward_manager import RewardTermCfg as RewTerm  # MJLab: isaaclab.managers → mjlab.managers
from mjlab.managers.scene_entity_config import SceneEntityCfg  # MJLab: isaaclab.managers → mjlab.managers
from mjlab.sensor.contact_sensor import ContactSensorCfg  # MJLab: isaaclab.sensors → mjlab.sensor
# MJLab: @configclass → @dataclass(kw_only=True)
# MJLab: isaaclab.markers VisualizationMarkersCfg not available in MJLab (MuJoCo viewer)

# MJLab: soccer.assets ASSET_DIR → MJLab uses XML-based ball asset (soccer_ball.xml)
# MJLab: soccer.robots.g1 G1_ACTION_SCALE, G1_CYLINDER_CFG → mjlab.asset_zoo.robots
# MJLab: soccer.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg LOW_FREQ_SCALE → not ported
# MJLab: soccer.tasks.tracking mdp → mjlab_playground.tasks.soccer.mdp
from ... import mdp  # MJLab: soccer.tasks.tracking → ...mdp
from ...tracking_env_cfg import TrackingEnvCfg, MySceneCfg, CurriculumCfg  # MJLab: tracking_env_cfg → tracking_env_config
from .flat_env_cfg import G1FlatEnvCfg  # MJLab: .flat_env_cfg → .flat_env_config

# MJLab: isaaclab.terrains TerrainImporterCfg, TerrainGeneratorCfg → not available (MJLab terrain system differs)
#
# MJLab: isaaclab.terrains terrain_gen not available
# MjLab: no TerrainGeneratorCfg

from mjlab.managers.termination_manager import TerminationTermCfg as DoneTerm  # MJLab: isaaclab.managers → mjlab.managers

SOCCER_BALL_RADIUS = 0.11

SOCCER_ASSET_PATH = Path(__file__).parents[2] / "mdp" / "soccer_ball.xml" # MJLab: SOCCER_ASSET_PATH = f"{ASSET_DIR}/soccer/soccer.usda" — USD asset not available; MJLab uses XML

def _get_ball_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(SOCCER_ASSET_PATH))


def get_soccer_ball_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_ball_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.7, 0.0, SOCCER_BALL_RADIUS),
        ),
    )


def _apply_soccer_obs(cfg):
    cfg.observations.policy.target_point_pos = ObsTerm(
        func=mdp.constant_target_point_pos,
        params={"command_name": "motion"},
    )

    cfg.observations.critic.target_point_pos = ObsTerm(
        func=mdp.constant_target_point_pos,
        params={"command_name": "motion"},
    )

    cfg.observations.policy.target_destination_pos_local = ObsTerm(
        func=mdp.target_destination_pos_local,
        params={"command_name": "motion"},
    )

    cfg.observations.critic.target_destination_pos_local = ObsTerm(
        func=mdp.target_destination_pos_local,
        params={"command_name": "motion"},
    )


def _apply_soccer_scene(cfg):
    # MJLab: cfg.scene.soccer_ball = cfg.scene.soccer_ball.replace(prim_path="{ENV_REGEX_NS}/SoccerBall") — USD prim_path manipulation not available
    cfg.scene.soccer_ball.init_state.pos = (0.0, 0.0, SOCCER_BALL_RADIUS)

    # MJLab: VisualizationMarkersCfg(target_point_marker_cfg / target_destination_marker_cfg) not available (MuJoCo viewer doesn't use markers)
    # MJLab: sim_utils.SphereCfg, PreviewSurfaceCfg not available

## Scene configuration

@dataclass(kw_only=True)  # MJLab: @configclass → @dataclass(kw_only=True)
class G1FlatSoccerSceneCfg(MySceneCfg):
    def __post_init__(self):
        super().__post_init__()
        # Keep parent terrain material settings and explicitly set restitution.
        # MJLab: self.terrain.physics_material = self.terrain.physics_material.replace(restitution=0.8) — physics_material not available (MuJoCo uses geom solref/solimp)
        pass  # MJLab: terrain restitution set via XML geom properties instead

    # MJLab: RigidObjectCfg(prim_path=..., spawn=sim_utils.UsdFileCfg(usd_path=SOCCER_ASSET_PATH, activate_contact_sensors=True)) → EntityCfg with XML
    # MJLab: soccer ball configured via get_soccer_ball_cfg() in flat_env_config.py instead
    # soccer_ball: EntityCfg = ...  # MJLab: RigidObjectCfg → EntityCfg; configured externally

    # MJLab: soccer_ball_contact ContactSensorCfg(prim_path=...) → ContactSensorCfg configured via ContactMatch in flat_env_config.py
    # soccer_ball_contact: ContactSensorCfg = ...  # MJLab: configured externally via scene.sensors
    

## Environment configuration

@dataclass(kw_only=True)  # MJLab: @configclass → @dataclass(kw_only=True)
class G1TerrainEnvCfg(G1FlatEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.class_type = mdp.commands_multi_motion_soccer.MotionCommand
        self.terminations.anchor_pos_z = DoneTerm(
            func=mdp.bad_anchor_pos_z_only,
            params={"command_name": "motion", "threshold": 0.25},  # Slightly larger threshold for robustness.
        )
        self.terminations.anchor_ori = DoneTerm(
            func=mdp.bad_anchor_ori,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
        )
        self.terminations.ee_body_pos = DoneTerm(
            func=mdp.bad_motion_body_pos_z_only,
            params={
                "command_name": "motion",
                "threshold": 0.25, # 0.75, # 0.25,
                "body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
            },
        )

        # MJLab: GRAVEL_TERRAINS_CFG = TerrainGeneratorCfg(...) — not available (MJLab terrain system differs)
        # MJLab: terrain_gen.HfRandomUniformTerrainCfg not available
        # MJLab: self.scene.terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="generator", terrain_generator=GRAVEL_TERRAINS_CFG) — not available


@dataclass(kw_only=True)  # MJLab: @configclass → @dataclass(kw_only=True)
class G1TerrainMotionEnvCfg(G1TerrainEnvCfg):
    # MJLab: scene redefinition not supported in @dataclass inheritance — set in __post_init__ instead
    def __post_init__(self):
        self.scene = G1FlatSoccerSceneCfg(num_envs=4096, env_spacing=2.5)  # MJLab: scene: G1FlatSoccerSceneCfg = ... → __post_init__ assignment
        super().__post_init__()
        self.commands.motion.sampling_strategy = "adaptive"
        _apply_soccer_obs(self)
        _apply_soccer_scene(self)


@dataclass(kw_only=True)  # MJLab: @configclass → @dataclass(kw_only=True)
class G1FlatMotionEnvCfg(G1FlatEnvCfg):
    # MJLab: scene redefinition not supported in @dataclass inheritance — set in __post_init__ instead
    def __post_init__(self):
        self.scene = G1FlatSoccerSceneCfg(num_envs=4096, env_spacing=2.5)  # MJLab: scene: G1FlatSoccerSceneCfg = ... → __post_init__ assignment
        super().__post_init__()
        self.commands.motion.class_type = mdp.commands_multi_motion_soccer.MotionCommand
        self.commands.motion.sampling_strategy = "uniform"
        _apply_soccer_obs(self)
        _apply_soccer_scene(self)


@dataclass(kw_only=True)  # MJLab: @configclass → @dataclass(kw_only=True)
class G1FlatProximityEnvCfg(G1FlatMotionEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.foot_cfg = SceneEntityCfg(
            "robot",
            body_names=[
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ],
        )

        self.waist_cfg = SceneEntityCfg(
            "robot",
            joint_names=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint"
            ],
        )

        self.commands.motion.curve_offset_range = {
            "radius": (-0.25, 0.25),
            "arc_angle": math.pi / 9,
            "height": SOCCER_BALL_RADIUS,
        }


        self.rewards.foot_distance = RewTerm(
            func=mdp.foot_distance,
            weight=0.2,
            params={
                "threshold": 0.24,
                "std": 0.5,
                "foot_cfg": self.foot_cfg,
            },
        )

        # self.rewards.feet_slip_penalty = RewTerm(
        #     func=mdp.feet_slip_penalty,
        #     weight=-1.0,
        #     params={
        #         "foot_cfg": self.foot_cfg,
        #         "slip_force_threshold": 5.0,
        #     },
        # )

        self.rewards.target_point_proximity = RewTerm(
            func=mdp.target_point_proximity,
            weight=1.0,
            params={
                "std": 4.0,
                "command_name": "motion",
            },
        )

        self.rewards.motion_global_anchor_pos = RewTerm(
            func=mdp.motion_global_anchor_position_error_exp,
            # weight=0.5,
            weight=0.0,
            params={"command_name": "motion", "std": 0.3},
        )

        self.rewards.motion_global_anchor_ori = RewTerm(
            func=mdp.motion_global_anchor_orientation_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.4},
        )

        self.rewards.waist_action_rate_l2 = RewTerm(
            func=mdp.waist_action_rate_l2_clip,
            weight=-2.5e-1,
            params={
                "waist_cfg": self.waist_cfg,
            },
        )

        self.rewards.pelvis_orientation = RewTerm(
            func=mdp.pelvis_orientation,
            weight=-1.0,
            params={"command_name": "motion",},
        )

        self.rewards.motion_body_pos = RewTerm(
            func=mdp.motion_relative_body_position_error_exp,
            weight=1.0,
            params={
                "command_name": "motion",
                "std": 0.3,
                "body_names" : [
                    "pelvis",
                    "left_hip_roll_link",
                    "left_knee_link",
                    # "left_ankle_roll_link",
                    "right_hip_roll_link",
                    "right_knee_link",
                    # "right_ankle_roll_link",
                    "torso_link",
                    "left_shoulder_roll_link",
                    "left_elbow_link",
                    "left_wrist_yaw_link",
                    "right_shoulder_roll_link",
                    "right_elbow_link",
                    "right_wrist_yaw_link",
                ],
            },
        )

        self.motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4, 
                "body_names" : [
                    "pelvis",
                    "left_hip_roll_link",
                    "left_knee_link",
                    # "left_ankle_roll_link",
                    "right_hip_roll_link",
                    "right_knee_link",
                    # "right_ankle_roll_link",
                    "torso_link",
                    "left_shoulder_roll_link",
                    "left_elbow_link",
                    "left_wrist_yaw_link",
                    "right_shoulder_roll_link",
                    "right_elbow_link",
                    "right_wrist_yaw_link",
                ],
            },
        )

        self.rewards.motion_foot_pos = RewTerm(
            func=mdp.motion_relative_foot_position_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.3,
                    "foot_body_names" : [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                ],
            },
        )




@dataclass(kw_only=True)  # MJLab: @configclass → @dataclass(kw_only=True)
class G1FlatKickEnvCfg(G1FlatProximityEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.target_point_contact = RewTerm(
            func=mdp.target_point_contact,
            weight=50.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10,
                "foot_cfg": self.foot_cfg,
            },
        )

        self.rewards.sideways_kick = RewTerm(
            func=mdp.sideways_kick,
            weight=50.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10,
                "foot_cfg": self.foot_cfg,
            },
        )

        
        self.rewards.ball_velocity_direction_alignment = RewTerm(
            func=mdp.ball_velocity_direction_alignment,
            weight=30.0,
            params={
                "command_name": "motion",
                "std": 0.8,
                "velocity_threshold": 0.5,
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10,
                "foot_cfg": self.foot_cfg,
            },
        )

        self.rewards.ball_speed_reward = RewTerm(
            func=mdp.ball_speed_reward,
            weight=10.0,
            params={
                "command_name": "motion",
                # "target_speed": 4.0,
                "std": 1.2,
                "velocity_threshold": 0.5,
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10,
                "foot_cfg": self.foot_cfg,
            },
        )

        self.rewards.ball_z_speed_penalty_reward = RewTerm(
            func=mdp.ball_z_speed_penalty_reward,
            weight=-0.0,
            params={
                "command_name": "motion",
                "std": 3,
                "velocity_threshold": 0.5,
            },
        )

@dataclass(kw_only=True)  # MJLab: @configclass → @dataclass(kw_only=True)
class G1FlatKickMovingEnvCfg(G1FlatKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Initial soccer-ball linear velocity configuration.
        self.commands.motion.enable_soccer_ball_init_vel = True  # Enable sampling of initial ball velocity.
        self.commands.motion.soccer_ball_init_lin_vel_range = {
            "x": (-0.3, 0.3),
            "y": (-0.3, 0.3),
            "z": (0.0, 0.0),
        }


@dataclass(kw_only=True)  # MJLab: @configclass → @dataclass(kw_only=True)
class G1FlatSoccerBlindEnvCfg(G1FlatKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # Custom blind-zone range: the ball is invisible when (x, y) distance is outside [min, max].
        self.commands.motion.blind_distance_min_range = (0.2, 0.8)  # Minimum distance sampling range.
        self.commands.motion.blind_distance_max_range = (1.8, 2.5)  # Maximum distance sampling range.
        
        self.observations.policy.target_point_pos = ObsTerm(
            func=mdp.blind_zone_target_point_pos,
            params={"command_name": "motion"},
        )

        self.observations.critic.target_point_pos = ObsTerm(
            func=mdp.blind_zone_target_point_pos,
            params={"command_name": "motion"},
        )


@dataclass(kw_only=True)  # MJLab: @configclass → @dataclass(kw_only=True)
class G1FlatSuperSoccerEnvCfg(G1FlatKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        self.observations.policy.motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        self.observations.policy.body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        self.observations.policy.body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        self.observations.policy.base_lin_vel = ObsTerm(func=mdp.base_lin_vel)


        self.observations.critic.projected_gravity = ObsTerm(func=mdp.projected_gravity)
        self.observations.critic.motion_ref_ang_vel = ObsTerm(func=mdp.motion_anchor_ang_vel, params={"command_name": "motion"})




@dataclass(kw_only=True)  # MJLab: @configclass → @dataclass(kw_only=True)
class G1FlatSoccerStudentEnvCfg(G1FlatKickEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        student_obs = self.observations.policy.copy()
        student_obs.target_point_pos = ObsTerm(
            func=mdp.target_point_pos_first_frame,
            params={"command_name": "motion"},
        )
        self.observations.StudentPolicyCfg = student_obs

        student_obs.target_destination_pos_local = ObsTerm(
            func=mdp.target_destination_pos_local_first_frame,
            params={"command_name": "motion"},
        )
