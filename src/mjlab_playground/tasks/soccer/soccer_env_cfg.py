
from pathlib import Path
import mujoco

from mjlab.entity import EntityCfg

from mjlab.managers.observation_manager import ObservationTermCfg as ObsTerm
from . import mdp

SOCCER_BALL_RADIUS = 0.11

SOCCER_ASSET_PATH = Path(__file__).parent / "mdp" / "soccer_ball.xml"

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
