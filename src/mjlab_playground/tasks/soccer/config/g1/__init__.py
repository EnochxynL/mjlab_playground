from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import g1_soccer_tracking_env_cfg, g1_soccer_destination_env_cfg
from .rl_cfg import g1_soccer_tracking_ppo_runner_cfg, g1_soccer_destination_ppo_runner_cfg

# Stage 1: motion-skill acquisition on gravel terrain (tracking + soccer obs, no kick rewards)
register_mjlab_task(
    task_id="Mjlab-SoccerTracking-Terrain-G1",
    env_cfg=g1_soccer_tracking_env_cfg(),
    play_env_cfg=g1_soccer_tracking_env_cfg(play=True),
    rl_cfg=g1_soccer_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

# Stage 2: kick-to-destination on flat ground (tracking + kick rewards + soccer obs)
register_mjlab_task(
    task_id="Mjlab-SoccerDestination-Flat-G1",
    env_cfg=g1_soccer_destination_env_cfg(),
    play_env_cfg=g1_soccer_destination_env_cfg(play=True),
    rl_cfg=g1_soccer_destination_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)
