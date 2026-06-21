"""Booster T1 soccer tasks."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import t1_soccer_destination_env_cfg, t1_soccer_tracking_env_cfg
from .rl_cfg import (
  t1_soccer_destination_ppo_runner_cfg,
  t1_soccer_tracking_ppo_runner_cfg,
)

# Stage 1: motion-skill acquisition (tracking + soccer obs, no kick rewards)
register_mjlab_task(
  task_id="Mjlab-SoccerTracking-T1",
  env_cfg=t1_soccer_tracking_env_cfg(),
  play_env_cfg=t1_soccer_tracking_env_cfg(play=True),
  rl_cfg=t1_soccer_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# Stage 2: kick-to-destination (tracking + kick rewards + soccer obs)
register_mjlab_task(
  task_id="Mjlab-SoccerDestination-T1",
  env_cfg=t1_soccer_destination_env_cfg(),
  play_env_cfg=t1_soccer_destination_env_cfg(play=True),
  rl_cfg=t1_soccer_destination_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
