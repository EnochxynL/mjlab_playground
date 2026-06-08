from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import g1_soccer_env_cfg
from .rl_cfg import g1_soccer_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Soccer-Flat-G1",
    env_cfg=g1_soccer_env_cfg(),
    play_env_cfg=g1_soccer_env_cfg(play=True),
    rl_cfg=g1_soccer_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)
