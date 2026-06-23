from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
    ainex_flat_env_cfg,
    ainex_rough_env_cfg,
)
from .rl_cfg import ainex_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-AiNex",
    env_cfg=ainex_rough_env_cfg(),
    play_env_cfg=ainex_rough_env_cfg(play=True),
    rl_cfg=ainex_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-AiNex",
    env_cfg=ainex_flat_env_cfg(),
    play_env_cfg=ainex_flat_env_cfg(play=True),
    rl_cfg=ainex_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
