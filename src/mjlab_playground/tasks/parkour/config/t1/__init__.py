"""Register Instinct MJ parkour T1 tasks."""

# Copyright (c) 2022-2025, The Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from instinct_mj.tasks.registry import register_instinct_task

from .agents.instinct_rl_amp_cfg import T1ParkourPPORunnerCfg
from .t1_parkour_target_amp_cfg import instinct_t1_parkour_amp_final_cfg

register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-T1-v0",
    env_cfg_factory=lambda: instinct_t1_parkour_amp_final_cfg(play=False),
    play_env_cfg_factory=lambda: instinct_t1_parkour_amp_final_cfg(play=True),
    instinct_rl_cfg_factory=T1ParkourPPORunnerCfg,
)


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-T1-Play-v0",
    env_cfg_factory=lambda: instinct_t1_parkour_amp_final_cfg(play=True),
    play_env_cfg_factory=lambda: instinct_t1_parkour_amp_final_cfg(play=True),
    instinct_rl_cfg_factory=T1ParkourPPORunnerCfg,
)
