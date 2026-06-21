"""在 MJLab 中预览 Booster T1 运动 .npz 文件。

仿照 ``scripts/soccer/g1_replay_npz_mjlab.py``，使用 MJLab Entity API
将 T1 关节和根状态写入场景，MuJoCo 原生被动 viewer 渲染。

.. code-block:: bash

    # Usage
    uv run python scripts/soccer/t1_replay_npz_mjlab.py \\
      --motion-path data/mjlab_playground-mjlab/soccer-standard/t1/soccer-standard-001_right.npz
"""

from __future__ import annotations

import argparse
import numpy as np
import torch
import time

import mujoco
import mujoco.viewer  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.observations import joint_pos_rel
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from mjlab_playground.asset_zoo.robots.booster_t1 import get_t1_robot_cfg
from mjlab_playground.tasks.soccer.mdp.commands import MotionLoader


def _make_replay_env_cfg() -> ManagerBasedRlEnvCfg:
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_t1_robot_cfg()},
            num_envs=1,
            extent=2.0,
        ),
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=0.5,
            ),
        },
        observations={
            "actor": ObservationGroupCfg(
                terms={
                    "joint_pos": ObservationTermCfg(func=joint_pos_rel),
                },
                concatenate_terms=True,
                enable_corruption=False,
            ),
        },
        commands={},
        events={},
        rewards={},
        terminations={},
        curriculum={},
        metrics={},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="Trunk",
        ),
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=4,
        episode_length_s=6.0,
    )


def run_simulator(env: ManagerBasedRlEnv, motion_path: str) -> None:
    robot = env.scene["robot"]

    sim_dt = env.cfg.sim.mujoco.timestep * env.cfg.decimation
    frame_dt = sim_dt

    motion = MotionLoader(motion_path, [0], device="cpu")
    motion_fps = getattr(motion, "fps", None)
    if motion_fps is not None:
        fps_array = np.asarray(motion_fps)
        if fps_array.size:
            fps_value = float(fps_array.reshape(-1)[0])
            if fps_value > 0:
                frame_dt = 1.0 / fps_value
    time_steps = 0

    mjm = env.unwrapped.sim.mj_model
    mjd = env.unwrapped.sim.mj_data
    simulation_app = mujoco.viewer.launch_passive(mjm, mjd)

    try:
        simulation_app.cam.lookat[:] = (0.0, 0.0, 0.74)
        simulation_app.cam.distance = 2.5
        simulation_app.cam.elevation = -15.0
        simulation_app.cam.azimuth = 90.0
    except AttributeError:
        pass

    while simulation_app.is_running():
        frame_start = time.perf_counter()
        time_steps += 1
        if time_steps >= motion.time_step_total:
            time_steps = 0

        root_states = torch.zeros(1, 13, dtype=torch.float32)
        root_states[:, :3] = motion.body_pos_w[time_steps]
        root_states[:, 3:7] = motion.body_quat_w[time_steps]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(
            motion.joint_pos[time_steps : time_steps + 1],
            motion.joint_vel[time_steps : time_steps + 1],
        )

        sim_data = env.unwrapped.sim.data
        if mjm.nq > 0:
            mjd.qpos[:] = sim_data.qpos[0].cpu().numpy()
            mjd.qvel[:] = sim_data.qvel[0].cpu().numpy()
        mujoco.mj_forward(mjm, mjd)
        simulation_app.sync()

        elapsed = time.perf_counter() - frame_start
        if elapsed < frame_dt:
            time.sleep(frame_dt - elapsed)

    simulation_app.close()
    env.close()


def main():
    parser = argparse.ArgumentParser(description="在 MJLab 中预览 Booster T1 运动 .npz")
    parser.add_argument(
        "--motion-path", type=str, required=True, help="T1 .npz 运动文件路径"
    )
    args = parser.parse_args()

    cfg = _make_replay_env_cfg()
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    run_simulator(env, args.motion_path)


if __name__ == "__main__":
    main()
