"""在 MJLab 中预览 G1 运动 .npz 文件。

基于 HumanoidSoccer/scripts/replay_npz.py（IsaacLab 版）改写，
将 IsaacLab API 替换为 MJLab 等效 API。
逐帧加载 G1 运动数据，通过 MJLab Entity API 将关节和根状态写入场景，
使用 MuJoCo 原生被动 viewer 渲染。不执行物理步进——纯运动学重放。

.. code-block:: bash

    # Usage
    uv run python scripts/soccer/g1_replay_npz_mjlab.py \\
      --motion-path data/soccer-standard/g1/soccer-standard-001_right.npz
"""

from __future__ import annotations

import argparse
import numpy as np
import torch
import time

import mujoco
import mujoco.viewer  # noqa: F401 — 被动 viewer 的懒加载子模块

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.observations import joint_pos_rel
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

##
# Pre-defined configs
##
from mjlab.asset_zoo.robots import get_g1_robot_cfg  # 替代 G1_CYLINDER_CFG
from mjlab_playground.tasks.soccer.mdp.commands import MotionLoader  # 替代 IsaacLab 版


def _make_replay_env_cfg() -> ManagerBasedRlEnvCfg:
    """Configuration for a replay motions scene.

    替代 IsaacLab 的 ``@configclass ReplayMotionsSceneCfg(InteractiveSceneCfg)``。
    IsaacLab 版包含 ground plane、dome light 和 G1 robot articulation；
    MJLab 版只需要平地 terrain + G1 机器人 entity，外加最小动作/观测层以
    满足 ManagerBasedRlEnv 初始化。
    """
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            # IsaacLab: ground = AssetBaseCfg(..., spawn=sim_utils.GroundPlaneCfg())
            # MJLab: TerrainEntityCfg(terrain_type="plane")
            terrain=TerrainEntityCfg(terrain_type="plane"),
            # IsaacLab: robot: ArticulationCfg = G1_CYLINDER_CFG.replace(...)
            # MJLab: EntityCfg via get_g1_robot_cfg()
            entities={"robot": get_g1_robot_cfg()},
            num_envs=1,
            extent=2.0,
        ),
        # 动作层：JointPositionAction 需要存在以满足管理器初始化，回放时不使用。
        # IsaacLab: cfg.actions["joint_pos"] = ...
        # MJLab: 同名配置，API 一致
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=0.5,
            ),
        },
        # 观测层：最小 actor 组以满足 env 初始化。
        # ObservationManager 要求至少一个 term，否则 stack 失败。
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
            body_name="pelvis",
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
    # Extract scene entities
    robot = env.scene["robot"]
    # breakpoint()
    # Define simulation stepping
    sim_dt = env.cfg.sim.mujoco.timestep * env.cfg.decimation
    frame_dt = sim_dt

    # registry_name = args_cli.registry_name
    # if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
    #     registry_name += ":latest"
    # import pathlib

    # import wandb

    # api = wandb.Api()
    # artifact = api.artifact(registry_name)
    # motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
    # motion_file = '/tmp/motion.npz'

    motion_file = motion_path

    motion = MotionLoader(
        motion_file,
        [0],  # IsaacLab: torch.tensor([0], device=sim.device) → MJLab: [0]
        device="cpu",  # IsaacLab: sim.device → MJLab: "cpu"
    )
    # MotionLoader now permutes joint_pos/joint_vel to MJCF order internally.
    motion_fps = getattr(motion, "fps", None)
    if motion_fps is not None:
        fps_array = np.asarray(motion_fps)
        if fps_array.size:
            fps_value = float(fps_array.reshape(-1)[0])
            if fps_value > 0:
                frame_dt = 1.0 / fps_value
    time_steps = 0  # 帧计数器 IsaacLab: time_steps = torch.zeros(scene.num_envs, device=sim.device); MJLab: 单环境 int 计数器

    # ── MuJoCo 被动 viewer ──────────────────────────────────────────
    # IsaacLab 用 sim.render() 在 Isaac Sim 内部渲染；
    # MJLab 的高层 NativeMujocoViewer 需要 policy 接口且会步进物理，
    # 不适合纯运动学回放。这里直接使用 MuJoCo 原生被动 viewer，
    mjm = env.unwrapped.sim.mj_model  # CPU MjModel
    mjd = env.unwrapped.sim.mj_data  # CPU MjData（单环境）
    simulation_app = mujoco.viewer.launch_passive(mjm, mjd)
    # 设置相机看向 pelvis。
    try:
        simulation_app.cam.lookat[:] = (0.0, 0.0, 0.74)
        simulation_app.cam.distance = 2.5
        simulation_app.cam.elevation = -15.0
        simulation_app.cam.azimuth = 90.0
    except AttributeError:
        pass


    # Simulation loop
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
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps : time_steps + 1], motion.joint_vel[time_steps : time_steps + 1])

        # ── 同步 Entity 写入 → CPU MjData → 渲染 ──────────────────
        # IsaacLab: scene.write_data_to_sim()
        #           sim.render()
        #           scene.update(sim_dt)
        # MJLab: Entity API 将状态写入了 batched sim.data，需要复制到
        # CPU MjData 供 viewer 渲染。参照 NativeMujocoViewer 内部做法。
        sim_data = env.unwrapped.sim.data
        if mjm.nq > 0:
            mjd.qpos[:] = sim_data.qpos[0].cpu().numpy()
            mjd.qvel[:] = sim_data.qvel[0].cpu().numpy()
        mujoco.mj_forward(mjm, mjd)
        simulation_app.sync()

        # MJLab: viewer 相机在初始化时已设置

        elapsed = time.perf_counter() - frame_start
        if elapsed < frame_dt:
            time.sleep(frame_dt - elapsed)

    simulation_app.close()
    env.close()

def main():
    parser = argparse.ArgumentParser(description="在 MJLab 中预览 G1 运动 .npz")
    parser.add_argument(
        "--motion-path", type=str, required=True, help="G1 .npz 运动文件路径"
    )
    args = parser.parse_args()


    cfg = _make_replay_env_cfg()
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    # Run the simulator
    run_simulator(env, args.motion_path)


if __name__ == "__main__":
    # run the main function
    main()
