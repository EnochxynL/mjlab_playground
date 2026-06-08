# mjlab playground

A collection of tasks built with [mjlab](https://github.com/mujocolab/mjlab), starting with ports from [MuJoCo Playground](https://playground.mujoco.org/).

## Tasks

| Task ID | Robot | Description | Preview |
|---------|-------|-------------|---------|
| **Getup** | | | |
| `Mjlab-Getup-Flat-Unitree-Go1` | Unitree Go1 | Fall recovery on flat terrain | <img src="https://raw.githubusercontent.com/mujocolab/mjlab_playground/assets/go1_getup_teaser.gif" width="200"/> |
| `Mjlab-Getup-Flat-Booster-T1` | Booster T1 | Fall recovery on flat terrain | <img src="https://raw.githubusercontent.com/mujocolab/mjlab_playground/assets/t1_getup_teaser.gif" width="200"/> |

## Getting Started

```bash
git clone https://github.com/mujocolab/mjlab_playground.git && cd mjlab_playground
uv sync
```

Train a task:

```bash
# uv run train <task-id> --num_envs 4096 # deprecated
uv run train <task-id> --env.scene.num-envs 4096
uv run train Mjlab-Getup-Flat-Booster-T1 --env.scene.num-envs 4096 # TODO
uv run train Mjlab-Velocity-Flat-Booster-T1 --env.scene.num-envs 4096 # TODO
uv run train Mjlab-Tracking-Flat-Booster-T1 --env.scene.num-envs 4096 # TODO
```

Play back a trained policy:

```bash
uv run play <task-id>
uv run play Mjlab-Getup-Flat-Booster-T1 --checkpoint_file=logs/rsl_rl/t1_getup/2026-05-01_15-50-00/model_2999.pt # TODO
```

### Getup training

On a single NVIDIA 5090, the Go1 getup task converges in ~2 minutes and T1 in ~8 minutes, but we continue training with a curriculum that progressively tightens action rate, joint velocity, and power penalties to produce smoother, safer policies.

<p align="center">
  <img src="https://raw.githubusercontent.com/mujocolab/mjlab_playground/assets/training_curves.png" width="80%"/>
</p>

## Citation

If you use this repository in your research, consider citing mjlab:

```bibtex
@misc{zakka2026mjlablightweightframeworkgpuaccelerated,
  title={mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning},
  author={Kevin Zakka and Qiayuan Liao and Brent Yi and Louis Le Lay and Koushil Sreenath and Pieter Abbeel},
  year={2026},
  eprint={2601.22074},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2601.22074},
}
```

## License

This repository is released under an [Apache-2.0 License](LICENSE).

# HumanoidSoccer

Stage 1：运动技能获取（4k iterations）

```sh
uv run train Mjlab-SoccerTracking-Terrain-G1 \
    --env.commands.motion.motion_files \
        data/soccer-standard/soccer-standard-001_right.npz \
        data/soccer-standard/soccer-standard-002_left.npz \
        data/soccer-standard/soccer-standard-003_left.npz \
        data/soccer-standard/soccer-standard-004_right.npz \
        data/soccer-standard/soccer-standard-005_right.npz \
        data/soccer-standard/soccer-standard-006_right.npz \
        data/soccer-standard/soccer-standard-007_left.npz \
        data/soccer-standard/soccer-standard-008_left.npz \
        data/soccer-standard/soccer-standard-009_right.npz \
        data/soccer-standard/soccer-standard-010_right.npz \
    --env.scene.num-envs 4096
```

输出目录：`logs/rsl_rl/g1_soccer_tracking/<timestamp>/`

Stage 2：踢球微调（30k iterations，需要从 Stage 1 加载权重）

由于 Stage 1 和 Stage 2 的实验名不同（`g1_soccer_tracking` vs `g1_soccer_destination`），`--agent.resume` 只会在当前实验目录下搜索。跨实验加载需要手动处理：

```sh
# 先把 Stage 1 的 checkpoint 复制到 Stage 2 的 log 目录下
STAGE1_RUN=$(ls -1t logs/rsl_rl/g1_soccer_tracking/ | head -1)
mkdir -p logs/rsl_rl/g1_soccer_destination/
cp -r logs/rsl_rl/g1_soccer_tracking/$STAGE1_RUN \
     logs/rsl_rl/g1_soccer_destination/stage1_pretrain
```

```sh
# 启动 Stage 2，从复制的 checkpoint 恢复
python -m mjlab.scripts.train Mjlab-SoccerDestination-Flat-G1 \
    --env.commands.motion.motion_files \
        doc/HumanoidSoccer/motions/soccer-standard/*.npz \
    --env.scene.num-envs 4096 \
    --agent.resume true \
    --agent.load_run "stage1_pretrain" \
    --agent.load_checkpoint "model_.*.pt"
```

验证训练结果（Play）

```sh
# 零动作演示（看场景是否正常）
python -m mjlab.scripts.play Mjlab-SoccerDestination-Flat-G1 \
    --agent zero \
    --env.commands.motion.motion_files doc/HumanoidSoccer/motions/soccer-standard/*.npz

# 加载训练好的策略
python -m mjlab.scripts.play Mjlab-SoccerDestination-Flat-G1 \
    --agent trained \
    --env.commands.motion.motion_files doc/HumanoidSoccer/motions/soccer-standard/*.npz \
    --checkpoint-file logs/rsl_rl/g1_soccer_destination/<run>/model_30000.pt
```

自动脚本使用方式

```sh
# 默认参数（4096 envs）
./progressive_soccer_train.sh

# 自定义运行名
./progressive_soccer_train.sh my_experiment

# 小规模测试
NUM_ENVS=64 STAGE1_ITERS=50 STAGE2_ITERS=100 ./progressive_soccer_train.sh debug
```