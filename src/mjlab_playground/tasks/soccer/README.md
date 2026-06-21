# Soccer Task — IsaacLab → MJLab Port

Humanoid soccer RL task ported from [HumanoidSoccer](https://github.com/KaydenKnapik/BoosterT1mjlab) (arXiv-2602.05310v1).

## File Mapping

| MJLab (`mjlab_playground/tasks/soccer/`) | IsaacLab (`soccer/tasks/tracking/`) |
|---|---|
| `__init__.py` | `__init__.py` |
| `config/__init__.py` | `config/__init__.py` |
| `config/g1/__init__.py` | `config/g1/__init__.py` |
| `config/g1/flat_env_cfg.py` | `config/g1/flat_env_cfg.py` |
| `config/g1/flat_soccer_env_cfg.py` | `config/g1/soccer_flat_env_cfg.py` |
| `config/g1/env_cfgs.py` | *(no IsaacLab counterpart)* |
| `config/g1/rl_cfg.py` | `config/g1/agents/rsl_rl_ppo_cfg.py` |
| `tracking_env_cfg.py` | `tracking_env_cfg.py` |
| `mdp/__init__.py` | `mdp/__init__.py` |
| `mdp/commands.py` | `mdp/commands.py` |
| `mdp/commands_multi_motion.py` | `mdp/commands_multi_motion.py` |
| `mdp/commands_multi_motion_soccer.py` | `mdp/commands_multi_motion_soccer.py` |
| `mdp/events.py` | `mdp/events.py` |
| `mdp/kick_detection.py` | `mdp/kick_detection.py` |
| `mdp/observations.py` | `mdp/observations.py` |
| `mdp/rewards.py` | `mdp/rewards.py` |
| `mdp/terminations.py` | `mdp/terminations.py` |
| `mdp/soccer_ball.xml` | *(no IsaacLab counterpart — MuJoCo XML asset)* |

## Structure Assessment

### Files identical to IsaacLab (after import changes)

These files have **identical function bodies, identical logic, identical comments, identical formatting** after imports:

- **`mdp/observations.py`** — All functions identical.
- **`mdp/rewards.py`** — All reward functions identical (except `gravity_vec_w`, `root_link_lin_vel_w` renames). `undesired_contacts` appended at end (ported from `isaaclab.envs.mdp.rewards`).
- **`mdp/terminations.py`** — All functions identical (except `gravity_vec_w` rename).
- **`mdp/kick_detection.py`** — Entire `KickContactTracker` class identical (except `force_history` rename and `body_link_pos_w` rename).
- **`mdp/commands.py`** — `MotionCommand` body identical (except `asset_name→entity_name`, `Articulation→Entity`, `body_pos_w→body_link_pos_w`, VisualizationMarkers removed).
- **`mdp/commands_multi_motion.py`** — `MultiMotionMotionCommand` body identical (same class of changes).
- **`mdp/commands_multi_motion_soccer.py`** — `SoccerMotionCommand` body identical (same class of changes, plus VisualizationMarkers removed).
- **`config/g1/flat_soccer_env_cfg.py`** — All class bodies have identical logic; identical comments, parameter values, weight values, indentation.

### Files with MJLab-only additions

- **`config/g1/env_cfgs.py`** — Factory functions replacing `@configclass` inheritance chain. No IsaacLab counterpart.
- **`config/g1/rl_cfg.py`** — MJLab-native PPO runner config. Different API from IsaacLab `RslRlPpoActorCriticCfg`, but semantic equivalence preserved.
- **`mdp/events.py`** — `_randomize_prop_by_op` locally defined. `randomize_rigid_body_material` completely re-implemented for MuJoCo.

### Files not ported

- `G1FlatWoStateEstimationEnvCfg` / `G1FlatLowFreqEnvCfg` — not needed for soccer task.
- `config/g1/agents/` — IsaacLab agent configs replaced by `rl_cfg.py`.

---

## Forced API Differences

Every difference below is **forced** by the MJLab ↔ IsaacLab API gap. Lines that could stay identical to IsaacLab were kept identical.

### 1. Configuration system: `@configclass` → `@dataclass(kw_only=True)`

| IsaacLab | MJLab | Reason |
|---|---|---|
| `from isaaclab.utils import configclass` | `from dataclasses import dataclass` | MJLab has no `@configclass` |
| `@configclass` | `@dataclass(kw_only=True)` | Python `@dataclass` is nearest equivalent |
| `field = SomeCfg(...)` (mutable default) | `field = field(default_factory=lambda: SomeCfg(...))` | `@dataclass` forbids mutable defaults |
| `scene: SubCfg = SubCfg(...)` (field redefinition) | moved to `__post_init__`: `self.scene = SubCfg(...)` | `@dataclass` forbids field redefinition in child class |
| `class TrackingEnvCfg(ManagerBasedRLEnvCfg)` | `class TrackingEnvCfg` (standalone) | `@dataclass` cannot redefine parent fields |

### 2. Package namespace

| IsaacLab | MJLab |
|---|---|
| `isaaclab.envs` | `mjlab.envs` |
| `isaaclab.managers` | `mjlab.managers` |
| `isaaclab.sensors` | `mjlab.sensor` |
| `isaaclab.scene` | `mjlab.scene` |
| `isaaclab.assets` | `mjlab.entity` |
| `isaaclab.terrains` | `mjlab.terrains` |
| `isaaclab.utils.math` | `mjlab.utils.lab_api.math` |
| `isaaclab.markers` | *(removed — MuJoCo viewer)* |

### 3. Type renames

| IsaacLab | MJLab | Reason |
|---|---|---|
| `ManagerBasedRLEnv` | `ManagerBasedRlEnv` | MJLab naming convention |
| `Articulation` | `Entity` | MJLab unified entity type (no Articulation/RigidObject split) |
| `RigidObject` | `Entity` | Same |
| `ArticulationCfg` | `EntityCfg` | Same |
| `RigidObjectCfg` | `EntityCfg` | Same |
| `InteractiveSceneCfg` | `SceneCfg` or standalone `MySceneCfg` | MJLab scene API differs |
| `TerrainImporterCfg` | `TerrainEntityCfg` | MJLab terrain is simpler (plane vs USD importer) |
| `AdditiveUniformNoiseCfg` | `UniformNoiseCfg` | MJLab noise class naming |

### 4. Parameter renames

| IsaacLab | MJLab | Files |
|---|---|---|
| `asset_name` | `entity_name` | `commands.py`, `commands_multi_motion.py`, `commands_multi_motion_soccer.py`, `tracking_env_cfg.py` |
| `joint_names` (action) | `actuator_names` | `tracking_env_cfg.py` |
| `prim_path` (sensor) | `name` + `entity` via `ContactMatch` | `flat_soccer_env_cfg.py`, `env_cfgs.py` |

### 5. Entity data property renames

All `data.body_*_w` → `data.body_link_*_w`, `data.root_*_w` → `data.root_link_*_w`:

| IsaacLab | MJLab |
|---|---|
| `data.body_pos_w` | `data.body_link_pos_w` |
| `data.body_quat_w` | `data.body_link_quat_w` |
| `data.body_lin_vel_w` | `data.body_com_lin_vel_w` |
| `data.body_ang_vel_w` | `data.body_link_ang_vel_w` |
| `data.root_pos_w` | `data.root_link_pos_w` |
| `data.root_lin_vel_w` | `data.root_link_lin_vel_w` |
| `data.GRAVITY_VEC_W` | `data.gravity_vec_w` |

Reason: MJLab uses `body_link_*` prefix for per-link properties, snake_case for property naming.
**Important:** `body_lin_vel_w` is a special case — in IsaacLab it is a compatibility alias that resolves to
`body_com_lin_vel_w` (COM frame velocity), NOT `body_link_lin_vel_w` (link frame velocity). MJLab exposes
both `body_com_lin_vel_w` and `body_link_lin_vel_w` as separate properties with different semantics
(link velocity = COM velocity + ω × (link_pos − com_pos)). The correct port uses `body_com_lin_vel_w`.

### 6. Contact sensor API

| IsaacLab | MJLab | Reason |
|---|---|---|
| `ContactSensorCfg(prim_path="{ENV_REGEX_NS}/SoccerBall", history_length=3, ...)` | `ContactSensorCfg(name="soccer_ball_contact", primary=ContactMatch(mode="body", pattern="soccer_ball", entity="soccer_ball"), secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"), ...)` | MJLab uses `ContactMatch` pattern matching instead of USD `prim_path` |
| `data.net_forces_w_history` [B, H, max_bodies, 3] | `data.force_history` [B, N, H, 3] | MJLab: [B, bodies, history, 3] vs IsaacLab: [B, history, max_bodies, 3] |
| `data.net_forces_w` | `data.force` (fallback) | Different field naming |

### 7. Scene and physics

| IsaacLab | MJLab | Reason |
|---|---|---|
| `sim.dt = 0.005` | `sim.mujoco.timestep = 0.005` | MuJoCo-specific simulation config |
| `sim.physics_material = ...` | *(removed)* | PhysX physics material — MuJoCo uses geom properties |
| `sim.render_interval = ...` | *(removed)* | MJLab viewer is separate |
| `sim.physx.gpu_max_rigid_patch_count` | *(removed)* | PhysX-specific |
| `viewer.eye / origin_type / asset_name` | `viewer.body_name` | MJLab viewer config differs |
| `TerrainImporterCfg(prim_path=..., terrain_type="plane", physics_material=..., visual_material=..., collision_group=...)` | `TerrainEntityCfg(terrain_type="plane")` | MJLab simpler terrain system |
| `TerrainGeneratorCfg` / `terrain_gen.HfRandomUniformTerrainCfg` | *(removed)* | MJLab no height-field terrain generation |
| `RigidBodyMaterialCfg(friction_combine_mode=..., static_friction=..., dynamic_friction=...)` | *(removed)* | MuJoCo uses `solref/solimp` for contact dynamics |
| `MdlFileCfg` / `DomeLightCfg` / `DistantLightCfg` | *(removed)* | MuJoCo uses XML lighting |
| `AssetBaseCfg(spawn=...)` for lights | *(removed)* | Same |

### 8. Asset format

| IsaacLab | MJLab | Reason |
|---|---|---|
| `RigidObjectCfg(prim_path=..., spawn=sim_utils.UsdFileCfg(usd_path=..., activate_contact_sensors=True))` | `EntityCfg(spec_fn=_get_ball_spec, init_state=...)` | MuJoCo uses MJCF/XML, not USD |
| `cfg.scene.soccer_ball.replace(prim_path=...)` | *(removed)* | USD prim_path manipulation not applicable |
| `SOCCER_ASSET_PATH = f"{ASSET_DIR}/soccer/soccer.usda"` | `Path(__file__).parents[2] / "mdp" / "soccer_ball.xml"` | Different asset format and location |

### 9. Debug visualization (all removed)

| IsaacLab | MJLab | Reason |
|---|---|---|
| `VisualizationMarkers` / `VisualizationMarkersCfg` | *(removed)* | MuJoCo viewer doesn't support Omniverse markers |
| `FRAME_MARKER_CFG` | *(removed)* | Same |
| `_set_debug_vis_impl()` / `_debug_vis_callback()` | *(removed)* | Marker-based debug visualization not available |
| `target_point_marker_cfg` / `target_destination_marker_cfg` | `None` | Placeholder |
| `anchor_visualizer_cfg` / `body_visualizer_cfg` | *(removed)* | Not available |

### 10. Material/COM randomization

| IsaacLab | MJLab | Reason |
|---|---|---|
| `randomize_rigid_body_material` (PhysX material buckets) | `dr.geom_friction` + `dr.mat_rgba` | MuJoCo has no PhysX material system |
| `asset.root_physx_view.get_coms()` / `set_coms()` | `env.sim.model.body_ipos` direct assignment | MuJoCo no PhysX COM views |
| `_randomize_prop_by_op` (imported from `isaaclab.envs.mdp.events`) | locally defined | Not re-exported by MJLab |

### 11. Environment registration

| IsaacLab | MJLab | Reason |
|---|---|---|
| `gym.register(id=..., entry_point="isaaclab.envs:ManagerBasedRLEnv", kwargs={...})` | `register_mjlab_task(task_id=..., env_cfg=..., play_env_cfg=..., rl_cfg=..., runner_cls=...)` | MJLab task registry |

### 12. RL configuration

| IsaacLab | MJLab | Reason |
|---|---|---|
| `RslRlOnPolicyRunnerCfg` + `RslRlPpoActorCriticCfg` (joint actor/critic) | `RslRlOnPolicyRunnerCfg(RslRlModelCfg, RslRlPpoAlgorithmCfg)` (separate actor + critic) | MJLab RL API structure |
| `actor_hidden_dims=[512, 256, 128]` / `critic_hidden_dims=[512, 256, 128]` | `actor=RslRlModelCfg(hidden_dims=(512, 256, 128))` / `critic=RslRlModelCfg(hidden_dims=(512, 256, 128))` | Different config nesting |
| `init_noise_std=1.0` | `distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}` | MJLab distribution specification |
| `empirical_normalization = True` | `obs_normalization=True` | Different flag name |
| `experiment_name = "g1_flat"` | `experiment_name = "g1_soccer"` | Task-specific name |

### 13. Motion command config

| IsaacLab | MJLab | Reason |
|---|---|---|
| `class_type: type = MotionCommand` | *(removed)* | MJLab uses `cfg.build(env)` instead of `class_type` dispatch |
| `motion_file: str = MISSING` | `motion_files: list[str] = field(default_factory=list)` | MJLab multi-motion support |
| *(none)* | `resampling_time_range: tuple = (1e9, 1e9)` | Required by MJLab `CommandTermCfg` |
| *(none)* | `debug_vis: bool = True` | Required by MJLab `CommandTermCfg` |

---

## Known Limitations

1. **Terrain randomization** — IsaacLab `HfRandomUniformTerrainCfg` gravel terrain not available. Stage 1 uses flat plane.
2. **Material restitution** — `PhysicsMaterial(restitution=0.8)` not directly mappable. MuJoCo uses `solref/solimp`.
3. **Visualization markers** — Debug markers not rendered in MuJoCo viewer.
4. **Contact sensor detail** — IsaacLab per-body contact history with `force_threshold` and `track_air_time` replaced by MJLab `ContactMatch`-based sensor. Functional equivalence verified, edge cases may differ.

---

## 关节顺序映射 (Joint Order Remapping)

### 问题

G1 的 29 个关节在 IsaacLab 和 MuJoCo/MJCF 中排列顺序不同：

- **IsaacLab 顺序**（按关节类型分组）：所有 `hip_pitch` → 所有 `hip_roll` → 所有 `hip_yaw` → ...
- **MuJoCo/MJCF 顺序**（按肢体分组）：左腿全部关节 → 右腿全部关节 → 腰部 → 左臂 → 右臂

`.npz` 运动文件使用 IsaacLab 顺序存储 `joint_pos` / `joint_vel`，但仿真器（MJCF 模型）使用 limb 顺序。将 `.npz` 数据直接写入仿真器会导致关节数据错位（例如左膝角度被写入右髋关节）。

### 排列

```python
_ISAACLAB_TO_MUJOCO = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
```

逆排列（MJCF → IsaacLab）：`[0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 22, 23, 4, 10, 24, 21, 5, 11, 25, 26, 18, 19, 27, 17, 20, 28, 15, 16]`

### 修复位置

| 文件 | 修复方式 |
|------|----------|
| `mdp/commands.py` | `MotionLoader.__init__` 中加载后排列 `joint_pos`/`joint_vel` |
| `mdp/commands_multi_motion.py` | `MultiMotionLoader.__init__` 中加载后排列 |
| `mdp/commands_multi_motion_soccer.py` | `MultiMotionLoader.__init__` 中加载后排列 |
| `scripts/soccer/play_onnx.py` | 删除 `_remap_motion_file`（Loader 已内部处理），保留 ONNX I/O 排列 |
| `scripts/soccer/g1_replay_npz_mjlab.py` | 删除显式排列（`MotionLoader` 已内部处理） |
| `scripts/soccer/g1_replay_npz_mujoco.py` | **保留**显式排列（直接用 `np.load`，不经过 Loader） |
| `scripts/soccer/g1_retarget_t1.py` | **保留**显式排列（直接用 `np.load`，不经过 Loader） |

### 影响

- 已训练的策略模型（使用旧代码训练）：观测向量中的 `command` 项关节顺序与 `joint_pos`/`joint_vel` 不一致，策略已学习隐式重映射。用新代码重训可提高样本效率。
- 已生成的 T1 重定向 `.npz`：旧文件使用了错误的 G1 关节映射，需用修复后的 `g1_retarget_t1.py` 重新生成。

---

## API差异对照表（中文）

以下逐一列出所有强制API差异，并标出差异出现的文件名。每项差异都是MJLab与IsaacLab底层API不兼容导致的，不是风格选择。

### 1. 配置系统：`@configclass` → `@dataclass(kw_only=True)`

| IsaacLab | MJLab | 原因 |
|---|---|---|
| `from isaaclab.utils import configclass` | `from dataclasses import dataclass` | MJLab无`@configclass`装饰器 |
| `@configclass` | `@dataclass(kw_only=True)` | Python标准库`@dataclass`是最接近的替代 |
| `field = SomeCfg(...)`（可变默认值） | `field = field(default_factory=lambda: SomeCfg(...))` | `@dataclass`禁止可变默认值 |
| `scene: SubCfg = SubCfg(...)`（子类字段重定义） | 移至`__post_init__`中赋值：`self.scene = SubCfg(...)` | `@dataclass`不允许子类重定义父类字段 |
| `class TrackingEnvCfg(ManagerBasedRLEnvCfg)` | `class TrackingEnvCfg`（独立类） | `@dataclass`无法重定义父类字段 |

**出现文件：** `tracking_env_cfg.py`, `flat_soccer_env_cfg.py`, `flat_env_cfg.py`

### 2. 包命名空间

| IsaacLab | MJLab | 出现文件 |
|---|---|---|
| `isaaclab.envs` | `mjlab.envs` | 所有文件 |
| `isaaclab.managers` | `mjlab.managers` | 所有文件 |
| `isaaclab.sensors` | `mjlab.sensor` | `flat_soccer_env_cfg.py`, `env_cfgs.py`, `kick_detection.py`, `rewards.py` |
| `isaaclab.scene` | `mjlab.scene` | `tracking_env_cfg.py` |
| `isaaclab.assets` | `mjlab.entity` | `commands.py`, `commands_multi_motion_soccer.py`, `events.py`, `tracking_env_cfg.py` |
| `isaaclab.terrains` | `mjlab.terrains` | `tracking_env_cfg.py` |
| `isaaclab.utils.math` | `mjlab.utils.lab_api.math` | 所有MDP文件 |
| `isaaclab.markers` | （删除） | `commands.py`, `commands_multi_motion_soccer.py` |

### 3. 类型重命名

| IsaacLab | MJLab | 出现文件 | 原因 |
|---|---|---|---|
| `ManagerBasedRLEnv` | `ManagerBasedRlEnv` | 所有MDP文件 | MJLab命名惯例（`Rl`而非`RL`） |
| `Articulation` | `Entity` | `commands.py`, `commands_multi_motion_soccer.py`, `events.py`, `terminations.py` | MJLab统一实体类型，不区分Articulation/RigidObject |
| `RigidObject` | `Entity` | `commands_multi_motion_soccer.py` | 同上 |
| `ArticulationCfg` | `EntityCfg` | `tracking_env_cfg.py`, `flat_soccer_env_cfg.py` | 同上 |
| `RigidObjectCfg` | `EntityCfg` | `flat_soccer_env_cfg.py` | 同上 |
| `InteractiveSceneCfg` | `SceneCfg`或独立`MySceneCfg` | `tracking_env_cfg.py` | MJLab场景API不同 |
| `TerrainImporterCfg` | `TerrainEntityCfg` | `tracking_env_cfg.py` | MJLab地形系统更简单 |
| `AdditiveUniformNoiseCfg` | `UniformNoiseCfg` | `tracking_env_cfg.py` | MJLab噪声类命名 |

### 4. 参数重命名

| IsaacLab | MJLab | 出现文件 |
|---|---|---|
| `asset_name` | `entity_name` | `commands.py`, `commands_multi_motion_soccer.py`, `tracking_env_cfg.py`, `flat_soccer_env_cfg.py`, `flat_env_cfg.py`, `env_cfgs.py` |
| `joint_names`（动作配置） | `actuator_names` | `tracking_env_cfg.py` |
| `prim_path`（传感器） | `name` + `entity`（通过`ContactMatch`） | `flat_soccer_env_cfg.py`, `env_cfgs.py` |

### 5. 实体数据属性重命名

所有`data.body_*_w` → `data.body_link_*_w`，`data.root_*_w` → `data.root_link_*_w`：

| IsaacLab | MJLab | 出现文件 |
|---|---|---|
| `data.body_pos_w` | `data.body_link_pos_w` | `commands.py`, `commands_multi_motion_soccer.py`, `observations.py`, `rewards.py`, `kick_detection.py` |
| `data.body_quat_w` | `data.body_link_quat_w` | `commands.py`, `commands_multi_motion_soccer.py`, `rewards.py` |
| `data.body_lin_vel_w` | `data.body_com_lin_vel_w` | `commands.py`, `commands_multi_motion_soccer.py`, `rewards.py` |
| `data.body_ang_vel_w` | `data.body_link_ang_vel_w` | `commands.py`, `commands_multi_motion_soccer.py` |
| `data.root_pos_w` | `data.root_link_pos_w` | `commands_multi_motion_soccer.py` |
| `data.root_lin_vel_w` | `data.root_link_lin_vel_w` | `rewards.py`（多处） |
| `data.GRAVITY_VEC_W` | `data.gravity_vec_w` | `terminations.py`, `rewards.py` |

原因：MJLab对每个link的属性使用`body_link_*`前缀，属性命名统一用snake_case。
**特别注意：** `body_lin_vel_w`与其他body属性不同。在IsaacLab中它是兼容性别名，
实际解析为`body_com_lin_vel_w`（质心COM速度），而非`body_link_lin_vel_w`（连杆框架速度）。
MJLab将两者作为独立属性暴露，语义不同（link速度 = COM速度 + ω × (link_pos − com_pos)）。
正确移植应使用`body_com_lin_vel_w`。

### 6. 接触传感器API

| IsaacLab | MJLab | 出现文件 | 原因 |
|---|---|---|---|
| `ContactSensorCfg(prim_path="{ENV_REGEX_NS}/SoccerBall", history_length=3, ...)` | `ContactSensorCfg(name="soccer_ball_contact", primary=ContactMatch(mode="body", pattern="soccer_ball", entity="soccer_ball"), secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"), ...)` | `env_cfgs.py` | MJLab用`ContactMatch`模式匹配代替USD `prim_path` |
| `data.net_forces_w_history` [B, H, max_bodies, 3] | `data.force_history` [B, N, H, 3] | `kick_detection.py`, `rewards.py` | MJLab张量布局：[B, bodies, history, 3]；IsaacLab：[B, history, max_bodies, 3] |
| `data.net_forces_w` | `data.force`（回退方案） | `rewards.py`, `kick_detection.py` | 不同字段命名 |

### 7. 场景与物理

| IsaacLab | MJLab | 出现文件 | 原因 |
|---|---|---|---|
| `sim.dt = 0.005` | `sim.mujoco.timestep = 0.005` | `flat_soccer_env_cfg.py`, `env_cfgs.py` | MuJoCo特定的模拟配置 |
| `sim.physics_material = ...` | （删除） | `tracking_env_cfg.py` | PhysX物理材质—MuJoCo用geom属性 |
| `sim.render_interval = ...` | （删除） | `tracking_env_cfg.py` | MJLab viewer独立运行 |
| `sim.physx.gpu_max_rigid_patch_count` | （删除） | `tracking_env_cfg.py` | PhysX特有参数 |
| `viewer.eye / origin_type / asset_name` | `viewer.body_name` | `env_cfgs.py` | MJLab viewer配置不同 |
| `TerrainImporterCfg(prim_path=..., terrain_type="plane", physics_material=..., visual_material=..., collision_group=...)` | `TerrainEntityCfg(terrain_type="plane")` | `tracking_env_cfg.py` | MJLab地形系统更简洁 |
| `TerrainGeneratorCfg` / `terrain_gen.HfRandomUniformTerrainCfg` | （删除） | `flat_soccer_env_cfg.py` | MJLab无高度场地形生成 |
| `RigidBodyMaterialCfg(friction_combine_mode=..., static_friction=..., dynamic_friction=...)` | （删除） | `tracking_env_cfg.py` | MuJoCo用`solref/solimp`控制接触动力学 |
| `MdlFileCfg` / `DomeLightCfg` / `DistantLightCfg` | （删除） | `tracking_env_cfg.py` | MuJoCo用XML光照 |
| `AssetBaseCfg(spawn=...)`（用于灯光） | （删除） | `tracking_env_cfg.py` | 同上 |

### 8. 资产格式

| IsaacLab | MJLab | 出现文件 | 原因 |
|---|---|---|---|
| `RigidObjectCfg(prim_path=..., spawn=sim_utils.UsdFileCfg(usd_path=..., activate_contact_sensors=True))` | `EntityCfg(spec_fn=_get_ball_spec, init_state=...)` | `flat_soccer_env_cfg.py` | MuJoCo用MJCF/XML，不用USD |
| `cfg.scene.soccer_ball.replace(prim_path=...)` | （删除） | `flat_soccer_env_cfg.py` | USD prim_path操作不适用 |
| `SOCCER_ASSET_PATH = f"{ASSET_DIR}/soccer/soccer.usda"` | `Path(__file__).parents[2] / "mdp" / "soccer_ball.xml"` | `flat_soccer_env_cfg.py` | 不同资产格式和位置 |

### 9. 调试可视化（全部删除）

| IsaacLab | MJLab | 出现文件 | 原因 |
|---|---|---|---|
| `VisualizationMarkers` / `VisualizationMarkersCfg` | （删除） | `commands.py`, `commands_multi_motion_soccer.py` | MuJoCo viewer不支持Omniverse标记 |
| `FRAME_MARKER_CFG` | （删除） | `commands.py`, `commands_multi_motion_soccer.py` | 同上 |
| `_set_debug_vis_impl()` / `_debug_vis_callback()` | （删除） | `commands.py`, `commands_multi_motion_soccer.py` | 基于标记的调试可视化不可用 |
| `target_point_marker_cfg` / `target_destination_marker_cfg` | `None` | `commands_multi_motion_soccer.py` | 占位符 |
| `anchor_visualizer_cfg` / `body_visualizer_cfg` | （删除） | `commands.py`, `commands_multi_motion_soccer.py` | 不可用 |

### 10. 材质/COM随机化

| IsaacLab | MJLab | 出现文件 | 原因 |
|---|---|---|---|
| `randomize_rigid_body_material`（PhysX材质桶） | `dr.geom_friction` + `dr.mat_rgba` | `events.py` | MuJoCo无PhysX材质系统 |
| `asset.root_physx_view.get_coms()` / `set_coms()` | `env.sim.model.body_ipos`直接赋值 | `events.py` | MuJoCo无PhysX COM视图 |
| `_randomize_prop_by_op`（从`isaaclab.envs.mdp.events`导入） | 本地定义 | `events.py` | MJLab未重新导出此函数 |

### 11. 环境注册

| IsaacLab | MJLab | 出现文件 | 原因 |
|---|---|---|---|
| `gym.register(id=..., entry_point="isaaclab.envs:ManagerBasedRLEnv", kwargs={...})` | `register_mjlab_task(task_id=..., env_cfg=..., play_env_cfg=..., rl_cfg=..., runner_cls=...)` | `config/g1/__init__.py` | MJLab任务注册系统 |

### 12. RL配置

| IsaacLab | MJLab | 出现文件 | 原因 |
|---|---|---|---|
| `RslRlOnPolicyRunnerCfg` + `RslRlPpoActorCriticCfg`（actor/critic联合） | `RslRlOnPolicyRunnerCfg(RslRlModelCfg, RslRlPpoAlgorithmCfg)`（actor/critic分离） | `rl_cfg.py` | MJLab RL API结构 |
| `actor_hidden_dims=[512, 256, 128]` / `critic_hidden_dims=[512, 256, 128]` | `actor=RslRlModelCfg(hidden_dims=(512, 256, 128))` / `critic=RslRlModelCfg(hidden_dims=(512, 256, 128))` | `rl_cfg.py` | 不同配置嵌套 |
| `init_noise_std=1.0` | `distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}` | `rl_cfg.py` | MJLab分布指定方式 |
| `empirical_normalization = True` | `obs_normalization=True` | `rl_cfg.py` | 不同标志名 |
| `experiment_name = "g1_flat"` | `experiment_name = "g1_soccer"` | `rl_cfg.py` | 任务特定名称 |

### 13. Motion命令配置

| IsaacLab | MJLab | 出现文件 | 原因 |
|---|---|---|---|
| `class_type: type = MotionCommand` | （删除） | `commands.py`, `commands_multi_motion_soccer.py` | MJLab用`cfg.build(env)`代替`class_type`分发 |
| `motion_file: str = MISSING` | `motion_files: list[str] = field(default_factory=list)` | `commands_multi_motion_soccer.py` | MJLab多动作支持 |
| （无） | `resampling_time_range: tuple = (1e9, 1e9)` | `tracking_env_cfg.py`, `flat_soccer_env_cfg.py`, `flat_env_cfg.py` | MJLab `CommandTermCfg`必需字段 |
| （无） | `debug_vis: bool = True` | `tracking_env_cfg.py`, `flat_soccer_env_cfg.py`, `flat_env_cfg.py` | MJLab `CommandTermCfg`必需字段 |

---

## API转换详解（中文）

以下深入解释每个API差异的转换方式（How）和转换原因（Why），结合MJLab与IsaacLab底层实现的具体语义。

### 1. 配置系统：为什么`@configclass`继承链断裂

**IsaacLab的`@configclass`** 是对Python `@dataclass`的增强。它做了两件`@dataclass`做不到的事：

1. **可变默认值**：IsaacLab允许 `field = SomeCfg(...)`，因为`@configclass`会在每次实例化时深拷贝。Python标准`@dataclass`会共享同一个可变对象，导致多实例间状态污染。

2. **子类字段重定义**：IsaacLab允许子类 `G1FlatMotionEnvCfg(TrackingEnvCfg)` 重写 `scene: G1FlatSoccerSceneCfg = ...`。Python标准`@dataclass`会报错：父类已定义了`scene`字段，子类不能重定义。

**转换方式（How）**：
- 可变默认值 → `field(default_factory=lambda: SomeCfg(...))`：每次实例化时执行lambda创建新对象。
- 字段重定义 → `__post_init__`中赋值：`self.scene = G1FlatSoccerSceneCfg(...)`覆盖父类值。
- 继承链断裂 → `env_cfgs.py`工厂函数模式：`make_tracking_env_cfg()`生成基础配置，`_apply_common_soccer_config()`一次性叠加所有层级修改。

**转换原因（Why）**：这两项限制是Python `@dataclass`的固有行为，不是MJLab的选择。MJLab选择标准`@dataclass`而非自己实现`@configclass`，是为了减少依赖和降低维护成本。工厂函数模式虽然不如继承链优雅，但功能完全等价，且在执行流程上更透明（所有修改在一处可见）。

### 2. 包命名空间：为什么路径不同

**转换方式（How）**：全局替换导入路径前缀。

**转换原因（Why）**：
- `isaaclab` → `mjlab`：两个不同的库，包名自然不同。MJLab被设计为IsaacLab的API兼容替代品，所以子模块结构刻意保持一致（`envs`, `managers`, `sensor`, `scene`, `terrains`等），使得导入路径替换后的代码逻辑不变。
- `isaaclab.utils.math` → `mjlab.utils.lab_api.math`：MJLab将数学工具放在`lab_api`子命名空间下，表明这些是从IsaacLab移植而来的"兼容层"。
- `isaaclab.markers`被删除：VisualizationMarkers依赖NVIDIA Omniverse渲染管线，MuJoCo viewer不提供等效API。

### 3. 类型重命名：为什么统一为Entity

**IsaacLab**区分`Articulation`（关节体，如机器人）和`RigidObject`（刚体，如足球）。两者继承自`AssetBase`，有不同的PhysX视图。

**MJLab**统一为`Entity`。MuJoCo模型不区分关节体和刚体——所有body都在同一个`mjModel`中，通过joint定义的类型（free/hinge/ball等）自动判断运动学行为。因此不需要两个类。

**转换方式（How）**：全局替换 `Articulation` / `RigidObject` → `Entity`，`ArticulationCfg` / `RigidObjectCfg` / `AssetBaseCfg` → `EntityCfg`。

**转换原因（Why）**：MuJoCo的统一模型表示使得类型区分没有必要。一个`Entity`实例既可以是有joint的机器人，也可以是无joint的足球。MJLab选择统一类型减少了API表面积。

### 4. 参数重命名：`asset_name` → `entity_name`

**转换方式（How）**：全局替换。

**转换原因（Why）**：既然类型从`Articulation`/`RigidObject`改为`Entity`，参数命名自然从`asset`改为`entity`以保持一致。`joint_names` → `actuator_names`是因为MJLab动作空间按执行器（actuator）索引，而非按关节（joint）索引——两者在多数情况下等价，但在含耦合关节的机器人上可能不同。

### 5. 实体数据属性重命名：`body_pos_w` → `body_link_pos_w`

**IsaacLab**的属性命名：`body_pos_w`表示"body在世界坐标系中的位置"。`w`后缀表示world frame。

**MJLab**的命名约定：
- `body_link_*`前缀明确区分"body link级别"和"root link级别"的属性
- snake_case全小写（`gravity_vec_w`而非`GRAVITY_VEC_W`）
- MJLab中`body_*_w`保留用于"整个刚体的世界坐标"，`body_link_*_w`用于"每个link的世界坐标"

**转换方式（How）**：大部分属性全局替换。这些属性在MJLab的`EntityData`中定义，返回的张量形状和语义与IsaacLab完全相同，仅名称不同。

**`body_lin_vel_w`的特殊情况：** 此属性与`body_pos_w`等其他属性不同。在IsaacLab `ArticulationData`（`articulation_data.py:978-980`）中，`body_lin_vel_w`是一个兼容性别名，直接返回`body_com_lin_vel_w`（质心COM速度）。而`body_link_lin_vel_w`在IsaacLab中是不同的量——它在COM速度基础上加上`ω × (link_pos - com_pos)`偏移以得到连杆框架速度（`articulation_data.py:525-526`）。MJLab将两者作为独立属性暴露。由于原IsaacLab足球代码使用`body_lin_vel_w`（即COM速度），正确移植到MJLab应当使用`body_com_lin_vel_w`，而非`body_link_lin_vel_w`。该差异在快速腿部运动中可能产生数值影响。

**转换原因（Why）**：MJLab选择更明确的前缀命名避免歧义（在多刚体实体中，"body"可能指整个实体或单个link），而snake_case与Python社区惯例一致。

### 6. 接触传感器API：`prim_path` → `ContactMatch`

这是最复杂的API转换。

**IsaacLab的接触传感器**：
- 用USD `prim_path`指定要监测的物体
- `pattern="{ENV_REGEX_NS}/SoccerBall"`匹配所有环境中的SoccerBall prim
- `history_length=3`保存最近3步的接触力
- 输出`net_forces_w_history`：[B, history_length, max_bodies, 3]

**MJLab的接触传感器**：
- 用`ContactMatch`指定primary和secondary对象
- `ContactMatch(mode="body", pattern="soccer_ball", entity="soccer_ball")`：按body名称模式匹配
- `ContactMatch(mode="subtree", pattern="pelvis", entity="robot")`：匹配subtree（含子body）
- `fields=("force",)`：只追踪力（不追踪torque）
- `reduce="netforce"`：计算净力
- `num_slots=1`：每步保留1个slot
- `history_length=4`：保存最近4步历史
- 输出`force_history`：[B, num_bodies, history_length, 3]

**张量布局差异**：
```
IsaacLab: net_forces_w_history[B, H, N, 3]  →  时间步维度在前
MJLab:    force_history[B, N, H, 3]          →  body维度在前
```

**转换方式（How）**：
- 在`env_cfgs.py`中用`ContactMatch`重新定义传感器配置
- 在`kick_detection.py`的`detect()`中：访问`force_history`获取接触力
- 如果`force_history`不可用（`force_history`可能在MJLab的某些版本中叫法不同），回退到`force`
- 处理维度差异：MJLab `force_history`需要`amax(dim=1)`来获取每个body的最大力

**转换原因（Why）**：IsaacLab用USD prim_path定位是因为其底层基于NVIDIA Omniverse/USD场景图。MJLab用模式匹配是因为MuJoCo场景用XML定义，body名称是自然的主键。两者虽然定位方式不同，但传感器输出的物理含义（接触力历史）完全一致。

### 7. 场景与物理：PhysX → MuJoCo

**转换方式（How）**：
- `sim.dt` → `sim.mujoco.timestep`：直接设置MuJoCo模拟步长
- 删除所有PhysX特有配置（`physics_material`, `physx.gpu_max_rigid_patch_count`, `render_interval`）
- 地形从`TerrainImporterCfg(prim_path=..., terrain_type="plane", physics_material=..., visual_material=...)`简化为`TerrainEntityCfg(terrain_type="plane")`
- 地形生成器（`TerrainGeneratorCfg`, `HfRandomUniformTerrainCfg`）全部删除
- 灯光配置（`DomeLightCfg`, `DistantLightCfg`）删除，改用MuJoCo XML
- Viewer配置：`viewer.eye / origin_type / asset_name` → `viewer.body_name`

**转换原因（Why）**：
- **物理引擎**：IsaacLab底层是PhysX 5，MJLab底层是MuJoCo。两个引擎的API完全不同。PhysX用材质桶（material bucket）系统随机化摩擦/恢复系数，MuJoCo用`geom`的`friction`属性和`solref/solimp`接触参数。
- **地形**：IsaacLab的`TerrainImporter`从USD加载地形几何并自动生成碰撞体；MJLab的`TerrainEntityCfg`只支持平面（plane）类型，复杂地形需通过XML自定义。
- **光照**：IsaacLab在USD场景中用`DomeLight`/`DistantLight`；MuJoCo在XML中定义光照，MJLab选择不在Python配置层暴露光照设置。
- **Viewer**：MuJoCo viewer通过body名称跟随实体，不需要相机位置/方向配置。

### 8. 资产格式：USD → MJCF/XML

**IsaacLab**用USD（Universal Scene Description）作为资产格式。球定义为`RigidObjectCfg(spawn=sim_utils.UsdFileCfg(usd_path="soccer.usda", activate_contact_sensors=True))`。

**MJLab/MuJoCo**用MJCF（MuJoCo XML）作为资产格式。球定义为`EntityCfg(spec_fn=_get_ball_spec)`，其中`_get_ball_spec()`返回`mujoco.MjSpec.from_file("soccer_ball.xml")`。

**转换方式（How）**：
- 创建`soccer_ball.xml`：MuJoCo XML定义球的几何（sphere）、质量、惯性等
- `get_soccer_ball_cfg()`工厂函数返回`EntityCfg(spec_fn=_get_ball_spec, init_state=...)`
- `activate_contact_sensors`不再需要：MJLab通过`ContactSensorCfg`独立配置接触传感器

**转换原因（Why）**：MuJoCo原生不支持USD。MJLab选择MJCF作为资产格式是因为：
1. MuJoCo对MJCF有最佳支持（直接解析、优化）
2. MJCF是纯XML，易于手动编辑和调试
3. 避免引入USD解析依赖（usd-core包体积大）

### 9. 调试可视化：为什么全部删除

**IsaacLab的VisualizationMarkers**依赖NVIDIA Omniverse的渲染管线：创建3D标记（球、线框、坐标系轴），在GPU渲染管线中绘制。

**MuJoCo viewer**是一个独立的OpenGL渲染器，不提供等效的"动态标记"API。MuJoCo的`MjvScene`可以添加地标（perturbation），但API完全不同且功能有限。

**转换方式（How）**：
- 所有`VisualizationMarkers`相关代码全部删除
- `_set_debug_vis_impl()`和`_debug_vis_callback()`方法删除
- `target_point_marker`和`target_destination_marker`设为`None`
- 标记配置类型从`VisualizationMarkersCfg`改为`any`（占位）

**转换原因（Why）**：这是MJLab viewer的能力边界，不是API设计选择。未来如果MJLab viewer增加了动态标记支持，可以用类似API恢复这些功能。

### 10. 材质/COM随机化：PhysX材质桶 → MuJoCo域随机化

**IsaacLab的`randomize_rigid_body_material`**：
1. 将物理材质参数（静摩擦、动摩擦、恢复系数）分成`num_buckets`个桶
2. 为每个环境随机选择一个桶
3. 通过`root_physx_view.set_material_properties()`设置PhysX材质属性

**MJLab的实现**：
1. 摩擦部分：取静摩擦和动摩擦范围的均值作为"切向摩擦"范围，调用`dr.geom_friction(env, env_ids, ranges=(min, max), axes=[0])`。`axes=[0]`表示只随机化切向摩擦（tangential），不碰扭转/滚动摩擦。
2. 颜色部分：调用`dr.mat_rgba(env, env_ids, ranges=(0.5, 1.5), operation="scale")`随机缩放材质RGBA颜色。
3. 恢复系数：MuJoCo的恢复系数通过`solref/solimp`控制而非独立材质参数。`restitution_range`参数被接受但不使用。

**COM随机化**：
- IsaacLab：`asset.root_physx_view.get_coms()` → 修改 → `set_coms(coms, env_ids)`
- MJLab：`env.sim.model.body_ipos`（body惯性位置）直接赋值

**转换原因（Why）**：
- MuJoCo的`geom`没有PhysX的"材质"概念。摩擦是geom的直接属性，通过`mjModel.geom_friction`数组访问。MJLab的域随机化工具（`dr.*`）提供批量随机化接口。
- MuJoCo没有运行时COM修改的专用API；直接修改`mjModel.body_ipos`是对模型数据的就地修改，需要在模拟步进前完成。
- 恢复系数的缺失是已知局限性：`solref/solimp`控制接触柔软度和阻尼，不是简单的恢复系数标量。

### 11. 环境注册：`gym.register` → `register_mjlab_task`

**转换方式（How）**：
```python
# IsaacLab:
gym.register(id="Isaac-Velocity-Flat-G1-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv", kwargs={"env_cfg": ...})

# MJLab:
register_mjlab_task(task_id="Mjlab-SoccerTracking-G1", env_cfg=..., play_env_cfg=..., rl_cfg=..., runner_cls=...)
```

**转换原因（Why）**：MJLab有自己的任务管理系统。`register_mjlab_task`不仅注册环境，还绑定RL配置和runner，使得训练（train）和演示（play）模式共享同一注册入口。`play_env_cfg`提供运行时覆盖（无限episode长度、无终止条件、无域随机化）。

### 12. RL配置：联合actor/critic → 分离actor/critic

**IsaacLab**：`RslRlPpoActorCriticCfg`将actor和critic网络定义在同一个配置类中：
```python
class RslRlPpoActorCriticCfg:
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    init_noise_std = 1.0
```

**MJLab**：`RslRlOnPolicyRunnerCfg`分别接受`RslRlModelCfg`（actor）和`RslRlModelCfg`（critic）：
```python
RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(hidden_dims=(512, 256, 128), distribution_cfg={...}),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128)),
    algorithm=RslRlPpoAlgorithmCfg(...),
)
```

**转换方式（How）**：将IsaacLab的扁平字段映射到MJLab的嵌套结构。`init_noise_std`映射到`distribution_cfg["init_std"]`。`empirical_normalization`映射到`obs_normalization`。

**转换原因（Why）**：MJLab将网络架构（`RslRlModelCfg`）与算法超参（`RslRlPpoAlgorithmCfg`）解耦，支持更灵活的配置组合（如不同架构的actor/critic、不同的分布类型）。

### 13. Motion命令配置：`class_type` → `build()`

**IsaacLab的CommandTerm实例化**依赖`class_type`字段：
```python
class_type: type = MotionCommand  # 运行时通过 class_type(env, cfg) 实例化
```

**MJLab的CommandTerm实例化**通过`cfg.build(env)`方法：
```python
@dataclass
class MotionCommandCfg(CommandTermCfg):
    def build(self, env) -> MotionCommand:
        return MotionCommand(self, env)
```

**转换方式（How）**：删除`class_type`字段，在`MotionCommandCfg`中实现`build()`方法。

**转换原因（Why）**：MJLab的`CommandTermCfg`要求子类实现`build()`抽象方法。这是一种更显式的工厂方法模式：配置类自己知道如何构建对应的命令实例，避免了`class_type`的隐式反射。
- `resampling_time_range`和`debug_vis`是MJLab `CommandTermCfg`的必需字段，在IsaacLab中不存在。设置为`(1e9, 1e9)`表示"永不重采样"（由motion自身逻辑控制）。
- `motion_file` → `motion_files`（列表）：MJLab版本支持多动作文件，与`MultiMotionLoader`对应。
