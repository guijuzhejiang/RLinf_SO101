# 05 · 自定义 PickPlace 任务

把 `pick_place.xml` 的场景搬到 ManiSkill3。代码在 [tasks/so101_pick_place.py](../../rlinf/envs/maniskill/tasks/so101_pick_place.py)、agent [agents/so101.py](../../rlinf/envs/maniskill/agents/so101.py)。设计原则看 [03](03-ManiSkill3入门.md)，详细改动看 [08](08-代码改动清单.md)。

## 原 MuJoCo 场景

| 项 | 值 |
|---|---|
| 机械臂 | SO-ARM 101，6 DOF |
| MJCF | `~/workspace/pycharm/Robot/assets/so101_pick101/so101_new_calib.xml` |
| 红方块 | 2 cm³，初始 `(0.18, -0.04, 0.01)` |
| 白盘子 | r=6cm × h=0.4cm，初始 `(0.28, 0.04, 0.002)` |
| 真机 FPS | 30 |
| Home pose | `qpos = [0, 0, 0, π/2, π/2, 0.3]` |

## MJCF 加载

SAPIEN 3.0.3 `Scene` 没 `create_mjcf_loader()`（是 ManiSkill 的 `ManiSkillScene` 子类加的）；mani_skill 3.0.1 的 `mjcf_loader.py` wrapper 有 bug。绕过 wrapper 直接用底层：

```python
from mani_skill.utils.building._mjcf_loader import MJCFLoader  # 下划线
loader = MJCFLoader()
loader.set_scene(scene)
art_builders, actor_builders, cameras = loader.parse(XML_PATH)
art = art_builders[0].build()
```

但实际接入 agent 时只要 `mjcf_path` 设好，ManiSkill 内部就会走对的逻辑，不用自己调 loader。

## reward 与 success

```
reward = 0.3 * reach        # (1 − tanh(5·|gripper − cube|))
       + 2.0 * grasp        # is_grasped
       + 1.5 * move         # is_grasped × (1 − tanh(5·|cube − plate|))
       + 2.0 * on_plate     # xy<5cm 且 z 合理
       + 5.0 * success      # 放下且在 plate 里
```

最后 `/ 11.0` 归一化到 [0,1]。

**成功判定**：cube xy 到 plate 中心 < 5cm、相对 plate 表面 z 在 4mm–4cm、**且松手**（避免抓着 hack 通关）。

## 真机相机参数

| 相机 | 真机 | sim |
|---|---|---|
| 顶部 | 夹爪前方 ~50cm、上方 ~50cm 俯视 | `eye=[0.70, 0, 0.55], target=[0.05, 0, 0.05]`, FOV 60° |
| 腕部 | MJCF `<camera name="wrist_cam">` | 偏到 gripper local -Y 侧绕过 wrist-roll 电机遮挡 |

## 桌面颜色对齐真机 + 预览脚本

ManiSkill 默认 `TableSceneBuilder` 的 `table.glb` 是冷色木桌，跟真机土黄色木桌不像。改色尝试过 mutate `part.material.set_base_color(...)`，CPU/GPU 后端都不生效（命中数 > 0 但渲染不变），原因可能是 GLB 用 specular workflow 或被命中的 part 不是相机看到的 tabletop mesh。最终方案：在 `_load_scene` 里调 `_build_table_color_overlay(color=[R,G,B,1])` 叠一层薄 box（visual + collision），跟 `build_cube(color=...)` 同机制 —— 材质 build 时设定，稳定生效。改色只需改 `color=` 实参。

**预览脚本**：[examples/embodiment/preview_table_color.py](../../examples/embodiment/preview_table_color.py)

| 项 | 说明 |
|---|---|
| 功能 | 不跑训练，单 env reset 一次，渲染当前 overlay 颜色 + cube/plate/机械臂场景 |
| 用途 | 改完 `color=` 后秒看效果，调到对齐真机照片再开训 |
| 输入 | `--out-dir ./preview_out`（默认）、`--seed 0`、`--sim-backend cpu\|gpu`（默认 cpu） |
| 输出 | 3 张 640×480 PNG：`render_camera.png`（录视频视角）、`base_camera.png`（模型输入）、`wrist_camera.png` |
| 跑法 | `python examples/embodiment/preview_table_color.py` |

启动时会 grep 源码里的 `TABLE_TINT = [...]` 行打印当前生效色（确认改了哪一版），无头模式默认 `MUJOCO_GL=egl`，远程 3090 也能跑。
