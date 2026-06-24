# 03 · ManiSkill3 入门

## 是什么

[ManiSkill3](https://github.com/haosulab/ManiSkill) = SAPIEN 物理引擎 + Gym 接口 + GPU 并行 sim。

- **GPU 并行 env**：单卡几百 env，吞吐比 CPU sim 高 10-100×
- 物理 SAPIEN（PhysX）+ 渲染 Vulkan
- 任务接口 `gymnasium.Env` 风格

跟 **MuJoCo** 是同类但不同栈：SAPIEN 能直接加载 MJCF（绕过 ManiSkill 的 `MJCFLoader` wrapper）。跟 **LeRobot**（真机库）没关系。

RLinf 通过 [maniskill_env.py](../../rlinf/envs/maniskill/maniskill_env.py) 把 ManiSkill 包成 rollout worker 协议的 vectorized env。

## 安装

⚠️ RLinf `embodied` extra **不包含 ManiSkill3**，要补装：

```bash
pip install "git+https://github.com/haosulab/ManiSkill.git@v3.0.0b22"
python -c "import mani_skill, sapien; print(mani_skill.__version__, sapien.__version__)"
```

资源（默认 `~/.maniskill/data/`）：

```bash
python -m mani_skill.utils.download_asset PickCube-v1
python -m mani_skill.utils.download_asset ycb

# RLinf 任务（PutOnPlateInScene25Main 等）需要背景图
cd rlinf/envs/maniskill && export HF_ENDPOINT=https://hf-mirror.com
hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets
```

## 跑通内置任务

```python
import gymnasium as gym
import mani_skill.envs

env = gym.make("PickCube-v1", obs_mode="rgb",
               control_mode="pd_ee_delta_pose",
               render_mode="rgb_array", num_envs=4)
obs, info = env.reset(seed=0)
for _ in range(10):
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
env.close()
```

看到 `reward.shape == (4,)` 即 GPU vectorized 起来了。

## 自定义任务的核心原则：跟 SFT 分布对齐

代码细节看 [05](05-自定义PickPlace任务.md)；这里只讲**为什么这么设计**。

**模型看得见的任何维度跟 SFT dataset 有偏差，rollout action 就是 OOD，PPO 救不回来**。

对齐 6 个维度（[so101_pick_place.py](../../rlinf/envs/maniskill/tasks/so101_pick_place.py)）：

| 维度 | 实现 |
|---|---|
| **控制/物理频率** | `SimConfig(sim_freq=300, control_freq=30)`。真机 30 FPS，必须严格对齐；300 是 PhysX 子步稳定需要 |
| **物体几何** | cube `half=0.008`（真物 1.6 cm）；plate `outer=0.08, base=4mm, rim=2cm`。SAPIEN primitive 无凹陷 mesh，盘子用底盘 + 16 段 box 拼外缘。cylinder 局部 X 是长轴，要 `quat=euler2quat(0, π/2, 0)` 才水平 |
| **画面物体分布**（最关键） | `plate=(0.28, -0.22)` 固定左下角；`cube x∈[0.22,0.30], y∈[-0.04,0.06]` 随机；`min_cube_plate_dist=0.10` 单位向量推开避免初始就 success |
| **相机视角** | 顶部 640×480 FOV 60°；名字必须叫 `base_camera`（RLinf `wrap_obs_mode=simple` 硬约定）；`render_camera` 复用同 pose，录视频跟模型看的一致 |
| **桌面颜色** | 真机暖黄木桌，PBR 出来偏深红。**只改 material `base_color` (`[1.0, 0.78, 0.50, 1.0]`)，不动 mesh**。PBR 颜色 = tint × wood_texture，物理行为不变 |
| **机器人** | `SUPPORTED_ROBOTS=["so101"]`；MJCF 直接加载（SAPIEN 原生支持），不用转 URDF |

### 对齐 vs 不对齐的边界

| 看得见 → 必须对齐 | 看不见 → 随便设计 |
|---|---|
| 几何、画面分布、相机、桌面颜色、控制频率 | reward 函数、success 判定、`max_episode_steps` |

### 对齐失败 = silent failure

不会报错，loss 也下降，但 `success_once` 永远 0。两个典型坑：

- **视觉端**：SFT 用 3 路（2 真 + 1 `empty_cameras=1` 占位），RL 只塞 2 路漏占位，VLM cross-attn token 序列长度少一路
- **控制端**：仿真喂 raw qpos（弧度），SFT 用 normalized 度数（差 57.3× ÷ std）

**接新 base 模型必须先端到端跑一次 obs/action 完整路径，把每个 tensor 的量级、分布、key 命名跟 SFT dataset 对一遍**。

## 常见问题

| 问题 | 解决 |
|---|---|
| Vulkan 报错 | `sudo apt install libvulkan1 vulkan-tools`；无 root 用 EGL：`export MUJOCO_GL=egl SAPIEN_RENDERER_DEFAULT_DEVICE=cuda:0` |
| N 个 env 显存爆 | 降 `num_envs` 或 `obs_mode="state"` 跳渲染 |
| FPS 慢 | `obs_mode="state"`，减相机数/分辨率，用 `pd_joint_pos` 省 IK |

## MuJoCo vs ManiSkill3

| | MuJoCo | ManiSkill3 |
|---|---|---|
| 描述 | `.xml` (MJCF) | Python API + URDF，也能 MJCF |
| 物理 | C++ | PhysX (GPU) |
| 渲染 | OpenGL/EGL | Vulkan |
| 并行 | 多进程 | 单进程 GPU vectorized |
| 相机 | `<camera>` XML | `CameraConfig` Python |

参考：[ManiSkill3 文档](https://maniskill.readthedocs.io/)，源码示例 `python -c "import mani_skill; print(mani_skill.__path__)"`。
