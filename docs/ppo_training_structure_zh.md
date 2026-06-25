# RLinf: Conda 环境下 IsaacLab 与 OpenPI 集成开发指南

本作业文档汇录了在非官方推荐的 Conda 环境下，配置运行 RLinf 具身智能训练时遇到的常见问题、环境补丁方案以及 PPO 训练逻辑解析。

---

## 一、 环境架构逻辑 (Conda vs. Venv)

RLinf 官方 `install.sh` 脚本默认强制使用 `.venv` 环境。在自定义 Conda 环境（如 `py312_cu121`）中运行时需遵守以下原则：

- **补丁手动化**：脚本中自动执行的源码修改（如 `transformers` 的猴子补丁）必须手动操作。
- **路径显式化**：必须手动设置 `ISAACSIM_PATH` 或 `ISAAC_PATH` 环境变量，否则 `AppLauncher` 无法加载 `SimulationApp`。
- **环境隔离**：在 Conda 激活状态下，确保不要意外触发 `install.sh` 再次创建 `.venv`。

---

## 二、 核心错误排查与修复 (Troubleshooting)

### 1. OpenPI 源码补丁报错 (`ValueError: transformers_replace`)
*   **现象**：模型初始化时提示 `transformers_replace is not installed correctly`。
*   **根因**：OpenPI 需要对 Hugging Face 的 `transformers` 库进行源码级修改才能运行，此修改文件位于 `openpi/src/openpi/models_pytorch/transformers_replace/`。
*   **修复**：手动将该目录下的所有内容拷贝到 Conda 环境的 `site-packages/transformers/` 目录下。
    ```bash
    cp -r /path/to/openpi/src/openpi/models_pytorch/transformers_replace/* /your/conda/env/lib/python3.12/site-packages/transformers/
    ```

### 2. Isaac Sim 导入失败 (`TypeError: 'NoneType' object is not callable`)
*   **现象**：`AppLauncher` 在创建 app 时报 `NoneType` 错误。
*   **根因**：程序未在系统路径中找到 Isaac Sim 安装目录，导致 `SimulationApp` 被解析为 `None`。
*   **修复**：在 shell 环境中导出正确的路径。
    ```bash
    export ISAACSIM_PATH="/path/to/your/isaac-sim"
    ```

### 3. 环境 ID 未注册 (`gymnasium.error.NameNotFound`)
*   **现象**：提示 `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0` 不存在。
*   **根因**：该 ID 是 RLinf 自定义的 Rewarded 变体，IsaacLab 官方源码（`isaaclab_tasks`）中未包含此注册。
*   **修复**：在 IsaacLab 源码的对应任务 `__init__.py` 中手动添加注册代码：
    ```python
    gym.register(
        id="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.stack_ik_rel_visuomotor_env_cfg:FrankaCubeStackVisuomotorEnvCfg",
            "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_image_200.json",
        },
        disable_env_checker=True,
    )
    ```

---

## 三、 PPO 训练模块交互与参数解析

### 1. 模块角色 (Analogy)

| 模块名称 | 形象比喻 | 职能描述 |
| :--- | :--- | :--- |
| **Env** | **考场** | 提供物理模拟、任务目标和奖励反馈。 |
| **Rollout** | **考生** | 带着当前策略去考试，记录下动作和得分，形成“经验包”。 |
| **Actor** | **大脑技能中心** | 负责决策（策略输出）。 |
| **Critic** | **评分预估中心** | 负责预测当前的潜在得分（价值估计）。 |

### 2. 交互流程
1.  **Actor** →(传出权重)→ **Rollout**
2.  **Rollout** ↔(互动)↔ **Env** (持续运行 `rollout_epoch` 次采集数据)
3.  **Rollout** →(传出经验数据包)→ **Actor/Critic**
4.  **Actor/Critic** (利用数据包进行 `update_epoch` 次学习)
5.  **Actor/Critic** →(传出新版本权重)→ **Rollout** (周而复始)

### 3. 关键参数对数据流的影响
*   **`rollout_epoch`**: 每次更新前要采集多少“轮”数据。单轮最大数据量由 `total_num_envs * max_steps_per_rollout_epoch` 决定。
*   **`update_epoch`**: 拿到一包数据后，模型会反复复习多少次。PPO 算法通常设置为 3-10 次以提高采样利用率。
*   **`global_batch_size`**: 每次优化权重时看的数据条数。对于 VLA 等大模型，通常受限于显存而设置得较小（如 8）。

---

## 四、 实践演算示例
以常用配置为例：`rollout_epoch: 2`, `total_num_envs: 32`, `update_epoch: 3`

1.  **准备**：32 个环境启动。
2.  **采样**：每个环境跑两轮，产生约 28,000 个采样点（Steps）。
3.  **学习**：模型分批（每批 8 个点）循环学习 3 遍，完成后更新一次权重，开启下一次大循环。

---

## 五、 显存优化指南 (VRAM Optimization)

针对 3090 (24GB) 或更小显存的显卡，运行 Pi05 等大模型训练时建议：

| 方案 | 参数设置 | 作用描述 |
| :--- | :--- | :--- |
| **使用 LoRA** | `is_lora: True` | 冻结主干模型，只训练极少量的适配器权重，大幅降低显存和优化器消耗。 |
| **微批次调优** | `micro_batch_size: 1` | **最有效手段**。通过降低单次反向传播的样本数，利用梯度累积（Grad Accumulation）保持总批大小。 |
| **显存卸载** | `enable_offload: True` | 将优化器状态和部分模型权重卸载到 CPU RAM（内存），腾出宝贵显存给激活值（Activations）。 |
| **混合精度** | `precision: "bf16"` | 使用 BFloat16 精度（3090 原生支持），相比单精度（Float32）可节省近 50% 显存。 |
| **减少环境数** | `total_num_envs` | 适当减少并行环境数量，可以减轻采样缓存和推理时的显存压力。 |

> [!WARNING]
> **注意**：在 OpenPI + FSDP 架构下，`gradient_checkpointing` 目前可能存在兼容性限制（默认为 False），建议优先通过调整 `micro_batch_size` 来解决 OOM 问题。
