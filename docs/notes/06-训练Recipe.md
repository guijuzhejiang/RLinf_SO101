# 06 · 训练 Recipe

两套：**SmolVLA 主**（已实施）+ **OpenPI 退路**。

## Recipe A · SmolVLA + LoRA RL（主）

加载已有 LoRA SFT ckpt 继续训，base 冻结。代码 [08](08-代码改动清单.md)，配置 [maniskill_ppo_smolvla_so101.yaml](../../examples/embodiment/config/maniskill_ppo_smolvla_so101.yaml)。

### 资产

| 项 | 值 |
|---|---|
| Base | `/media/zzg/GJ_disk01/pretrained_model/lerobot/smolvla_base` |
| LoRA ckpt | `~/workspace/pycharm/lerobot/.../smolvla_lora32_so101_pickplace_real_fps30_hil_v1/.../last/pretrained_model/` |
| LoRA | rank=32, alpha=8（必须跟 SFT 一致） |
| action_dim / chunk_size | 6 / 50（30 FPS ≈ 1.67s） |

LoRA target_modules 正则带 `model.` 前缀，PEFT 必须包在 `SmolVLAPolicy` 顶层（不是 `.model`），否则全 miss。

### 启动

```bash
conda activate py311_leisaac
cd /home/zzg/workspace/pycharm/RLinf
bash examples/embodiment/run_embodiment.sh maniskill_ppo_smolvla_so101
```

启动前确认：LoRA 路径对、ManiSkill3 装好 + `MUJOCO_GL=egl`、`ray stop --force` 清残留。

启动后看：`success_once` 不为 0、`policy_loss` 不爆、`kl` 不漂（漂得严重 `kl_beta` 从 0 调到 1e-3）。

## Recipe B · OpenPI π₀-base + LoRA RL（退路）

SmolVLA 接入卡住（dtype / shape / normalizer）时切。RLinf first-class，模板齐全。

步骤：
1. 用 LeRobot 的 SmolVLA dataset 跑一次 OpenPI LoRA SFT（改 dataset_id + model name）
2. 复制 `maniskill_ppo_openpi.yaml` → SO-101 版
3. 复用 `agents/so101.py` + `tasks/so101_pick_place.py`（跟 SmolVLA 无关）

OpenPI 模板：

```yaml
actor.model:
  model_type: openpi
  num_action_chunks: 5
  action_dim: 7              # OpenPI 默认 7 dim，要改 6
  add_value_head: True
  openpi:
    config_name: pi0_maniskill
    noise_method: flow_sde
```

OpenPI 3B，3090 必须 `is_lora=True` + 小 batch，吞吐比 SmolVLA 低但代码成熟。

## 通用 trade-off

| 参数 | 值 | 说明 |
|---|---|---|
| `lr` actor | `5e-6` | LoRA RL 保留 SFT 必须小 |
| `value_lr` | `1e-4` | value head 新建可大 |
| `total_num_envs` | 8 起步 → 16 | 看显存 |
| `kl_beta` | 0 → 1e-3 ~ 1e-2 | `train/kl` 失控时调 |
| `temperature` | train 1.0 / eval 0.0 | 探索 vs 贪心 |

## 验证脚本

启动前单独验证组件（在 `~/workspace/pycharm/Robot/rlinf_bridge/data/`）：

```bash
python verify_so101_mjcf.py         # SAPIEN 加载 MJCF
MUJOCO_GL=egl python verify_pickplace_task.py  # 任务 reset/step
python verify_smolvla_lora_load.py  # PEFT 套 SmolVLA
python verify_smolvla_flow_sde.py   # flow_sde wrapper 全链路
```

4 个都过了再启训练。
