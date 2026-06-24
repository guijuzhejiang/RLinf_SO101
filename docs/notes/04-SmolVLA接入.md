# 04 · SmolVLA 接入 RLinf

实施细节看 [08](08-代码改动清单.md)。本文是设计依据。

## SmolVLA vs 其它 VLA

| | OpenVLA | OpenPI π₀ | SmolVLA |
|---|---|---|---|
| 参数 | 7B | 3B | 450M |
| Backbone | Llama-7B + SigLIP | PaliGemma 3B | SmolVLM2 256M |
| Action decoder | 离散 token | Flow matching | Flow matching |
| 接口 | HF `generate()` | LeRobot policy | LeRobot policy |

SmolVLA 跟 OpenPI 都是 LeRobot + flow matching，所以 wrapper **照抄 [openpi/](../../rlinf/models/embodiment/openpi/) 的目录结构和接口**，把模型和数据换成 SmolVLA 即可。OpenVLA-OFT 不可参考（离散 token 跟 flow 不兼容）。

## 调用链

```
YAML model_type=smolvla
  → config.py SupportedModel
  → rlinf/models/embodiment/smolvla/__init__.py get_model(cfg)
  → SmolVLAPolicy.from_pretrained → PeftModel.from_pretrained → SmolVLAForRLActionPrediction
  → forward(forward_type=DEFAULT) → default_forward → logprobs/values/entropy
  → predict_action_batch(env_obs) → actions, {prev_logprobs, prev_values, ...}
```

## SmolVLA 内部结构

```
SmolVLAPolicy (451M)
└── model: VLAFlowMatching
    ├── vlm_with_expert (448M)
    │   ├── vlm: SmolLM2-500M (350M, 含 SigLIP)
    │   └── lm_expert: LlamaModel (98M, 16 层，专门生成 action)
    └── 5 个 projection (机器人 ↔ VLM 接口)
        ├── state_proj         32 → 960   关节状态 → VLM hidden
        ├── action_in_proj     32 → 960   含 noise 的 action → expert
        ├── action_out_proj    960 → 32   expert → 真实 action (velocity)
        ├── action_time_mlp_in 1920 → 960 timestep+action emb 拼接
        └── action_time_mlp_out 960 → 960 timestep emb MLP
```

**5 个 projection 的本质**：机器人世界 ↔ VLM 表示空间的所有接口。承担"把机器人任务知识灌进模型"的薄层。

**术语区别**：SmolVLA "projection" ≠ VLA 架构里的 "projector"（vision encoder → LLM 的桥接 MLP，OpenVLA/LLaVA 有，SmolVLA 没有）。

## state / action / chunk 三个维度

第一天最容易混的概念：

| 名字 | shape | 含义 |
|---|---|---|
| **state** | `[6]` | 当前 6 个关节实测位置（输入） |
| **单步 action** | `[6]` | 下一步 6 个关节目标位置（输出） |
| **action chunk** | `[50, 6]` | 未来连续 50 步的目标关节序列（一次决策） |
| **flow noise** | `[50, 6]` | SDE 注入的 Gaussian，跟 chunk 同 shape |

**state 和单步 action 数字都是 6 但语义不同**：state 输入"我在哪"，action 输出"我要去哪"。能对齐是因为选了 `pd_joint_pos`，action 直接是关节目标角；换 `pd_ee_delta_pose` action 就变成末端位姿增量，两个 6 不是一个 6。

**action chunk 不是 6 维**，是 `[50, 6] = 300 个数`。flow matching 把整 chunk 当**一次**联合 Gaussian 采样，所以 `algorithm.logprob_type: chunk_level` 必须配 —— `token_level` 把 300 值当 300 次独立决策做 clip，违反 flow_sde 设计。

**padding 到 32 的原因**：SmolVLA `max_state_dim = max_action_dim = 32`，SO-101 只用前 6 维，后 26 维填 0，切别的机器人架构不用动。

## LeRobot 的 LoRA SFT 范式

LeRobot 把 PEFT default 写死在 SmolVLA 源码里：

```python
common_projections = "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
target_modules = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
```

**冻结/LoRA 分布**：

| 模块 | LoRA | 个数 |
|---|---|---|
| `vlm.vision_model` (SigLIP) | ❌ 完全冻结 | - |
| `vlm.text_model` | ❌ 完全冻结 | - |
| `lm_expert.q_proj / v_proj` | ✅ rank=32 | 32 |
| `lm_expert.k_proj / o_proj / mlp.*` | ❌ 冻结 | - |
| 5 个 projection | ✅ rank=32 | 5 |
| `value_head` (RL 新加) | ❌ 全量 trainable | - |

**总数 37 个 nn.Linear，trainable ≈ 1.49M / 451M = 0.33%**，`r=32, alpha=8, scale=0.25`（比常规 α=r 保守 4 倍）。

**为什么只 LoRA q/v 不动 k/o/MLP**（LoRA 原论文结论）：
- q+v 是 minimal optimal，同预算效果最好
- q "我关注什么"，v "从被关注位置提取什么"，attention 语义核心
- k 跟 q 对偶，边际收益小；o 只是线性 reshape；MLP 不参与跨 token 信息流

**为什么 VLM 完全不动**：SmolVLA 只 450M，VLM 全 LoRA 化反而破坏 SmolLM2 的通用视-语理解。真正承担任务的薄层 = 5 projection + lm_expert attention 核心。

## RLinf 的 3 种 LoRA 范式对比

| 范式 | 用谁 | 冻结/LoRA | trainable% |
|---|---|---|---|
| **A. 广撒网** | OpenVLA / OpenVLA-OFT | 全冻结 base，vision + projector + LLM 几乎所有 Linear | ~1-2% |
| **B. 半冻结** | OpenPI `is_lora=True`（罕用） | VLM 冻结，LoRA 加 VLM；expert 全量训 | 部分 |
| **C. 最小扰动** | **SmolVLA（我们）** | 全冻结 + VLM 不加 LoRA；只 lm_expert q/v + 5 projection | **0.33%** |

OpenPI / Pi0 / GR00T / DexBotic 默认 `is_lora=False`（full fine-tune）。

**我们走 C 不走 A**：SmolVLA SFT 的 `adapter_config.json` target_modules 是 LeRobot 正则；RLinf 外层 PEFT 是写死字符串列表，两种语法不兼容。而且外层走 `LoraConfig + get_peft_model` **创建全新 LoRA**，不读 `lora_path` → 把已有 LoRA 扔掉重新随机初始化。所以必须绕开外层自己跑 `PeftModel.from_pretrained` 加载 SFT LoRA。

代价是 FSDP1 writeback bug，已修（`_fsdp_ignored_modules` + `_fsdp_force_lora_wrap` 双标志），见 [11 #2.5](11-踩坑总结.md)。
