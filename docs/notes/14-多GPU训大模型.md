# 14 · 多 GPU 训 π0.5 类大模型

下次上 π0.5（3B）/ OpenPI π0（3B）/ GR00T（2B+），单卡装不下。本文是配置升级清单。

## 显存预算（bf16，base 冻结 LoRA）

| 模型 | 参数 | 单卡推理 | 单卡 LoRA 训练 |
|---|---|---|---|
| SmolVLA | 450M | 2-3G | 12-16G（3090 够） |
| π0 base | 3B | 8-10G | 30-40G（A100 40G 紧凑） |
| π0.5 | 3.4B | 9-11G | 35-45G（A100 80G 舒服） |
| GR00T N1 | 2B | 5-7G | 20-28G（4090 紧，A100 舒服） |

**RL 比 SFT 显存高**：要同时保住 rollout（推理）+ actor（更新），collocated 同卡等于双倍。RL 多卡几乎必需。

## 部署模式

### Mode 1：collocated（SmolVLA 当前）

```yaml
cluster.component_placement:
  actor,env,rollout: 0-0
```

actor update 时 rollout 模型 offload CPU；rollout 时 actor offload。**适合**：单卡显存够装 actor + env。

### Mode 2：actor / rollout 分卡（推荐扩展）

```yaml
cluster.component_placement:
  env: 0-0
  actor: 0-0
  rollout: 1-1     # 独占 GPU 1
```

actor 和 rollout 并行；NCCL sync weight。**适合**：2 卡，π0 / GR00T 起步。

### Mode 3：FSDP shard 大模型

```yaml
cluster.component_placement:
  actor: 0-3,      # FSDP 跨 4 卡
  rollout: 4-5,    # TP 跨 2 卡
  env: 6-7

actor.fsdp_config:
  sharding_strategy: "full_shard"
  use_orig_params: True
```

**适合**：π0.5 / 大模型 + 多卡服务器。

## 升级清单（单卡 → 多卡）

目标：**π0.5 LoRA RL，2 张 A100 80G**。

### 1. cluster placement

```yaml
cluster:
  num_nodes: 1
  component_placement:
    actor: 0-0       # actor + env
    env: 0-0
    rollout: 1-1
```

### 2. FSDP 配置

A100 80G 装 3B base + LoRA + activation 富裕，`no_shard` 更快（无 communication）：

```yaml
actor.fsdp_config:
  strategy: "fsdp"
  sharding_strategy: "no_shard"
  use_orig_params: True
  mixed_precision:
    param_dtype: bf16
    reduce_dtype: fp32
    buffer_dtype: fp32
  gradient_checkpointing: False   # 显存够关掉换速度
```

H100 80G 4-8 卡上 π0.5 full FT：

```yaml
actor.fsdp_config:
  sharding_strategy: "full_shard"
  use_orig_params: False         # full FT 时 requires_grad 均匀，关掉省 mem
  mixed_precision:
    param_dtype: bf16
    reduce_dtype: bf16           # 通信用 bf16 省一半带宽
    buffer_dtype: fp32
  gradient_checkpointing: True
  auto_wrap_policy:              # 按 transformer block 包
    module_classes: [LlamaDecoderLayer, ...]
```

### 3. rollout 切 SGLang

```yaml
rollout:
  generation_backend: "sglang"
  gpu_memory_utilization: 0.85
  tensor_parallel_size: 1
```

注意：π0 / π0.5 / GR00T 大部分还没合并进 SGLang 主线，要自适配或暂用 huggingface。

### 4. weight syncer

```yaml
weight_syncer:
  type: "nccl_patch_syncer"      # GPU-GPU NCCL 直接同步
```

LoRA RL 只 sync LoRA delta，通信量小。

### 5. batch size

```yaml
env.train.total_num_envs: 32      # A100 能 batch 32
actor:
  micro_batch_size: 4             # 80G 舒服
  global_batch_size: 32           # gradient accumulation = 8
```

### 6. 启动

```bash
export RAY_DISABLE_DOCKER_CPU_WARNING=1
ray start --head --num-gpus=2
python examples/embodiment/train_embodied_agent.py --config-name maniskill_ppo_pi05_so101
```

多机：

```bash
# 头节点
export RLINF_NODE_RANK=0
export RAY_HEAD_NODE_IP=<...>
ray start --head ...

# 其他节点
export RLINF_NODE_RANK=1
ray start --address=<head_ip>:<port>
```

**必须先 export 再 ray start**，Ray 启动一次性快照环境变量。

## 多卡新坑

| | 现象 | 解 |
|---|---|---|
| **A** | FSDP `full_shard` + LoRA 必须 `use_orig_params=True` | 失去通信优化但必须 |
| **B** | weight sync 成新瓶颈 | LoRA 只 sync delta OK（几 MB），full FT sync 几 G 会卡 → SHM 或 NCCL 直传 |
| **C** | NCCL deadlock | `export NCCL_BLOCKING_WAIT=1 NCCL_TIMEOUT=1800` |
| **D** | actor offload 后 OOM | 缓存碎片化 → 每次 reload 前 `torch.cuda.empty_cache()` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| **E** | ckpt 一致性 | 多卡 FSDP 每卡只有 shard → `save_full_state_dict`（慢但单文件）或 sharded（快但要保持卡数一致） |

## 推荐分阶段推进

1. 单 A100 80G 上 π0.5 LoRA 跑通（不改架构只改 yaml `model_path` + 显存参数）
2. actor / rollout 分 2 卡（Mode 2）
3. actor FSDP shard（full_shard）跑 full FT
4. 上 SGLang rollout，吞吐 ×3

每步跑稳了再推下一步，不要一次性全改。
