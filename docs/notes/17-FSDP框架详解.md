# 17 · FSDP 框架详解（RL 训 VLA 视角）

代码：[fsdp.py:419 wrap_model](../../rlinf/hybrid_engines/fsdp/strategy/fsdp.py#L419)。配置：[maniskill_ppo_smolvla_so101.yaml:183](../../examples/embodiment/config/maniskill_ppo_smolvla_so101.yaml#L183) `fsdp_config`。

## 1. 一句话

**FSDP = Fully Sharded Data Parallel**，PyTorch 原生分布式策略。模型参数 / 梯度 / 优化器状态按 rank 切片存到不同 GPU，**用时 all-gather 凑齐，用完释放**。等价 DeepSpeed ZeRO-3 的 PyTorch 官方实现。

### 跟 DDP 的本质区别

| | DDP | FSDP |
|---|---|---|
| 模型权重 | 每卡**完整**一份 | 每卡只存一**片** |
| 梯度 | 每卡完整 | 每卡只存自己片 |
| Opt state | 每卡完整（AdamW 2× 参数 fp32） | 每卡只存自己片 |
| 通信 | 每 step 一次 all-reduce grad | 每层 forward 前 all-gather param + backward 后 reduce-scatter grad |
| 显存 | 高（多卡冗余） | 低（线性 1/N） |
| 通信量 | 低 | 高（trade compute for memory） |
| 能跑多大 | 单卡装得下 | 多卡总和能装下 |

**用 FSDP 的根本原因**：训 3B+ VLA（π0.5、GR00T）单卡 80G 也装不下。

## 2. 为什么 RL 训 VLA 单卡也用

SmolVLA 450M 单卡装得下，RLinf 仍包成 FSDP `no_shard`。看似多此一举，**其实不是**，单卡 FSDP 提供 4 个 DDP 没有的能力：

1. **统一 mixed precision API**：bf16 param + fp32 reduce + fp32 buffer 一行配
2. **offload 钩子**：[fsdp.py:461](../../rlinf/hybrid_engines/fsdp/strategy/fsdp.py#L461) `offload_param_and_grad` 直接操作 `flat_param.data` 整片搬 CPU。**RL collocated 训练核心机制**
3. **gradient checkpointing 集成**：FSDP 知道 wrap unit，checkpoint 边界自动对齐
4. **无缝升级**：今天 `no_shard`，明天 π0.5 直接 `full_shard`，代码一行不动

```yaml
fsdp_config:
  strategy: "fsdp"
  sharding_strategy: "no_shard"
  gradient_checkpointing: False
  use_orig_params: True              # LoRA 必需
  mixed_precision:
    param_dtype: bf16                # forward / backward
    reduce_dtype: fp32               # grad reduce 保精度
    buffer_dtype: fp32               # BN / LN running stats
```

## 3. 工作原理 · 一次 forward+backward

假设 4 卡 `full_shard`，16 transformer block 每块一个 wrap unit。

### 初始化

```
Param P 总 4 GB，切 4 片，每卡 1 GB（flat_param.data）
其余 3 片在别人卡上
```

`wrap_model` ([fsdp.py:419](../../rlinf/hybrid_engines/fsdp/strategy/fsdp.py#L419)) 完成：
- `auto_wrap_policy`：决定哪些 module 一起 flatten。Transformer 按 block 切，每 `LlamaDecoderLayer` 一个 unit
- `flat_param`：unit 内所有 `nn.Parameter` 物理拼成一维 buffer 按 rank 切。后续通信和 offload 都对 flat_param 操作

### Forward

```
For block i in [0..15]:
    [1] all_gather(flat_param[i])    ← 4 卡凑齐
    [2] block_i.forward(x)
    [3] free unsharded flat_param    ← 立刻释放
```

任一时刻 GPU 只有"当前层" full param，其他层只 1/N。峰值显存 ≈ 1/N 模型 + 1 层 full + activation。

`forward_prefetch=True` 时 block i forward 没结束就开始 all_gather block i+1，掩盖延迟。

### Backward

```
For block i in [15..0]:
    [1] all_gather(flat_param[i])
    [2] block_i.backward(grad)
    [3] reduce_scatter(grad)         ← grad 按 rank reduce-scatter，每卡留自己片
    [4] free unsharded flat_param + full grad
```

反向 reduce-scatter 一步搞定 DDP 里 all-reduce 的活，**显存峰值不暴涨**（DDP 反向是 full grad）。

### Optimizer step

```
For each shard on this rank:
    optimizer.step(grad_shard, param_shard)   只对自己 1/N 片 update
    optimizer state (m, v) 也 1/N
```

**FSDP 比 DDP 省显存最猛的地方**：AdamW m+v 都是 fp32（param 量级）。3B 模型 bf16 weight = 6 GB，但 AdamW state = 24 GB。FSDP 切成 N 份每卡 24/N。

## 4. 四种 sharding_strategy

代码 [utils.py:727](../../rlinf/hybrid_engines/fsdp/utils.py#L727) `get_sharding_strategy`。

| 策略 | 切什么 | 不切 | 显存 | 通信 | 何时 |
|---|---|---|---|---|---|
| `no_shard` | 啥都不切 | param/grad/opt 完整 | 最高（≈DDP） | 最少 | 单卡，或多卡富余 |
| `shard_grad_op` | grad + opt | param 完整 | 中（ZeRO-2） | 中 | 中等模型，多卡带宽紧 |
| `hybrid_shard` | 节点内 full，节点间复制 | — | 中 | 跨节点无 | 多节点，跨节点带宽差 |
| `full_shard` | param + grad + opt 全切 | 啥都不留 | 最低（ZeRO-3） | 最多 | 大模型，单卡装不下 |

- SmolVLA 单卡：`no_shard`，模型一卡放下，切了引入伪通信开销
- π0.5：`full_shard` + 2-4 卡，见 [14](14-多GPU训大模型.md)

## 5. `use_orig_params=True` 为什么 LoRA 必须

LoRA 训练同 transformer block 里：base `requires_grad=False`，lora_A/B `requires_grad=True`。

FSDP 默认 `use_orig_params=False` 用 `flat_param` 把同 unit 所有 param 拍平成一维 buffer 优化通信粒度。**要求 flat_param 里所有元素 `requires_grad` 一致**——LoRA 混着冻+训违反约束，init 时 assert。

`use_orig_params=True` 保留每个 param 原始 `nn.Parameter` view：
- flat_param 物理上还是连续 buffer
- 每个 `nn.Parameter` 是 buffer 上的 view，独立 `requires_grad`
- 通信粒度变细，对 LoRA 这种小 trainable subset 不亏

**代价**：
- `clip_grad_norm_` 实现要分 sharded / non-sharded 走（[fsdp.py:589](../../rlinf/hybrid_engines/fsdp/strategy/fsdp.py#L589)），RLinf 已处理
- `model.parameters().dtype` 返回原 fp32，看真实 dtype 要 `module.weight.dtype` 直接读（dtype mismatch 踩坑就因这）

## 6. Mixed Precision 三个 dtype

```yaml
mixed_precision:
  param_dtype: bf16       # forward / backward
  reduce_dtype: fp32      # 跨卡 reduce
  buffer_dtype: fp32      # BN / LN running mean
```

- **`param_dtype: bf16`**：FSDP 把 fp32 master cast bf16 给 forward，省一半显存。VLA transformer bf16 完全够
- **`reduce_dtype: fp32`**：跨卡 reduce grad 升回 fp32 避免精度累积。bf16 只 7 bit mantissa，N 卡 reduce round-off 漂移。单卡（`no_shard`）配不配影响不大，但保留以便升多卡
- **`buffer_dtype: fp32`**：Buffer = 不参与梯度但跟着 forward 的状态。SmolVLA 主要是 position embedding cache

**master copy 机制**：FSDP 维护 fp32 weight master + bf16 working。forward/backward bf16；opt.step fp32（保精度）；step 完 cast 回 bf16。Opt state (m, v) 也 fp32。

## 7. Offload：RL 训练 FSDP 最关键的 RL-specific 用法

VLA RL collocated 同卡 actor + rollout 抢显存。RLinf 用 FSDP offload 做"谁干活谁上 GPU，闲着下 CPU"：

```
Rollout 阶段：
  actor.offload_param_and_grad(actor_fsdp, offload_grad=True)
  rollout.onload_param(rollout_model, device='cuda')
  跑 8 × 200 step 收集 trajectory

Update 阶段：
  rollout.offload_param(rollout_model)
  actor.onload_param_and_grad(actor_fsdp, device='cuda')
  跑 PPO loss + backward + step
```

代码：
- offload [fsdp.py:461](../../rlinf/hybrid_engines/fsdp/strategy/fsdp.py#L461)
- onload [fsdp.py:506](../../rlinf/hybrid_engines/fsdp/strategy/fsdp.py#L506)
- opt offload [fsdp.py:552](../../rlinf/hybrid_engines/fsdp/strategy/fsdp.py#L552)

**实现要点**：
- 直接搬 `flat_param.data` 整片 tensor 到 CPU。粒度粗 + `non_blocking=True` 异步
- offload 后立刻调 `_rebind_handle_views(handle)` ([fsdp.py:374](../../rlinf/hybrid_engines/fsdp/strategy/fsdp.py#L374)) 重新绑 view，否则 onload 回来 view 错位 hard-to-debug
- offload 后 `clear_memory()` 强制释放 PyTorch 缓存的 GPU 内存

**DDP 没有这能力，所以单卡 RL 也用 FSDP**。

## 8. auto_wrap_policy 决定 FSDP 切到多细

代码 [utils.py:get_fsdp_wrap_policy](../../rlinf/hybrid_engines/fsdp/utils.py)，调 PyTorch `transformer_auto_wrap_policy`。

粒度 tradeoff：

| 粒度 | 例 | 显存 | 速度 |
|---|---|---|---|
| 整个模型 | 1 unit | 高（峰值 = 全模型） | 快（通信少） |
| 每 transformer block | SmolVLM2 16 block | 中 | 中 |
| 每 Linear | 几百 unit | 最低 | 最慢（通信爆炸） |

SmolVLA wrap 边界是 `SmolVLMDecoderLayer` + `SmolVLA action expert block`。block 既是计算单元也是通信单元。

**调试**：

```python
import torch.distributed.fsdp as fsdp
fsdp._debug_level = "DEBUG"
```

## 9. FSDP1 vs FSDP2

| | FSDP1 (`strategy: "fsdp"`) | FSDP2 (`strategy: "fsdp2"`) |
|---|---|---|
| API | `FullyShardedDataParallel(model)` 包整个 | `fully_shard(module)` 按模块包 |
| 内部 | `flat_param`（拍平） | `DTensor` |
| LoRA 兼容 | 需 `use_orig_params=True` | 原生兼容 |
| TP/PP 组合 | 困难 | 容易，DTensor 统一抽象 |
| 成熟度 | 高 | 越来越主流，PyTorch 2.4+ 推荐 |

SmolVLA 用 FSDP1 是 LoRA debug 时栈更熟悉。新项目无 legacy 直接 FSDP2。

## 10. 多 GPU 通信成本估算

3B + `full_shard` + 4 卡：

```
Forward all_gather   : N_block × P × 2 bytes
Backward all_gather  : 同上
Backward reduce_scat : N_block × P × 4 bytes (reduce_dtype=fp32)
Per step total      ≈ N_block × P × 8 bytes
                     = 16 × 3B × 8 = 384 GB / step  per all 4 卡
```

A100 NVLink 单链路 600 GB/s，4 卡 mesh ~1.8 TB/s。算上 overlap 实际 step 时间通信占 30-50%。

**PCIe 4.0 单卡 32 GB/s，4 卡 PCIe 集群通信能堵 90%+。FSDP 必须 NVLink**，否则速度断崖。

## 11. RL 训练 FSDP 易踩的坑

1. **`use_orig_params=False` 配 LoRA**：init assert。修：True
2. **`mixed_precision` bf16 但代码手动 `.to(fp32)`**：mat dtype mismatch。修：全链路 bf16，只在出口 `.float()`
3. **offload 后忘 `_rebind_handle_views`**：onload 回来 weight 指向 stale storage。修：用 RLinf `offload_param_and_grad` 不手动搬
4. **rollout 不 offload + actor full_shard**：单卡 collocated 时 rollout 一直占 1-2G，actor train 炸。修：`enable_offload: True`
5. **`sync_module_states=True` 但 rank 0 没正确 init**：FSDP 让 rank 0 广播给其他 rank，rank 0 错全错。修：检查 `param_init_fn`
6. **gradient_checkpointing + FSDP1 + custom forward**：checkpoint 重跑 FSDP all-gather，不慢但易 deadlock。修：用 `apply_activation_checkpointing` 不手动 `torch.utils.checkpoint`
7. **`clip_grad_norm` sharded 模式用错 API**：要走 `model.clip_grad_norm_(max_norm)`（FSDP root 方法），不是 `torch.nn.utils.clip_grad_norm_`。RLinf [fsdp.py:589](../../rlinf/hybrid_engines/fsdp/strategy/fsdp.py#L589) 已分支处理

## 12. 一句话总结

FSDP 把模型/梯度/optimizer 切到多卡上，"用时凑齐用完释放"让多卡能训单卡装不下的模型；mixed precision + offload + LoRA + RL collocated 这几个 RL-specific 能力 DDP 给不了，所以即使单卡训 VLA 也要用 FSDP。

工程取舍：

- 单卡 SmolVLA → `no_shard` + `use_orig_params=True` + bf16 + offload
- 多卡 π0.5 → `full_shard` + `use_orig_params=True` + bf16 + 节点内 NVLink + auto_wrap 按 block 切
- 单节点多卡 + 多节点 → `hybrid_shard` 节点内 shard、节点间复制

参考：[14](14-多GPU训大模型.md) 多卡升级、[16](16-训练时进程与显存分布.md) 显存分布、[15](15-配置文件参数解读.md) fsdp_config 字段；[FSDP1 tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) / [FSDP2 design](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)。
