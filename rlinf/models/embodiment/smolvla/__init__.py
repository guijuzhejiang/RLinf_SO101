# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""SmolVLA model wrapper for RLinf.

Architecture:
    SmolVLAPolicy
    └── .model → VLAFlowMatching (450M)
        ├── .vlm_with_expert (SmolVLM2 350M + LlamaModel expert 98M)
        └── 5 projection Linears (state/action_in/out, action_time_mlp_in/out)

LoRA target (from your SFT adapter_config.json):
    37 Linears = 16 layers × (q_proj + v_proj) in lm_expert + 5 projections
    Trainable params ≈ 3-8M (about 1% of base 450M)

PEFT wrap target: SmolVLAPolicy (顶层 p)，不是 p.model
    因为 adapter_config target_modules regex 前缀是 "model."
"""
import os
from typing import Optional

import torch
from omegaconf import DictConfig, open_dict


def get_model(cfg: DictConfig, torch_dtype: Optional[torch.dtype] = None):
    """加载 SmolVLA base + (可选) PEFT LoRA adapter，包成 RLinf BasePolicy。

    Args:
        cfg: 来自 actor.model 或 rollout.model 的子 config，期望字段：
            model_path: base SmolVLA 路径（含 config.json + model.safetensors）
            lora_path: (可选) PEFT adapter 路径（含 adapter_config.json + adapter_model.safetensors）
            is_lora: (bool) 是否加载 LoRA
            train_lora_only: (bool) 是否只训 LoRA（冻结 base）
        torch_dtype: 模型 dtype，默认 bfloat16
    """
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from rlinf.models.embodiment.smolvla.smolvla_action_model import (
        SmolVLAForRLActionPrediction,
    )

    dtype = torch_dtype or torch.bfloat16

    # 1. 加载 base SmolVLA
    print(f"[SmolVLA] Loading base from {cfg.model_path}")
    base_policy = SmolVLAPolicy.from_pretrained(cfg.model_path)

    # 2. 加载 PEFT LoRA（如果指定）
    is_lora = bool(cfg.get("is_lora", False))
    lora_path = cfg.get("lora_path", None)
    train_lora_only = bool(cfg.get("train_lora_only", True))

    if is_lora and lora_path:
        import dataclasses

        from peft import PeftModel

        adapter_cfg = os.path.join(lora_path, "adapter_config.json")
        assert os.path.exists(adapter_cfg), (
            f"adapter_config.json not found at {lora_path}"
        )

        # peft 在 inject_adapter 时会调 config.to_dict() 然后用 dict.get()
        # 检查 tie_word_embeddings。SmolVLAConfig 是 LeRobot dataclass，没有
        # to_dict / get，会报 AttributeError。补一个 to_dict 让 peft 走通。
        cfg_obj = getattr(base_policy, "config", None)
        if (
            cfg_obj is not None
            and dataclasses.is_dataclass(cfg_obj)
            and not hasattr(cfg_obj, "to_dict")
        ):
            cfg_dict = {"tie_word_embeddings": False}  # SmolVLA 没有 tied embed
            cfg_obj.to_dict = lambda d=cfg_dict: d  # type: ignore[attr-defined]

        # ⚠️ PEFT wrap 在顶层 SmolVLAPolicy（不是 .model）
        # 因为 adapter_config target_modules 用的是 "model.xxx" 前缀
        base_policy = PeftModel.from_pretrained(
            base_policy,
            lora_path,
            is_trainable=train_lora_only,
        )
        print(f"[SmolVLA] Loaded LoRA from {lora_path}")

        # Tier 0.A 修复：SmolVLA SFT 时 image_features 是 3 路 + empty_cameras=1
        # from_pretrained 会把 empty_cameras 重置成 0；不补回来 → VLM 输入少一路 → 完全 OOD
        inner = base_policy
        while hasattr(inner, "base_model"):  # 穿过 PeftModel 套层
            inner = inner.base_model
        while hasattr(inner, "model") and not hasattr(inner, "config"):
            inner = inner.model
        inner.config.empty_cameras = 1
        # 验证一下打个 log
        print(f"[SmolVLA] empty_cameras set to {inner.config.empty_cameras}, "
              f"image_features keys = {list(inner.config.image_features.keys())}")

        if hasattr(base_policy, "print_trainable_parameters"):
            base_policy.print_trainable_parameters()

        # 自己 PEFT 完后，让外层 rlinf.models.__init__.get_model 跳过它的
        # PEFT 分支 —— 那段写死的 target_modules 跟 SmolVLA SFT 的 regex 不
        # 匹配，再套一层只会出错或包出空 LoRA 壳。
        with open_dict(cfg):
            cfg.is_lora = False

    # 3. 冻结非 LoRA 参数（双保险）
    if train_lora_only:
        for name, p in base_policy.named_parameters():
            if "lora_" not in name:
                p.requires_grad = False
        trainable = sum(
            p.numel() for p in base_policy.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in base_policy.parameters())
        print(
            f"[SmolVLA] Trainable: {trainable / 1e6:.2f}M / "
            f"{total / 1e6:.2f}M ({100 * trainable / total:.2f}%)"
        )

    # 3.5 加载 LeRobot 训练时的 normalize stats（独立于模型权重，必须手动挂）
    #
    # LeRobot pipeline 走的是【外部 NormalizerProcessorStep → 模型只见 N(0,1) →
    # 外部 UnnormalizerProcessorStep】，stats 存在 lora_path 下的两个 safetensors：
    #   - policy_preprocessor_step_5_normalizer_processor.safetensors   (state 入)
    #   - policy_postprocessor_step_0_unnormalizer_processor.safetensors (action 出)
    #
    # 而 SmolVLAPolicy.from_pretrained / PeftModel.from_pretrained 都【不会】自动
    # 加载这两个文件，也不会把 normalize 接进 forward。dataset 单位是【度】(feetech
    # 真机协议)，模型只见 normalized 度数。RLinf rollout 端如果直接喂 raw qpos (弧度)
    # + 直接拿 model 输出当 action 送 env，相当于：
    #   1) state 端：模型见到的不是 normalized 而是 raw 弧度 → 输入 OOD
    #   2) action 端：模型输出的是 normalized 而不是度数 → 输出 OOD
    # 两边都漂离 SFT 分布 → 训练表现就是 success_rate=0 + 机械臂瞎动。
    #
    # 这里把 stats 一次加载好挂到 wrapper，让 _build_lerobot_batch 在 state 入口做
    # normalize，predict_action_batch 在 action 出口做 unnormalize + deg→rad。
    norm_stats = None
    if lora_path:
        from safetensors.torch import load_file

        pre_path = os.path.join(
            lora_path, "policy_preprocessor_step_5_normalizer_processor.safetensors"
        )
        post_path = os.path.join(
            lora_path,
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        )
        if os.path.exists(pre_path) and os.path.exists(post_path):
            pre_sd = load_file(pre_path)
            post_sd = load_file(post_path)
            norm_stats = {
                "state_mean": pre_sd["observation.state.mean"].float(),
                "state_std": pre_sd["observation.state.std"].float().clamp(min=1e-8),
                "action_mean": post_sd["action.mean"].float(),
                "action_std": post_sd["action.std"].float().clamp(min=1e-8),
            }
            print(
                f"[SmolVLA] Loaded normalize stats from {lora_path}\n"
                f"  state_mean (deg) = {[round(float(x), 2) for x in norm_stats['state_mean']]}\n"
                f"  state_std  (deg) = {[round(float(x), 2) for x in norm_stats['state_std']]}\n"
                f"  action_mean(deg) = {[round(float(x), 2) for x in norm_stats['action_mean']]}\n"
                f"  action_std (deg) = {[round(float(x), 2) for x in norm_stats['action_std']]}"
            )
        else:
            print(
                f"[SmolVLA][WARN] normalize stats not found at {lora_path} —— "
                f"模型将看到 raw state, 输出 normalized action, 几乎一定不 work"
            )

    # 4. 包成 RLinf 接口
    model = SmolVLAForRLActionPrediction(
        lerobot_policy=base_policy.to(dtype),
        config=cfg,
    )
    # 把 normalize stats 注册成 buffer（自动跟着 model.to(device/dtype) 走）
    if norm_stats is not None:
        for k, v in norm_stats.items():
            model.register_buffer(k, v, persistent=False)
        model._has_norm_stats = True
    else:
        model._has_norm_stats = False
    # 把整个 wrapper（含 value_head）也转到目标 dtype——避免 rollout/actor 两端
    # value_head weight dtype 跟 suffix_out 不一致。FSDP `use_orig_params=True`
    # 下 `.parameters().dtype` 看到的是原始 dtype 而非 FSDP cast 后的 dtype，
    # 所以运行时按 weight dtype 走会踩坑。直接整体 bf16 最省心。
    model = model.to(dtype)

    # 5. 收集完全冻结的最顶层子树, 让 FSDP 把它们 ignored。这是绕过 PyTorch FSDP1
    #    在 use_orig_params=True 下对冻结参数 _writeback_orig_params shape-mismatch
    #    bug 的关键: FSDP1 forward 把 module.weight 从 Parameter 降级为 Tensor,
    #    冻结参数不参与 backward 永远不会被恢复, 下次 forward writeback 必 raise
    #    (实测 SmolVLA patch_embedding [768,3,16,16] vs flat numel [589824] 就是
    #    这个 bug 的发作). LoRA 训练正好只需训 LoRA + value_head, base 全部冻结,
    #    把冻结子树 ignored 后 FSDP 只管 trainable params, 既绕开 bug 又减少显存.
    if train_lora_only:
        ignored = []

        def _visit(mod):
            params = list(mod.parameters(recurse=True))
            if not params:
                return
            if all(not p.requires_grad for p in params):
                ignored.append(mod)
                return  # 整子树冻结, 不再下钻 (避免重复 ignore)
            for child in mod.children():
                _visit(child)

        for child in model.children():
            _visit(child)

        model._fsdp_ignored_modules = ignored

        # 6. 同时强制 wrap_policy 给 trainable 叶子 (LoRA A/B + value_head 里的 Linear)
        #    每个独立 wrap 成 FSDP 单元 —— 否则它们会和其他 root 参数一起塞进同一个
        #    FlatParameter, 继续撞 _writeback_orig_params 的 shape-mismatch bug
        #    (实测 LoRA A [32,720] 也会触发同一个 bug). cfg.is_lora 已被设 False 防止
        #    外层 PEFT 二次 wrap, 这里用独立标志告诉 fsdp.py 启用 LoRA wrap policy.
        model._fsdp_force_lora_wrap = True

        n_ignored_params = sum(
            sum(p.numel() for p in m.parameters()) for m in ignored
        )
        print(
            f"[SmolVLA] FSDP ignored_modules: {len(ignored)} subtrees, "
            f"{n_ignored_params / 1e6:.2f}M params (all frozen)"
        )

    return model
