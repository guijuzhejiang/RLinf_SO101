# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""SmolVLA RL action model：把 LeRobot SmolVLAPolicy 包装成 RLinf BasePolicy。

实现要点：
    1. 复用 LeRobot SmolVLAPolicy 自带的 `prepare_images` / `prepare_state` 做预处理；
       语言部分用 HuggingFace AutoTokenizer 自己 tokenize。
    2. 直接调用底层 ``VLAFlowMatching`` 的 ``embed_prefix`` / ``denoise_step`` /
       ``action_out_proj``，绕过 SmolVLAPolicy 的 chunk-queue 推理，自己写 SDE 采样
       与 log-prob 计算。
    3. flow_sde 实现移植自
       ``rlinf/models/embodiment/openpi/openpi_action_model.py``（OpenPI Pi0 RL 版本），
       关键公式：
           x0_pred = x_t - v_t * t
           x1_pred = x_t + v_t * (1 - t)
           sigma  = noise_level * sqrt(t / (1 - t))
           x_t_mean = x0_pred * (1 - (t - dt)) + x1_pred * (t - dt - sigma^2 * dt / (2t))
           x_t_std  = sqrt(dt) * sigma
       一个轨迹随机挑一步注入 SDE 噪声，其它步走 ODE；log prob 等同于这一步的 Gaussian
       概率。训练时 critic-style forward 复用同一公式重新算 mean/std → log prob。

外部依赖：
    - ``self.policy`` 可能被 PEFT (LoRA) 包过，因此通过 ``_flow_model`` 属性穿透
      ``base_model.model.model`` 取到 ``VLAFlowMatching``。
"""
from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
from torch import nn

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.modules.value_head import ValueHead


class SmolVLAForRLActionPrediction(BasePolicy, nn.Module):
    """SmolVLA RL wrapper.

    Args:
        lerobot_policy: ``lerobot.policies.smolvla.SmolVLAPolicy`` (PEFT 包过亦可)。
        config: ``OmegaConf`` / dict，包含 flow_sde 超参与 action 维度。
    """

    SUPPORTED_NOISE_METHODS = {"flow_ode", "flow_sde"}

    def __init__(self, lerobot_policy: nn.Module, config: Any):
        nn.Module.__init__(self)
        self.policy = lerobot_policy
        self.config = config

        # --- flow / RL 超参 ---
        self.noise_method = str(config.get("noise_method", "flow_sde"))
        if self.noise_method not in self.SUPPORTED_NOISE_METHODS:
            raise ValueError(
                f"noise_method={self.noise_method} 暂未实现，"
                f"支持 {self.SUPPORTED_NOISE_METHODS}"
            )
        self.noise_level = float(config.get("noise_level", 0.5))
        self.noise_anneal = bool(config.get("noise_anneal", False))
        self.noise_params = list(config.get("noise_params", [0.7, 0.3, 400]))
        self.num_steps = int(config.get("num_steps", 10))
        self.safe_get_logprob = bool(config.get("safe_get_logprob", False))
        self.ignore_last = bool(config.get("ignore_last", False))
        self.joint_logprob = bool(config.get("joint_logprob", False))
        self.global_step = 0

        # action 维度（环境实际维度 vs SmolVLA pad 后维度）
        self.action_env_dim = int(config.get("action_dim", 6))
        # chunk_size：rollout 时往 env 一次回的 step 数；SmolVLA 默认 50
        self.action_chunk = int(config.get("chunk_size", 50))

        # critic 输入预处理（参考 OpenPI flow_sde）
        # chunk_critic_input=True：只用 action 部分 suffix_out（去掉 pad）做 critic
        # detach_critic_input=True：critic 不回传梯度给 VLM/expert（actor/critic 解耦）
        self.chunk_critic_input = bool(config.get("chunk_critic_input", True))
        self.detach_critic_input = bool(config.get("detach_critic_input", False))

        # value head（GRPO 不要；PPO 用 ValueHead 多层 MLP，与 OpenVLA 对齐）
        self.with_value_head = bool(
            config.get("with_value_head", config.get("add_value_head", False))
        )
        if self.with_value_head:
            expert_hidden = self._flow_model.vlm_with_expert.expert_hidden_size
            value_hidden = tuple(config.get("value_hidden_sizes", (512, 128)))
            # 注意：ValueHead._init_weights 走 kaiming_normal_，PyTorch 的
            # calculate_gain 不支持 "gelu"。OpenVLA 配置里写 gelu 但本机
            # PyTorch 2.5 会报错；用 relu 与 OpenPI 默认一致，gain 公式正确。
            value_activation = str(config.get("value_activation", "relu"))
            self.value_head = ValueHead(
                input_dim=expert_hidden,
                hidden_sizes=value_hidden,
                output_dim=1,
                activation=value_activation,
                bias_last=True,  # 与 OpenPI 一致，最后一层 free bias 数值更稳
            )
        else:
            self.value_head = None

        # tokenizer：SmolVLA 用 SmolVLM2 的 tokenizer
        self._tokenizer = None
        self._tokenizer_max_len = int(
            self._policy_config.tokenizer_max_length
            if hasattr(self._policy_config, "tokenizer_max_length")
            else 48
        )

        # image_features 一组 key（如 "observation.images.top"）
        cfg = self._policy_config
        if hasattr(cfg, "image_features") and cfg.image_features:
            self._image_keys = list(cfg.image_features)
        else:
            self._image_keys = [
                "observation.images.top",
                "observation.images.wrist",
            ]

    # ============================================================
    # Accessors（穿透 PEFT 拿到底层 SmolVLAPolicy / VLAFlowMatching）
    # ============================================================
    @property
    def _base_policy(self):
        """剥掉 PEFT，得到 SmolVLAPolicy 实例。"""
        p = self.policy
        if hasattr(p, "base_model"):
            p = p.base_model.model
        return p

    @property
    def _flow_model(self):
        """SmolVLAPolicy.model == VLAFlowMatching。"""
        return self._base_policy.model

    @property
    def _policy_config(self):
        return self._base_policy.config

    # ============================================================
    # HF 接口转发：FSDP manager 要求 module.gradient_checkpointing_enable()
    # SmolVLAPolicy / VLAFlowMatching 都不是 HF PreTrainedModel，没有这个方法；
    # 只有底下的 SmolVLM2 (vlm) 和 LlamaModel (lm_expert) 才有。
    # 这里转发到两个子模型上即可。
    # ============================================================
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        vlm_with_expert = self._flow_model.vlm_with_expert
        kwargs = (
            {"gradient_checkpointing_kwargs": gradient_checkpointing_kwargs}
            if gradient_checkpointing_kwargs is not None
            else {}
        )
        for sub in (vlm_with_expert.vlm, vlm_with_expert.lm_expert):
            if hasattr(sub, "gradient_checkpointing_enable"):
                sub.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        vlm_with_expert = self._flow_model.vlm_with_expert
        for sub in (vlm_with_expert.vlm, vlm_with_expert.lm_expert):
            if hasattr(sub, "gradient_checkpointing_disable"):
                sub.gradient_checkpointing_disable()

    # ============================================================
    # Tokenization
    # ============================================================
    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            vlm_name = self._policy_config.vlm_model_name
            self._tokenizer = AutoTokenizer.from_pretrained(vlm_name)
        return self._tokenizer

    def _tokenize_tasks(self, tasks: list[str], device: torch.device):
        """tokenize prompts，返回 (tokens, attention_mask)。

        与 LeRobot SmolVLANewLineProcessor 行为对齐：末尾确保 ``\\n``。
        """
        tokenizer = self._ensure_tokenizer()
        normalized = [t if t.endswith("\n") else f"{t}\n" for t in tasks]
        enc = tokenizer(
            normalized,
            padding="max_length",
            truncation=True,
            max_length=self._tokenizer_max_len,
            return_tensors="pt",
        )
        return (
            enc["input_ids"].to(device=device),
            enc["attention_mask"].to(device=device, dtype=torch.bool),
        )

    # ============================================================
    # Observation adapter
    # ============================================================
    def _build_lerobot_batch(self, env_obs: dict) -> dict[str, torch.Tensor]:
        """RLinf env_obs (main_images / wrist_images / states / task_descriptions)
        → LeRobot SmolVLAPolicy 的 batch 格式。
        """
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
            OBS_STATE,
        )

        flow = self._flow_model
        device = next(flow.parameters()).device
        # state_proj 决定 prefix 内部 dtype（bfloat16 / fp32 都可能）
        param_dtype = flow.state_proj.weight.dtype
        batch: dict[str, torch.Tensor] = {}

        # === images ===
        # 约定：image_keys[0] = top, image_keys[1] = wrist；多出来的相机由
        # SmolVLAPolicy.prepare_images 自动用 0 填充 (config.empty_cameras)
        #
        # RLinf maniskill_env 在 wrap_obs_mode=simple 下产出：
        #   main_images       [B, H, W, 3]    ← base_camera
        #   extra_view_images [B, N, H, W, 3] ← 其它相机按字典序 stack（N=0 时是 None）
        # 任务里只额外定义了 wrist_camera，所以 extra_view_images 第 0 路就是 wrist。
        image_inputs = []
        if "main_images" in env_obs and env_obs["main_images"] is not None:
            image_inputs.append(env_obs["main_images"])
        # 兼容两种 key 命名：直接 wrist_images，或 extra_view_images 的第 0 路
        wrist = env_obs.get("wrist_images")
        if wrist is None:
            extra = env_obs.get("extra_view_images")
            if extra is not None and getattr(extra, "ndim", 0) == 5 and extra.shape[1] > 0:
                wrist = extra[:, 0]
        if wrist is not None:
            image_inputs.append(wrist)
        for key, img in zip(self._image_keys, image_inputs, strict=False):
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            # NHWC → NCHW
            if img.ndim == 4 and img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2).contiguous()
            batch[key] = img.to(device=device, dtype=param_dtype)

        # === state ===
        # 完整链路 (从 env 到模型)：
        #   1. env.qpos      [B, 22] flatten      弧度 (SAPIEN 原生 SI 单位)
        #   2. 切前 6 维     [B, 6]                qpos 部分 (后 16 是 qvel + tcp + plate)
        #   3. rad → deg     [B, 6]                feetech 电机协议单位 (LeRobot dataset 单位)
        #   4. normalize     [B, 6]                (state_deg - mean) / std → 模型见到的 N(0,1)
        #   5. prepare_state 自动 pad 到 32 维     模型实际输入
        #
        # LeRobot pipeline 走的是【NormalizerProcessorStep → 模型 → UnnormalizerProcessorStep】
        # 三段式，模型本身只在 normalized 域工作。我们 RLinf 接入时这两段 processor 都没接，
        # 必须手动接通——否则模型见到 raw deg 完全 OOD（差 std=40+ 倍）。
        state = env_obs["states"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        if state.shape[-1] > 6:
            state = state[..., :6]
        state = state.float().to(device=device) * (180.0 / math.pi)  # rad → deg
        # MEAN_STD normalize (跟 LeRobot policy_preprocessor.json 里的 norm_map STATE=MEAN_STD 一致)
        if getattr(self, "_has_norm_stats", False):
            state = (state - self.state_mean) / self.state_std
        batch[OBS_STATE] = state.to(dtype=param_dtype)

        # === language tokens ===
        tasks = env_obs.get("task_descriptions")
        if tasks is None:
            bsize = batch[OBS_STATE].shape[0]
            tasks = [
                "pick up the red cube and place it on the white plate"
            ] * bsize
        tok_ids, tok_mask = self._tokenize_tasks(list(tasks), device)
        batch[OBS_LANGUAGE_TOKENS] = tok_ids
        batch[OBS_LANGUAGE_ATTENTION_MASK] = tok_mask
        return batch

    def _prepare_prefix_inputs(self, lerobot_batch: dict):
        """从 LeRobot batch 抽出 prefix 所需 5 元组。"""
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )

        base = self._base_policy
        images, img_masks = base.prepare_images(lerobot_batch)
        state = base.prepare_state(lerobot_batch)
        lang_tokens = lerobot_batch[OBS_LANGUAGE_TOKENS]
        lang_masks = lerobot_batch[OBS_LANGUAGE_ATTENTION_MASK]
        return images, img_masks, lang_tokens, lang_masks, state

    # ============================================================
    # Flow-SDE 核心
    # ============================================================
    def set_global_step(self, step: int):
        self.global_step = step

    def sample_noise(self, shape, device):
        return self._flow_model.sample_noise(shape, device)

    def _get_timesteps(self, denoise_steps: int, device):
        t = torch.linspace(1.0, 1.0 / denoise_steps, denoise_steps, device=device)
        t = torch.cat([t, torch.tensor([0.0], device=device)])
        return t

    def _get_noise_level(self, device, dtype, sample_method: str | None = None):
        method = sample_method or self.noise_method
        if method == "flow_ode":
            return torch.zeros((), device=device, dtype=dtype)
        if self.noise_anneal:
            ns, ne, anneal = self.noise_params
            level = ns + (ne - ns) * min(self.global_step, anneal) / max(anneal, 1)
        else:
            level = self.noise_level
        return torch.tensor(level, device=device, dtype=dtype)

    def get_logprob_norm(
        self, sample: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ):
        """logN(sample | mu, sigma)，sigma==0 时返回 0（ODE 退化）。"""
        if self.safe_get_logprob:
            return -torch.pow(sample - mu, 2)
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        const = -torch.log(sigma_safe) - 0.5 * torch.log(
            2 * torch.pi * torch.ones_like(sample)
        )
        expo = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
        lp = const + expo
        return torch.where(mask, torch.zeros_like(lp), lp)

    def gaussian_entropy(self, sigma: torch.Tensor):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        return 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe ** 2))

    def _build_prefix_cache(
        self, images, img_masks, lang_tokens, lang_masks, state
    ):
        """对图+语言+state 计算 KV cache（之后每个 denoise step 复用）。"""
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        flow = self._flow_model
        prefix_embs, prefix_pad_masks, prefix_att_masks = flow.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_kv = flow.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=flow.config.use_cache,
            fill_kv_cache=True,
        )
        return prefix_pad_masks, past_kv

    def _get_velocity(self, x_t, timestep, prefix_pad_masks, past_kv):
        """v_t = action_out_proj(suffix_out)，同时返回 suffix_out 供 value_head 使用。

        内联 SmolVLA ``VLAFlowMatching.denoise_step`` 逻辑，只是额外把 ``suffix_out``
        作为返回值之一（原版只返回 v_t）。这一步是计算 value 的关键中间产物。
        """
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        flow = self._flow_model
        proj_dtype = flow.action_in_proj.weight.dtype

        suffix_embs, suffix_pad_masks, suffix_att_masks = flow.embed_suffix(
            x_t.to(dtype=proj_dtype), timestep.to(dtype=proj_dtype)
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = flow.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_kv,
            inputs_embeds=[None, suffix_embs],
            use_cache=flow.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -flow.config.chunk_size:]
        # OpenPI 在这里强制 `suffix_out.to(fp32)`（参考 openpi_action_model.py:874）。
        # OpenPI 走 fp32 weight 路线（precision: null + amp_autocast），upcast 是 no-op。
        # 我们走 bf16 weight 路线（3090 显存），fp32 upcast 反而引发 fp32 input × bf16 weight
        # 的 mat mismatch，所以这里改成跟 weight dtype 走（bf16 路线下保持 bf16）。
        proj_w_dtype = flow.action_out_proj.weight.dtype
        suffix_out = suffix_out.to(dtype=proj_w_dtype)
        v_t = flow.action_out_proj(suffix_out)
        return v_t.to(dtype=x_t.dtype), suffix_out

    def _compute_value_from_suffix(self, suffix_out: torch.Tensor) -> torch.Tensor:
        """从 expert suffix_out 池化算 value。参考 OpenPI ``_compute_value_from_suffix``。

        Args:
            suffix_out: [B, chunk_size, expert_hidden]，dtype=fp32

        Returns:
            values: [B]，与样本一一对应
        """
        if self.value_head is None:
            return torch.zeros(suffix_out.shape[0], device=suffix_out.device)
        if self.chunk_critic_input:
            pooled = torch.mean(suffix_out[:, : self.action_chunk], dim=1)
        else:
            pooled = torch.mean(suffix_out, dim=1)
        if self.detach_critic_input:
            pooled = pooled.detach()
        # 不做任何 dtype 转换：get_model 已经把整个 wrapper（含 value_head）
        # 转到目标 dtype（bf16），FSDP `param_dtype=bf16` 也保持一致；
        # suffix_out 来自 base policy 同 dtype，pooled 与 weight dtype 自动匹配。
        return self.value_head(pooled)[:, 0]

    def sample_mean_var_val(
        self, x_t, idx, prefix_pad_masks, past_kv, sample_method: str, denoise_steps: int
    ):
        """一步 denoise 的 mean / std 计算 + value，对齐 OpenPI flow_sde。

        Returns:
            x_t_mean, x_t_std, v_t, suffix_out
            （suffix_out 给 value_head 用；其余跟 OpenPI 一致）
        """
        bsize = x_t.shape[0]
        device = x_t.device
        if isinstance(idx, int):
            idx = torch.tensor(idx, device=device).expand(bsize)
        else:
            idx = idx.to(device)
        noise_level = self._get_noise_level(device, x_t.dtype, sample_method)
        timesteps = self._get_timesteps(denoise_steps, device).to(dtype=x_t.dtype)

        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]

        v_t, suffix_out = self._get_velocity(x_t, t_input, prefix_pad_masks, past_kv)

        delta_b = delta[:, None, None].expand_as(x_t)
        t_b = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_b
        x1_pred = x_t + v_t * (1 - t_b)

        if sample_method == "flow_ode":
            x0_w = 1.0 - (t_b - delta_b)
            x1_w = t_b - delta_b
            x_t_std = torch.zeros_like(t_b)
        elif sample_method == "flow_sde":
            denom = torch.where(timesteps == 1, timesteps[1], timesteps)
            sigma_ratio = timesteps / (1 - denom)
            sigmas = noise_level * torch.sqrt(sigma_ratio)[:-1]
            sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
            x0_w = torch.ones_like(t_b) - (t_b - delta_b)
            x1_w = t_b - delta_b - sigma_i ** 2 * delta_b / (2 * t_b)
            x_t_std = torch.sqrt(delta_b) * sigma_i
        else:
            raise ValueError(f"Unsupported sample method: {sample_method}")

        x_t_mean = x0_pred * x0_w + x1_pred * x1_w
        return x_t_mean, x_t_std, v_t, suffix_out

    # ============================================================
    # Rollout sampling
    # ============================================================
    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        mode: str = "train",
    ) -> dict[str, torch.Tensor]:
        """完整 num_steps 反推 + SDE log prob 采集。"""
        bsize = state.shape[0]
        device = state.device

        prefix_pad_masks, past_kv = self._build_prefix_cache(
            images, img_masks, lang_tokens, lang_masks, state
        )
        action_shape = (
            bsize,
            self._flow_model.config.chunk_size,
            self._flow_model.config.max_action_dim,
        )
        x_t = self.sample_noise(action_shape, device)

        # train 模式：随机挑一步注入噪声；eval：全 ODE
        if mode == "train":
            high = self.num_steps - 1 - int(self.ignore_last)
            pick = random.randint(0, max(high, 0))
            denoise_inds = torch.full((self.num_steps,), pick, dtype=torch.long)
        else:
            denoise_inds = torch.full((self.num_steps,), -1, dtype=torch.long)
        denoise_inds = denoise_inds[None].repeat(bsize, 1).to(device)

        chains = [x_t]
        log_probs = []
        values_per_step = []
        for idx in range(self.num_steps):
            method = self.noise_method if idx == int(denoise_inds[0, idx]) else "flow_ode"
            x_t_mean, x_t_std, _, suffix_out = self.sample_mean_var_val(
                x_t, idx, prefix_pad_masks, past_kv, method, self.num_steps
            )
            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            lp = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
            chains.append(x_t)
            log_probs.append(lp)
            # 每个 denoise step 都算一次 value（最终用 SDE 步的）
            values_per_step.append(self._compute_value_from_suffix(suffix_out))

        chains = torch.stack(chains, dim=1)  # [B, num_steps+1, chunk, dim]
        log_probs = torch.stack(log_probs, dim=1)
        # 只取被注入噪声那一步的 log prob（非 joint 模式）
        bidx = torch.arange(bsize, device=device)
        sampled_lp = log_probs[bidx, denoise_inds[:, 0]]
        # 裁到环境实际 action 维度 + 真实 chunk size
        sampled_lp = sampled_lp[
            :, : self.action_chunk, : self.action_env_dim
        ]
        # value：取被注入噪声那一步的 value（也可改成 mean）
        # 形状 [B, 1] 与 OpenPI 对齐——RLinf 用 final_values[:, :1] 访问 bootstrap
        values_per_step = torch.stack(values_per_step, dim=1)  # [B, num_steps]
        sampled_values = values_per_step[bidx, denoise_inds[:, 0]].unsqueeze(-1)  # [B, 1]

        return {
            "actions": x_t,  # [B, chunk, max_action_dim]
            "chains": chains,
            "prev_logprobs": sampled_lp,
            "prev_values": sampled_values,
            "denoise_inds": denoise_inds,
        }

    def predict_action_batch(
        self,
        env_obs: dict,
        mode: str = "train",
        compute_values: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """RLinf rollout 入口。"""
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )

        batch = self._build_lerobot_batch(env_obs)
        images, img_masks, lang_tokens, lang_masks, state = (
            self._prepare_prefix_inputs(batch)
        )
        outputs = self.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, mode=mode
        )
        # model 输出始终在 NORMALIZED 域 (mean≈0, std≈1)。
        # 取环境维度的切片仍然是 normalized。
        actions_norm = outputs["actions"][:, : self.action_chunk, : self.action_env_dim]

        # === action 出口：normalized → deg → rad，供 env 执行 ===
        # 跟 LeRobot UnnormalizerProcessorStep + SAPIEN SI 单位约定接通。
        # forward_inputs["action"] 必须保留 normalized 域 —— PPO 的 default_forward
        # 重算 log_prob 时模型还是在 normalized 域算 Gaussian，必须用同域 action
        # 才能算出和采样时一致的 ratio。
        if getattr(self, "_has_norm_stats", False):
            actions_env_deg = actions_norm * self.action_std + self.action_mean
        else:
            actions_env_deg = actions_norm
        actions = actions_env_deg * (math.pi / 180.0)  # deg → rad，env 直接吃这个

        # forward_inputs 必须 round-trip 给 default_forward 重算 log prob，
        # 因此保留 chains / denoise_inds 以及 tokenized obs。
        # ⚠️ "action" 字段是 normalized 域（跟 sample_actions/log_prob 同域）。
        # env 用的 rad 版本另存，不进 forward_inputs（重算 log_prob 用不到）。
        forward_inputs: dict[str, Any] = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
            OBS_LANGUAGE_TOKENS: lang_tokens,
            OBS_LANGUAGE_ATTENTION_MASK: lang_masks,
            "action": actions_norm.reshape(actions_norm.shape[0], -1).contiguous(),
            "model_action": outputs["actions"]
            .reshape(outputs["actions"].shape[0], -1)
            .contiguous(),
        }
        # 把图像 / state 透传到 forward_inputs 方便重算
        for k, v in batch.items():
            if k in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK):
                continue
            forward_inputs[k] = v.detach()

        # PPO loss 要求 logprobs/values/advantages 必须是 fp32 才数值稳定。
        # 我们 weight 走 bf16 路线（3090 显存约束），输出边界 cast 回 fp32。
        # `.float()` 不会断梯度。
        result = {
            "prev_logprobs": outputs["prev_logprobs"].float(),
            "prev_values": outputs["prev_values"].float(),
            "forward_inputs": forward_inputs,
        }
        return actions, result

    # ============================================================
    # 训练时重算 log prob / entropy
    # ============================================================
    def get_log_prob_value(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        chains,
        denoise_inds,
        compute_values: bool = True,
    ):
        bsize = state.shape[0]
        device = state.device
        prefix_pad_masks, past_kv = self._build_prefix_cache(
            images, img_masks, lang_tokens, lang_masks, state
        )

        # 只重算被采的那一步（非 joint 模式）
        denoise_ind = denoise_inds[:, 0]
        bidx = torch.arange(bsize, device=device)
        x_t_prev = chains[bidx, denoise_ind]
        x_t_next = chains[bidx, denoise_ind + 1]
        x_t_mean, x_t_std, _, suffix_out = self.sample_mean_var_val(
            x_t_prev,
            denoise_ind,
            prefix_pad_masks,
            past_kv,
            self.noise_method,
            self.num_steps,
        )
        log_prob = self.get_logprob_norm(x_t_next, x_t_mean, x_t_std)
        if self.noise_method == "flow_sde":
            entropy = self.gaussian_entropy(x_t_std)
        else:
            entropy = torch.zeros_like(log_prob)

        if compute_values and self.value_head is not None:
            values = self._compute_value_from_suffix(suffix_out)
        else:
            values = torch.zeros(bsize, device=device)
        return log_prob, entropy, values

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        compute_values: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """训练 forward：重算 logprobs / entropy / values。

        参考 OpenVLA ``default_forward`` 模式：从模型 hidden state 算 value。
        SmolVLA 这里的"hidden state" = 同一 denoise 步的 ``suffix_out``。
        """
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )

        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]

        # 构造一个 lerobot-style 子 batch（图像/state 已是 tensor 直接复用）
        base = self._base_policy
        device = chains.device
        sub_batch = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in forward_inputs.items()
            if k in self._image_keys
        }
        sub_batch["observation.state"] = forward_inputs["observation.state"].to(device)
        sub_batch[OBS_LANGUAGE_TOKENS] = forward_inputs[OBS_LANGUAGE_TOKENS].to(device)
        sub_batch[OBS_LANGUAGE_ATTENTION_MASK] = forward_inputs[
            OBS_LANGUAGE_ATTENTION_MASK
        ].to(device)
        images, img_masks = base.prepare_images(sub_batch)
        state = base.prepare_state(sub_batch)
        lang_tokens = sub_batch[OBS_LANGUAGE_TOKENS]
        lang_masks = sub_batch[OBS_LANGUAGE_ATTENTION_MASK]

        log_probs, entropy, values = self.get_log_prob_value(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            chains,
            denoise_inds,
            compute_values=compute_values,
        )
        # 裁到环境实际维度
        log_probs = log_probs[:, : self.action_chunk, : self.action_env_dim]
        entropy = entropy[:, : self.action_chunk, : self.action_env_dim]
        # entropy 聚合成 [B, 1] 对齐 loss-mask shape
        entropy = entropy.mean(dim=[1, 2], keepdim=False)[:, None]
        # PPO loss 强制 fp32 检查（actor_loss 里 assert logprobs.dtype == fp32）。
        # weight bf16 → forward 输出 bf16 → 边界 cast 回 fp32，不断梯度。
        return {
            "logprobs": log_probs.float(),
            "values": values.float(),
            "entropy": entropy.float(),
        }
