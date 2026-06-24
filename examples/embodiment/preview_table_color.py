#!/usr/bin/env python
"""预览 SO101 PickPlace 场景的桌面颜色。

改完 rlinf/envs/maniskill/tasks/so101_pick_place.py 里的 TABLE_TINT 后跑这个，
立刻看到桌面 + plate + cube + 机械臂的渲染图，不用整个训练流程。

用法：
    python examples/embodiment/preview_table_color.py
        # 截图保存到 ./preview_out/{render_camera,base_camera,wrist_camera}.png

    python examples/embodiment/preview_table_color.py --out-dir /tmp/tint_preview
    python examples/embodiment/preview_table_color.py --seed 42  # 换 cube/plate 摆位

无头模式：默认 MUJOCO_GL=egl，远程服务器也能跑。
"""
import argparse
import os
import re

# ⚠️ MUJOCO_GL 必须在 sapien / gymnasium / mani_skill import 之前设置，
# 否则 SAPIEN 拿到的 GL 后端就锁定了，再改环境变量无效。
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
from PIL import Image

# 触发自定义 task @register_env("SO101PickPlace-v0")，必须 import
import rlinf.envs.maniskill.tasks.so101_pick_place  # noqa: F401


TASK_SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "rlinf", "envs", "maniskill", "tasks", "so101_pick_place.py",
)


def _read_current_tint() -> str | None:
    """从 task 源码里直接 grep TABLE_TINT 行，方便打印当前生效值（确认你改了哪一版）。"""
    try:
        with open(TASK_SRC, "r", encoding="utf-8") as f:
            for line in f:
                m = re.match(r"\s*TABLE_TINT\s*=\s*(\[[^\]]+\])", line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return None


def _to_uint8_rgb(img) -> np.ndarray:
    """sapien / torch 渲染输出 → HxWx3 uint8。"""
    if hasattr(img, "cpu"):
        img = img.cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 4:           # [B, H, W, C] → [H, W, C]
        img = img[0]
    if img.dtype != np.uint8:   # float [0,1] → uint8
        img = (img * 255).clip(0, 255).astype(np.uint8)
    if img.shape[-1] == 4:      # RGBA → RGB
        img = img[..., :3]
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="./preview_out", help="截图保存目录")
    parser.add_argument("--seed", type=int, default=0, help="reset seed，换 cube/plate 摆位")
    parser.add_argument(
        "--sim-backend", default="cpu", choices=["cpu", "gpu"],
        help="CPU 后端材质变更立即生效，用来验证 GPU 后端是否有材质缓存问题。"
             "训练时 (run_embodiment.sh) 跑的是 gpu，要保证修改在 gpu 也生效。"
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    current_tint = _read_current_tint()
    if current_tint:
        print(f"[i] 当前 TABLE_TINT = {current_tint}  (源码: {TASK_SRC})")

    print(f"[1/3] 启动 SO101PickPlace-v0 (单 env, sim_backend={args.sim_backend}) ...")
    env = gym.make(
        "SO101PickPlace-v0",
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        num_envs=1,
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
    )
    obs, _info = env.reset(seed=args.seed)
    # update_render 强制把当前物理状态推到 GPU 渲染缓冲，
    # 否则首帧 render() 可能拿到 stale 帧
    env.unwrapped.scene.update_render()

    print("[2/3] 渲染 render_camera (录视频用，与 base_camera 同 pose) ...")
    render_img = _to_uint8_rgb(env.render())
    render_out = os.path.join(args.out_dir, "render_camera.png")
    Image.fromarray(render_img).save(render_out)
    print(f"    [OK] {render_out}  ({render_img.shape[1]}x{render_img.shape[0]})")

    print("[3/3] 渲染 sensor 相机 (base_camera + wrist_camera) ...")
    # reset 拿到的 obs 里就有 sensor_data，不用再 get_obs
    sensor = obs.get("sensor_data", {}) if isinstance(obs, dict) else {}
    if not sensor:
        # 兜底：手动取一次
        sensor = env.unwrapped.get_obs().get("sensor_data", {})
    saved = []
    for cam_name, cam_data in sensor.items():
        if not (isinstance(cam_data, dict) and "rgb" in cam_data):
            continue
        img = _to_uint8_rgb(cam_data["rgb"])
        out = os.path.join(args.out_dir, f"{cam_name}.png")
        Image.fromarray(img).save(out)
        print(f"    [OK] {out}  ({img.shape[1]}x{img.shape[0]})")
        saved.append(cam_name)
    if not saved:
        print("    [warn] 未拿到任何 sensor RGB；obs_mode 是否真的是 rgb？")

    env.close()

    print()
    print("=" * 60)
    print("预览完成。对比真机照片调色 (改 so101_pick_place.py 里 TABLE_TINT)：")
    print("  - 想更暖/更橙 → R 拉高、B 拉低，如 [0.95, 0.72, 0.40, 1]")
    print("  - 想更亮       → 整体往 1.0 拉，如 [1.0, 0.85, 0.55, 1]")
    print("  - 想更暗沉     → 整体压低，如 [0.65, 0.50, 0.28, 1]")
    print("  - 改完再跑本脚本即可对比新颜色。")
    print("=" * 60)


if __name__ == "__main__":
    main()
