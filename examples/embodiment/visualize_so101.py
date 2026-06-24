#!/usr/bin/env python
"""可视化 SO-101 机器人在 ManiSkill 里的加载结果。

用途：
    1. 验证 SO101 Agent 类能否成功加载 MJCF + 14 个 STL mesh
    2. 验证 home keyframe pose 合理
    3. 验证 wrist_camera mount 在正确 link 上
    4. 保存截图供肉眼对比真机

无头（headless）模式默认开启，结果保存为 PNG。
有 GUI 时可加 --gui 弹窗交互查看。
"""
import argparse
import os

import gymnasium as gym
import numpy as np
import sapien

# 触发 SO101 Agent 注册（关键 import）
import rlinf.envs.maniskill.agents.so101  # noqa: F401
from rlinf.envs.maniskill.agents.so101 import SO101


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="弹 SAPIEN viewer 交互查看")
    parser.add_argument(
        "--out-dir",
        default="/home/zzg/workspace/pycharm/Robot/rlinf_bridge/data/visualize_out",
        help="无头模式截图保存目录",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.gui:
        os.environ.setdefault("MUJOCO_GL", "egl")

    # 起一个最小 ManiSkill env，只为加载 SO101
    # 用内置 EmptyEnv-v1 作为最简场景；如果没有，退到任意 panda 类任务
    # ManiSkill 没有官方 "EmptyEnv"，用 PickCube-v1 并覆盖 robot_uids
    print("[1/4] 起 ManiSkill env (PickCube-v1, robot_uids='so101')...")
    try:
        env = gym.make(
            "PickCube-v1",
            obs_mode="state",
            control_mode="pd_joint_pos",
            robot_uids="so101",  # 这里强制用 SO-101
            num_envs=1,
            render_mode="rgb_array",
            human_render_camera_configs=dict(width=1024, height=768),
        )
    except Exception as e:
        # PickCube 不支持 so101，那就靠 register_so101 把它注册进去
        print(f"[!] 标准 PickCube 不支持 so101: {type(e).__name__}: {e}")
        print("    尝试先 import register_so101 注册...")
        from rlinf.envs.maniskill.agents.register_so101 import register_so101_to_tasks
        register_so101_to_tasks()
        env = gym.make(
            "PickCube-v1",
            obs_mode="state",
            control_mode="pd_joint_pos",
            robot_uids="so101",
            num_envs=1,
            render_mode="rgb_array",
        )

    obs, info = env.reset(seed=0)
    print(f"    DOF: {env.unwrapped.agent.robot.dof}")
    print(f"    qpos shape: {env.unwrapped.agent.robot.get_qpos().shape}")

    print("[2/4] 检查关节名 + link 名...")
    for j in env.unwrapped.agent.robot.get_active_joints():
        print(f"      joint: {j.name}")
    for l in env.unwrapped.agent.robot.get_links():
        print(f"      link: {l.name}")

    print("[3/4] 应用 home keyframe...")
    if hasattr(env.unwrapped.agent, "keyframes") and "home" in env.unwrapped.agent.keyframes:
        kf = env.unwrapped.agent.keyframes["home"]
        import torch
        qpos = torch.from_numpy(np.array(kf.qpos)).float().reshape(1, -1)
        env.unwrapped.agent.robot.set_qpos(qpos)
    env.unwrapped.scene.update_render()

    print("[4/4] 渲染并保存截图...")
    # human_render_camera 截图
    img = env.render()
    if hasattr(img, "cpu"):
        img = img.cpu().numpy()
    if img.ndim == 4:
        img = img[0]
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    # 保存
    try:
        from PIL import Image
        out_path = os.path.join(args.out_dir, "so101_home_pose.png")
        Image.fromarray(img).save(out_path)
        print(f"    [OK] 截图保存到: {out_path}")
    except ImportError:
        import imageio
        out_path = os.path.join(args.out_dir, "so101_home_pose.png")
        imageio.imwrite(out_path, img)
        print(f"    [OK] 截图保存到: {out_path}")

    # 尝试拿腕部相机的图（如果 _sensor_configs 起作用）
    if hasattr(env.unwrapped.agent, "_sensor_configs"):
        try:
            obs = env.unwrapped.get_obs()
            if isinstance(obs, dict) and "sensor_data" in obs:
                for cam_name, cam_data in obs["sensor_data"].items():
                    if isinstance(cam_data, dict) and "rgb" in cam_data:
                        rgb = cam_data["rgb"]
                        if hasattr(rgb, "cpu"):
                            rgb = rgb.cpu().numpy()
                        if rgb.ndim == 4:
                            rgb = rgb[0]
                        if rgb.dtype != np.uint8:
                            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
                        cam_out = os.path.join(args.out_dir, f"so101_{cam_name}.png")
                        from PIL import Image as PILImage
                        PILImage.fromarray(rgb).save(cam_out)
                        print(f"    [OK] {cam_name} 保存到: {cam_out}")
        except Exception as e:
            print(f"    [warn] 拿 sensor_data 失败 (不影响主流程): {e}")

    if args.gui:
        print("[GUI] 启动 viewer，按 q 退出...")
        viewer = env.unwrapped.scene.create_viewer()
        viewer.set_camera_xyz(x=0.6, y=0.6, z=0.6)
        viewer.set_camera_rpy(r=0, p=-np.pi / 6, y=np.pi * 0.75)
        while not viewer.closed:
            env.unwrapped.scene.update_render()
            viewer.render()

    env.close()
    print("\n[DONE] 如果截图里能看到 SO-101 摆出 home pose，Phase 1 通过。")
    print("       检查点：")
    print("         - 机械臂没散架（关节连接正确）")
    print("         - 末端朝下（wrist_flex/roll 都是 π/2）")
    print("         - 夹爪略合（gripper=0.3）")
    print("         - 腕部相机视角合理（朝向夹爪末端正前方）")


if __name__ == "__main__":
    main()
