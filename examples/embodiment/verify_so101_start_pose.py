"""快速验证 SO-101 仿真起手位是否与真机一致。

跑法：
  python examples/embodiment/verify_so101_start_pose.py

输出：
  /tmp/so101_sim_start_pose.png

把这张图跟真机第一帧照片并排看：
  - shoulder 段应该竖直朝上
  - forearm 折回，gripper 朝下
  - 夹爪接近闭合
如果仿真出来是镜像/反向（比如 arm 平躺而不是竖直）→ MJCF 关节符号跟
LeRobot motor 不一致，需要在 so101.py 的 keyframe 里翻转对应关节的符号
（最常见是 shoulder_lift 或 elbow_flex）。
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import imageio
import numpy as np

import mani_skill.envs  # noqa: F401
import rlinf.envs.maniskill.agents.so101  # noqa: F401 触发 agent 注册
import rlinf.envs.maniskill.tasks.so101_pick_place  # noqa: F401

env = gym.make(
    "SO101PickPlace-v0",
    obs_mode="rgb",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    num_envs=1,
)
obs, _ = env.reset(seed=0)

# 取相机帧
img = env.unwrapped.render_rgb_array()
if hasattr(img, "cpu"):
    img = img.cpu().numpy()
img = np.asarray(img)
if img.ndim == 4:
    img = img[0]

out = "/tmp/so101_sim_start_pose.png"
imageio.imwrite(out, img.astype(np.uint8))

# 同时打印实际 qpos 跟 keyframe 是否对得上
qpos = env.unwrapped.agent.robot.get_qpos()
if hasattr(qpos, "cpu"):
    qpos = qpos.cpu().numpy()
qpos = np.asarray(qpos).reshape(-1)
joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
print("\n实际 reset 后的 qpos：")
for n, q in zip(joints, qpos):
    print(f"  {n:15s} = {q:+.4f} rad = {np.degrees(q):+7.2f} deg")

print(f"\n仿真起手帧已存到: {out}")
print("跟真机第一帧并排看 —— 形状应该一致（shoulder 竖直 + forearm 折下 + gripper 朝下）")
print("如果是平躺 / 镜像 → 某个关节 MJCF 符号反了，回到 so101.py keyframe 把对应值取负")

env.close()
