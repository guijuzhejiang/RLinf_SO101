# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""
SO-101 Pick-and-Place Task for ManiSkill.

任务：用 SO-101 抓取桌面上的红色 2cm 方块，放进白色盘子里。
真机对齐：30 FPS, 640x480, 顶部相机 + 腕部相机。

参考: april5129/RLinf fork 的 so101_pick_cube.py（改写成 pick_place + 双相机 + plate）
"""
import math
from typing import Any, Dict

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

# 触发 SO101 agent 注册
import rlinf.envs.maniskill.agents.so101  # noqa: F401

# build_cylinder 默认 cylinder 沿局部 X 轴；想让它水平躺在桌面，
# 需要绕 Y 轴旋转 90°，使 X 局部轴对齐世界 Z 轴。
_PLATE_FLAT_QUAT = euler2quat(0.0, np.pi / 2, 0.0)  # (w,x,y,z)


@register_env("SO101PickPlace-v0", max_episode_steps=400)
class SO101PickPlaceEnv(BaseEnv):
    """SO-101 红色 2cm 方块 → 白色盘子。

    Randomization:
        - Cube xy ∈ [0.15, 0.25] × [-0.10, 0.10]
        - Plate xy ∈ [0.20, 0.30] × [-0.10, 0.10]，强制与 cube 距离 > 0.10

    Success:
        - Cube xy 投影 < plate 半径
        - Cube z 在 plate 上方 (释放后) 且不过高
        - 末端不再抓 cube (放下了)
    """

    SUPPORTED_ROBOTS = ["so101"]
    agent: Any  # SO101

    # 物体常量（与真机对齐）
    CUBE_HALF_SIZE = 0.012                # 2.4 cm 方块

    # 真机盘子是家用碗碟形：外缘高 2 cm + 中间凹陷底 4 mm。
    # 仿真用 "底盘 + 16 段 box 拼成外缘环" 模拟（SAPIEN primitive 不支持凹陷形）。
    PLATE_OUTER_RADIUS = 0.08             # 8 cm 外半径
    PLATE_RIM_WIDTH = 0.01                # 1 cm 边宽（径向厚度）
    PLATE_INNER_RADIUS = PLATE_OUTER_RADIUS - PLATE_RIM_WIDTH  # 0.07 m，凹陷区域半径
    PLATE_BASE_THICKNESS = 0.004          # 4 mm 凹陷底
    PLATE_RIM_HEIGHT = 0.02               # 2 cm 外缘最高点（含底）
    PLATE_RIM_SEGMENTS = 16               # 外缘用 16 段 box 拼圆环

    # 成功判定：cube 中心落进凹陷区域且贴在凹陷底面附近。
    # 注：判定都是 cube_pos - plate_pos 的相对量，plate origin 在 actor 底面。
    # xy：cube 中心离 plate 中心 < inner_radius - cube_half - margin
    SUCCESS_XY_THRESH = PLATE_INNER_RADIUS - CUBE_HALF_SIZE - 0.005    # ≈ 0.057
    # z：cube 中心相对 plate 底面的高度
    # 最小：cube 立在凹陷底上 → base_thickness + cube_half = 0.012
    # 最大：cube 不能飞过外缘 → rim_height + cube_half = 0.028
    SUCCESS_Z_MIN = PLATE_BASE_THICKNESS + CUBE_HALF_SIZE              # 0.012
    SUCCESS_Z_MAX = PLATE_RIM_HEIGHT + CUBE_HALF_SIZE                  # 0.028

    # 真机 dataset 里 plate 是固定的（front 视角左下角），cube 在 arm 前方
    # 桌面中间随机；仿真对齐这个分布 —— SFT 的视觉先验才能复用。
    # PLATE_FIXED_XY: 仿真世界坐标 (x, y)，对应 base_camera 画面左下角。
    # 若跑视觉脚本发现 plate 在画面右下，把 y 符号反过来。
    PLATE_FIXED_XY = (0.28, -0.22)
    # cube 在 arm 正前方桌面中线附近随机。
    # x ∈ [0.22, 0.30]：最近离底座 22cm（≥ home pose 末端 x≈0.20，确保 cube 在夹爪
    #   前方而不是脚下）；最远 30cm（不超过 SO-101 工作半径 ~30-35cm，避免训出
    #   "够不到"的策略）。
    # y ∈ [-0.04, 0.06]：桌面中线附近，离 plate (0.28, -0.22) 的 y 方向有间距。
    CUBE_INIT_REGION = ((0.22, 0.30), (-0.04, 0.06))
    MIN_CUBE_PLATE_DIST = 0.10

    def __init__(
        self,
        *args,
        robot_uids="so101",
        # 默认 0.02 rad ≈ 1.15°；但 lerobot_start curled keyframe 的
        # shoulder_lift = -1.725 rad 距离 MJCF 限位 -1.745 只有 0.02 rad，
        # noise>=0.02 会有约 50% 概率超限被 clip 到边界，破坏起手位。
        # 收紧到 0.003 rad ≈ 0.17°，比 dataset 内 curled 簇本身的 std (~3°) 还小，
        # 但保证 reset 后绝不超限。
        robot_init_qpos_noise=0.003,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ---------- 仿真参数 ----------
    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=300,  # 300 / 30(control) = 10 倍率，整除
            control_freq=30,  # 真机 30 FPS
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2 ** 25,
                max_rigid_patch_count=2 ** 18,
            ),
        )

    # ---------- 相机 ----------
    @property
    def _default_sensor_configs(self):
        """顶部相机（RLinf simple wrap_obs_mode 约定：main 相机名必须叫 base_camera）。

        真机参考图：相机大约在 arm 末端正上偏前 ~30cm 处俯视，
        画面下方能看到桌面 + 盘子，画面上方是机械臂 wrist/夹爪。
        v2 (0.40,0,0.35) → v3：v2 太近导致 60° FOV 装不下夹爪，
        eye 沿视线反方向回退到 (0.55,0,0.40)，target 微调到 (0.15,0,0.03)
        让桌面 + 全身 arm + 夹爪 + 盘子全进画面。
        """
        top_pose = sapien_utils.look_at(
            eye=[0.44, 0.0, 0.40],
            target=[0.15, 0.0, 0.03],
        )
        return [
            CameraConfig(
                uid="base_camera",   # ← RLinf wrap_obs_mode=simple 期望的 main 相机名
                pose=top_pose,
                width=640,
                height=480,
                fov=np.deg2rad(60),
                near=0.01,
                far=2.0,
            ),
            # wrist_camera 由 SO101 agent 在 _sensor_configs 里提供；
            # 名字按字典序排序后会进 simple 模式的 extra_view_images 第 0 位
        ]

    @property
    def _default_human_render_camera_configs(self):
        """录视频用的相机。RLinf 视频录制无法只录某一路 sensor，所以让 render_camera
        复用 base_camera 的 pose / fov，再设 render_mode='rgb_array' 就只录到
        front 顶部视角（跟模型实际看的 base_camera 像素级一致）。
        """
        pose = sapien_utils.look_at(
            eye=[0.44, 0.0, 0.40], target=[0.15, 0.0, 0.03]   # 与 base_camera 一致
        )
        return CameraConfig(
            "render_camera", pose, width=640, height=480, fov=np.deg2rad(60),
        )

    # ---------- 场景 ----------
    def _load_agent(self, options):
        super()._load_agent(options, sapien.Pose(p=[0.0, 0.0, 0.0]))

    def _load_scene(self, options):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        #   - 实际目标：暖黄木 [0.82, 0.67, 0.39, 1] (取色自真机照片 RGB≈210,170,100)。
        TABLE_TINT = [0.92, 0.73, 0.45, 1.0]  # 保留纹理

        for sub in self.table_scene.table._objs:
            comp = sub.find_component_by_type(sapien.render.RenderBodyComponent)
            if comp is None:
                continue

            for shape in comp.render_shapes:
                for part in shape.parts:
                    if part.material.base_color_texture is not None:
                        part.material.set_base_color(TABLE_TINT)
                        part.material.set_base_color_texture(None)

        # 红色 2cm 方块
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.CUBE_HALF_SIZE,
            color=[0.85, 0.10, 0.10, 1.0],
            name="red_cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0.20, 0.0, self.CUBE_HALF_SIZE + 0.001]),
        )

        # 白色盘子：底盘薄圆柱 + 16 段 box 拼成的外缘环，origin 在 actor 底面。
        self.plate = self._build_plate_with_rim(
            initial_pose=sapien.Pose(p=[0.28, 0.0, 0.001]),  # 底面贴桌面 + 1mm margin
        )

    def _build_plate_with_rim(self, initial_pose: sapien.Pose):
        """凹陷盘子：薄底盘 + N 段 box 拼接的外缘环。

        几何（actor origin 在最底面 z=0）：
            - 底盘：cylinder, z ∈ [0, PLATE_BASE_THICKNESS]
            - 外缘：N 段 box，z ∈ [PLATE_BASE_THICKNESS, PLATE_RIM_HEIGHT]
            - 凹陷区域：半径 < PLATE_INNER_RADIUS、底面 z = PLATE_BASE_THICKNESS
        """
        builder = self.scene.create_actor_builder()
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([0.95, 0.95, 0.95, 1.0])
        mat.set_roughness(0.6)
        mat.set_metallic(0.0)

        # 1) 凹陷底盘（薄圆柱，水平躺平：cylinder 局部 X → 世界 Z）
        base_pose = sapien.Pose(
            p=[0.0, 0.0, self.PLATE_BASE_THICKNESS / 2],
            q=_PLATE_FLAT_QUAT.tolist(),
        )
        builder.add_cylinder_visual(
            radius=self.PLATE_OUTER_RADIUS,
            half_length=self.PLATE_BASE_THICKNESS / 2,
            pose=base_pose,
            material=mat,
        )
        builder.add_cylinder_collision(
            radius=self.PLATE_OUTER_RADIUS,
            half_length=self.PLATE_BASE_THICKNESS / 2,
            pose=base_pose,
        )

        # 2) 外缘环：N 段 box 拼接（N=16 视觉够圆 + 物理稳定）
        n_seg = self.PLATE_RIM_SEGMENTS
        rim_seg_h = self.PLATE_RIM_HEIGHT - self.PLATE_BASE_THICKNESS  # 段高 16mm
        rim_center_z = self.PLATE_BASE_THICKNESS + rim_seg_h / 2       # 段中心 z
        seg_radial = self.PLATE_RIM_WIDTH                              # 径向厚度
        r_seg_center = self.PLATE_OUTER_RADIUS - seg_radial / 2        # 段中心半径
        seg_chord = 2 * math.pi * r_seg_center / n_seg                 # 段切向弦长
        for i in range(n_seg):
            angle = 2 * math.pi * i / n_seg
            cx = r_seg_center * math.cos(angle)
            cy = r_seg_center * math.sin(angle)
            q = euler2quat(0.0, 0.0, angle).tolist()  # box local X = 径向，Y = 切向
            seg_pose = sapien.Pose(p=[cx, cy, rim_center_z], q=q)
            half_size = [seg_radial / 2, seg_chord / 2, rim_seg_h / 2]
            builder.add_box_visual(pose=seg_pose, half_size=half_size, material=mat)
            builder.add_box_collision(pose=seg_pose, half_size=half_size)

        builder.initial_pose = initial_pose
        return builder.build_dynamic(name="white_plate")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # ★ TableSceneBuilder.initialize() 内部只识别 so100，对 so101 走 fall-through
            # 既不 reset qpos 也不 set_pose，机器人停在 MJCF 加载后的随机状态。
            # 这里手动 reset 到 LeRobot dataset CURLED 簇 mean（前 50 ep 同一次规范录制
            # 的起手位）。注意 dataset 是双峰分布（前 50 curled + 后 200 extended），
            # 不能简单 median 否则落到中间过渡区。详见 so101.py lerobot_start keyframe 注释。
            #
            # ⚠️ qpos 必须用 unbatched [6]，让 ManiSkill GPU sim 走广播路径。
            #   - 第一次 manual env.reset()：env_idx 是全集（b=num_envs），broadcast OK
            #   - 后续 auto_reset：env_idx 可能是子集（b<num_envs），用 batched [b,6]
            #     时 ManiSkill GPU set_qpos 的 reset_mask 协议会不一致，导致 partial
            #     reset 时 keyframe 没真正写进 GPU 缓冲。表现就是 0.mp4 起手 curled，
            #     1.mp4+ 起手回到 extended（保留上一个 epoch 末态的姿势）。
            #   - put_carrot_on_plate.py 等工作中的 task 用的都是 unbatched，
            #     是 ManiSkill 多 env reset 的 canonical 姿势。
            # 顺序也按 put_carrot_on_plate 走：先 set_pose，再 reset(qpos)。
            self.agent.robot.set_pose(sapien.Pose([0.0, 0.0, 0.0]))
            start_qpos = self.agent.keyframes["lerobot_start"].qpos  # [6]
            if self.robot_init_qpos_noise > 0:
                # 噪声在 numpy 域加完后保持 1D [6]，让 ManiSkill 广播；
                # 不能用 [b,6] 否则触发上述 partial-reset bug。
                noise = self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(start_qpos)
                )
                start_qpos = start_qpos + noise
            self.agent.reset(init_qpos=start_qpos)

            # plate 固定（对齐真机：dataset 里 plate 一直在 front 视角左下角）
            # 新 plate origin 在底面，z = 0.001 让底面贴桌面 + 1mm margin 防穿透
            px, py = self.PLATE_FIXED_XY
            plate_xyz = torch.zeros((b, 3))
            plate_xyz[:, 0] = px
            plate_xyz[:, 1] = py
            plate_xyz[:, 2] = 0.001

            # cube 在 plate 的另一侧（+y）随机；若仍落进 plate 附近就推开
            (cxmin, cxmax), (cymin, cymax) = self.CUBE_INIT_REGION
            cube_xyz = torch.zeros((b, 3))
            cube_xyz[:, 0] = torch.rand(b) * (cxmax - cxmin) + cxmin
            cube_xyz[:, 1] = torch.rand(b) * (cymax - cymin) + cymin
            cube_xyz[:, 2] = self.CUBE_HALF_SIZE + 0.001  # 略高于桌面

            # cube 离 plate 太近时往 plate 的反方向推 (plate 固定，移 cube)
            diff = cube_xyz[:, :2] - plate_xyz[:, :2]
            d = torch.linalg.norm(diff, dim=1, keepdim=True)
            too_close = (d < self.MIN_CUBE_PLATE_DIST).squeeze(-1)
            d_safe = torch.where(d < 1e-6, torch.ones_like(d), d)
            unit = diff / d_safe
            pushed = plate_xyz[:, :2] + unit * self.MIN_CUBE_PLATE_DIST
            cube_xyz[too_close, :2] = pushed[too_close]

            # plate 整体 quat = identity；底盘 cylinder 在 _build_plate_with_rim
            # 里已经用 _PLATE_FLAT_QUAT 摆成水平，无需在 actor 层面再转。
            q_identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(b, 1)
            self.cube.set_pose(Pose.create_from_pq(p=cube_xyz, q=q_identity))
            self.plate.set_pose(Pose.create_from_pq(p=plate_xyz, q=q_identity))

    # ---------- 评估 ----------
    def evaluate(self) -> Dict[str, Any]:
        is_grasped = self.agent.is_grasping(self.cube)
        cube_pos = self.cube.pose.p
        plate_pos = self.plate.pose.p

        # cube 在 plate xy 范围内
        xy_diff = cube_pos[..., :2] - plate_pos[..., :2]
        xy_dist = torch.linalg.norm(xy_diff, dim=-1)
        in_xy = xy_dist < self.SUCCESS_XY_THRESH

        # cube z 在合理高度
        rel_z = cube_pos[..., 2] - plate_pos[..., 2]
        in_z = (rel_z >= self.SUCCESS_Z_MIN) & (rel_z <= self.SUCCESS_Z_MAX)

        on_plate = in_xy & in_z
        success = on_plate & ~is_grasped  # 必须松手

        return {
            "success": success,
            "is_grasped": is_grasped,
            "cube_on_plate": on_plate,
            "cube_to_plate_xy_dist": xy_dist,
        }

    # ---------- 观测扩展 ----------
    def _get_obs_extra(self, info: Dict) -> Dict[str, Any]:
        obs = dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            plate_pos=self.plate.pose.p,
        )
        if self._obs_mode in ("state", "state_dict"):
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                tcp_to_cube_pos=self.cube.pose.p - self.agent.tcp_pose.p,
                cube_to_plate_pos=self.plate.pose.p - self.cube.pose.p,
            )
        return obs

    # ---------- Reward ----------
    def compute_dense_reward(self, obs, action, info) -> torch.Tensor:
        tcp_pos = self.agent.tcp_pose.p
        cube_pos = self.cube.pose.p
        plate_pos = self.plate.pose.p

        # 1) 接近 cube
        reach_d = torch.linalg.norm(tcp_pos - cube_pos, axis=1)
        reach_r = 1.0 - torch.tanh(5.0 * reach_d)

        # 2) 抓住 cube
        grasp_r = info["is_grasped"].float() * 2.0

        # 3) 抓住后搬到 plate 上方
        target = plate_pos.clone()
        target[:, 2] += 0.10
        move_d = torch.linalg.norm(cube_pos - target, axis=1)
        move_r = info["is_grasped"].float() * (1.0 - torch.tanh(5.0 * move_d)) * 1.5

        # 4) 落到 plate 里
        on_plate_r = info["cube_on_plate"].float() * 2.0

        # 5) 成功 (放下 + 在 plate 里)
        success_bonus = info["success"].float() * 5.0

        return reach_r * 0.3 + grasp_r + move_r + on_plate_r + success_bonus

    def compute_normalized_dense_reward(self, obs, action, info) -> torch.Tensor:
        # 最大可能 reward ≈ 0.3 + 2.0 + 1.5 + 2.0 + 5.0 = 10.8
        return self.compute_dense_reward(obs, action, info) / 11.0

    def get_language_instruction(self) -> list[str]:
        return ["pick up the red cube and place it on the white plate"] * self.num_envs
