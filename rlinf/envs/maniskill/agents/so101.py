# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SO-ARM 101 Robot Agent for ManiSkill.

适配本项目：直接加载用户已有的 MJCF（含完整 STL mesh），不使用 URDF。
- 6 DOF: shoulder_pan / shoulder_lift / elbow_flex / wrist_flex / wrist_roll / gripper
- 2 cameras: top_camera (由 task 提供) + wrist_camera (mounted on gripper link)
- TCP: 用 gripper link + offset 计算（MJCF 里没有专门 TCP link）

参考: april5129/RLinf fork 的 agents/so101.py（适配本项目 MJCF 改造）
"""

import copy
import hashlib
import os
import re
import tempfile

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor

# 用户已有的 SO-101 MJCF (含 STL mesh，已通过 SAPIEN 加载验证)
SO101_MJCF_SRC = (
    "/home/zzg/workspace/pycharm/Robot/assets/so101_pick101/so101_new_calib.xml"
)

# 真机配色：机械臂主体白色，夹爪黑色（与 MJCF 默认金色 / 金色 jaw 不同）
_ARM_WHITE = "0.95 0.95 0.95 1"
_GRIPPER_BLACK = "0.05 0.05 0.05 1"

_MATERIAL_OVERRIDES = {
    # 主体 (原 1 0.82 0.12 金色) → 白
    "base_motor_holder_so101_v1_material": _ARM_WHITE,
    "base_so101_v2_material": _ARM_WHITE,
    "waveshare_mounting_plate_so101_v2_material": _ARM_WHITE,
    "motor_holder_so101_base_v1_material": _ARM_WHITE,
    "rotation_pitch_so101_v1_material": _ARM_WHITE,
    "upper_arm_so101_v1_material": _ARM_WHITE,
    "under_arm_so101_v1_material": _ARM_WHITE,
    "motor_holder_so101_wrist_v1_material": _ARM_WHITE,
    "wrist_roll_pitch_so101_v2_material": _ARM_WHITE,
    # 手腕末端 (wrist_roll_follower mesh 含 static finger 区域) → 黑色，
    # 真机这一段从末端电机壳到下爪 fingertip 整片是黑色，白色会显得下爪也是白
    "wrist_roll_follower_so101_v1_material": _GRIPPER_BLACK,
    # 活动夹爪 (原金色) → 黑色，对齐真机
    "moving_jaw_so101_v1_material": _GRIPPER_BLACK,
    # sts3215_* 已经是黑色，保持
}


def _build_recolored_mjcf(src_path: str) -> str:
    """生成颜色覆盖后的 MJCF 临时副本，路径稳定（hash 缓存）。

    覆盖 rgba 字段：金色机械臂部件→白色，moving_jaw→黑色。
    其他几何引用 (mesh file) 不动，保持相对路径生效（同目录拷贝）。
    """
    with open(src_path, "r", encoding="utf-8") as f:
        xml = f.read()

    for mat_name, rgba in _MATERIAL_OVERRIDES.items():
        # 匹配 <material name="X" rgba="..."/>
        pattern = rf'(<material\s+name="{re.escape(mat_name)}"\s+rgba=")[^"]+(")'
        xml = re.sub(pattern, rf'\g<1>{rgba}\g<2>', xml)

    # hash 缓存：源文件 mtime + 覆盖规则
    digest_src = (
        xml.encode("utf-8")
        + str(_MATERIAL_OVERRIDES).encode("utf-8")
    )
    digest = hashlib.md5(digest_src).hexdigest()[:10]

    src_dir = os.path.dirname(os.path.abspath(src_path))
    cache_path = os.path.join(src_dir, f"so101_recolored_{digest}.xml")
    if not os.path.exists(cache_path):
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(xml)
    return cache_path


SO101_MJCF_PATH = _build_recolored_mjcf(SO101_MJCF_SRC)


@register_agent()
class SO101(BaseAgent):
    """SO-ARM 101 Robot Agent.

    6-DOF robotic arm from LeRobot:
    - 5 arm joints + 1 gripper joint
    - Loaded from MJCF (MuJoCo XML)
    """

    uid = "so101"
    mjcf_path = SO101_MJCF_PATH
    urdf_path = None  # 显式禁用 URDF，强制走 MJCF 路径

    # MJCF 里没有 _materials 块，ManiSkill 会用默认；夹爪摩擦在 MJCF 里已经设置
    urdf_config = None

    keyframes = dict(
        # MuJoCo XML keyframe "home" 一致：末端朝下，夹爪略合
        home=Keyframe(
            qpos=np.array([0, 0, 0, np.pi / 2, np.pi / 2, 0.3]),
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        ),
        zero=Keyframe(
            qpos=np.array([0.0] * 6),
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        ),
        rest=Keyframe(
            qpos=np.array([0, -1.5708, 1.5708, 0.66, 0, -1.1]),
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        ),
        # 对齐 LeRobot real-hil dataset (so101_pickplace_real_hil_fps30_combined_v1)
        # 该 dataset 起手位是【双峰分布】（按 episode_index 看）：
        #   前 50 ep:  CURLED 缩起姿势（std<7°，明显是同一次规范录制）
        #   后 200 ep: EXTENDED 平伸姿势（std 15-33°，多次松散录制）
        # 用前 50 ep 的 cluster mean 作为 keyframe（curled 视觉与"机械臂收起来准备
        # 抓"的真机直觉一致；如果想匹配 extended 用 250 ep median 反而落到中间过渡区）。
        # 度数：[-3.38, -98.81, +93.49, +64.00, +0.70, +3.57]
        # ⚠️ shoulder_lift = -98.81° = -1.725 rad 距离 MJCF 限位 -1.745 只有 0.02 rad
        # = 1.2°，task `_initialize_episode` 里的 init_qpos_noise 必须 ≤ 0.003 rad
        # 否则有概率超限被 clip 到边界。
        lerobot_start=Keyframe(
            qpos=np.deg2rad(
                np.array([-3.38, -98.81, 93.49, 64.00, 0.70, 3.57])
            ),
            pose=sapien.Pose(),  # 底座原点；real setup 底座固定在桌面边缘
        ),
    )

    # 关节名（已经过 verify_so101_mjcf.py 验证，与 MJCF 一致）
    arm_joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]
    gripper_joint_names = ["gripper"]

    # MJCF 实际 link 名（不是 fork URDF 的 Fixed_Jaw / Moving_Jaw）
    # 用 gripper link 作为 TCP 基准；moving_jaw_so101_v1 是活动夹爪
    GRIPPER_LINK_NAME = "gripper"
    MOVING_JAW_LINK_NAME = "moving_jaw_so101_v1"
    # TCP 相对 gripper link 的偏移（沿夹爪末端方向，初始 4cm，需在 visualize 中调）
    TCP_OFFSET = sapien.Pose(p=[0.0, 0.0, 0.04])

    @property
    def _controller_configs(self):
        pd_joint_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            lower=None,
            upper=None,
            stiffness=[1e3] * 6,
            damping=[1e2] * 6,
            force_limit=100,
            normalize_action=False,
        )

        # Delta 位置控制：用于 RL 探索时小幅度增量
        pd_joint_delta_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            [-0.05, -0.05, -0.05, -0.05, -0.05, -0.2],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.2],
            stiffness=[1e3] * 6,
            damping=[1e2] * 6,
            force_limit=100,
            use_delta=True,
            use_target=False,
        )

        pd_joint_target_delta_pos = copy.deepcopy(pd_joint_delta_pos)
        pd_joint_target_delta_pos.use_target = True

        controller_configs = dict(
            pd_joint_delta_pos=pd_joint_delta_pos,
            pd_joint_pos=pd_joint_pos,
            pd_joint_target_delta_pos=pd_joint_target_delta_pos,
        )
        return deepcopy_dict(controller_configs)

    @property
    def _sensor_configs(self):
        """腕部相机：mount 在 gripper link 上，盯着两个 jaw + 工作区。

        gripper link local frame（实测 home pose）：
            local +X → world +Z (向上)
            local +Y → world +Y (向右)
            local +Z → world -X (后方)
        因此 jaw 沿 local -Z (= world +X, 向前) 延伸：
            static fingertip ~ (-0.012, 0, -0.104)
            moving jaw     ~ (0.020, 0.019, -0.023)
        但 (0, 0, -0.10) 这条线被 wrist-roll 电机挡住，
        参照 MJCF <camera name="wrist_cam"> 把摄像头偏到 -Y 侧 (gripper 一侧)，
        斜视过去能看清开合状态。
        """
        wrist_pose = sapien_utils.look_at(
            eye=[0.008, -0.065, -0.019],   # 与 MJCF wrist_cam 一致：侧贴 gripper
            target=[-0.012, 0.0, -0.104],  # 静止 fingertip 位置（实际 jaw 尖）
            up=[0.0, 0.0, 1.0],            # 让 gripper 后端在画面上方
        )
        return [
            CameraConfig(
                uid="wrist_camera",
                pose=wrist_pose,
                width=640,
                height=480,
                fov=np.deg2rad(70.5),
                near=0.01,
                far=2.0,
                mount=self.robot.links_map[self.GRIPPER_LINK_NAME],
            )
        ]

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        # MJCF 里没有专门的 Fixed_Jaw_tip 等 link
        # gripper 是固定夹爪基座，moving_jaw_so101_v1 是活动颚
        self.finger1_link = self.robot.links_map[self.GRIPPER_LINK_NAME]
        self.finger2_link = self.robot.links_map[self.MOVING_JAW_LINK_NAME]
        # 没有 tip link，用同 link 占位
        self.finger1_tip = self.finger1_link
        self.finger2_tip = self.finger2_link
        # TCP：用 gripper link，pose 经过 TCP_OFFSET 偏移
        self.tcp = self.finger1_link

    @property
    def tcp_pose(self):
        """TCP pose = gripper link pose 乘以 TCP_OFFSET。"""
        return self.tcp.pose * self.TCP_OFFSET

    @property
    def tcp_pos(self):
        return self.tcp_pose.p

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=110):
        """基于接触力判断是否抓住物体。

        Args:
            object: 目标物体 Actor
            min_force: 最小接触力 (N)
            max_angle: 接触方向与夹爪法线夹角阈值 (度)
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold=0.2):
        """判断机器人是否静止（不包括 gripper）。"""
        qvel = self.robot.get_qvel()[:, :-1]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """构造抓取 pose。"""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)
