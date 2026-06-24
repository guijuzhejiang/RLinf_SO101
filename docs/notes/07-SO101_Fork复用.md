# 07 · april5129 RLinf Fork 的 SO-101 资产

[april5129/RLinf](https://github.com/april5129/RLinf) 2025-12 提交了 SO-101 + OpenPI π₀.₅ 接入，官方未合并。本地克隆在 `~/workspace/pycharm/Robot/rlinf_bridge/reference/april5129_fork/`。

**我们没直接拷 fork 代码**：fork 用 URDF + 数字 joint 名（`"1"..."6"`），跟我们 MJCF 语义化命名（`shoulder_pan, ...`）不兼容。fork 起**架构参考**作用。

## 参考点

| Fork 文件 | 借鉴 |
|---|---|
| `agents/so101.py` | `@register_agent`、`_controller_configs`、`is_grasping` |
| `tasks/so101_pick_cube.py` | reward 设计思路（reach + grasp + bonus） |
| `models/embodiment/openpi/policies/so101_policy.py` | LeRobot policy → OpenPI wrapper 模式（启发 SmolVLA wrapper） |
| `dataconfig/so101_dataconfig.py` | LeRobot normalize stats 加载方式（留作后续） |
| URDF + 14 STL | ❌ 不用，用自己的 MJCF |

## 不能直接拷的原因

| | Fork URDF | 我们 MJCF |
|---|---|---|
| Joint 名 | `"1".."6"` 硬编码 | `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` |
| Link 名 | 与 fork agent 写死 | 跟 MJCF 一致 |
| 格式 | URDF + STL | MJCF + STL |

拷 fork agent 要改 5 处 link 名 + 路径换 MJCF，工作量跟自己写差不多。最终方案：MJCF + 自写 agent，借鉴架构思路。
