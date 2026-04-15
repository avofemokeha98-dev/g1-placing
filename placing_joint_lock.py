# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 Placing / MARL 共用：锁死关节名模式、解析 ID、将目标角写回（与单机 ``g1_placing_env`` 行为一致）。"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation

# ---------------------------------------------------------------------------
# 与 ``g1_placing_env._initialize_locked_joints`` 保持同一顺序与内容
# ---------------------------------------------------------------------------
LOCKED_FINGER_JOINT_NAME_PATTERNS: list[str] = [
    ".*_five_joint",
    ".*_three_joint",
    ".*_six_joint",
    ".*_four_joint",
    ".*_zero_joint",
    ".*_one_joint",
    ".*_two_joint",
]

LOCKED_ELBOW_JOINT_NAME_PATTERNS: list[str] = [
    ".*_elbow_pitch_joint",
    ".*_elbow_roll_joint",
]

# 腰部不锁死，由策略控制（与 placing 一致：空列表）
LOCKED_TORSO_JOINT_NAME_PATTERNS: list[str] = []

LOCKED_ANKLE_JOINT_NAME_PATTERNS: list[str] = [
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]

LOCKED_SHOULDER_JOINT_NAME_PATTERNS: list[str] = [
    ".*shoulder_pitch.*joint",
    ".*shoulder_roll.*joint",
    ".*shoulder_yaw.*joint",
]


def all_locked_joint_name_patterns() -> list[str]:
    return (
        LOCKED_FINGER_JOINT_NAME_PATTERNS
        + LOCKED_ELBOW_JOINT_NAME_PATTERNS
        + LOCKED_TORSO_JOINT_NAME_PATTERNS
        + LOCKED_ANKLE_JOINT_NAME_PATTERNS
        + LOCKED_SHOULDER_JOINT_NAME_PATTERNS
    )


def collect_locked_joint_ids(robot: Articulation) -> list[int]:
    """解析 ``robot`` 上所有锁死关节的索引（去重排序）。"""
    out: list[int] = []
    for pattern in all_locked_joint_name_patterns():
        joint_ids, _ = robot.find_joints(pattern)
        out.extend(joint_ids)
    return sorted(set(out))


def apply_locked_joint_targets(
    robot: Articulation,
    joint_pos_target: torch.Tensor,
    default_pos: torch.Tensor,
    locked_joint_ids: list[int],
) -> None:
    """将锁死关节的目标位置写回：踝/肩/肘 pitch&roll → 0；手指等其余锁死关节 → ``default_pos``。"""
    if not locked_joint_ids:
        return

    ankle_pitch_ids, _ = robot.find_joints(".*_ankle_pitch_joint")
    ankle_roll_ids, _ = robot.find_joints(".*_ankle_roll_joint")
    ankle_joint_ids = ankle_pitch_ids + ankle_roll_ids

    shoulder_pitch_ids, _ = robot.find_joints(".*shoulder_pitch.*joint")
    shoulder_roll_ids, _ = robot.find_joints(".*shoulder_roll.*joint")
    shoulder_yaw_ids, _ = robot.find_joints(".*shoulder_yaw.*joint")
    shoulder_joint_ids = shoulder_pitch_ids + shoulder_roll_ids + shoulder_yaw_ids

    elbow_pitch_ids, _ = robot.find_joints(".*_elbow_pitch_joint")
    elbow_roll_ids, _ = robot.find_joints(".*_elbow_roll_joint")
    elbow_joint_ids = elbow_pitch_ids + elbow_roll_ids

    for ankle_id in ankle_joint_ids:
        if ankle_id in locked_joint_ids:
            joint_pos_target[:, ankle_id] = 0.0
    for shoulder_id in shoulder_joint_ids:
        if shoulder_id in locked_joint_ids:
            joint_pos_target[:, shoulder_id] = 0.0
    for elbow_pitch_id in elbow_pitch_ids:
        if elbow_pitch_id in locked_joint_ids:
            joint_pos_target[:, elbow_pitch_id] = 0.0
    for elbow_roll_id in elbow_roll_ids:
        if elbow_roll_id in locked_joint_ids:
            joint_pos_target[:, elbow_roll_id] = 0.0
    for joint_id in locked_joint_ids:
        if (
            joint_id not in ankle_joint_ids
            and joint_id not in shoulder_joint_ids
            and joint_id not in elbow_joint_ids
        ):
            joint_pos_target[:, joint_id] = default_pos[:, joint_id]
