# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Placing 奖励中的关节位置越界：参照 ``参考/unitree_rl_gym-main`` 的 ``LeggedRobot`` 逻辑。

- 硬限位数值来自 ``legged_gym/.../resources/robots/g1_description/g1_12dof.urdf``（髋 pitch/roll/yaw）。
- 与 ``legged_gym/envs/g1/g1_config.py`` 中 ``rewards.soft_dof_pos_limit = 0.9`` 一致：以 URDF 行程中点为轴，
  将可用区间缩窄为 ``soft`` 倍后再作为惩罚边界（越界量与 ``_reward_dof_pos_limits`` 同型）。
"""

from __future__ import annotations

import re


def _soften_interval(lo: float, hi: float, soft: float) -> tuple[float, float]:
    if soft <= 0.0:
        raise ValueError(f"soft_dof_pos_limit must be > 0, got {soft}")
    m = 0.5 * (lo + hi)
    r = hi - lo
    return m - 0.5 * r * soft, m + 0.5 * r * soft


def reward_joint_limit_interval(
    joint_name: str,
    *,
    soft_dof_pos_limit: float,
    torso_half_rad: float,
) -> tuple[float, float] | None:
    """返回该关节在奖励中使用的 ``[lower, upper]``；``None`` 表示不参与关节限位惩罚映射。"""
    jn = joint_name.lower()
    s = float(soft_dof_pos_limit)

    if re.search(r"hip_pitch", jn):
        return _soften_interval(-2.5307, 2.8798, s)

    if re.search(r"hip_roll", jn):
        # 左右髋 roll 在 URDF 中镜像不对称
        if jn.startswith("left"):
            return _soften_interval(-0.5236, 2.9671, s)
        if jn.startswith("right"):
            return _soften_interval(-2.9671, 0.5236, s)
        return _soften_interval(-0.5236, 0.5236, s)

    if re.search(r"hip_yaw", jn):
        return _soften_interval(-2.7576, 2.7576, s)

    if re.search(r"torso", jn):
        t = float(torso_half_rad)
        return (-t, t)

    return None
