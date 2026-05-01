"""地面平面内的参考路径（世界系水平面 XY，无竖直起伏）。

几何定义（路径系：原点 = 重置时刻根位置在地面的投影，+X = 当时机头航向）：

1. 沿 +X **直线**前进 ``straight_m`` 米（默认 5 m）；
2. 再沿 **四分之一圆**（90°）转弯，**弧长** ``arc_length_m`` 米（默认 10 m，半径 R=2L/π）。

由弧长进度可得到同一曲线上的世界系切向速度 (vx, vy) 与绕竖直轴角速度 ω。
"""
from __future__ import annotations
import math

import numpy as np
import torch
import trimesh


def quarter_circle_radius_from_arc_length(arc_length_m: float) -> float:
    """90° 圆弧：弧长 L = (π/2) R → R = 2L/π。"""
    return 2.0 * float(arc_length_m) / math.pi


def total_path_length_m(straight_m: float, arc_length_m: float) -> float:
    return float(straight_m) + float(arc_length_m)


def reference_path_polyline_world(
    origin_xy: torch.Tensor,
    psi0: torch.Tensor,
    z_world: float,
    straight_m: float,
    arc_radius_m: float,
    arc_length_m: float,
    *,
    turn_left: bool = True,
    n_straight: int = 48,
    n_arc: int = 36,
) -> torch.Tensor:
    """在世界系 **XY 地面**上生成参考路径采样点：直线段 + 四分之一圆弧；全程 Z = ``z_world``。

    直线沿航向 ``psi0`` 从 ``origin_xy`` 出发，长度 ``straight_m``；圆弧弧长 ``arc_length_m``、半径 ``arc_radius_m``。

    Args:
        origin_xy: (2,) 世界系路径起点
        psi0: 标量张量，起始航向（rad）
        z_world: 贴地可视化高度（m）
        arc_length_m: 90° 圆弧弧长（m）

    Returns:
        ``[N, 3]`` 连续折线，``N = n_straight + n_arc - 1``（直线末点与圆弧起点去重）。
    """
    device = origin_xy.device
    dtype = origin_xy.dtype
    L1 = float(straight_m)
    R = float(arc_radius_m)
    L_arc = float(arc_length_m)
    n_s = int(max(2, n_straight))
    n_a = int(max(2, n_arc))

    s_straight = torch.linspace(0.0, L1, n_s, device=device, dtype=dtype)
    dx = s_straight * torch.cos(psi0)
    dy = s_straight * torch.sin(psi0)
    pts_straight_xy = origin_xy.unsqueeze(0) + torch.stack([dx, dy], dim=1)

    arc_start_xy = pts_straight_xy[-1]
    direction = 1.0 if turn_left else -1.0
    cx = arc_start_xy[0] - R * torch.sin(psi0) * direction
    cy = arc_start_xy[1] + R * torch.cos(psi0) * direction

    s_arc = torch.linspace(0.0, L_arc, n_a, device=device, dtype=dtype)
    theta = s_arc / R
    half_pi = torch.as_tensor(0.5 * math.pi, device=device, dtype=dtype)
    angle0 = psi0 - half_pi * direction
    current_angles = angle0 + theta * direction

    arc_x = cx + R * torch.cos(current_angles)
    arc_y = cy + R * torch.sin(current_angles)
    pts_arc_xy = torch.stack([arc_x, arc_y], dim=1)

    all_xy = torch.cat([pts_straight_xy, pts_arc_xy[1:]], dim=0)
    z_coords = torch.full((all_xy.shape[0], 1), float(z_world), device=device, dtype=dtype)
    return torch.cat([all_xy, z_coords], dim=1)


def polyline_ground_ribbon_trimesh(points_xyz: np.ndarray, half_width_m: float) -> trimesh.Trimesh:
    """将地面折线挤出为带状网格（连续路面），法线朝上。

    Args:
        points_xyz: ``[N, 3]`` 世界系路径采样点（近似共面）。
        half_width_m: 带条半宽（米），总宽度为 ``2 * half_width_m``。

    Returns:
        双面可见的三角条带网格，无物理、仅用于可视化。
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.shape[0] < 2:
        raise ValueError('polyline_ground_ribbon_trimesh requires at least 2 points')
    if half_width_m <= 0.0:
        raise ValueError('half_width_m must be positive')
    n = pts.shape[0]
    tangents = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        if i == 0:
            d = pts[1] - pts[0]
        elif i == n - 1:
            d = pts[-1] - pts[-2]
        else:
            d = pts[i + 1] - pts[i - 1]
        norm = float(np.linalg.norm(d))
        if norm < 1e-09:
            d = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            d = d / norm
        tangents[i] = d
    perp = np.stack([-tangents[:, 1], tangents[:, 0], np.zeros(n, dtype=np.float64)], axis=1)
    pn = np.linalg.norm(perp[:, :2], axis=1, keepdims=True) + 1e-09
    perp[:, 0:2] /= pn
    w = float(half_width_m)
    left = pts + w * perp
    right = pts - w * perp
    vertices = np.zeros((2 * n, 3), dtype=np.float64)
    vertices[0::2] = left
    vertices[1::2] = right
    faces: list[list[int]] = []
    for i in range(n - 1):
        faces.append([2 * i, 2 * i + 1, 2 * (i + 1)])
        faces.append([2 * i + 1, 2 * (i + 1) + 1, 2 * (i + 1)])
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)


def reference_path_velocity_world(
    progress_m: torch.Tensor,
    speed_m_s: float,
    straight_m: float,
    arc_radius_m: float,
    arc_length_m: float,
    psi0: torch.Tensor,
    *,
    turn_left: bool=True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """根据沿**地面路径**的弧长进度 ``progress_m``（米）返回世界系水平速度。

    路径贴地（竖直方向无分量）：+X 为重置时刻机头航向，原点为重置时刻根在地面投影。
    第一段直线长 ``straight_m``；第二段为弧长 ``arc_length_m`` 的四分之一圆（90°）。
    ``turn_left=True`` 为左转（逆时针）。

    Args:
        progress_m: (num_envs,) 从回合开始累计的弧长（建议每步 += speed * dt）。
        psi0: (num_envs,) 重置时刻根航向（绕世界 Z，与 euler_xyz_from_quat 的 yaw 一致）。
        turn_left: True 时圆心在直线终点左侧（+Y 侧），False 为右转。

    Returns:
        (vx_w, vy_w, omega_z_w)，各 (num_envs,)，单位 m/s 与 rad/s；走完路径后为 0。
    """
    L1 = float(straight_m)
    R = float(arc_radius_m)
    L2 = float(arc_length_m)
    Ltot = L1 + L2
    V = float(speed_m_s)
    sign = 1.0 if turn_left else -1.0
    s = progress_m
    at_end = s >= Ltot
    s_eff = torch.clamp(s, max=Ltot)
    theta = torch.where(s_eff <= L1, torch.zeros_like(s_eff), sign * (s_eff - L1) / R)
    psi = psi0 + theta
    cos_p = torch.cos(psi)
    sin_p = torch.sin(psi)
    vx_w = torch.where(at_end, torch.zeros_like(s), V * cos_p)
    vy_w = torch.where(at_end, torch.zeros_like(s), V * sin_p)
    on_arc = (~at_end) & (s_eff > L1)
    omega_mag = sign * V / R
    omega_z_w = torch.where(on_arc, torch.full_like(s, omega_mag), torch.zeros_like(s))
    return (vx_w, vy_w, omega_z_w)
