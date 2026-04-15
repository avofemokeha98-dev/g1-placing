# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""三机协同抬六边形木板：MARL 场景配置（与单机 Placing 同 G1 灵巧手观测维度）。"""

from __future__ import annotations

import math
from dataclasses import MISSING

import torch

from isaaclab_assets.robots.unitree import G1_CFG

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

import isaaclab.sim as sim_utils

from .g1_placing_env_cfg import _HOST_PD_ACTUATORS
from .hex_prism_mesh import MeshHexPrismCfg

# 与 g1_placing_env_cfg（灵巧手 g1.usd）对齐
G1_DEXTEROUS_NUM_JOINTS = 37
G1_DEXTEROUS_OBS_DIM = 13 + 2 * G1_DEXTEROUS_NUM_JOINTS + 3 + 16
# 高层「脚目标」模式：前 106 维 = 根+关节+脚目标(根系)+脚（与踩点小脑一致）；后 3 维 = 木板质心相对根（根系）
G1_MARL_OBS_DIM = 106 + 3
G1_MARL_ACTION_DIM = 3 + G1_DEXTEROUS_NUM_JOINTS

AGENT_IDS = ("robot_0", "robot_1", "robot_2")

# 单机 G1 Placing 小脑（RSL-RL）默认 checkpoint，相对仓库根；train_marl2 / play_marl2 / env 自动加载均应对齐此路径
MARL_DEFAULT_CEREBELLUM_RELPATH = "logs/rsl_rl/g1_placing/2026-04-11_09-57-49/model_29999.pt"


def _yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    q = quat_from_euler_xyz(
        torch.zeros(1, dtype=torch.float32),
        torch.zeros(1, dtype=torch.float32),
        torch.tensor([yaw], dtype=torch.float32),
    )
    return (float(q[0, 0]), float(q[0, 1]), float(q[0, 2]), float(q[0, 3]))


def _equilateral_triangle_robots(
    radius_m: float, root_z: float
) -> list[tuple[tuple[float, float, float], tuple[float, float, float, float]]]:
    """等边三角形顶点：机器人位于半径 `radius_m` 的圆上，水平面内朝向场景中心。"""
    out: list[tuple[tuple[float, float, float], tuple[float, float, float, float]]] = []
    for k in range(3):
        theta = math.pi / 2.0 + 2.0 * math.pi * k / 3.0
        x = radius_m * math.cos(theta)
        y = radius_m * math.sin(theta)
        dx = -x
        dy = -y
        yaw = math.atan2(dy, dx)
        out.append(((x, y, root_z), _yaw_to_quat_wxyz(yaw)))
    return out


def _marl_default_joint_pos() -> dict:
    """与单机 Placing 相同：全身以 ``G1_CFG.init_state.joint_pos`` 为默认（髋/膝/踝/手指等一致）。

    手臂单独覆盖：站立时 **大臂近似与地面垂直**、**小臂近似与地面平行**（G1 约定见 ``unitree.py``：
    ``elbow_pitch=0`` 为约 90° 弯，``π/2`` 为直臂）。肩/肘 roll 置 0 对称；若视口上手臂朝向仍不理想，可微调
    ``shoulder_pitch``（小弧度）。
    """
    j = dict(G1_CFG.init_state.joint_pos)
    j.pop(".*_elbow_pitch_joint", None)
    j.update(
        {
            "left_shoulder_pitch_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "left_elbow_pitch_joint": 0.0,
            "right_elbow_pitch_joint": 0.0,
            "left_elbow_roll_joint": 0.0,
            "right_elbow_roll_joint": 0.0,
        }
    )
    return j


def _make_robot_cfg(
    name: str,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
) -> ArticulationCfg:
    # 与单机 Placing 一致：手写 _setup_scene 时不会走 InteractiveScene 对 {ENV_REGEX_NS} 的 format，须用绝对路径。
    # ``name`` 必须与 USD 下 prim 名一致（当前 Robot_0 / Robot_1 / Robot_2）。若视口「只有板」先确认此处与 Stage 路径匹配。
    return G1_CFG.replace(
        prim_path=f"/World/envs/env_.*/{name}",
        init_state=G1_CFG.init_state.replace(pos=pos, rot=rot, joint_pos=_marl_default_joint_pos()),
        spawn=G1_CFG.spawn.replace(
            rigid_props=G1_CFG.spawn.rigid_props.replace(
                max_depenetration_velocity=10.0,
            ),
            articulation_props=G1_CFG.spawn.articulation_props.replace(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
        actuators=_HOST_PD_ACTUATORS,
    )


# 机器人根 z 与木板初始高度 **独立**（提高木板时只改 `_HEX_PLANK_INIT_Z_M`）。
# 须与 ``G1_CFG.init_state.pos[2]``、单机 ``g1_placing_env_cfg.reset_root_height`` 一致 (0.74)，否则默认站姿下足端会离地数厘米。
_ROBOT_ROOT_Z_M = 0.74
# 木板质心初始高度（世界系 z）；未做「+3 cm」抬高前即为此值。
_HEX_PLANK_INIT_Z_M = 0.92

_TRI = _equilateral_triangle_robots(radius_m=0.55, root_z=_ROBOT_ROOT_Z_M)


@configclass
class G1PlacingMarlEnvCfg(DirectMARLEnvCfg):
    """三台 G1 + 六边形木板（3 kg，外接圆半径更大）场景：等边三角站位；观测维与单机 Placing 一致。奖励：木板高度带内（见 env）；终止：超时、倒地、板掉落。"""

    decimation = 4
    episode_length_s = 15.0

    # 三台 G1 + 六边形板 × 大量并行 env：默认 PhysX GPU 补丁/配对缓冲偏小时易出现
    # convexCoreConvexNphase_Kernel / prepareLostFoundPairs / updateFrictionPatches fail to launch。
    # 参考 manager_based 重接触任务（如 in-hand）抬高 gpu_* 上限；仍报错可试减小 num_envs 或 sim.device="cpu"。
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_found_lost_pairs_capacity=2**23,
            gpu_found_lost_aggregate_pairs_capacity=2**26,
            gpu_collision_stack_size=2**27,
        ),
    )

    observation_spaces: dict = MISSING
    action_spaces: dict = MISSING
    # MAPPO/skrl 集中式 critic 需要非空 state；``-1`` = 拼接全体智能体观测（见 DirectMARLEnv.state）
    state_space = -1
    possible_agents = list(AGENT_IDS)

    # 视口：勿用默认 (7.5,7.5,7.5) 远景，否则六边形板占满画面、三台 G1 像黑点，易被误认为「只剩木板」。
    # origin_type=env：相机相对 env_0 原点，与 GridCloner 下机器人/板所在区域一致（world 原点可能对不齐第一个 env）。
    viewer: ViewerCfg = ViewerCfg(
        eye=(3.0, 2.8, 1.65),
        lookat=(0.0, 0.0, 0.82),
        origin_type="env",
        env_index=0,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # 三台 prim_path 形如 ``/World/envs/env_.*/Robot_0``；与 ``g1_placing_marl_env._setup_scene`` 中 scene 键 robot_0 等对应。
    robot_0_cfg: ArticulationCfg = _make_robot_cfg("Robot_0", _TRI[0][0], _TRI[0][1])
    robot_1_cfg: ArticulationCfg = _make_robot_cfg("Robot_1", _TRI[1][0], _TRI[1][1])
    robot_2_cfg: ArticulationCfg = _make_robot_cfg("Robot_2", _TRI[2][0], _TRI[2][1])

    # True：生成六边形木板（碰撞/视觉）；观测含板相对根 3 维，奖励/终止含板相关项。
    use_hex_plank: bool = True

    hex_plank_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/HexPlank",
        spawn=MeshHexPrismCfg(
            radius=0.52,
            height=0.03,
            rigid_props=RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.42, 0.22)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, _HEX_PLANK_INIT_Z_M), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    ground_plane_cfg: sim_utils.GroundPlaneCfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.7,
        ),
    )

    action_scale = 0.25
    action_rate_limit_rad_per_step = 0.08

    # 与 g1_placing_env 一致：躯干 ±rad；髋类为 URDF + soft_dof_pos_limit（见 placing_joint_limits）
    torso_joint_limit_rad: float = 0.35
    soft_dof_pos_limit: float = 0.9

    # --- Episode 终止（见 ``g1_placing_marl_env._get_dones``）---
    # 团队级：``robot_0/1/2`` 共用同一 ``terminated`` / ``time_out``。
    # ``terminated = (任一机器人根/质心最低高度过低) OR 木板掉落``。
    # 倒地：``min(root_pos_z, root_link_z, root_com_z) < termination_fall_root_height_m``（单机 ~0.37；多机抬板易下蹲，0.45 较稳）。
    termination_fall_root_height_m: float = 0.45
    # 木板 root 世界 z 低于该值则判掉落
    termination_plank_height_m: float = 0.6
    # 相对复位瞬间下落；首帧物理稳定时常 >0.12 导致误触发，略放宽。
    termination_plank_drop_m: float | None = 0.22
    # episode_length_buf 达到该值之后才判「机器人倒地 / 木板掉落」（仍判超时）。防首帧误杀导致 mean episode≈1。
    termination_min_env_steps_before_fall: int = 3

    # 木板质心高度（世界系 z）低于该阈值时每步施加团队惩罚（各智能体均分总惩罚）；无带内正奖励
    reward_plank_low_penalty_height_m: float = 0.7
    reward_plank_low_penalty_weight: float = -2.0

    # === 原地静止任务 奖励权重 ===
    reward_action_penalty_weight: float = -0.0001  # 动作幅度惩罚（三台动作 L2 之和）
    reward_plank_tilt_weight: float = -1.0  # 木板倾斜惩罚
    # 每步生存奖励：写入团队标量后再 /n_agents，使每名智能体每步 +reward_survival_weight
    reward_survival_weight: float = 0.5

    # === 强制静止 / 开环测试（调试用）===
    # True：每步将各智能体 action 置零（见 ``g1_placing_marl_env._pre_physics_step``），排除策略输出干扰，便于确认 USD/PhysX 是否正常生成三台 G1。
    # 若仍为「只有木板」，请核对 ``robot_*_cfg`` 的 ``prim_path``（须与 Stage 下 ``/World/envs/env_*/Robot_*`` 一致）。
    # **正常 MARL 训练前务必将此项改回 False。**
    debug_force_zero_action: bool = True

    # 高层通过「根坐标系下脚目标点」驱动（无速度接口时）；False 时退化为仅 37 维关节、106 维观测（木板 3 维在观测中段）
    use_high_level_foot_target: bool = True

    # 冻结单机踩点小脑（106→37）：True 时关节目标由 checkpoint 推理；False 时关节用高层策略输出的后 37 维
    use_frozen_cerebellum: bool = True
    # 默认即下方 MARL 小脑路径；可改为 None 仅走 env 内候选，或填其它 .pt
    cerebellum_checkpoint: str | None = MARL_DEFAULT_CEREBELLUM_RELPATH

    # ------------------ 漂移边界与相位节拍（根坐标系，米）；动作经 tanh 后线性映射到 [low, high]；Z 钉死为 0 ------------------
    # 大脑只输出整体平移漂移 (Drift)；Y 向另由 env 内节拍对 ±交替，对齐小脑步态半周期
    foot_target_offset_bound_x: tuple[float, float] = (-0.05, 0.10)
    foot_target_offset_bound_y: tuple[float, float] = (-0.05, 0.05)
    foot_target_offset_bound_z: tuple[float, float] = (0.0, 0.0)

    # 相位节拍：每 N 个 env 步将脚目标 Y 漂移乘以 +1/-1，与单机小脑步态周期对齐
    marl_phase_period_steps: int = 16

    # 指数平滑系数：越大越跟得上高层指令，越小越平滑
    foot_target_smooth_alpha: float = 0.45
    # 每环境步对平滑后目标在根坐标系各轴上的最大变化（米），抑制 PD 冲击
    foot_target_max_delta_m: float = 0.035

    def __post_init__(self):
        if self.use_high_level_foot_target:
            if self.use_frozen_cerebellum:
                # 关键修复：高层只负责输出 (dx, dy, dz) 的脚部目标偏移！
                ad = 3
            else:
                ad = G1_MARL_ACTION_DIM
            od = G1_MARL_OBS_DIM
        else:
            od, ad = G1_DEXTEROUS_OBS_DIM, G1_DEXTEROUS_NUM_JOINTS

        self.observation_spaces = {AGENT_IDS[0]: od, AGENT_IDS[1]: od, AGENT_IDS[2]: od}
        self.action_spaces = {AGENT_IDS[0]: ad, AGENT_IDS[1]: ad, AGENT_IDS[2]: ad}
