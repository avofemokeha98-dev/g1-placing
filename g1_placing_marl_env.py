# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""三台 G1 + 可选六边形木板（DirectMARLEnv）。``cfg.use_hex_plank=False`` 时不生成木板；否则木板高度带内奖励，episode 在超时、倒地或板掉落时结束。"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.envs.common import ActionType, AgentID, ObsType, StateType
from isaaclab.sim.spawners.from_files import spawn_ground_plane
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    quat_rotate_inverse,
)

from .g1_placing_marl_env_cfg import AGENT_IDS, G1PlacingMarlEnvCfg, MARL_DEFAULT_CEREBELLUM_RELPATH
from .marl_foot_target_interpolation import smooth_foot_target_offset
from .placing_joint_lock import apply_locked_joint_targets, collect_locked_joint_ids
from .placing_joint_limits import reward_joint_limit_interval


class G1PlacingMarlEnv(DirectMARLEnv):
    cfg: G1PlacingMarlEnvCfg

    _AGENT_ORDER: ClassVar[tuple[str, ...]] = AGENT_IDS
    _articulation_count_mismatch_warned: ClassVar[bool] = False

    def __init__(self, cfg: G1PlacingMarlEnvCfg, render_mode: str | None = None, **kwargs):
        self._prev_joint_pos_target: dict[str, torch.Tensor | None] = {a: None for a in self._AGENT_ORDER}
        self._locked_joint_ids: list[int] | None = None
        self._locked_joint_initialized = False
        self._cerebellum_policy: Any = None
        self._cerebellum_cfg_load_attempted = False
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_limit_map: dict[int, list[float]] | None = None
        self._joint_limit_initialized = False

        # 高层：纯漂移（策略→边界映射后平滑）与节拍相位后的脚目标（喂小脑/观测）
        self._marl_drift_smooth: dict[str, torch.Tensor] = {}
        self._foot_target_offset_smooth: dict[str, torch.Tensor] = {}
        if self.cfg.use_high_level_foot_target:
            for aid in self._AGENT_ORDER:
                self._marl_drift_smooth[aid] = torch.zeros((self.num_envs, 3), device=self.device)
                self._foot_target_offset_smooth[aid] = torch.zeros((self.num_envs, 3), device=self.device)

        self.robot_0 = self.scene.articulations["robot_0"]
        self.robot_1 = self.scene.articulations["robot_1"]
        self.robot_2 = self.scene.articulations["robot_2"]
        self.hex_plank: RigidObject | None
        if self.cfg.use_hex_plank:
            self.hex_plank = self.scene.rigid_objects["hex_plank"]
        else:
            self.hex_plank = None
        self._robots: dict[str, Articulation] = {
            "robot_0": self.robot_0,
            "robot_1": self.robot_1,
            "robot_2": self.robot_2,
        }

        # 三台须为同一套 G1 资产（关节名与顺序一致），否则共享小脑 / 共用 joint_id 映射会出错
        self._assert_three_robots_identical_kinematics()

        # 本 episode 开始时木板 root 世界 z（用于相对下落终止，避免板卡在中等高度时绝对阈值永远不触发）
        self._plank_z_at_reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # 解析并缓存小脑对齐的脚目标物理边界（根坐标系，米）；_pre_physics_step 中 tanh 后线性映射用
        if self.cfg.use_high_level_foot_target:
            self._target_bound_low = torch.tensor(
                [
                    self.cfg.foot_target_offset_bound_x[0],
                    self.cfg.foot_target_offset_bound_y[0],
                    self.cfg.foot_target_offset_bound_z[0],
                ],
                device=self.device,
                dtype=torch.float32,
            )
            self._target_bound_high = torch.tensor(
                [
                    self.cfg.foot_target_offset_bound_x[1],
                    self.cfg.foot_target_offset_bound_y[1],
                    self.cfg.foot_target_offset_bound_z[1],
                ],
                device=self.device,
                dtype=torch.float32,
            )
        else:
            self._target_bound_low = self._target_bound_high = None

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._maybe_warn_articulation_physics_mismatch()
        return obs, info

    def _maybe_warn_articulation_physics_mismatch(self) -> None:
        """GPU PhysX 窄相报错后，常见症状：板（简单刚体）仍在，G1（Articulation）未进求解器 → 只见板悬空下落。"""
        if G1PlacingMarlEnv._articulation_count_mismatch_warned:
            return
        expected = self.num_envs
        bad: list[str] = []
        for aid, r in self._robots.items():
            try:
                n = int(r.num_instances)
            except Exception:
                continue
            if n != expected:
                bad.append(f"{aid}={n}")
        if not bad:
            return
        G1PlacingMarlEnv._articulation_count_mismatch_warned = True
        print(
            "[ERROR] G1PlacingMarlEnv: 机器人 Articulation 在 PhysX 中实例数与并行环境数不一致 "
            f"({', '.join(bad)}，期望各为 num_envs={expected})。"
            "典型现象：**视口只有六边形板、不见三台 G1，板从初始高度下落**。\n"
            "  常见原因：GPU PhysX 接触/窄相 kernel 失败（如 convexCoreConvexNphase_Kernel fail to launch），"
            "复杂人形未正确注册，而程序化刚体仍参与模拟。\n"
            "  处理：播放使用 `MARL_PLAY_CPU_PHYSICS=1 bash scripts/play_marl2.sh ...` 或 "
            "`play.py --cpu_physics`；或提高 PhysX gpu_* 缓冲后仍失败则改用 CPU 仿真。"
        )

    def _assert_three_robots_identical_kinematics(self) -> None:
        """保证 robot_0/1/2 结构等价：共享一份小脑权重、共用从 robot_0 解析的关节索引。"""
        n0 = tuple(self.robot_0.joint_names)
        n1 = tuple(self.robot_1.joint_names)
        n2 = tuple(self.robot_2.joint_names)
        if n0 != n1 or n0 != n2:
            raise RuntimeError(
                "G1PlacingMarlEnv: 三台机器人必须具有相同的 joint_names（顺序与名称），"
                "当前小脑与锁关节/限位均按同一套索引作用于三台。 "
                f"len(robot_0)={len(n0)} len(robot_1)={len(n1)} len(robot_2)={len(n2)}"
            )

    def set_cerebellum_policy(self, policy: Any) -> None:
        """注入单机踩点小脑 ``obs (N,106) -> joint (N,37)``（与 ``cerebellum_loader.load_frozen_policy`` 一致）。"""
        self._cerebellum_policy = policy

    def _try_load_cerebellum_from_cfg(self) -> None:
        """若尚未注入 ``_cerebellum_policy``，按 cfg 或默认候选路径加载单机 Placing 小脑 checkpoint。"""
        if self._cerebellum_policy is not None:
            return
        if self._cerebellum_cfg_load_attempted:
            return
        self._cerebellum_cfg_load_attempted = True
        if not getattr(self.cfg, "use_frozen_cerebellum", True):
            print("[INFO] G1PlacingMarlEnv: use_frozen_cerebellum=False, cerebellum not loaded from cfg.")
            return
        if not self.cfg.use_high_level_foot_target:
            print("[INFO] G1PlacingMarlEnv: use_high_level_foot_target=False, skipping cerebellum load.")
            return

        placing_dir = Path(__file__).resolve().parent
        repo_root = placing_dir.parents[5]
        default_log_ckpt = repo_root / MARL_DEFAULT_CEREBELLUM_RELPATH

        ckpt = getattr(self.cfg, "cerebellum_checkpoint", None)
        candidates: list[Path] = []
        if ckpt is not None and (not isinstance(ckpt, str) or ckpt.strip() != ""):
            p = Path(os.path.expanduser(str(ckpt)))
            candidates.append(p)
            if not p.is_file():
                candidates.append(placing_dir / str(ckpt))
                candidates.append(repo_root / str(ckpt))
        else:
            candidates.append(default_log_ckpt)
            candidates.append(placing_dir / "model_39999.pt")

        cand: Path | None = None
        for c in candidates:
            if c.is_file():
                cand = c
                break
        if cand is None:
            print(
                "[WARN] G1PlacingMarlEnv: cerebellum checkpoint not found. Tried: "
                + ", ".join(str(c) for c in candidates)
                + ". Set cfg.cerebellum_checkpoint or pass --cerebellum_checkpoint."
            )
            return

        from .cerebellum_loader import load_frozen_policy

        try:
            self._cerebellum_policy = load_frozen_policy(str(cand.resolve()), self.device)
            print(f"[INFO] G1PlacingMarlEnv: loaded cerebellum from {cand.resolve()}")
        except Exception as err:
            print(f"[WARN] G1PlacingMarlEnv: cerebellum load failed ({cand}): {err}")

    def _cerebellum_joint_act(self, robot: Articulation, aid: str) -> torch.Tensor:
        """同一套冻结小脑网络：按各机器人自身 obs 前 106 维推理；权重不区分 agent（``aid`` 仅兼容可选签名）。"""
        o106 = self._single_robot_obs(robot, aid)[:, :106]
        cb = self._cerebellum_policy
        with torch.no_grad():
            try:
                return cb(o106, aid)
            except TypeError:
                return cb(o106)

    def _initialize_locked_joints(self) -> None:
        """关节名与 ``g1_placing_env`` 共用 ``collect_locked_joint_ids``。

        仅用 ``robot_0`` 解析关节索引；假定三台与 ``_assert_three_robots_identical_kinematics`` 一致，故同一索引适用于 robot_1/2。
        """
        if self._locked_joint_initialized:
            return
        self._locked_joint_ids = collect_locked_joint_ids(self.robot_0)
        self._locked_joint_initialized = True

    def _initialize_joint_limits(self) -> None:
        """与 ``g1_placing_env._initialize_joint_limits`` 相同：``placing_joint_limits.reward_joint_limit_interval``。

        仅用 ``robot_0`` 的 ``joint_names`` 建 ``joint_id -> 限位``；三台结构一致时 id 对齐。
        """
        if self._joint_limit_initialized:
            return

        self._joint_limit_map = {}
        soft = float(getattr(self.cfg, "soft_dof_pos_limit", 0.9))
        torso_half = float(getattr(self.cfg, "torso_joint_limit_rad", 0.35))
        joint_names = self.robot_0.joint_names
        for joint_id, joint_name in enumerate(joint_names):
            lim = reward_joint_limit_interval(
                joint_name,
                soft_dof_pos_limit=soft,
                torso_half_rad=torso_half,
            )
            if lim is not None:
                self._joint_limit_map[joint_id] = list(lim)

        self._joint_limit_initialized = True

    def _apply_locked_joint_targets(
        self,
        robot: Articulation,
        joint_pos_target: torch.Tensor,
        default_pos: torch.Tensor,
    ) -> None:
        """与 ``g1_placing_env`` 共用 ``placing_joint_lock.apply_locked_joint_targets``。"""
        apply_locked_joint_targets(robot, joint_pos_target, default_pos, self._locked_joint_ids)

    def _setup_scene(self) -> None:
        self.robot_0 = Articulation(self.cfg.robot_0_cfg)
        self.robot_1 = Articulation(self.cfg.robot_1_cfg)
        self.robot_2 = Articulation(self.cfg.robot_2_cfg)
        self.hex_plank = RigidObject(self.cfg.hex_plank_cfg) if self.cfg.use_hex_plank else None

        spawn_ground_plane(prim_path="/World/ground", cfg=self.cfg.ground_plane_cfg)
        self.scene.clone_environments(copy_from_source=False)

        self.scene.articulations["robot_0"] = self.robot_0
        self.scene.articulations["robot_1"] = self.robot_1
        self.scene.articulations["robot_2"] = self.robot_2
        if self.cfg.use_hex_plank:
            self.scene.rigid_objects["hex_plank"] = self.hex_plank

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    @staticmethod
    def _get_feet_body_ids(robot: Articulation) -> list[int]:
        left_ankle_ids, _ = robot.find_bodies(".*left.*ankle.*link")
        right_ankle_ids, _ = robot.find_bodies(".*right.*ankle.*link")
        feet_body_ids: list[int] = []
        if len(left_ankle_ids) > 0:
            feet_body_ids.append(int(left_ankle_ids[0]))
        if len(right_ankle_ids) > 0:
            feet_body_ids.append(int(right_ankle_ids[0]))
        return feet_body_ids[:2]

    @staticmethod
    def root_offset_to_world(root_pos: torch.Tensor, root_quat: torch.Tensor, offset_root: torch.Tensor) -> torch.Tensor:
        """根坐标系偏移 → 世界系点（与 quat_apply 约定一致）。"""
        return root_pos + quat_apply(root_quat, offset_root)

    @staticmethod
    def world_point_to_root_offset(root_pos: torch.Tensor, root_quat: torch.Tensor, point_w: torch.Tensor) -> torch.Tensor:
        """世界系点 → 相对根的局部偏移（与木板 / 脚目标同一套 root 定义）。"""
        return quat_rotate_inverse(root_quat, point_w - root_pos)

    def _robot_min_root_height_z(self, robot: Articulation) -> torch.Tensor:
        """取 root 位姿 / link / 质心 三者世界系 z 的最小值，避免「根 actor 仍高、人已躺平」漏检。"""
        zp = robot.data.root_pos_w[:, 2]
        zl = robot.data.root_link_pos_w[:, 2]
        zc = robot.data.root_com_pos_w[:, 2]
        return torch.minimum(torch.minimum(zp, zl), zc)

    def _pre_physics_step(self, actions: dict[AgentID, ActionType]) -> None:
        for aid in self._AGENT_ORDER:
            self.actions[aid].copy_(actions[aid])

        if getattr(self.cfg, "debug_force_zero_action", False):
            for aid in self._AGENT_ORDER:
                self.actions[aid].zero_()

        if self.cfg.use_high_level_foot_target:
            alpha = float(self.cfg.foot_target_smooth_alpha)
            max_d = float(self.cfg.foot_target_max_delta_m)

            # 相位节拍：前半周期左脚相位、后半周期右脚相位（步长为 marl_phase_period_steps）
            phase_period = max(1, int(getattr(self.cfg, "marl_phase_period_steps", 16)))
            is_left_phase = (self.episode_length_buf % (phase_period * 2)) < phase_period

            # 左脚默认停靠 Y=+0.12，右脚 Y=-0.12；与平滑后的漂移离散叠加，不再对 Y 符号跳变做二次平滑
            _tmp0 = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            leg_offset_y = torch.where(is_left_phase, _tmp0 + 0.12, _tmp0 - 0.12)

            for aid in self._AGENT_ORDER:
                act = self.actions[aid]
                norm_act = torch.tanh(act[:, :3])

                dx = self._target_bound_low[0] + (norm_act[:, 0] + 1.0) * 0.5 * (
                    self._target_bound_high[0] - self._target_bound_low[0]
                )
                dy = self._target_bound_low[1] + (norm_act[:, 1] + 1.0) * 0.5 * (
                    self._target_bound_high[1] - self._target_bound_low[1]
                )
                dz = torch.zeros_like(dx)
                raw_drift = torch.stack([dx, dy, dz], dim=1)

                prev_drift = self._marl_drift_smooth[aid]
                smoothed_drift = smooth_foot_target_offset(
                    raw_drift, prev_drift, alpha=alpha, max_delta_m=max_d
                )
                self._marl_drift_smooth[aid] = smoothed_drift

                final_target = smoothed_drift.clone()
                final_target[:, 1] = final_target[:, 1] + leg_offset_y
                self._foot_target_offset_smooth[aid] = final_target

    def _apply_action(self) -> None:
        scale = float(self.cfg.action_scale)
        rate = float(self.cfg.action_rate_limit_rad_per_step)

        self._initialize_locked_joints()
        self._initialize_joint_limits()
        self._try_load_cerebellum_from_cfg()

        for aid in self._AGENT_ORDER:
            robot = self._robots[aid]
            act = self.actions[aid]
            if self.cfg.use_high_level_foot_target:
                if self._cerebellum_policy is not None:
                    joint_act = self._cerebellum_joint_act(robot, aid)
                else:
                    joint_act = act[:, 3:]
            else:
                joint_act = act
            default_pos = robot.data.default_joint_pos.clone()
            joint_pos_target = default_pos + joint_act * scale
            self._apply_locked_joint_targets(robot, joint_pos_target, default_pos)

            if self._joint_limit_map:
                for joint_id, (low, high) in self._joint_limit_map.items():
                    joint_pos_target[:, joint_id] = torch.clamp(
                        joint_pos_target[:, joint_id], low, high
                    )

            if rate > 0.0 and self._prev_joint_pos_target[aid] is not None:
                prev = self._prev_joint_pos_target[aid]
                joint_pos_target = torch.clamp(joint_pos_target, prev - rate, prev + rate)

            self._prev_joint_pos_target[aid] = joint_pos_target.clone()
            robot.set_joint_position_target(joint_pos_target, joint_ids=None)

    def _single_robot_obs(self, robot: Articulation, aid: str) -> torch.Tensor:
        root_pos = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        root_lin_vel_w = robot.data.root_lin_vel_w
        root_ang_vel_w = robot.data.root_ang_vel_w

        # --- 消除 yaw：给小脑「朝北」系下的姿态与速度（单机训练时以 yaw≈0 为主）---
        _roll, _pitch, yaw = euler_xyz_from_quat(root_quat_w)
        yaw_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        yaw_quat_inv = quat_inv(yaw_quat)

        root_quat_fake = quat_mul(yaw_quat_inv, root_quat_w)
        root_lin_vel_fake = quat_apply(yaw_quat_inv, root_lin_vel_w)
        root_ang_vel_fake = quat_apply(yaw_quat_inv, root_ang_vel_w)

        num_envs = root_pos.shape[0]
        if self.cfg.use_hex_plank and self.hex_plank is not None:
            plank_pos_w = self.hex_plank.data.root_pos_w
            plank_offset_local = self.world_point_to_root_offset(root_pos, root_quat_w, plank_pos_w)
        else:
            plank_offset_local = torch.zeros((num_envs, 3), device=self.device, dtype=root_pos.dtype)

        feet_body_ids = self._get_feet_body_ids(robot)
        num_feet = min(len(feet_body_ids), 2)
        feet_contact_state = torch.zeros((num_envs, num_feet), device=self.device)
        feet_pos_local = torch.zeros((num_envs, num_feet, 3), device=self.device)
        feet_quat_local = torch.zeros((num_envs, num_feet, 4), device=self.device)

        if num_feet == 2:
            body_pos_w = robot.data.body_pos_w
            body_quat_w = robot.data.body_quat_w
            fi = feet_body_ids
            feet_pos_w = body_pos_w[:, fi, :]
            feet_quat_w = body_quat_w[:, fi, :]
            feet_z = feet_pos_w[:, :, 2]
            feet_contact_state = (feet_z < 0.08).float()

            feet_offset_w = feet_pos_w - root_pos.unsqueeze(1)
            rq = root_quat_w.unsqueeze(1).expand(-1, num_feet, -1)
            feet_pos_local = quat_rotate_inverse(rq, feet_offset_w)
            root_quat_inv_exp = quat_inv(root_quat_w).unsqueeze(1).expand(-1, num_feet, -1)
            feet_quat_local = quat_mul(root_quat_inv_exp, feet_quat_w)

        feet_contact_flat = feet_contact_state
        feet_pos_flat = feet_pos_local.reshape(num_envs, -1)
        feet_quat_flat = feet_quat_local.reshape(num_envs, -1)

        if self.cfg.use_high_level_foot_target:
            # 给小脑「眼罩」：观测里抹掉根位置 XY，只保留 Z，避免三角站位时绝对坐标 OOD
            root_pos_fake = root_pos.clone()
            root_pos_fake[:, :2] = 0.0

            hl_offset_yaw_aligned = self._foot_target_offset_smooth[aid]

            offset_w = quat_apply(yaw_quat, hl_offset_yaw_aligned)

            base_target_w = root_pos.clone()
            base_target_w[:, 2] = 0.07

            final_target_w = base_target_w + offset_w
            target_offset_w = final_target_w - root_pos

            foot_target_local = quat_rotate_inverse(root_quat_w, target_offset_w)

            return torch.cat(
                (
                    root_pos_fake,
                    root_quat_fake,
                    root_lin_vel_fake,
                    root_ang_vel_fake,
                    robot.data.joint_pos,
                    robot.data.joint_vel,
                    foot_target_local,
                    feet_contact_flat,
                    feet_pos_flat,
                    feet_quat_flat,
                    plank_offset_local,
                ),
                dim=-1,
            )

        return torch.cat(
            (
                root_pos,
                root_quat_fake,
                root_lin_vel_fake,
                root_ang_vel_fake,
                robot.data.joint_pos,
                robot.data.joint_vel,
                plank_offset_local,
                feet_contact_flat,
                feet_pos_flat,
                feet_quat_flat,
            ),
            dim=-1,
        )

    def _get_observations(self) -> dict[AgentID, ObsType]:
        return {aid: self._single_robot_obs(self._robots[aid], aid) for aid in self._AGENT_ORDER}

    def _get_states(self) -> StateType:
        """仅当 ``cfg.state_space`` 为正整数（自定义全局状态维）时需要实现；``state_space=-1`` 时基类用观测拼接，不会调用此处。"""
        raise NotImplementedError(
            "G1PlacingMarlEnv: 自定义 state_space>0 时请实现 _get_states；默认请使用 cfg.state_space=-1（MAPPO 联合观测）。"
        )

    def _get_rewards(self) -> dict[AgentID, torch.Tensor]:
        # 2. 静止约束 A：动作幅度惩罚 (Action L2 Penalty)
        action_penalty = torch.zeros(self.num_envs, device=self.device)
        for aid in self._AGENT_ORDER:
            action_penalty += torch.sum(torch.square(self.actions[aid]), dim=1)
        w_action = getattr(self.cfg, "reward_action_penalty_weight", -0.0001)
        r_action = w_action * action_penalty

        if self.cfg.use_hex_plank and self.hex_plank is not None:
            # 1. 基础惩罚：木板过低
            plank_z = self.hex_plank.data.root_pos_w[:, 2]
            pen_h = float(self.cfg.reward_plank_low_penalty_height_m)
            w_low = float(self.cfg.reward_plank_low_penalty_weight)
            too_low = plank_z < pen_h
            # 2. 木板倾斜
            plank_quat = self.hex_plank.data.root_quat_w
            plank_up_z = 1.0 - 2.0 * (plank_quat[:, 1] ** 2 + plank_quat[:, 2] ** 2)
            tilt_error = 1.0 - plank_up_z
            w_tilt = getattr(self.cfg, "reward_plank_tilt_weight", -1.0)
            r_tilt = w_tilt * tilt_error
            r_team = (w_low * too_low.float()) + r_action + r_tilt
        else:
            r_team = r_action

        n_agents = len(self._AGENT_ORDER)
        w_surv = float(getattr(self.cfg, "reward_survival_weight", 0.5))
        r_team = r_team + w_surv * float(n_agents)
        r_each = r_team / float(n_agents)
        return {aid: r_each for aid in self._AGENT_ORDER}

    def _get_dones(self) -> tuple[dict[AgentID, torch.Tensor], dict[AgentID, torch.Tensor]]:
        """三机抬板：Episode 结束条件（**团队级**，``robot_0/1/2`` 共用同一 ``terminated``、``time_out``）。

        - ``time_out``：``episode_length_buf >= max_episode_length``。
        - ``terminated``：``robot_team_fallen | plank_dropped``。
          - ``robot_team_fallen``：**三台中任意一台**满足
            ``min(root_pos_z, root_link_z, root_com_z) < termination_fall_root_height_m``。
          - ``plank_dropped``：木板 ``z < termination_plank_height_m`` 或相对复位时下落
            ``> termination_plank_drop_m``。
        - ``termination_min_env_steps_before_fall``：前几步不判倒地/板落（仍判超时）。
        """
        max_len = self.max_episode_length
        time_out = self.episode_length_buf >= max_len

        fall_th = float(self.cfg.termination_fall_root_height_m)
        plank_th = float(self.cfg.termination_plank_height_m)

        h0 = self._robot_min_root_height_z(self.robot_0)
        h1 = self._robot_min_root_height_z(self.robot_1)
        h2 = self._robot_min_root_height_z(self.robot_2)
        h_stack = torch.stack((h0, h1, h2), dim=1)
        robot_fallen = torch.any(h_stack < fall_th, dim=1)

        # 木板掉落：绝对高度过低 **或** 相对本 episode 起始下落过大（无木板时永不因板终止）
        if self.cfg.use_hex_plank and self.hex_plank is not None:
            plank_z = self.hex_plank.data.root_pos_w[:, 2]
            plank_too_low = plank_z < plank_th
            plank_drop = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            drop_m = getattr(self.cfg, "termination_plank_drop_m", None)
            if drop_m is not None and float(drop_m) > 0.0:
                plank_drop = (self._plank_z_at_reset - plank_z) > float(drop_m)
            plank_dropped = plank_too_low | plank_drop
        else:
            plank_dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 前几步仅推进物理/避免缓冲区未稳定时误触发「倒地/板落」（仍允许 time_out）
        min_step = int(getattr(self.cfg, "termination_min_env_steps_before_fall", 1))
        if min_step > 1:
            gate = self.episode_length_buf >= min_step
            robot_fallen = robot_fallen & gate
            plank_dropped = plank_dropped & gate

        terminated = (robot_fallen | plank_dropped).to(dtype=torch.bool)
        td = {aid: terminated for aid in self._AGENT_ORDER}
        to = {aid: time_out for aid in self._AGENT_ORDER}
        return td, to

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            return
        ids = env_ids
        if not isinstance(ids, torch.Tensor):
            ids = torch.as_tensor(list(ids), device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return

        # --- 1. 重置高层漂移与脚目标缓冲 ---
        if self.cfg.use_high_level_foot_target:
            init = torch.zeros((ids.shape[0], 3), device=self.device)
            for aid in self._AGENT_ORDER:
                self._marl_drift_smooth[aid][ids] = init
                self._foot_target_offset_smooth[aid][ids] = init

        # --- 2. 物理重置：三台机器人 ---
        for aid in self._AGENT_ORDER:
            r = self._robots[aid]

            # 获取默认关节状态
            joint_pos = r.data.default_joint_pos[ids].clone()
            joint_vel = r.data.default_joint_vel[ids].clone()

            # 获取默认根状态（必须加上 env_origins 偏移！）
            default_root_state = r.data.default_root_state[ids].clone()
            default_root_state[:, :3] += self.scene.env_origins[ids]
            default_root_state[:, 7:13] = 0.0  # 速度清零

            # 【关键修复】将位姿和速度强制写回底层的物理引擎
            r.write_root_pose_to_sim(default_root_state[:, :7], ids)
            r.write_root_velocity_to_sim(default_root_state[:, 7:], ids)
            r.write_joint_state_to_sim(joint_pos, joint_vel, None, ids)
            # 必须传入 env_ids=ids，否则默认写满 num_envs 行，与 joint_pos 行数 (len(ids)) 不一致会报错
            r.set_joint_position_target(joint_pos, joint_ids=None, env_ids=ids)

            # 重置 PD 目标缓存，防止刚重置时产生巨大的瞬间扭矩
            if self._prev_joint_pos_target[aid] is not None:
                self._prev_joint_pos_target[aid][ids] = joint_pos.clone()

        # --- 3. 物理重置：木板 ---
        if self.cfg.use_hex_plank and self.hex_plank is not None:
            plank_root_state = self.hex_plank.data.default_root_state[ids].clone()
            plank_root_state[:, :3] += self.scene.env_origins[ids]
            plank_root_state[:, 7:13] = 0.0  # 速度清零

            self.hex_plank.write_root_pose_to_sim(plank_root_state[:, :7], ids)
            self.hex_plank.write_root_velocity_to_sim(plank_root_state[:, 7:], ids)

            # --- 4. 更新相关记录缓存 ---
            self._plank_z_at_reset[ids] = self.hex_plank.data.root_pos_w[ids, 2].clone()

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        pass
