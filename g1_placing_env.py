
from __future__ import annotations

import re
import math
from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply,
    quat_apply_yaw,
    quat_inv,
    quat_mul,
    quat_rotate_inverse,
    sample_uniform,
)

from .g1_placing_env_cfg import G1PlacingEnvCfg
from .placing_joint_lock import apply_locked_joint_targets, collect_locked_joint_ids
from .placing_joint_limits import reward_joint_limit_interval


class G1PlacingEnv(DirectRLEnv):
    """G1 踩点环境：矩形采样地面目标点、动态摆动相引导（滞空/离地/摆速）、踩点奖励与稳定性/平滑正则。"""

    cfg: G1PlacingEnvCfg

    def __init__(self, cfg: G1PlacingEnvCfg, render_mode: str | None = None, **kwargs):
        # 点位生成器状态
        self._foot_target_positions: torch.Tensor | None = None  # 目标点位位置（世界坐标系）
        self._target_generation_time: torch.Tensor | None = None  # 点位生成时间
        self._target_hit: torch.Tensor | None = None  # 是否踩到点位
        self._target_foot_indices: torch.Tensor | None = None  # 摆动脚索引（0=左脚，1=右脚）
        self._last_touchdown_time: torch.Tensor | None = None  # 最后一次落地时间（用于超时刷新）
        self._user_target_mode: torch.Tensor | None = None  # 用户指定目标模式：True 时跳过自动生成，踩中后清除
        self._target_regenerate_deadline: torch.Tensor | None = None  # 自动模式踩中后：仿真时刻达到该值才生成下一目标（nan=未调度）
        self._swing_air_accum_s: torch.Tensor | None = None  # 本目标周期内摆动脚累计滞空时间（秒），用于完美落地滞空奖励

        self._swing_foot_lifted: torch.Tensor | None = None  # 本目标周期内摆动脚是否曾离地（fallback 踩点判定）
        self._swing_foot_contact_prev: torch.Tensor | None = None  # 上步摆动脚触地（用于 touchdown 与路径起点）
        # 可视化标记（显示目标点位）
        self._target_markers: VisualizationMarkers | None = None
        # 触地：物理模式用 ContactSensor；否则用高度+连续帧+速度启发式
        self._foot_contact_sensor: ContactSensor | None = None
        self._foot_contact_body_sensor_indices: torch.Tensor | None = None  # (2,) 左/右脚在传感器 body 维中的索引
        self._ankle_contact_consecutive_count: torch.Tensor | None = None
        self._feet_contact_cache: torch.Tensor | None = None  # 每步只更新一次，供 obs 与 reward 共用
        self._feet_air_time: torch.Tensor | None = None  # (num_envs, 2) 左/右脚滞空；super 后用 num_envs/device 分配

        super().__init__(cfg, render_mode, **kwargs)

        # 左/右脚滞空状态；(num_envs,2) 须在 super 之后分配（num_envs/device 由父类就绪）
        self._feet_air_time = torch.zeros((self.num_envs, 2), device=self.device)
        self._left_knee_idx = self.robot.find_joints(".*left_knee_joint")[0][0]
        self._right_knee_idx = self.robot.find_joints(".*right_knee_joint")[0][0]

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        
        self._prev_joint_vel = None
        
        self._prev_actions = None
        
        self._locked_joint_ids = None
        self._locked_joint_initialized = False

        self._ankle_locked_angle_pitch = 0.0
        self._ankle_locked_angle_roll = 0.0

        # 关节奖励限位见 ``placing_joint_limits``（髋：g1_12dof.urdf + soft_dof_pos_limit；躯干：cfg）
        self._joint_limit_map = None
        self._joint_limit_initialized = False

        # 宇树 hip_pos：髋 roll/yaw 绝对角平方惩罚（见 _initialize_hip_pos_penalty_joints）
        self._hip_pos_joint_ids: torch.Tensor | None = None
        self._hip_pos_joint_initialized = False

        # 关节目标变化率限制：上一步的关节目标（用于抑制高频）
        self._prev_joint_pos_target: torch.Tensor | None = None

        # 物理干扰课程：下次推力触发的 episode 内时间（秒）；与 event_curriculum 阶段联动
        self._event_next_push_at: torch.Tensor | None = None
        self._event_mass_asset_cfg: SceneEntityCfg | None = None  # lazy resolve for randomize_rigid_body_mass
        if getattr(cfg, "event_curriculum_enabled", False):
            self._event_next_push_at = torch.zeros(self.num_envs, device=self.device)

    def _get_event_curriculum_phase(self) -> dict | None:
        """由 common_step_counter 得到训练 iteration，减去 event_curriculum_base_iteration 后选阶段。"""
        if not getattr(self.cfg, "event_curriculum_enabled", False):
            return None
        cur = getattr(self.cfg, "event_curriculum", None)
        if not cur:
            return None
        step = int(self.common_step_counter)
        spi = int(getattr(self.cfg, "event_curriculum_steps_per_iteration", 24))
        base = int(getattr(self.cfg, "event_curriculum_base_iteration", 0))
        it = max(0, step // max(1, spi) - base)
        for p in cur:
            start = int(p.get("iter_start", 0))
            end = p.get("iter_end")
            if it < start:
                continue
            if end is not None and it >= int(end):
                continue
            return p
        return cur[-1] if cur else None

    def _apply_event_curriculum_mass(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        phase = self._get_event_curriculum_phase()
        if phase is None:
            return
        mass = phase.get("mass_add_kg")
        if mass is None:
            return
        eids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)
        if eids.numel() == 0:
            return
        if self._event_mass_asset_cfg is None:
            names = list(getattr(self.cfg, "event_torso_body_names", (".*torso.*",)))
            self._event_mass_asset_cfg = SceneEntityCfg("robot", body_names=names)
            self._event_mass_asset_cfg.resolve(self.scene)
        mdp.randomize_rigid_body_mass(
            self,
            eids,
            asset_cfg=self._event_mass_asset_cfg,
            mass_distribution_params=(float(mass[0]), float(mass[1])),
            operation="add",
        )

    def _reschedule_event_push_times_on_reset(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        if self._event_next_push_at is None:
            return
        eids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)
        if eids.numel() == 0:
            return
        lo, hi = getattr(self.cfg, "event_push_interval_range_s", (2.0, 4.0))
        self._event_next_push_at[eids] = sample_uniform(
            float(lo), float(hi), (eids.shape[0],), self.device
        )

    def _apply_event_curriculum_push_interval(self) -> None:
        if self._event_next_push_at is None:
            return
        phase = self._get_event_curriculum_phase()
        if phase is None or not phase.get("push_enabled", False):
            return
        if "push_xy" not in phase:
            return
        px = float(phase["push_xy"])
        pz = float(phase.get("push_z", 0.05))
        vel_range = {"x": (-px, px), "y": (-px, px), "z": (-pz, pz)}
        t = self.episode_length_buf.float() * self.step_dt
        due = t >= self._event_next_push_at
        if not torch.any(due):
            return
        env_ids = torch.where(due)[0]
        mdp.push_by_setting_velocity(self, env_ids, vel_range, SceneEntityCfg("robot"))
        lo, hi = getattr(self.cfg, "event_push_interval_range_s", (2.0, 4.0))
        interval = sample_uniform(float(lo), float(hi), (env_ids.shape[0],), self.device)
        self._event_next_push_at[env_ids] = t[env_ids] + interval

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """与 DirectRLEnv.step 一致，但在 interval 处改为课程化推力（cfg.events 为 None 时使用）。"""
        action = action.to(self.device)
        if self.cfg.action_noise_model:
            action = self._action_noise_model.apply(action)

        self._pre_physics_step(action)

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)
        elif getattr(self.cfg, "event_curriculum_enabled", False):
            self._apply_event_curriculum_push_interval()

        self.obs_buf = self._get_observations()

        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _get_feet_body_ids(self) -> list:
        """返回左右脚踝 body 索引（Articulation 内顺序，用于位姿与 ContactSensor 脚索引对齐）。"""
        # 【核心修复】：必须精确定位到 roll_link（脚底板），绝不能用 .*ankle.*，否则会抓到不触地的 pitch_link！
        left_ankle_ids, _ = self.robot.find_bodies(".*left_ankle_roll_link")
        right_ankle_ids, _ = self.robot.find_bodies(".*right_ankle_roll_link")

        feet_body_ids = []
        if len(left_ankle_ids) > 0:
            feet_body_ids.append(left_ankle_ids[0])
        if len(right_ankle_ids) > 0:
            feet_body_ids.append(right_ankle_ids[0])
        if len(feet_body_ids) > 2:
            feet_body_ids = feet_body_ids[:2]

        return feet_body_ids

    def _resolve_foot_contact_sensor_body_indices(self) -> None:
        """将 Articulation 左右踝 body 名映射到 ContactSensor 的 body 维索引（仅解析一次）。"""
        if self._foot_contact_body_sensor_indices is not None:
            return
        sensor = self._foot_contact_sensor
        if sensor is None or not sensor.is_initialized:
            return
        feet_body_ids = self._get_feet_body_ids()
        if len(feet_body_ids) < 2:
            return
        names = [self.robot.body_names[i] for i in feet_body_ids[:2]]
        idxs: list[int] = []
        for n in names:
            bids, _ = sensor.find_bodies("^" + re.escape(n) + "$")
            if len(bids) != 1:
                return
            idxs.append(int(bids[0]))
        self._foot_contact_body_sensor_indices = torch.tensor(idxs, device=self.device, dtype=torch.long)

    def get_feet_contact_state(self) -> torch.Tensor | None:
        """足部触地 (num_envs, 2) bool，左/右脚。

        默认（``foot_contact_use_physics_sensor``）：与 H1 速度任务一致，用 ``ContactSensor`` 的
        ``net_forces_w`` 范数与 ``foot_contact_force_threshold``（写入传感器的 ``force_threshold``）判定。

        否则：脚踝高度 + 线速度 + 连续帧启发式。每步只更新一次并写 ``_feet_contact_cache``。
        """
        feet_body_ids = self._get_feet_body_ids()
        if len(feet_body_ids) < 2:
            return None
        if self._feet_contact_cache is not None:
            return self._feet_contact_cache

        use_phys = getattr(self.cfg, "foot_contact_use_physics_sensor", True)
        if (
            use_phys
            and self._foot_contact_sensor is not None
            and self._foot_contact_sensor.is_initialized
        ):
            self._resolve_foot_contact_sensor_body_indices()
            if self._foot_contact_body_sensor_indices is not None:
                idx = self._foot_contact_body_sensor_indices
                forces = self._foot_contact_sensor.data.net_forces_w[:, idx, :]
                fn = torch.norm(forces, dim=-1)
                th = float(getattr(self.cfg, "foot_contact_force_threshold", 1.0))
                self._feet_contact_cache = fn > th
                return self._feet_contact_cache

        if self._ankle_contact_consecutive_count is None:
            return None
        feet_body_ids = feet_body_ids[:2]
        body_pos_w = self.robot.data.body_pos_w
        body_lin_vel_w = self.robot.data.body_lin_vel_w
        feet_z = body_pos_w[:, feet_body_ids, 2]
        feet_vel = body_lin_vel_w[:, feet_body_ids, :]
        vel_norm = torch.norm(feet_vel, dim=-1)
        h_th = float(getattr(self.cfg, "foot_contact_height_threshold", 0.075))
        v_th = float(getattr(self.cfg, "foot_contact_velocity_threshold", 0.08))
        # 相对本环境地面原点，避免 env_origins.z 抬高时绝对世界高度永远 > h_th 导致「永不触地」→ contact_no_vel 恒 0
        orig_z = self.scene.env_origins[:, 2:3]
        height_ag = feet_z - orig_z
        low_height = height_ag < h_th
        small_vel = vel_norm < v_th
        candidate = low_height & small_vel
        n_frames = int(getattr(self.cfg, "foot_contact_consecutive_frames", 2))
        count = self._ankle_contact_consecutive_count
        count = torch.where(candidate, torch.clamp(count + 1, max=n_frames), torch.zeros_like(count))
        self._ankle_contact_consecutive_count = count
        contact = count >= n_frames
        self._feet_contact_cache = contact
        return contact

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=self.cfg.ground_plane_cfg)

        self.scene.clone_environments(copy_from_source=False)

        self.scene.articulations["robot"] = self.robot

        if getattr(self.cfg, "foot_contact_use_physics_sensor", True):
            fth = float(getattr(self.cfg, "foot_contact_force_threshold", 1.0))
            ccfg = ContactSensorCfg(
                prim_path=f"{self.scene.env_regex_ns}/Robot/.*",
                history_length=0,
                track_air_time=False,
                force_threshold=fth,
            )
            self._foot_contact_sensor = ContactSensor(ccfg)
            self.scene._sensors["foot_contact"] = self._foot_contact_sensor
            self._foot_contact_body_sensor_indices = None

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        num_envs = self.scene.num_envs
        self._foot_target_positions = torch.zeros((num_envs, 3), device=self.device)
        self._swing_air_accum_s = torch.zeros(num_envs, device=self.device)

        self._target_generation_time = torch.full((num_envs,), float('-inf'), device=self.device)
        self._target_hit = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._target_foot_indices = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._last_touchdown_time = torch.full((num_envs,), float('-inf'), device=self.device)
        self._user_target_mode = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._next_is_follow = torch.zeros(num_envs, dtype=torch.bool, device=self.device)  # True=下一脚为跟随点（24cm），False=下一脚为目标点（随机）
        self._target_regenerate_deadline = torch.full((num_envs,), float("nan"), device=self.device)

        self._swing_foot_lifted = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._swing_foot_contact_prev = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self._foot_land_rewarded = torch.zeros(num_envs, dtype=torch.bool, device=self.device)  # 落地奖励是否已发放（每目标一次，用于 fallback）
        # 触地判定：每脚连续满足“脚踝高度<7.5cm且速度小”的帧数 (num_envs, 2)
        self._ankle_contact_consecutive_count = torch.zeros((num_envs, 2), dtype=torch.long, device=self.device)
        self._feet_contact_cache = None

        target_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/foot_target_markers",
            markers={
                "target": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                ),
            },
        )
        self._target_markers = VisualizationMarkers(target_marker_cfg)
        self._target_markers.set_visibility(True)

        self._reward_components = [
            "rew_pitch_roll_angle",
            "rew_pitch_roll_ang_vel",
            "rew_lin_vel_z",
            "rew_height",
            "rew_joint_velocity",
            "rew_joint_acceleration",
            "rew_foot_hit",
            "rew_feet_air_time",
            "rew_foot_clearance",
            "rew_distance_attraction",
            "rew_swing_knee",
            "rew_joint_limit",
            "rew_action_rate",
            "rew_contact_no_vel",
            "rew_hip_pos",
        ]
        self._episode_reward_sums = {name: torch.zeros(num_envs, dtype=torch.float32, device=self.device) 
                                      for name in self._reward_components}

        self._last_episode_reward_means = {name: 0.0 for name in self._reward_components}

    def _pre_physics_step(self, actions: torch.Tensor) -> None:

        if hasattr(self, 'actions') and self.actions is not None:
            if self._prev_actions is not None:
                self._prev_actions = self.actions.clone()
            else:
                self._prev_actions = self.actions.clone()
        else:
            if self._prev_actions is None:
                self._prev_actions = actions.clone()
        
        self.actions = actions.clone()
        if getattr(self.cfg, "force_zero_action", False):
            self.actions.zero_()

        self._check_and_generate_targets()

        # 持续更新标记位置（确保标记始终显示最新位置）；踩中后的刷新在 _get_rewards 末尾完成
        self._update_target_markers()

    def _update_target_markers(self) -> None:
        """更新目标点位可视化标记，使小球正确反映当前点位世界坐标位置。
        
        每步先 _check_and_generate_targets 再本函数，故刷新后同一步内即更新。
        固定传入 num_envs 个位置（float32），保证 PointInstancer 实例数稳定、每次都能刷新；
        无目标的环境用 (0,0,-100) 置于地下；每次同时传 marker_indices 以强制视图更新。
        """
        if self._target_markers is None or self._foot_target_positions is None:
            return
        num_envs = self._foot_target_positions.shape[0]
        # 用“已生成过目标”判断，比用位置!=0 更稳健（避免目标恰好在原点被误隐藏）
        has_valid = (self._target_generation_time is not None) & (
            self._target_generation_time > float("-inf")
        )
        # 固定数量：每 env 一个槽位，有目标用当前目标位置，无目标用地下位置
        marker_positions = self._foot_target_positions.clone()
        marker_positions[~has_valid] = torch.tensor(
            (0.0, 0.0, -100.0), device=self.device, dtype=marker_positions.dtype
        )
        trans_np = marker_positions.detach().cpu().float().numpy()
        # 每次传 translations + marker_indices，确保 PointInstancer 视图刷新
        self._target_markers.visualize(
            translations=trans_np,
            marker_indices=np.zeros(num_envs, dtype=np.int32),
        )

    def _initialize_locked_joints(self):
        if self._locked_joint_initialized:
            return
        self._locked_joint_ids = collect_locked_joint_ids(self.robot)
        self._locked_joint_initialized = True

    def _initialize_joint_limits(self):
        if self._joint_limit_initialized:
            return

        self._joint_limit_map = {}
        soft = float(getattr(self.cfg, "soft_dof_pos_limit", 0.9))
        torso_half = float(getattr(self.cfg, "torso_joint_limit_rad", 0.35))
        joint_names = self.robot.joint_names
        for joint_id, joint_name in enumerate(joint_names):
            lim = reward_joint_limit_interval(
                joint_name,
                soft_dof_pos_limit=soft,
                torso_half_rad=torso_half,
            )
            if lim is not None:
                self._joint_limit_map[joint_id] = list(lim)

        self._joint_limit_initialized = True

    def _initialize_hip_pos_penalty_joints(self) -> None:
        """与宇树 ``g1_env._reward_hip_pos`` 一致：惩罚 Σ q²（髋 roll、髋 yaw，不含 pitch）。"""
        if self._hip_pos_joint_initialized:
            return
        ids: list[int] = []
        for joint_id, joint_name in enumerate(self.robot.joint_names):
            ln = joint_name.lower()
            if re.search(r"hip_roll", ln) or re.search(r"hip_yaw", ln):
                ids.append(joint_id)
        if ids:
            self._hip_pos_joint_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
        else:
            self._hip_pos_joint_ids = None
        self._hip_pos_joint_initialized = True

    def _apply_action(self) -> None:
        """应用动作到机器人
        
        动作为相对参考姿态的增量：target = reference_pose + action * action_scale。
        参考姿态与 reset 一致（default + 双膝微屈），避免第一步就命令到 [−0.25,0.25] 绝对位姿导致突然伸膝/飞起。
        锁死关节（脚踝、肩部等）保持默认位置。
        """
        self._initialize_locked_joints()

        default_pos = self.robot.data.default_joint_pos.clone()
        # 动作为增量：target = default + action * scale
        joint_pos_target = default_pos + self.actions * self.cfg.action_scale
        
        # 锁死指定关节（实现见 ``placing_joint_lock``，与 MARL 共用）
        if self._locked_joint_ids and len(self._locked_joint_ids) > 0:
            apply_locked_joint_targets(self.robot, joint_pos_target, default_pos, self._locked_joint_ids)
        
        # 关节目标限幅，避免超出关节限位导致大扭矩“弹飞”
        if self._joint_limit_map:
            for joint_id, (low, high) in self._joint_limit_map.items():
                joint_pos_target[:, joint_id] = torch.clamp(
                    joint_pos_target[:, joint_id], low, high
                )

        # 关节目标变化率限制：每步最大变化，抑制高频运动
        rate_limit = getattr(self.cfg, "action_rate_limit_rad_per_step", 0.0)
        if rate_limit > 0 and self._prev_joint_pos_target is not None:
            delta = joint_pos_target - self._prev_joint_pos_target
            delta = torch.clamp(delta, -rate_limit, rate_limit)
            joint_pos_target = self._prev_joint_pos_target + delta
        self._prev_joint_pos_target = joint_pos_target.clone()

        self.robot.set_joint_position_target(joint_pos_target, joint_ids=None)

    def _get_observations(self) -> dict:
        """获取观察值（106维）
        
        观察空间组成：
        - root状态：位置(3) + 朝向(4) + 线速度(3) + 角速度(3) = 13维
        - 关节状态：位置(37) + 速度(37) = 74维
        - 目标点：相对根的位置(3)；无有效目标（``_foot_target_positions`` 全 0）时为当前摆动脚脚踝在根系下位置
        - 脚部状态：接触(2) + 位置(6) + 朝向(8) = 16维
        总计：106维
        """
        self._feet_contact_cache = None  # 每步只在本步首次调用 get_feet_contact_state 时更新
        # 机器人根状态
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        root_lin_vel = self.robot.data.root_lin_vel_w
        root_ang_vel = self.robot.data.root_ang_vel_w

        # 获取脚部 body 索引（左右脚踝，无接触传感器时接触状态为零）
        feet_body_ids = self._get_feet_body_ids()
        num_envs = root_pos.shape[0]
        num_feet = min(len(feet_body_ids), 2) if len(feet_body_ids) > 0 else 2
        if len(feet_body_ids) > 2:
            feet_body_ids = feet_body_ids[:2]

        feet_contact_state = torch.zeros((num_envs, num_feet), device=self.device)
        feet_pos_local = torch.zeros((num_envs, num_feet, 3), device=self.device)
        feet_quat_local = torch.zeros((num_envs, num_feet, 4), device=self.device)

        if len(feet_body_ids) > 0:
            feet_contact = self.get_feet_contact_state()
            if feet_contact is not None:
                feet_contact_state = feet_contact.float()

            body_pos_w = self.robot.data.body_pos_w
            body_quat_w = self.robot.data.body_quat_w
            feet_pos_w = body_pos_w[:, feet_body_ids, :]
            feet_quat_w = body_quat_w[:, feet_body_ids, :]

            feet_offset_w = feet_pos_w - root_pos.unsqueeze(1)
            root_quat_expanded = root_quat.unsqueeze(1).expand(-1, num_feet, -1)
            feet_pos_local = quat_rotate_inverse(root_quat_expanded, feet_offset_w)

            root_quat_inv = quat_inv(root_quat).unsqueeze(1).expand(-1, num_feet, -1)
            feet_quat_local = quat_mul(root_quat_inv, feet_quat_w)

        # 目标点相对根（body 系）：无有效目标（全 0）时默认 = 当前摆动脚脚踝在根系下的位置，避免退化为「指向世界原点」
        if self._foot_target_positions is not None:
            target_offset_w = self._foot_target_positions - root_pos
            target_offset_local = quat_rotate_inverse(root_quat, target_offset_w)
            has_target = torch.any(self._foot_target_positions != 0, dim=1)
            if len(feet_body_ids) > 0 and num_feet >= 1 and self._target_foot_indices is not None:
                row = torch.arange(num_envs, device=self.device)
                ti = self._target_foot_indices.clamp(min=0, max=num_feet - 1)
                swing_local = feet_pos_local[row, ti]
                target_offset_local = torch.where(has_target.unsqueeze(1), target_offset_local, swing_local)
        else:
            target_offset_local = torch.zeros((num_envs, 3), device=self.device)

        feet_contact_flat = feet_contact_state
        feet_pos_flat = feet_pos_local.reshape(num_envs, -1)
        feet_quat_flat = feet_quat_local.reshape(num_envs, -1)

        obs = torch.cat(
            (
                root_pos,
                root_quat,
                root_lin_vel,
                root_ang_vel,
                self.joint_pos,
                self.joint_vel,
                target_offset_local,
                feet_contact_flat,
                feet_pos_flat,
                feet_quat_flat,
            ),
            dim=-1,
        )

        # policy 观测开关：False 时暂时关闭 policy 观测（检修用），返回零向量
        if getattr(self.cfg, "policy_observation_enabled", True):
            observations = {"policy": obs}
        else:
            observations = {"policy": torch.zeros_like(obs)}
        return observations

    def collect_reference_motions(self, num_samples: int) -> torch.Tensor:

        obs_dict = self._get_observations()
        current_obs = obs_dict["policy"]

        num_envs = current_obs.shape[0]
        if num_envs >= num_samples:

            indices = torch.randperm(num_envs, device=self.device)[:num_samples]
            reference_states = current_obs[indices]
        else:

            num_repeats = (num_samples + num_envs - 1) // num_envs
            reference_states = current_obs.repeat(num_repeats, 1)[:num_samples]

            noise_scale = 0.01
            noise = torch.randn_like(reference_states) * noise_scale
            reference_states = reference_states + noise

        return reference_states

    def _generate_follow_target(self, env_ids: torch.Tensor, hit_foot_indices: torch.Tensor) -> None:
        """生成跟随点：另一脚跟上，保持双腿 24cm 间距。
        核心修复：使用 Root 横向（body Y 轴）进行平移，确保双脚并排站立，而非前后错开。
        """
        if len(env_ids) == 0:
            return

        feet_body_ids = self._get_feet_body_ids()
        if len(feet_body_ids) < 2:
            return
        feet_body_ids = feet_body_ids[:2]

        body_pos_w = self.robot.data.body_pos_w
        feet_pos_w = body_pos_w[env_ids][:, feet_body_ids, :]

        # 获取刚刚踩中目标的那只脚的 XY 坐标 (即新的支撑脚)
        swing_xy = torch.where(
            hit_foot_indices.unsqueeze(1).expand(-1, 2) == 0,
            feet_pos_w[:, 0, :2],
            feet_pos_w[:, 1, :2],
        )

        # ---------------- 核心修改部分 ----------------
        # 提取当前机器人的朝向 (Root Quat)
        root_quat = self.robot.data.root_quat_w[env_ids]

        # 构建一个局部 Y 轴向量 (0, 1, 0) 表示机器人的正左方
        local_y = torch.zeros(len(env_ids), 3, device=self.device, dtype=root_quat.dtype)
        local_y[:, 1] = 1.0

        # 将局部 Y 轴旋转到世界坐标系下
        world_y = quat_apply_yaw(root_quat, local_y)[:, :2]

        # 判断方向：
        # 如果是左脚踩中(hit_foot_indices == 0)，那么右脚需要跟随，目标点应该在左脚的负 Y 方向（右侧），乘数为 -1.0
        # 如果是右脚踩中(hit_foot_indices == 1)，那么左脚需要跟随，目标点应该在右脚的正 Y 方向（左侧），乘数为 1.0
        direction_multiplier = torch.where(hit_foot_indices == 0, -1.0, 1.0).unsqueeze(1)

        spacing_m = getattr(self.cfg, "foot_spacing_stand_m", 0.24)

        # 新的跟随点：踩中脚的位置 + 垂直于身体朝向的 24cm 偏移
        target_xy = swing_xy + spacing_m * direction_multiplier * world_y
        # ----------------------------------------------

        target_positions = torch.zeros(len(env_ids), 3, device=self.device, dtype=feet_pos_w.dtype)
        target_positions[:, :2] = target_xy
        target_positions[:, 2] = self.scene.env_origins[env_ids, 2]

        # 更新摆动脚（1 - 当前踩中脚 = 另一只脚）
        swing_foot_indices = 1 - hit_foot_indices
        self._foot_target_positions[env_ids] = target_positions
        self._target_foot_indices[env_ids] = swing_foot_indices

    def _generate_random_target(self, env_ids: torch.Tensor, *, alternate_swing: bool = False) -> None:
        """点位生成器：随机目标为脚踝周围 body 系矩形（前/后/横向见 cfg），z 为各 env 地面标高；与跟随点（24cm）交替。
        序列：左脚踩目标 → 右脚跟（24cm）→ 右脚踩目标 → 左脚跟（24cm）→ ...
        """
        if len(env_ids) == 0:
            return
        if self._target_regenerate_deadline is not None:
            self._target_regenerate_deadline[env_ids] = float("nan")
        num_generate = len(env_ids)
        root_quat = self.robot.data.root_quat_w[env_ids]
        feet_body_ids = self._get_feet_body_ids()
        if len(feet_body_ids) > 2:
            feet_body_ids = feet_body_ids[:2]
        if len(feet_body_ids) < 2:
            return
        body_pos_w = self.robot.data.body_pos_w
        feet_pos_w = body_pos_w[env_ids][:, feet_body_ids, :]
        left_foot_xy = feet_pos_w[:, 0, :2]
        right_foot_xy = feet_pos_w[:, 1, :2]

        if alternate_swing:
            hit_foot = self._target_foot_indices[env_ids]
            use_follow = self._next_is_follow[env_ids]
            self._next_is_follow[env_ids] = ~use_follow
            env_ids_follow = env_ids[use_follow]
            env_ids_random = env_ids[~use_follow]
            if len(env_ids_follow) > 0:
                # 踩中随机目标 → 另一脚跟（24cm）
                self._generate_follow_target(env_ids_follow, hit_foot[use_follow])
            if len(env_ids_random) > 0:
                # 踩中跟随点 → 同一脚踩新随机目标
                swing_r = hit_foot[~use_follow]
                self._generate_random_target_internal(env_ids_random, swing_r, root_quat[~use_follow], left_foot_xy[~use_follow], right_foot_xy[~use_follow])
        else:
            swing_foot_indices = torch.randint(0, 2, (num_generate,), device=self.device)
            self._next_is_follow[env_ids] = True
            self._generate_random_target_internal(env_ids, swing_foot_indices, root_quat, left_foot_xy, right_foot_xy)

        current_time = self.episode_length_buf[env_ids].float() * self.step_dt
        self._target_generation_time[env_ids] = current_time
        self._target_hit[env_ids] = False
        if self._last_touchdown_time is not None:
            self._last_touchdown_time[env_ids] = float('-inf')
        self._swing_foot_lifted[env_ids] = False
        if self._swing_foot_contact_prev is not None:
            self._swing_foot_contact_prev[env_ids] = True
        if self._foot_land_rewarded is not None:
            self._foot_land_rewarded[env_ids] = False
        if self._swing_air_accum_s is not None:
            self._swing_air_accum_s[env_ids] = 0.0
        self._feet_air_time[env_ids] = 0.0

    def _generate_random_target_internal(
        self, env_ids: torch.Tensor, swing_foot_indices: torch.Tensor,
        root_quat: torch.Tensor, left_foot_xy: torch.Tensor, right_foot_xy: torch.Tensor
    ) -> None:
        """内部：在摆动脚踝处 body 系矩形内均匀采样地面目标点（z = 该 env 地面世界坐标，与 env_origins.z 一致）。"""
        num_generate = len(env_ids)
        center_xy = torch.where(
            swing_foot_indices.unsqueeze(1).expand(-1, 2) == 0,
            left_foot_xy,
            right_foot_xy,
        )

        target_z = self.scene.env_origins[env_ids, 2]
        # =====================================================================
        # 获取配置参数
        # =====================================================================
        x_back = getattr(self.cfg, "foot_target_rect_x_back", 0.08)
        x_forward = getattr(self.cfg, "foot_target_rect_x_forward", 0.30)
        y_outward = getattr(self.cfg, "foot_target_rect_y_outward", 0.15)
        y_inward = getattr(self.cfg, "foot_target_rect_y_inward", 0.02)
        is_idle = torch.rand(num_generate, device=self.device) < 0.10
        # =====================================================================
        # 1. 独立生成 X 和 Y 的范围
        # =====================================================================
        local_x = sample_uniform(-x_back, x_forward, (num_generate,), self.device)

        # 核心拦截：判断当前是哪只脚在摆动 (0是左脚，1是右脚)
        # 假设局部坐标系：+Y 是向左。
        # 对于左脚：+Y 是外(outward), -Y 是内(inward) -> 范围 [-inward, outward]
        # 对于右脚：-Y 是外(outward), +Y 是内(inward) -> 范围 [-outward, inward]
        is_left_foot = swing_foot_indices == 0
        y_min = torch.where(
            is_left_foot,
            torch.full((num_generate,), -y_inward, device=self.device, dtype=center_xy.dtype),
            torch.full((num_generate,), -y_outward, device=self.device, dtype=center_xy.dtype),
        )
        y_max = torch.where(
            is_left_foot,
            torch.full((num_generate,), y_outward, device=self.device, dtype=center_xy.dtype),
            torch.full((num_generate,), y_inward, device=self.device, dtype=center_xy.dtype),
        )

        # 使用基础的 rand 生成 0-1 的随机数，再映射到 [y_min, y_max] 区间
        rand_y = torch.rand(num_generate, device=self.device, dtype=center_xy.dtype)
        local_y = y_min + rand_y * (y_max - y_min)
        # =====================================================================
        # 2. 镂空盲区外推 (逻辑保持不变)
        # =====================================================================
        min_dist = getattr(self.cfg, "foot_target_min_distance", 0.10)
        dist = torch.sqrt(local_x**2 + local_y**2).clamp(min=1e-6)
        too_close = dist < min_dist
        safe_r = sample_uniform(min_dist, x_forward, (num_generate,), self.device)
        scale = safe_r / dist

        local_x = torch.where(too_close, local_x * scale, local_x)
        local_y = torch.where(too_close, local_y * scale, local_y)

        # 重新裁剪边界，防止外推时越界 (这里的裁剪也要用刚刚算好的非对称边界)
        local_x = torch.clamp(local_x, -x_back, x_forward)
        local_y = torch.max(torch.min(local_y, y_max), y_min)
        # 待机模式兜底
        local_x = torch.where(is_idle, torch.zeros_like(local_x), local_x)
        local_y = torch.where(is_idle, torch.zeros_like(local_y), local_y)
        local_positions = torch.stack(
            [local_x, local_y, torch.zeros(num_generate, device=self.device, dtype=center_xy.dtype)],
            dim=1,
        )
        # 转换到世界坐标系
        world_offset = quat_apply_yaw(root_quat, local_positions)
        target_positions = torch.zeros(num_generate, 3, device=self.device, dtype=center_xy.dtype)
        target_positions[:, :2] = center_xy + world_offset[:, :2]
        target_positions[:, 2] = target_z

        # =====================================================================
        # 保存状态
        # =====================================================================
        self._foot_target_positions[env_ids] = target_positions
        self._target_foot_indices[env_ids] = swing_foot_indices

    def _check_and_generate_targets(self) -> None:
        """点位生成器入口：首次生成、踩中延迟到期后或踩中立即（delay=0）刷新。用户模式下跳过自动生成。"""
        if self._foot_target_positions is None:
            return
        skip_auto = self._user_target_mode if self._user_target_mode is not None else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        t_now = self.episode_length_buf.float() * self.step_dt
        if self._target_regenerate_deadline is not None:
            delay_ready = (
                torch.isfinite(self._target_regenerate_deadline)
                & (t_now >= self._target_regenerate_deadline)
                & ~skip_auto
            )
            env_ids_delay = torch.where(delay_ready)[0]
            if len(env_ids_delay) > 0:
                self._target_regenerate_deadline[env_ids_delay] = float("nan")
                self._generate_random_target(env_ids_delay, alternate_swing=True)

        not_generated = torch.isinf(self._target_generation_time) & (self._target_generation_time < 0) & ~skip_auto
        env_ids_init = torch.where(not_generated)[0]
        if len(env_ids_init) > 0:
            self._generate_random_target(env_ids_init)
        if self._target_hit is not None:
            env_ids_hit = torch.where(self._target_hit & ~skip_auto)[0]
            if len(env_ids_hit) > 0:
                self._generate_random_target(env_ids_hit, alternate_swing=True)
            env_ids_user_hit = torch.where(self._target_hit & skip_auto)[0]
            if len(env_ids_user_hit) > 0:
                self._clear_user_target(env_ids_user_hit)

    def _clear_user_target(self, env_ids: torch.Tensor) -> None:
        """清除用户目标（踩中后调用）。保持用户模式，等待用户再次按 T 设置新目标。"""
        if len(env_ids) == 0:
            return
        if self._target_regenerate_deadline is not None:
            self._target_regenerate_deadline[env_ids] = float("nan")
        self._foot_target_positions[env_ids] = 0.0
        self._target_generation_time[env_ids] = float('-inf')
        self._target_hit[env_ids] = False
        if self._swing_air_accum_s is not None:
            self._swing_air_accum_s[env_ids] = 0.0
        self._feet_air_time[env_ids] = 0.0
        # 保持 _user_target_mode=True，继续跳过自动生成，等待用户按 T

    def set_user_target(
        self,
        env_id: int,
        position_world: tuple[float, float, float] | list[float],
        swing_foot_index: int | None = None,
    ) -> bool:
        """设置用户指定的踩点目标（世界坐标）。用于交互式 play 模式。
        
        Args:
            env_id: 环境索引（通常为 0）
            position_world: 目标点世界坐标 (x, y, z)；z 会被覆盖为该 env 地面标高（与 env_origins.z 一致）
            swing_foot_index: 摆动脚索引 0=左脚 1=右脚，None 时选离目标更近的脚
        
        Returns:
            是否设置成功
        """
        if self._foot_target_positions is None or env_id < 0 or env_id >= self.num_envs:
            return False
        env_ids = torch.tensor([env_id], device=self.device, dtype=torch.long)
        pos = torch.tensor([position_world], device=self.device, dtype=torch.float32)
        pos[0, 2] = self.scene.env_origins[env_id, 2]
        feet_body_ids = self._get_feet_body_ids()
        if len(feet_body_ids) < 2:
            return False
        feet_body_ids = feet_body_ids[:2]
        feet_pos_w = self.robot.data.body_pos_w[env_ids][:, feet_body_ids, :]
        if swing_foot_index is None:
            dist_left = torch.norm(feet_pos_w[0, 0, :] - pos[0], dim=0).item()
            dist_right = torch.norm(feet_pos_w[0, 1, :] - pos[0], dim=0).item()
            swing_foot_index = 0 if dist_left <= dist_right else 1
        swing_foot_index = max(0, min(1, swing_foot_index))
        self._foot_target_positions[env_ids] = pos
        self._target_foot_indices[env_ids] = swing_foot_index
        self._target_generation_time[env_ids] = (
            self.episode_length_buf[env_ids].float() * self.step_dt
        )
        self._target_hit[env_ids] = False
        self._user_target_mode[env_ids] = True
        if self._target_regenerate_deadline is not None:
            self._target_regenerate_deadline[env_ids] = float("nan")
        if self._swing_air_accum_s is not None:
            self._swing_air_accum_s[env_ids] = 0.0
        self._feet_air_time[env_ids] = 0.0
        self._next_is_follow[env_ids] = False
        self._update_target_markers()
        return True

    def _reward_curriculum_components(
        self, step: int, curriculum: list, mode: str = "step", steps_per_iter: int = 24
    ) -> list[str] | None:
        """解析 reward_curriculum，返回当前阶段启用的奖励组件名列表；无匹配则 None（表示全部启用）。
        mode: 'step' 用 step_start/step_end；'iteration' 用 iter_start/iter_end，且 step→iter = step // steps_per_iter。"""
        if mode == "iteration":
            tick = step // max(1, steps_per_iter)
            start_key, end_key = "iter_start", "iter_end"
        else:
            tick = step
            start_key, end_key = "step_start", "step_end"
        phase = None
        for p in curriculum:
            if not isinstance(p, dict):
                continue
            start = p.get(start_key, 0)
            end = p.get(end_key)
            if tick < start:
                continue
            if end is not None and tick >= end:
                continue
            phase = p
        if phase is None:
            return None
        comp = phase.get("components")
        return comp if isinstance(comp, (list, tuple)) else None

    def _expand_deprecated_reward_curriculum_components(
        self, names: list[str] | tuple[str, ...]
    ) -> list[str]:
        """将课程里已废弃的组件名映射为当前实现键，保证 ``comp`` 合分与 TensorBoard 日志一致。"""
        legacy = frozenset({"rew_foot_path_tracking"})
        gait = (
            "rew_feet_air_time",
            "rew_foot_clearance",
            "rew_distance_attraction",
        )
        seen: dict[str, None] = {}
        out: list[str] = []
        for n in names:
            if n == "rew_swing_velocity":
                n = "rew_distance_attraction"
            if n in legacy:
                for g in gait:
                    if g not in seen:
                        seen[g] = None
                        out.append(g)
            else:
                if n not in seen:
                    seen[n] = None
                    out.append(n)
        return out

    def _foot_hit_curriculum_radius_xy_m(self) -> float:
        """地面踩点：水平命中半径（米），随训练 iteration 收紧。"""
        step = getattr(self, "common_step_counter", 0)
        steps_per_iter = getattr(self.cfg, "reward_curriculum_steps_per_iteration", 24)
        iter_cur = step // max(1, steps_per_iter)
        iter_start = getattr(self.cfg, "foot_target_hit_threshold_iter_start", 1500)
        iter_end = getattr(self.cfg, "foot_target_hit_threshold_iter_end", 5000)
        thresh_start = float(getattr(self.cfg, "foot_target_hit_threshold", 0.08))
        thresh_end = float(getattr(self.cfg, "foot_target_hit_threshold_end", 0.04))
        if iter_cur < iter_start:
            return thresh_start
        if iter_cur >= iter_end:
            return thresh_end
        t = (iter_cur - iter_start) / max(1, iter_end - iter_start)
        return thresh_start + t * (thresh_end - thresh_start)

    def _get_rewards(self) -> torch.Tensor:
        """踩点、动态步态引导（滞空/离地/距离吸引）、稳定性/平滑正则（含宇树式 hip_pos）；可选奖励课程按 iteration 屏蔽部分项。"""
        # 提取机器人状态
        root_quat = self.robot.data.root_quat_w
        root_ang_vel_w = self.robot.data.root_ang_vel_w  # (ωx, ωy, ωz) 世界系
        roll_angle, pitch_angle, _ = euler_xyz_from_quat(root_quat)
        roll_angle = torch.atan2(torch.sin(roll_angle), torch.cos(roll_angle))  # 归一化到[-π, π]
        pitch_angle = torch.atan2(torch.sin(pitch_angle), torch.cos(pitch_angle))
        
        # 关节加速度（控制步内有限差分），与宇树 ``legged_robot._reward_dof_acc`` 同型：‖(q̇_t−q̇_{t−1})/dt‖² 再求和
        joint_vel = self.robot.data.joint_vel
        if self._prev_joint_vel is not None:
            dt = self.cfg.sim.dt * self.cfg.decimation
            joint_acc = (joint_vel - self._prev_joint_vel) / dt
            self._prev_joint_vel = joint_vel.clone()
        else:
            joint_acc = torch.zeros_like(joint_vel)
            self._prev_joint_vel = joint_vel.clone()
        
        joint_pos = self.robot.data.joint_pos
        num_envs = joint_pos.shape[0]
        # 先为「未生成点位」的环境（含刚 reset 的）生成目标，避免本步用旧目标/旧摆动脚做判定
        self._check_and_generate_targets()
        rew_feet_air_time = torch.zeros(num_envs, device=self.device)
        rew_foot_clearance = torch.zeros(num_envs, device=self.device)
        rew_distance_attraction = torch.zeros(num_envs, device=self.device)
        rew_swing_knee = torch.zeros(num_envs, device=self.device)

        env_indices = torch.arange(num_envs, device=self.device)
        # 本步落地事件（prev 未触地 & curr 触地）；在抬脚块内用 prev 计算后复用至落地块，避免 prev 被覆盖后误用
        touchdown_event = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        
        # 计算 has_target（每个环境是否有目标点）
        has_target = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        if self._foot_target_positions is not None:
            has_target = torch.any(self._foot_target_positions != 0, dim=1)
        
        # 检查是否有任何环境有目标（用于全局检查）
        has_target_valid = torch.any(has_target).item() if isinstance(has_target, torch.Tensor) else False

        feet_body_ids = self._get_feet_body_ids()
        if len(feet_body_ids) > 2:
            feet_body_ids = feet_body_ids[:2]

        feet_pos_w = None
        swing_foot_contact = None
        hit_target = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        if len(feet_body_ids) >= 2:
            body_pos_w = self.robot.data.body_pos_w
            feet_pos_w = body_pos_w[:, feet_body_ids, :]

        if len(feet_body_ids) >= 2:
            feet_contact = self.get_feet_contact_state()
            if feet_contact is not None:
                swing_foot_contact = feet_contact[env_indices, self._target_foot_indices]
                # 须在踩中刷新目标之前累计滞空；若晚于 _check_and_generate_targets，_feet_air_time 已被清零，滞空奖励≈0
                self._feet_air_time = self._feet_air_time + self.step_dt

            if self._target_foot_indices is not None and has_target_valid and swing_foot_contact is not None:
                self._swing_foot_lifted = self._swing_foot_lifted | (~swing_foot_contact)

                if self._swing_foot_contact_prev is not None:
                    touchdown_event = (~self._swing_foot_contact_prev) & swing_foot_contact
                    self._swing_foot_contact_prev = swing_foot_contact.clone()
                else:
                    touchdown_event = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

            # 地面踩点：摆动脚踝与目标水平对齐；竖直相对「目标地面 z + 踝高」，触地时允许小幅 slack
            foot_hit_reward = torch.zeros(num_envs, device=self.device)
            
            if (self._target_foot_indices is not None and
                self._foot_target_positions is not None and
                len(feet_body_ids) >= 2 and
                feet_pos_w is not None and
                has_target_valid):
                
                swing_foot_pos = feet_pos_w[env_indices, self._target_foot_indices, :]
                ankle_ground_h = float(getattr(self.cfg, "foot_ankle_ground_height", 0.07))
                target_xy = self._foot_target_positions[:, :2]
                target_z_ground = self._foot_target_positions[:, 2]
                expected_ankle_z = target_z_ground + ankle_ground_h
                d_xy = torch.norm(swing_foot_pos[:, :2] - target_xy, dim=1)
                swing_foot_z = swing_foot_pos[:, 2]
                d_z_err = torch.abs(swing_foot_z - expected_ankle_z)
                z_tolerance = float(getattr(self.cfg, "foot_target_hit_z_tolerance", 0.04))
                z_contact_slack = float(getattr(self.cfg, "foot_target_hit_z_contact_slack", 0.02))
                z_max = float(getattr(self.cfg, "foot_target_hit_z_max", 0.09))
                thresh_xy = self._foot_hit_curriculum_radius_xy_m()
                xy_ok = d_xy < thresh_xy
                z_ok_contact = swing_foot_contact & (
                    swing_foot_z <= (target_z_ground + ankle_ground_h + z_contact_slack)
                ) if swing_foot_contact is not None else torch.zeros(num_envs, dtype=torch.bool, device=self.device)
                z_ok = (d_z_err < z_tolerance) | z_ok_contact
                geometry_ok = xy_ok & z_ok

                target_elapsed = (self.episode_length_buf.float() * self.step_dt - self._target_generation_time).clamp(min=0.0)
                min_elapsed = getattr(self.cfg, "foot_target_hit_min_elapsed_s", 0.05)
                touchdown_with_target = (touchdown_event & has_target) if swing_foot_contact is not None else torch.zeros(num_envs, dtype=torch.bool, device=self.device)
                fallback_contact = (
                    has_target
                    & xy_ok
                    & swing_foot_contact
                    & self._swing_foot_lifted
                    & ~self._foot_land_rewarded
                ) if swing_foot_contact is not None else torch.zeros(num_envs, dtype=torch.bool, device=self.device)
                height_fallback = (
                    has_target
                    & xy_ok
                    & (swing_foot_z < (target_z_ground + z_max))
                    & ~self._foot_land_rewarded
                    & (target_elapsed >= min_elapsed)
                )
                landing_evidence = touchdown_with_target | fallback_contact | height_fallback

                hit_target = has_target & geometry_ok & landing_evidence
                current_time = self.episode_length_buf.float() * self.step_dt

                d_eff = torch.sqrt(
                    torch.square(d_xy / max(thresh_xy, 1e-6))
                    + torch.square(d_z_err / max(ankle_ground_h, 1e-6))
                ).clamp(min=1e-6)
                sigma_hit = getattr(self.cfg, "foot_hit_sigma", 0.03)
                sigma_eff = sigma_hit / max(thresh_xy, 1e-6)
                hit_reward_base = torch.exp(-0.5 * torch.square(d_eff / sigma_eff))
                reward_frame = landing_evidence & geometry_ok
                foot_hit_reward = torch.where(
                    reward_frame,
                    hit_reward_base,
                    torch.zeros_like(hit_reward_base)
                )
                target_air_time = self._feet_air_time[env_indices, self._target_foot_indices]
                target_time = float(getattr(self.cfg, "feet_air_time_target", 0.8))
                sigma_time = float(getattr(self.cfg, "feet_air_time_sigma", 0.15))

                # 高斯时间得分；仅在摆动脚触地瞬间兑现（不看踩点几何）
                time_score = torch.exp(-0.5 * torch.square((target_air_time - target_time) / sigma_time))
                rew_feet_air_time = torch.where(
                    touchdown_event,
                    time_score,
                    torch.zeros_like(time_score),
                )
                if self._foot_land_rewarded is not None:
                    self._foot_land_rewarded = self._foot_land_rewarded | landing_evidence
                delay_s = float(getattr(self.cfg, "foot_target_regenerate_delay_s", 0.5))
                skip_tm = (
                    self._user_target_mode
                    if self._user_target_mode is not None
                    else torch.zeros(num_envs, dtype=torch.bool, device=self.device)
                )
                user_hit = hit_target & skip_tm
                auto_hit = hit_target & ~skip_tm
                if delay_s > 0.0:
                    self._target_hit = self._target_hit | user_hit
                    if self._target_regenerate_deadline is not None:
                        sched = auto_hit & torch.isnan(self._target_regenerate_deadline)
                        self._target_regenerate_deadline = torch.where(
                            sched,
                            current_time + delay_s,
                            self._target_regenerate_deadline,
                        )
                else:
                    self._target_hit = self._target_hit | hit_target
                if self._last_touchdown_time is not None:
                    self._last_touchdown_time = torch.where(
                        landing_evidence,
                        current_time,
                        self._last_touchdown_time
                    )
            # =====================================================================
            # 新版动态步态引导：Clearance + 距离吸引（滞空累计见上；触地清零见下）
            # =====================================================================
            if len(feet_body_ids) >= 2 and feet_pos_w is not None:
                contact_mask = feet_contact if feet_contact is not None else self.get_feet_contact_state()
                if contact_mask is not None:
                    if (
                        has_target_valid
                        and self._target_foot_indices is not None
                        and self._foot_target_positions is not None
                        and swing_foot_contact is not None
                    ):
                        orig_z = self.scene.env_origins[:, 2]
                        target_clear_h = float(getattr(self.cfg, "foot_clearance_height", 0.09))

                        is_swinging = ~contact_mask[env_indices, self._target_foot_indices]
                        swing_foot_pos = feet_pos_w[env_indices, self._target_foot_indices, :]
                        swing_foot_z = swing_foot_pos[:, 2]
                        target_pos = self._foot_target_positions

                        body_lin_vel_w = self.robot.data.body_lin_vel_w
                        feet_vel = body_lin_vel_w[:, feet_body_ids[:2], :]
                        swing_foot_vel = feet_vel[env_indices, self._target_foot_indices, :]
                        vxy = swing_foot_vel[:, :2]
                        vxy_n = torch.norm(vxy, dim=1)

                        rel_h = swing_foot_z - orig_z
                        quad_clear = (has_target & is_swinging).float() * torch.square(rel_h - target_clear_h)
                        slide = (has_target & swing_foot_contact & (vxy_n > 0.02)).float() * torch.square(vxy_n)
                        rew_foot_clearance = quad_clear + slide

                        target_xy = target_pos[:, :2]
                        swing_xy = swing_foot_pos[:, :2]
                        dist_to_target = torch.norm(swing_xy - target_xy, dim=1)
                        rew_distance_attraction = torch.where(
                            has_target & is_swinging,
                            torch.exp(-0.5 * torch.square(dist_to_target / 0.15)),
                            torch.zeros_like(dist_to_target),
                        )
                        # 升级版：摆动相动态膝盖弯曲引导 (Phase-Based Sine Trajectory)
                        swing_knee_pos = torch.where(
                            self._target_foot_indices == 0,
                            self.joint_pos[:, self._left_knee_idx],
                            self.joint_pos[:, self._right_knee_idx],
                        )
                        target_time_cfg = float(getattr(self.cfg, "feet_air_time_target", 0.8))
                        phase = (target_air_time / max(target_time_cfg, 1e-6)).clamp(min=0.0, max=1.0)
                        max_knee_angle = 0.8
                        dynamic_target_knee = max_knee_angle * torch.sin(math.pi * phase)
                        knee_error = (dynamic_target_knee - swing_knee_pos).clamp(min=0.0)
                        rew_swing_knee = torch.where(
                            has_target & is_swinging,
                            torch.square(knee_error),
                            torch.zeros_like(knee_error),
                        )

                    self._feet_air_time = torch.where(
                        contact_mask,
                        torch.zeros_like(self._feet_air_time),
                        self._feet_air_time,
                    )

                if torch.any(hit_target):
                    self._check_and_generate_targets()
                    self._update_target_markers()

        self._initialize_joint_limits()

        joint_limit_violation = torch.zeros(num_envs, device=self.device)
        buffer_width = self.cfg.joint_limit_buffer

        if self._joint_limit_map:
            for joint_id, (min_limit, max_limit) in self._joint_limit_map.items():
                q = joint_pos[:, joint_id]

                lower_buffer_start = min_limit - buffer_width
                lower_penalty = torch.where(
                    q >= min_limit,
                    torch.zeros_like(q),
                    torch.where(
                        q >= lower_buffer_start,
                        min_limit - q,
                        min_limit - q
                    )
                )

                upper_buffer_end = max_limit + buffer_width
                upper_penalty = torch.where(
                    q <= max_limit,
                    torch.zeros_like(q),
                    torch.where(
                        q <= upper_buffer_end,
                        q - max_limit,
                        q - max_limit
                    )
                )

                joint_limit_violation += lower_penalty + upper_penalty

        root_height = self.robot.data.root_pos_w[:, 2]
        height_target = self.cfg.height_target
        height_band = self.cfg.height_penalty_band
        height_violation = (torch.abs(root_height - height_target) > height_band).float()

        # pitch/roll 角惩罚不再随摆动脚离地加重（已去掉空中/抬脚悬停相关姿态加罚）
        pitch_roll_swing_multiplier = torch.ones(num_envs, device=self.device, dtype=torch.float32)

        # 俯仰/横滚角速度惩罚：scale * (ωx² + ωy²)，ωx≈roll_rate, ωy≈pitch_rate
        scale_ang_vel = getattr(self.cfg, "rew_scale_pitch_roll_ang_vel", -1e-3)
        pitch_roll_ang_vel_sq = torch.square(root_ang_vel_w[:, 0]) + torch.square(root_ang_vel_w[:, 1])
        rew_pitch_roll_ang_vel = scale_ang_vel * pitch_roll_ang_vel_sq

        # ------------------------------------------------------------------
        # Transplanted from Unitree Baseline: Posture & Smoothness Regularization
        # ------------------------------------------------------------------

        # 1. Action Rate Penalty (Anti-Jitter/Parkinson's)
        rew_action_rate = torch.zeros(num_envs, device=self.device)
        if self._prev_actions is not None and hasattr(self, "actions") and self.actions is not None:
            action_diff = self.actions - self._prev_actions
            scale_action_rate = getattr(self.cfg, "rew_scale_action_rate", -0.0001)
            rew_action_rate = scale_action_rate * torch.sum(torch.square(action_diff), dim=1)

        # 2. Contact No Velocity Penalty (Anti-Slip/Ice-skating)
        rew_contact_no_vel = torch.zeros(num_envs, device=self.device)
        if len(feet_body_ids) >= 2:
            fc = self.get_feet_contact_state()
            if fc is not None:
                body_lin_vel_w = self.robot.data.body_lin_vel_w
                feet_vel_xy = body_lin_vel_w[:, feet_body_ids[:2], :2]
                contact_expanded = fc.unsqueeze(-1)
                contact_vel_sq = torch.sum(torch.square(feet_vel_xy * contact_expanded.float()), dim=(1, 2))
                scale_contact_no_vel = getattr(self.cfg, "rew_scale_contact_no_vel", -2.0)
                rew_contact_no_vel = scale_contact_no_vel * contact_vel_sq

        self._initialize_hip_pos_penalty_joints()
        rew_hip_pos = torch.zeros(num_envs, device=self.device)
        if self._hip_pos_joint_ids is not None and self._hip_pos_joint_ids.numel() > 0:
            q_hip = joint_pos[:, self._hip_pos_joint_ids]
            scale_hip = float(getattr(self.cfg, "rew_scale_hip_pos", -0.5))
            rew_hip_pos = scale_hip * torch.sum(torch.square(q_hip), dim=1)

        reward_result = compute_rewards(
            self.cfg.rew_scale_pitch_roll_angle,
            self.cfg.rew_scale_lin_vel_z,
            self.cfg.rew_scale_height,
            self.cfg.rew_scale_joint_velocity,
            self.cfg.rew_scale_joint_acceleration,
            self.cfg.rew_scale_foot_hit,
            0.0,  # 废弃的 rew_scale_foot_path_tracking（不再计算摆线追踪）
            self.cfg.rew_scale_joint_limit,
            self.robot.data.root_lin_vel_w[:, 2],
            pitch_angle,
            roll_angle,
            joint_vel,
            joint_acc,
            foot_hit_reward,
            torch.zeros_like(foot_hit_reward),  # 废弃的 foot_path_tracking_reward
            joint_limit_violation,
            height_violation,
            pitch_roll_swing_multiplier,
        )

        (total_reward,
         rew_pitch_roll_angle, rew_lin_vel_z, rew_height,
         rew_joint_velocity, rew_joint_acceleration,
         rew_foot_hit, rew_joint_limit) = reward_result

        total_reward = total_reward + rew_pitch_roll_ang_vel
        total_reward = total_reward + rew_action_rate + rew_contact_no_vel + rew_hip_pos
        # 应用新的动态步态奖励（根据 cfg 权重缩放）
        rew_feet_air_time = rew_feet_air_time * getattr(self.cfg, "rew_scale_feet_air_time", 8.0)
        rew_foot_clearance = rew_foot_clearance * getattr(self.cfg, "rew_scale_foot_clearance", -1.0)
        rew_distance_attraction = rew_distance_attraction * getattr(
            self.cfg,
            "rew_scale_distance_attraction",
            getattr(self.cfg, "rew_scale_swing_velocity", 2.0),
        )
        rew_swing_knee = rew_swing_knee * getattr(self.cfg, "rew_scale_swing_knee", -2.0)
        total_reward = total_reward + rew_feet_air_time + rew_foot_clearance + rew_distance_attraction + rew_swing_knee

        curriculum = getattr(self.cfg, "reward_curriculum", None)
        active_components = None  # None = 无课程，全部启用
        if curriculum and len(curriculum) > 0:
            step = getattr(self, "common_step_counter", 0)
            mode = getattr(self.cfg, "reward_curriculum_mode", "step")
            steps_per_iter = getattr(self.cfg, "reward_curriculum_steps_per_iteration", 24)
            active = self._reward_curriculum_components(step, curriculum, mode=mode, steps_per_iter=steps_per_iter)
            if active:
                active = self._expand_deprecated_reward_curriculum_components(active)
            active_components = active
            if active:
                # 与 _reward_components 对齐；勿再使用 rew_foot_path_tracking（已并入下方三项）
                comp = {
                    "rew_pitch_roll_angle": rew_pitch_roll_angle,
                    "rew_pitch_roll_ang_vel": rew_pitch_roll_ang_vel,
                    "rew_lin_vel_z": rew_lin_vel_z,
                    "rew_height": rew_height,
                    "rew_joint_velocity": rew_joint_velocity,
                    "rew_joint_acceleration": rew_joint_acceleration,
                    "rew_foot_hit": rew_foot_hit,
                    "rew_feet_air_time": rew_feet_air_time,
                    "rew_foot_clearance": rew_foot_clearance,
                    "rew_distance_attraction": rew_distance_attraction,
                    "rew_swing_knee": rew_swing_knee,
                    "rew_joint_limit": rew_joint_limit,
                    "rew_action_rate": rew_action_rate,
                    "rew_contact_no_vel": rew_contact_no_vel,
                    "rew_hip_pos": rew_hip_pos,
                }
                total_reward = sum(comp[n] for n in active if n in comp)
            elif active is not None:
                total_reward = torch.zeros_like(rew_pitch_roll_angle)

        if hasattr(self, '_episode_reward_sums'):
            # 有课程时只累加当前阶段启用的分量，未启用的累加 0，这样 TensorBoard/日志与分阶段一致
            def _add_if_active(name: str, value: torch.Tensor) -> None:
                if active_components is None or (active_components and name in active_components):
                    self._episode_reward_sums[name] += value
                else:
                    self._episode_reward_sums[name] += torch.zeros_like(value)

            _add_if_active("rew_pitch_roll_angle", rew_pitch_roll_angle)
            _add_if_active("rew_pitch_roll_ang_vel", rew_pitch_roll_ang_vel)
            _add_if_active("rew_lin_vel_z", rew_lin_vel_z)
            _add_if_active("rew_height", rew_height)
            _add_if_active("rew_joint_velocity", rew_joint_velocity)
            _add_if_active("rew_joint_acceleration", rew_joint_acceleration)
            _add_if_active("rew_foot_hit", rew_foot_hit)
            _add_if_active("rew_feet_air_time", rew_feet_air_time)
            _add_if_active("rew_foot_clearance", rew_foot_clearance)
            _add_if_active("rew_distance_attraction", rew_distance_attraction)
            _add_if_active("rew_swing_knee", rew_swing_knee)
            _add_if_active("rew_joint_limit", rew_joint_limit)
            _add_if_active("rew_action_rate", rew_action_rate)
            _add_if_active("rew_contact_no_vel", rew_contact_no_vel)
            _add_if_active("rew_hip_pos", rew_hip_pos)

        # 踩中后：用户模式仍当步清目标；自动模式由 foot_target_regenerate_delay_s 调度下一目标
        self._check_and_generate_targets()
        self._update_target_markers()
        
        return total_reward

    def get_reward_breakdown(self) -> dict:
        if not hasattr(self, '_last_episode_reward_means'):
            return {"error": "奖励统计尚未初始化"}
        
        total = sum(self._last_episode_reward_means.values())
        percentages = {}
        
        for name, value in self._last_episode_reward_means.items():
            if abs(total) > 1e-6:
                percentages[name] = (value / total) * 100
            else:
                percentages[name] = 0.0
        
        return {
            "components": self._last_episode_reward_means.copy(),
            "percentages": percentages,
            "total": total
        }
    
    def print_reward_breakdown(self):
        breakdown = self.get_reward_breakdown()
        if "error" in breakdown:
            print(f"[WARNING] {breakdown['error']}")
            return
        
        print("\n" + "=" * 60)
        print("奖励分量统计（Episode平均值）")
        print("=" * 60)
        print(f"{'奖励分量':<30} {'平均值':>12} {'占比':>10}")
        print("-" * 60)
        
        sorted_items = sorted(
            breakdown["components"].items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        for name, value in sorted_items:
            pct = breakdown["percentages"].get(name, 0.0)

            sign = "+" if value >= 0 else ""
            print(f"{name:<30} {sign}{value:>11.2f} {pct:>9.1f}%")
        
        print("-" * 60)
        print(f"{'总计':<30} {breakdown['total']:>12.2f} {'100.0':>9}%")
        print("=" * 60 + "\n")

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        root_height = self.robot.data.root_pos_w[:, 2]
        fallen = root_height < (self.cfg.reset_root_height * 0.5)

        return fallen, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        if hasattr(self, '_episode_reward_sums') and len(env_ids) > 0:

            env_ids_tensor = torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
            
            # 当前课程阶段（与 _get_rewards 一致）：未启用组件在日志中记 0，避免 TensorBoard 在阶段前就显示非零
            curriculum = getattr(self.cfg, "reward_curriculum", None)
            active_components = None
            if curriculum and len(curriculum) > 0:
                step = getattr(self, "common_step_counter", 0)
                mode = getattr(self.cfg, "reward_curriculum_mode", "step")
                steps_per_iter = getattr(self.cfg, "reward_curriculum_steps_per_iteration", 24)
                active_components = self._reward_curriculum_components(step, curriculum, mode=mode, steps_per_iter=steps_per_iter)
                if active_components:
                    active_components = self._expand_deprecated_reward_curriculum_components(active_components)

            total_sum = 0.0
            for name in self._reward_components:
                mean_val = torch.mean(self._episode_reward_sums[name][env_ids_tensor]).item()
                self._last_episode_reward_means[name] = mean_val
                # 仅当前阶段启用的组件计入 total，与 _get_rewards 的 total_reward 一致
                if active_components is None or (active_components and name in active_components):
                    total_sum += mean_val

            if not hasattr(self, 'extras') or self.extras is None:
                self.extras = {}
            if "log" not in self.extras:
                self.extras["log"] = {}

            # 日志：未启用组件记 0，避免“阶段前就有数值”
            for name in self._reward_components:
                mean_val = self._last_episode_reward_means[name]
                if active_components is not None and active_components and name not in active_components:
                    mean_val = 0.0
                self.extras["log"][f"reward/{name}"] = mean_val
            self.extras["log"]["reward/total"] = total_sum

            for name in self._reward_components:
                self._episode_reward_sums[name][env_ids_tensor] = 0.0

        super()._reset_idx(env_ids)

        if self._ankle_contact_consecutive_count is not None and len(env_ids) > 0:
            env_ids_t = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._ankle_contact_consecutive_count[env_ids_t] = 0

        self._initialize_locked_joints()

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        # 小噪声：仅加在上肢/手指，腿部与躯干保持默认角；对腿加噪声会改变链长几何，易出现脚踝略低于地面
        noise = sample_uniform(-0.02, 0.02, joint_pos.shape, joint_pos.device)
        if joint_pos.shape[1] == len(self.robot.joint_names):
            leg_substr = ("hip_", "knee_", "ankle_", "torso_joint")
            for j, name in enumerate(self.robot.joint_names):
                if any(s in name for s in leg_substr):
                    noise[:, j] = 0.0
        joint_pos += noise

        if self._locked_joint_ids and len(self._locked_joint_ids) > 0:
            default_pos = self.robot.data.default_joint_pos[env_ids].clone()
            apply_locked_joint_targets(self.robot, joint_pos, default_pos, self._locked_joint_ids)

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        default_root_state[:, 2] = self.cfg.reset_root_height
        # 根线速度、角速度显式置零，避免 asset 默认非零导致一开局“弹飞”
        default_root_state[:, 7:13] = 0.0

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        
        if self._prev_joint_vel is not None:
            self._prev_joint_vel[env_ids] = joint_vel
        else:
            self._prev_joint_vel = joint_vel.clone()
        
        if self._prev_actions is not None:
            self._prev_actions[env_ids] = 0.0
        else:

            num_envs = self.num_envs
            num_actions = self.cfg.action_space
            self._prev_actions = torch.zeros((num_envs, num_actions), device=self.device)

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # 关键：PD 目标必须与当前复位姿态一致，否则 reset 后 write_data_to_sim 会把目标写成 0，一步仿真即产生巨大扭矩导致弹飞
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        if self._prev_joint_pos_target is not None:
            self._prev_joint_pos_target[env_ids] = joint_pos  # 同步率限制缓存，避免 reset 后首步突变

        if self._foot_target_positions is not None:
            env_ids_tensor = (
                env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            )

            self._target_generation_time[env_ids_tensor] = float('-inf')
            self._target_hit[env_ids_tensor] = False
            if self._target_regenerate_deadline is not None:
                self._target_regenerate_deadline[env_ids_tensor] = float("nan")
            if self._swing_air_accum_s is not None:
                self._swing_air_accum_s[env_ids_tensor] = 0.0
        if self._foot_land_rewarded is not None:
            self._foot_land_rewarded[env_ids] = False
        # 踩点/抬脚相关状态一并重置，避免新 episode 沿用上一局残留导致整局无法触发踩点
        if self._swing_foot_contact_prev is not None:
            self._swing_foot_contact_prev[env_ids] = True
        if self._swing_foot_lifted is not None:
            self._swing_foot_lifted[env_ids] = False
        if hasattr(self, "_feet_air_time"):
            self._feet_air_time[env_ids] = 0.0
        if self._ankle_contact_consecutive_count is not None:
            self._ankle_contact_consecutive_count[env_ids, :] = 0
        if self._target_foot_indices is not None:
            self._target_foot_indices[env_ids] = 0
        if self._last_touchdown_time is not None:
            self._last_touchdown_time[env_ids] = float('-inf')
        if hasattr(self, "_next_is_follow"):
            self._next_is_follow[env_ids] = False

        if getattr(self.cfg, "event_curriculum_enabled", False) and self._event_next_push_at is not None:
            self._apply_event_curriculum_mass(env_ids)
            self._reschedule_event_push_times_on_reset(env_ids)

@torch.jit.script
def compute_rewards(
    rew_scale_pitch_roll_angle: float,
    rew_scale_lin_vel_z: float,
    rew_scale_height: float,
    rew_scale_joint_velocity: float,
    rew_scale_joint_acceleration: float,
    rew_scale_foot_hit: float,
    rew_scale_foot_path_tracking: float,
    rew_scale_joint_limit: float,
    root_lin_vel_z: torch.Tensor,
    pitch_angle: torch.Tensor,
    roll_angle: torch.Tensor,
    joint_vel: torch.Tensor,
    joint_acc: torch.Tensor,
    foot_hit: torch.Tensor,
    foot_path_tracking_reward: torch.Tensor,
    joint_limit_violation: torch.Tensor,
    height_violation: torch.Tensor,
    pitch_roll_swing_multiplier: torch.Tensor,
):
    rew_lin_vel_z = rew_scale_lin_vel_z * torch.square(root_lin_vel_z)
    rew_height = rew_scale_height * height_violation

    pitch_roll_angle_penalty = torch.square(pitch_angle) + torch.square(roll_angle)
    joint_acceleration_penalty = torch.sum(torch.square(joint_acc), dim=1)
    rew_pitch_roll_angle = rew_scale_pitch_roll_angle * pitch_roll_angle_penalty * pitch_roll_swing_multiplier
    rew_joint_acceleration = rew_scale_joint_acceleration * joint_acceleration_penalty
    rew_joint_velocity = rew_scale_joint_velocity * torch.sum(torch.square(joint_vel), dim=1)

    rew_foot_hit = rew_scale_foot_hit * foot_hit
    rew_foot_path_tracking = rew_scale_foot_path_tracking * foot_path_tracking_reward
    rew_joint_limit = rew_scale_joint_limit * joint_limit_violation

    total_reward = (
        rew_lin_vel_z
        + rew_height
        + rew_pitch_roll_angle
        + rew_joint_velocity
        + rew_joint_acceleration
        + rew_foot_hit
        + rew_foot_path_tracking
        + rew_joint_limit
    )

    return (
        total_reward,
        rew_pitch_roll_angle,
        rew_lin_vel_z,
        rew_height,
        rew_joint_velocity,
        rew_joint_acceleration,
        rew_foot_hit,
        rew_joint_limit,
    )
