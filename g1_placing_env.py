from __future__ import annotations
import re
from collections.abc import Sequence
import numpy as np
import torch
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_yaw, quat_from_angle_axis, quat_inv, quat_mul, quat_rotate_inverse, sample_uniform
from .g1_placing_env_cfg import G1PlacingEnvCfg
from .placing_joint_lock import apply_locked_joint_targets, collect_locked_joint_ids
from .placing_joint_limits import reward_joint_limit_interval
from .placing_reference_path import polyline_ground_ribbon_trimesh, quarter_circle_radius_from_arc_length, reference_path_polyline_world, reference_path_velocity_world

class G1PlacingEnv(DirectRLEnv):
    cfg: G1PlacingEnvCfg

    def __init__(self, cfg: G1PlacingEnvCfg, render_mode: str | None=None, **kwargs):
        self._foot_target_positions: torch.Tensor | None = None
        self._target_generation_time: torch.Tensor | None = None
        self._target_hit: torch.Tensor | None = None
        self._target_foot_indices: torch.Tensor | None = None
        self._last_touchdown_time: torch.Tensor | None = None
        self._user_target_mode: torch.Tensor | None = None
        self._target_regenerate_deadline: torch.Tensor | None = None
        self._swing_air_accum_s: torch.Tensor | None = None
        self._swing_foot_lifted: torch.Tensor | None = None
        self._swing_foot_contact_prev: torch.Tensor | None = None
        self._swing_foot_path_start: torch.Tensor | None = None
        self._path_start_time: torch.Tensor | None = None
        self._path_start_updated_for_target: torch.Tensor | None = None
        self._aerial_hold_start_time: torch.Tensor | None = None
        self._target_markers: VisualizationMarkers | None = None
        self._path_line_markers: VisualizationMarkers | None = None
        self._foot_contact_sensor: ContactSensor | None = None
        self._foot_contact_body_sensor_indices: torch.Tensor | None = None
        self._ankle_contact_consecutive_count: torch.Tensor | None = None
        self._feet_contact_cache: torch.Tensor | None = None
        super().__init__(cfg, render_mode, **kwargs)
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self._swing_foot_path_start = torch.zeros((self.num_envs, 3), device=self.device)
        self._path_start_time = torch.full((self.num_envs,), float('-inf'), device=self.device)
        self._path_start_updated_for_target = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._aerial_hold_start_time = torch.full((self.num_envs,), float('-inf'), device=self.device)
        self._prev_joint_vel = None
        self._prev_actions = None
        self._locked_joint_ids = None
        self._locked_joint_initialized = False
        self._ankle_locked_angle_pitch = 0.0
        self._ankle_locked_angle_roll = 0.0
        self._joint_limit_map = None
        self._joint_limit_initialized = False
        self._hip_pos_joint_ids: torch.Tensor | None = None
        self._hip_pos_joint_initialized = False
        self._prev_joint_pos_target: torch.Tensor | None = None
        self._event_next_push_at: torch.Tensor | None = None
        self._event_mass_asset_cfg: SceneEntityCfg | None = None
        self._ref_path_static_visual_spawned = False
        if getattr(cfg, 'event_curriculum_enabled', False):
            self._event_next_push_at = torch.zeros(self.num_envs, device=self.device)
        self._ensure_static_ref_path_visual()

    def _get_event_curriculum_phase(self) -> dict | None:
        if not getattr(self.cfg, 'event_curriculum_enabled', False):
            return None
        cur = getattr(self.cfg, 'event_curriculum', None)
        if not cur:
            return None
        step = int(self.common_step_counter)
        spi = int(getattr(self.cfg, 'event_curriculum_steps_per_iteration', 24))
        base = int(getattr(self.cfg, 'event_curriculum_base_iteration', 0))
        it = max(0, step // max(1, spi) - base)
        for p in cur:
            start = int(p.get('iter_start', 0))
            end = p.get('iter_end')
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
        mass = phase.get('mass_add_kg')
        if mass is None:
            return
        eids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)
        if eids.numel() == 0:
            return
        if self._event_mass_asset_cfg is None:
            names = list(getattr(self.cfg, 'event_torso_body_names', ('.*torso.*',)))
            self._event_mass_asset_cfg = SceneEntityCfg('robot', body_names=names)
            self._event_mass_asset_cfg.resolve(self.scene)
        mdp.randomize_rigid_body_mass(self, eids, asset_cfg=self._event_mass_asset_cfg, mass_distribution_params=(float(mass[0]), float(mass[1])), operation='add')

    def _reschedule_event_push_times_on_reset(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        if self._event_next_push_at is None:
            return
        eids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)
        if eids.numel() == 0:
            return
        (lo, hi) = getattr(self.cfg, 'event_push_interval_range_s', (2.0, 4.0))
        self._event_next_push_at[eids] = sample_uniform(float(lo), float(hi), (eids.shape[0],), self.device)

    def _apply_event_curriculum_push_interval(self) -> None:
        if self._event_next_push_at is None:
            return
        phase = self._get_event_curriculum_phase()
        if phase is None or not phase.get('push_enabled', False):
            return
        if 'push_xy' not in phase:
            return
        px = float(phase['push_xy'])
        pz = float(phase.get('push_z', 0.05))
        vel_range = {'x': (-px, px), 'y': (-px, px), 'z': (-pz, pz)}
        t = self.episode_length_buf.float() * self.step_dt
        due = t >= self._event_next_push_at
        if not torch.any(due):
            return
        env_ids = torch.where(due)[0]
        mdp.push_by_setting_velocity(self, env_ids, vel_range, SceneEntityCfg('robot'))
        (lo, hi) = getattr(self.cfg, 'event_push_interval_range_s', (2.0, 4.0))
        interval = sample_uniform(float(lo), float(hi), (env_ids.shape[0],), self.device)
        self._event_next_push_at[env_ids] = t[env_ids] + interval

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
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
        (self.reset_terminated[:], self.reset_time_outs[:]) = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).view(-1)
        if reset_env_ids.numel() > 0:
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
        if self.cfg.events:
            if 'interval' in self.event_manager.available_modes:
                self.event_manager.apply(mode='interval', dt=self.step_dt)
        elif getattr(self.cfg, 'event_curriculum_enabled', False):
            self._apply_event_curriculum_push_interval()
        self.obs_buf = self._get_observations()
        if self.cfg.observation_noise_model:
            self.obs_buf['policy'] = self._observation_noise_model.apply(self.obs_buf['policy'])
        self._advance_reference_path_progress(reset_env_ids)
        return (self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras)

    def _get_feet_body_ids(self) -> list:
        (left_ankle_ids, _) = self.robot.find_bodies('.*left_ankle_roll_link')
        (right_ankle_ids, _) = self.robot.find_bodies('.*right_ankle_roll_link')
        feet_body_ids = []
        if len(left_ankle_ids) > 0:
            feet_body_ids.append(left_ankle_ids[0])
        if len(right_ankle_ids) > 0:
            feet_body_ids.append(right_ankle_ids[0])
        if len(feet_body_ids) > 2:
            feet_body_ids = feet_body_ids[:2]
        return feet_body_ids

    def _resolve_foot_contact_sensor_body_indices(self) -> None:
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
            (bids, _) = sensor.find_bodies('^' + re.escape(n) + '$')
            if len(bids) != 1:
                return
            idxs.append(int(bids[0]))
        self._foot_contact_body_sensor_indices = torch.tensor(idxs, device=self.device, dtype=torch.long)

    def get_feet_contact_state(self) -> torch.Tensor | None:
        feet_body_ids = self._get_feet_body_ids()
        if len(feet_body_ids) < 2:
            return None
        if self._feet_contact_cache is not None:
            return self._feet_contact_cache
        use_phys = getattr(self.cfg, 'foot_contact_use_physics_sensor', True)
        if use_phys and self._foot_contact_sensor is not None and self._foot_contact_sensor.is_initialized:
            self._resolve_foot_contact_sensor_body_indices()
            if self._foot_contact_body_sensor_indices is not None:
                idx = self._foot_contact_body_sensor_indices
                forces = self._foot_contact_sensor.data.net_forces_w[:, idx, :]

                # 1. 核心修复：只取 Z 轴法向力，加绝对值防止穿模拉扯报错
                fn_z = torch.abs(forces[..., 2])

                # 2. 读取双阈值
                th_touchdown = float(getattr(self.cfg, 'foot_contact_touchdown_force', 80.0))
                th_liftoff = float(getattr(self.cfg, 'foot_contact_liftoff_force', 20.0))

                # 3. 施密特触发器 (迟滞滤波)逻辑
                was_contact = self._persistent_feet_contact
                is_contact = torch.where(
                    was_contact,
                    fn_z >= th_liftoff,   # 如果本来在地上，力要小于20N才算离地
                    fn_z >= th_touchdown  # 如果本来在空中，力要大于80N才算踩实
                )

                self._persistent_feet_contact = is_contact
                self._feet_contact_cache = is_contact
                return self._feet_contact_cache
        if self._ankle_contact_consecutive_count is None:
            return None
        feet_body_ids = feet_body_ids[:2]
        body_pos_w = self.robot.data.body_pos_w
        body_lin_vel_w = self.robot.data.body_lin_vel_w
        feet_z = body_pos_w[:, feet_body_ids, 2]
        feet_vel = body_lin_vel_w[:, feet_body_ids, :]
        vel_norm = torch.norm(feet_vel, dim=-1)
        h_th = float(getattr(self.cfg, 'foot_contact_height_threshold', 0.075))
        v_th = float(getattr(self.cfg, 'foot_contact_velocity_threshold', 0.08))
        orig_z = self.scene.env_origins[:, 2:3]
        height_ag = feet_z - orig_z
        low_height = height_ag < h_th
        small_vel = vel_norm < v_th
        candidate = low_height & small_vel
        n_frames = int(getattr(self.cfg, 'foot_contact_consecutive_frames', 2))
        count = self._ankle_contact_consecutive_count
        count = torch.where(candidate, torch.clamp(count + 1, max=n_frames), torch.zeros_like(count))
        self._ankle_contact_consecutive_count = count
        contact = count >= n_frames
        self._feet_contact_cache = contact
        return contact

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path='/World/ground', cfg=self.cfg.ground_plane_cfg)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations['robot'] = self.robot
        if getattr(self.cfg, 'foot_contact_use_physics_sensor', True):
            fth = float(getattr(self.cfg, 'foot_contact_force_threshold', 1.0))
            ccfg = ContactSensorCfg(prim_path=f'{self.scene.env_regex_ns}/Robot/.*', history_length=0, track_air_time=False, force_threshold=fth)
            self._foot_contact_sensor = ContactSensor(ccfg)
            self.scene._sensors['foot_contact'] = self._foot_contact_sensor
            self._foot_contact_body_sensor_indices = None
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func('/World/Light', light_cfg)
        num_envs = self.scene.num_envs
        self._foot_target_positions = torch.zeros((num_envs, 3), device=self.device)
        self._swing_air_accum_s = torch.zeros(num_envs, device=self.device)
        self._target_generation_time = torch.full((num_envs,), float('-inf'), device=self.device)
        self._target_hit = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._target_foot_indices = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._last_touchdown_time = torch.full((num_envs,), float('-inf'), device=self.device)
        self._user_target_mode = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._next_is_follow = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._target_regenerate_deadline = torch.full((num_envs,), float('nan'), device=self.device)
        self._swing_foot_lifted = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._swing_foot_contact_prev = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self._foot_land_rewarded = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._ankle_contact_consecutive_count = torch.zeros((num_envs, 2), dtype=torch.long, device=self.device)
        self._feet_contact_cache = None
        self._persistent_feet_contact = torch.zeros((num_envs, 2), dtype=torch.bool, device=self.device)
        target_marker_cfg = VisualizationMarkersCfg(prim_path='/Visuals/foot_target_markers', markers={'target': sim_utils.SphereCfg(radius=0.03, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)))})
        self._target_markers = VisualizationMarkers(target_marker_cfg)
        self._target_markers.set_visibility(True)
        path_line_cfg = VisualizationMarkersCfg(
            prim_path='/Visuals/foot_path_line_markers',
            markers={
                'path_line': sim_utils.CylinderCfg(
                    radius=0.004,
                    height=1.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                )
            },
        )
        self._path_line_markers = VisualizationMarkers(path_line_cfg)
        self._path_line_markers.set_visibility(True)
        self._reward_components = ['rew_pitch_roll_angle', 'rew_pitch_roll_ang_vel', 'rew_height', 'rew_joint_velocity', 'rew_joint_acceleration', 'rew_foot_hit', 'rew_foot_path_tracking', 'rew_joint_limit', 'rew_action_rate', 'rew_contact_no_vel', 'rew_hip_pos']
        self._episode_reward_sums = {name: torch.zeros(num_envs, dtype=torch.float32, device=self.device) for name in self._reward_components}
        self._last_episode_reward_means = {name: 0.0 for name in self._reward_components}
        arc_len = float(getattr(self.cfg, 'ref_path_quarter_arc_length_m', 10.0))
        self._ref_path_arc_radius_m = quarter_circle_radius_from_arc_length(arc_len)
        self._ref_path_progress_m = torch.zeros(num_envs, device=self.device)
        self._ref_path_psi0 = torch.zeros(num_envs, device=self.device)
        self._ref_path_origin_xy = torch.zeros((num_envs, 2), device=self.device)

    def _advance_reference_path_progress(self, skip_env_ids: torch.Tensor) -> None:
        """本步刚 reset 的环境不累计弧长，保证首帧观测对应 s=0。"""
        if not getattr(self.cfg, 'ref_path_velocity_command_enabled', False):
            return
        if self._ref_path_progress_m is None:
            return
        V = float(getattr(self.cfg, 'ref_path_speed_m_s', 0.5))
        delta = torch.full((self.num_envs,), V * self.step_dt, device=self.device, dtype=self._ref_path_progress_m.dtype)
        if skip_env_ids.numel() > 0:
            delta[skip_env_ids] = 0.0
        self._ref_path_progress_m += delta

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if hasattr(self, 'actions') and self.actions is not None:
            self._prev_actions = self.actions.clone()
        elif self._prev_actions is None:
            self._prev_actions = actions.clone()
        self.actions = actions.clone()
        if getattr(self.cfg, 'force_zero_action', False):
            self.actions.zero_()
        self._check_and_generate_targets()
        self._update_target_markers()
        self._ensure_static_ref_path_visual()

    def _update_target_markers(self) -> None:
        if self._target_markers is None or self._foot_target_positions is None:
            return
        num_envs = self._foot_target_positions.shape[0]
        has_valid = (self._target_generation_time is not None) & (self._target_generation_time > float('-inf'))
        marker_positions = self._foot_target_positions.clone()
        marker_positions[~has_valid] = torch.tensor((0.0, 0.0, -100.0), device=self.device, dtype=marker_positions.dtype)
        trans_np = marker_positions.detach().cpu().float().numpy()
        self._target_markers.visualize(translations=trans_np, marker_indices=np.zeros(num_envs, dtype=np.int32))
        self._update_path_line_markers()

    def _ensure_static_ref_path_visual(self) -> None:
        """参考路径在世界系中是固定几何：首次就绪时生成一条贴地 USD 带状网格（连续路面），不随回合刷新。"""
        if self._ref_path_static_visual_spawned:
            return
        if not getattr(self.cfg, 'ref_path_visualization_enabled', True):
            self._ref_path_static_visual_spawned = True
            return
        env_id = int(getattr(self.cfg, 'ref_path_visualize_env_id', 0))
        if env_id < 0 or env_id >= self.num_envs:
            env_id = 0
        if not hasattr(self.robot, 'data') or self.robot.data.default_root_state is None:
            return
        prim_path = '/Visuals/ref_path_road_ribbon'
        if prim_utils.is_prim_path_valid(prim_path):
            self._ref_path_static_visual_spawned = True
            return
        drs = self.robot.data.default_root_state[env_id].clone()
        drs[:3] = drs[:3] + self.scene.env_origins[env_id]
        drs[2] = float(getattr(self.cfg, 'reset_root_height', 0.74))
        origin_xy = drs[:2]
        (_, _, yaw0) = euler_xyz_from_quat(drs[3:7].unsqueeze(0))
        psi0 = yaw0[0]
        z_g = float(self.scene.env_origins[env_id, 2]) + float(getattr(self.cfg, 'ref_path_visualize_z_offset_m', 0.005))
        straight = float(getattr(self.cfg, 'ref_path_straight_m', 5.0))
        arc_len = float(getattr(self.cfg, 'ref_path_quarter_arc_length_m', 10.0))
        turn_left = bool(getattr(self.cfg, 'ref_path_turn_left', True))
        n_s = int(getattr(self.cfg, 'ref_path_visualize_n_straight', 48))
        n_a = int(getattr(self.cfg, 'ref_path_visualize_n_arc', 36))
        pts = reference_path_polyline_world(
            origin_xy,
            psi0,
            z_g,
            straight,
            self._ref_path_arc_radius_m,
            arc_len,
            turn_left=turn_left,
            n_straight=n_s,
            n_arc=n_a,
        )
        if pts.ndim != 2 or pts.shape[-1] != 3 or pts.shape[0] < 2:
            return
        trans_np = pts.detach().cpu().float().numpy()
        half_w = float(getattr(self.cfg, 'ref_path_visualize_road_half_width_m', 0.15))
        ribbon = polyline_ground_ribbon_trimesh(trans_np, half_w)
        mesh_cfg = MeshCfg(
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.35, 0.1),
                roughness=0.55,
                metallic=0.05,
            ),
        )
        _spawn_mesh_geom_from_mesh(prim_path, mesh_cfg, ribbon, None, None, None)
        self._ref_path_static_visual_spawned = True

    def _update_ref_path_trajectory_markers(self) -> None:
        """兼容旧调用点：参考路径已改为 ``_ensure_static_ref_path_visual`` 一次性绘制。"""
        self._ensure_static_ref_path_visual()

    def _update_path_line_markers(self) -> None:
        if self._path_line_markers is None:
            return
        if self._foot_target_positions is None or self._swing_foot_path_start is None or self._target_generation_time is None:
            return
        env_id = 0
        if env_id >= self.num_envs or self._target_generation_time[env_id] <= float('-inf'):
            off = np.array([[0.0, 0.0, -100.0]], dtype=np.float32)
            self._path_line_markers.visualize(
                translations=off,
                orientations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                scales=np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
                marker_indices=np.zeros(1, dtype=np.int32),
            )
            return
        start = self._swing_foot_path_start[env_id]
        target = self._foot_target_positions[env_id]
        base_h = float(getattr(self.cfg, 'foot_ankle_ground_height', 0.07))
        aerial_thresh = float(getattr(self.cfg, 'foot_target_aerial_ground_threshold', 0.075))
        target_z = target[2]
        dist_xy = torch.norm(target[:2] - start[:2]).clamp(min=1e-4)
        dist_min = float(getattr(self.cfg, 'foot_target_min_distance', 0.25))
        dist_max = float(getattr(self.cfg, 'foot_target_max_distance', 0.40))
        peak_min = float(getattr(self.cfg, 'foot_path_peak_height_min', 0.10))
        peak_max = float(getattr(self.cfg, 'foot_path_peak_height_max', 0.15))
        dist_ratio = ((dist_xy - dist_min) / (dist_max - dist_min + 1e-6)).clamp(0.0, 1.0)
        peak_h = peak_min + (peak_max - peak_min) * dist_ratio
        n_pts = 21
        t = torch.linspace(0.0, 1.0, n_pts, device=self.device, dtype=target.dtype)
        # XY 五次多项式
        t_xy = 10 * t**3 - 15 * t**4 + 6 * t**5
        ref_xy = (1.0 - t_xy).unsqueeze(1) * start[:2].unsqueeze(0) + t_xy.unsqueeze(1) * target[:2].unsqueeze(0)

        # Z 三次贝塞尔 (仅用于地面到地面连线可视化)
        p0_z = torch.full((n_pts, 1), base_h, device=self.device, dtype=target.dtype)
        p3_z = torch.full((n_pts, 1), base_h, device=self.device, dtype=target.dtype)
        pull_factor = 1.6
        p1_z = p0_z + (peak_h - base_h) * pull_factor
        p2_z = p3_z + (peak_h - base_h) * pull_factor

        t_z = t.unsqueeze(1)
        inv_t = 1.0 - t_z
        ref_z_ground = (inv_t**3) * p0_z + 3 * (inv_t**2) * t_z * p1_z + 3 * inv_t * (t_z**2) * p2_z + (t_z**3) * p3_z
        if target_z > aerial_thresh:
            ref_z = ref_z_ground * (1.0 - t_z) + target_z.unsqueeze(0).unsqueeze(1) * t_z
        else:
            ref_z = ref_z_ground
        pts = torch.cat([ref_xy, ref_z], dim=1)
        p0 = pts[:-1]
        p1 = pts[1:]
        seg = p1 - p0
        seg_len = torch.norm(seg, dim=1).clamp(min=1e-6)
        mid = 0.5 * (p0 + p1)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=pts.dtype).unsqueeze(0).expand_as(seg)
        dir_unit = seg / seg_len.unsqueeze(1)
        axis = torch.cross(z_axis, dir_unit, dim=1)
        axis_norm = torch.norm(axis, dim=1, keepdim=True)
        fallback_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=pts.dtype).unsqueeze(0).expand_as(axis)
        axis_unit = torch.where(axis_norm > 1e-6, axis / axis_norm.clamp(min=1e-6), fallback_axis)
        dot = torch.clamp(torch.sum(z_axis * dir_unit, dim=1), -1.0, 1.0)
        angle = torch.acos(dot)
        orient = quat_from_angle_axis(angle, axis_unit)
        scales = torch.stack(
            [
                torch.full_like(seg_len, 1.0),
                torch.full_like(seg_len, 1.0),
                seg_len,
            ],
            dim=1,
        )
        self._path_line_markers.visualize(
            translations=mid.detach().cpu().float().numpy(),
            orientations=orient.detach().cpu().float().numpy(),
            scales=scales.detach().cpu().float().numpy(),
            marker_indices=np.zeros(seg_len.shape[0], dtype=np.int32),
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
        soft = float(getattr(self.cfg, 'soft_dof_pos_limit', 0.9))
        torso_half = float(getattr(self.cfg, 'torso_joint_limit_rad', 0.35))
        joint_names = self.robot.joint_names
        for (joint_id, joint_name) in enumerate(joint_names):
            lim = reward_joint_limit_interval(joint_name, soft_dof_pos_limit=soft, torso_half_rad=torso_half)
            if lim is not None:
                self._joint_limit_map[joint_id] = list(lim)
        self._joint_limit_initialized = True

    def _initialize_hip_pos_penalty_joints(self) -> None:
        if self._hip_pos_joint_initialized:
            return
        ids: list[int] = []
        for (joint_id, joint_name) in enumerate(self.robot.joint_names):
            ln = joint_name.lower()
            # 严格对齐宇树：约束 hip_roll 和 hip_pitch，放开 hip_yaw
            if re.search('hip_roll', ln) or re.search('hip_pitch', ln):
                ids.append(joint_id)
        if ids:
            self._hip_pos_joint_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
        else:
            self._hip_pos_joint_ids = None
        self._hip_pos_joint_initialized = True

    def _apply_action(self) -> None:
        self._initialize_locked_joints()
        default_pos = self.robot.data.default_joint_pos.clone()
        joint_pos_target = default_pos + self.actions * self.cfg.action_scale
        if self._locked_joint_ids and len(self._locked_joint_ids) > 0:
            apply_locked_joint_targets(self.robot, joint_pos_target, default_pos, self._locked_joint_ids)
        if self._joint_limit_map:
            for (joint_id, (low, high)) in self._joint_limit_map.items():
                joint_pos_target[:, joint_id] = torch.clamp(joint_pos_target[:, joint_id], low, high)
        rate_limit = getattr(self.cfg, 'action_rate_limit_rad_per_step', 0.0)
        if rate_limit > 0 and self._prev_joint_pos_target is not None:
            delta = joint_pos_target - self._prev_joint_pos_target
            delta = torch.clamp(delta, -rate_limit, rate_limit)
            joint_pos_target = self._prev_joint_pos_target + delta
        self._prev_joint_pos_target = joint_pos_target.clone()
        self.robot.set_joint_position_target(joint_pos_target, joint_ids=None)

    def _ref_path_command_world(self) -> torch.Tensor:
        """参考路径在世界系下的期望 (vx, vy, ωz)，供路径驱动落点等内部逻辑使用；不并入 policy 观测。未启用或缓冲未就绪时为 0。"""
        num_envs = self.num_envs
        dtype = self.robot.data.root_pos_w.dtype
        out = torch.zeros((num_envs, 3), device=self.device, dtype=dtype)
        if not getattr(self.cfg, 'ref_path_velocity_command_enabled', False):
            return out
        if self._ref_path_progress_m is None or self._ref_path_psi0 is None:
            return out
        straight = float(getattr(self.cfg, 'ref_path_straight_m', 5.0))
        arc_len = float(getattr(self.cfg, 'ref_path_quarter_arc_length_m', 10.0))
        turn_left = bool(getattr(self.cfg, 'ref_path_turn_left', True))
        V = float(getattr(self.cfg, 'ref_path_speed_m_s', 0.5))
        (vx_w, vy_w, wz_w) = reference_path_velocity_world(self._ref_path_progress_m, V, straight, self._ref_path_arc_radius_m, arc_len, self._ref_path_psi0, turn_left=turn_left)
        out = torch.stack([vx_w, vy_w, wz_w], dim=1)
        return out

    def _path_driven_target_env_mask(self, env_ids: torch.Tensor) -> torch.Tensor:
        if env_ids.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        if not getattr(self.cfg, 'path_driven_target_enabled', False):
            return torch.zeros(env_ids.shape[0], dtype=torch.bool, device=self.device)
        p = float(getattr(self.cfg, 'path_driven_prob', 1.0))
        if p <= 0.0:
            return torch.zeros(env_ids.shape[0], dtype=torch.bool, device=self.device)
        skip = self._user_target_mode[env_ids] if self._user_target_mode is not None else torch.zeros(env_ids.shape[0], dtype=torch.bool, device=self.device)
        out = ~skip
        if p < 1.0:
            out = out & (torch.rand(env_ids.shape[0], device=self.device) < p)
        return out

    def _get_observations(self) -> dict:
        self._feet_contact_cache = None
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        root_lin_vel = self.robot.data.root_lin_vel_w
        root_ang_vel = self.robot.data.root_ang_vel_w
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
        if self._foot_target_positions is not None:
            target_offset_w = self._foot_target_positions - root_pos
            target_offset_local = quat_rotate_inverse(root_quat, target_offset_w)
            has_target = torch.any(self._foot_target_positions != 0, dim=1)
            if len(feet_body_ids) > 0 and num_feet >= 1 and (self._target_foot_indices is not None):
                row = torch.arange(num_envs, device=self.device)
                ti = self._target_foot_indices.clamp(min=0, max=num_feet - 1)
                swing_local = feet_pos_local[row, ti]
                target_offset_local = torch.where(has_target.unsqueeze(1), target_offset_local, swing_local)
        else:
            target_offset_local = torch.zeros((num_envs, 3), device=self.device)
        feet_contact_flat = feet_contact_state
        feet_pos_flat = feet_pos_local.reshape(num_envs, -1)
        feet_quat_flat = feet_quat_local.reshape(num_envs, -1)
        # 盲态踩点：策略仅见本体感受 + 目标相对根坐标 + 脚状态；参考路径速度仍用于环境内落点生成，不进入 obs
        obs = torch.cat((root_pos, root_quat, root_lin_vel, root_ang_vel, self.joint_pos, self.joint_vel, target_offset_local, feet_contact_flat, feet_pos_flat, feet_quat_flat), dim=-1)
        if getattr(self.cfg, 'policy_observation_enabled', True):
            observations = {'policy': obs}
        else:
            observations = {'policy': torch.zeros_like(obs)}
        return observations

    def collect_reference_motions(self, num_samples: int) -> torch.Tensor:
        obs_dict = self._get_observations()
        current_obs = obs_dict['policy']
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
        if len(env_ids) == 0:
            return
        feet_body_ids = self._get_feet_body_ids()
        if len(feet_body_ids) < 2:
            return
        feet_body_ids = feet_body_ids[:2]
        body_pos_w = self.robot.data.body_pos_w
        feet_pos_w = body_pos_w[env_ids][:, feet_body_ids, :]
        swing_xy = torch.where(hit_foot_indices.unsqueeze(1).expand(-1, 2) == 0, feet_pos_w[:, 0, :2], feet_pos_w[:, 1, :2])
        root_quat = self.robot.data.root_quat_w[env_ids]
        local_y = torch.zeros(len(env_ids), 3, device=self.device, dtype=root_quat.dtype)
        local_y[:, 1] = 1.0
        world_y = quat_apply_yaw(root_quat, local_y)[:, :2]
        direction_multiplier = torch.where(hit_foot_indices == 0, -1.0, 1.0).unsqueeze(1)
        spacing_m = getattr(self.cfg, 'foot_spacing_stand_m', 0.24)
        target_xy = swing_xy + spacing_m * direction_multiplier * world_y
        target_positions = torch.zeros(len(env_ids), 3, device=self.device, dtype=feet_pos_w.dtype)
        target_positions[:, :2] = target_xy
        target_positions[:, 2] = self.scene.env_origins[env_ids, 2]
        swing_foot_indices = 1 - hit_foot_indices
        self._foot_target_positions[env_ids] = target_positions
        self._target_foot_indices[env_ids] = swing_foot_indices
        row_idx = torch.arange(len(env_ids), device=self.device)
        if self._swing_foot_path_start is not None:
            self._swing_foot_path_start[env_ids] = feet_pos_w[row_idx, swing_foot_indices, :].clone()
        if self._path_start_time is not None:
            self._path_start_time[env_ids] = self.episode_length_buf[env_ids].float() * self.step_dt
        if self._path_start_updated_for_target is not None:
            self._path_start_updated_for_target[env_ids] = False
        if self._aerial_hold_start_time is not None:
            self._aerial_hold_start_time[env_ids] = float('-inf')

    def _generate_path_driven_target(self, env_ids: torch.Tensor, desired_vel_xy: torch.Tensor) -> None:
        """根据期望世界系水平线速度，用 Raibert 式启发将运动意图分解为摆动脚落脚点。

        Args:
            env_ids: 环境索引，shape ``(K,)``。
            desired_vel_xy: ``(K, 2)``，与 ``root_lin_vel_w[:, :2]`` 同为世界系 XY 速度。

        说明:
            使用当前 ``_target_foot_indices[env_ids]`` 作为摆动脚 (0 左脚 / 1 右脚)。
            在踩中或延迟刷新后若需换脚，请在本函数调用前执行
            ``self._target_foot_indices[env_ids] = 1 - self._target_foot_indices[env_ids]``。
        """
        if env_ids.numel() == 0:
            return
        if self._foot_target_positions is None or self._target_foot_indices is None:
            return
        feet_body_ids = self._get_feet_body_ids()
        if len(feet_body_ids) < 2:
            return
        feet_body_ids = feet_body_ids[:2]
        k = int(env_ids.numel())
        if tuple(desired_vel_xy.shape) != (k, 2):
            raise ValueError(f'desired_vel_xy 必须为 (K, 2) = ({k}, 2)，当前为 {tuple(desired_vel_xy.shape)}')
        root_pos = self.robot.data.root_pos_w[env_ids]
        root_quat = self.robot.data.root_quat_w[env_ids]
        root_vel = self.robot.data.root_lin_vel_w[env_ids]
        swing_foot_indices = self._target_foot_indices[env_ids]
        T_stance = float(getattr(self.cfg, 'foot_path_duration_base_s', 0.55))
        hip_width = float(getattr(self.cfg, 'foot_spacing_stand_m', 0.24)) / 2.0
        direction_multiplier = torch.where(swing_foot_indices == 0, 1.0, -1.0).to(dtype=root_pos.dtype)
        local_hip_pos = torch.zeros((k, 3), device=self.device, dtype=root_pos.dtype)
        local_hip_pos[:, 1] = hip_width * direction_multiplier
        world_hip_offset = quat_apply_yaw(root_quat, local_hip_pos)
        p_hip = root_pos[:, :2] + world_hip_offset[:, :2]
        p_symmetry = (T_stance / 2.0) * desired_vel_xy
        k_v = float(getattr(self.cfg, 'path_driven_raibert_kv', 0.05))
        v_error = root_vel[:, :2] - desired_vel_xy
        p_feedback = k_v * v_error
        target_xy = p_hip + p_symmetry + p_feedback
        row_idx = torch.arange(k, device=self.device)
        stance_foot_indices = 1 - swing_foot_indices
        body_pos_subset = self.robot.data.body_pos_w[env_ids]
        feet_pos_w = body_pos_subset[:, feet_body_ids, :]
        stance_foot_pos = feet_pos_w[row_idx, stance_foot_indices, :2]
        dist = torch.norm(target_xy - stance_foot_pos, dim=1)
        max_reach = float(getattr(self.cfg, 'foot_target_max_distance', 0.15))
        exceed_mask = dist > max_reach
        if torch.any(exceed_mask):
            scale = max_reach / dist[exceed_mask]
            target_xy[exceed_mask] = stance_foot_pos[exceed_mask] + (target_xy[exceed_mask] - stance_foot_pos[exceed_mask]) * scale.unsqueeze(1)
        target_positions = torch.zeros(k, 3, device=self.device, dtype=root_pos.dtype)
        target_positions[:, :2] = target_xy
        target_positions[:, 2] = self.scene.env_origins[env_ids, 2]
        self._foot_target_positions[env_ids] = target_positions
        swing_idx = self._target_foot_indices[env_ids]
        if self._swing_foot_path_start is not None:
            self._swing_foot_path_start[env_ids] = feet_pos_w[row_idx, swing_idx, :].clone()
        if self._path_start_time is not None:
            self._path_start_time[env_ids] = self.episode_length_buf[env_ids].float() * self.step_dt
        if self._path_start_updated_for_target is not None:
            self._path_start_updated_for_target[env_ids] = False
        if self._aerial_hold_start_time is not None:
            self._aerial_hold_start_time[env_ids] = float('-inf')
        if self._target_regenerate_deadline is not None:
            self._target_regenerate_deadline[env_ids] = float('nan')
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

    def _generate_random_target(self, env_ids: torch.Tensor, *, alternate_swing: bool=False) -> None:
        if len(env_ids) == 0:
            return
        if self._target_regenerate_deadline is not None:
            self._target_regenerate_deadline[env_ids] = float('nan')
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
                self._generate_follow_target(env_ids_follow, hit_foot[use_follow])
            if len(env_ids_random) > 0:
                swing_r = hit_foot[~use_follow]
                self._generate_random_target_internal(env_ids_random, swing_r, root_quat[~use_follow], left_foot_xy[~use_follow], right_foot_xy[~use_follow])
        else:
            swing_foot_indices = torch.randint(0, 2, (num_generate,), device=self.device)
            self._next_is_follow[env_ids] = True
            self._generate_random_target_internal(env_ids, swing_foot_indices, root_quat, left_foot_xy, right_foot_xy)
        if self._swing_foot_path_start is not None and self._target_foot_indices is not None:
            row_idx = torch.arange(len(env_ids), device=self.device)
            swing_idx = self._target_foot_indices[env_ids]
            self._swing_foot_path_start[env_ids] = feet_pos_w[row_idx, swing_idx, :].clone()
        if self._path_start_time is not None:
            self._path_start_time[env_ids] = self.episode_length_buf[env_ids].float() * self.step_dt
        if self._path_start_updated_for_target is not None:
            self._path_start_updated_for_target[env_ids] = False
        if self._aerial_hold_start_time is not None:
            self._aerial_hold_start_time[env_ids] = float('-inf')
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

    def _generate_random_target_internal(self, env_ids: torch.Tensor, swing_foot_indices: torch.Tensor, root_quat: torch.Tensor, left_foot_xy: torch.Tensor, right_foot_xy: torch.Tensor) -> None:
        num_generate = len(env_ids)
        center_xy = torch.where(swing_foot_indices.unsqueeze(1).expand(-1, 2) == 0, left_foot_xy, right_foot_xy)
        target_z = self.scene.env_origins[env_ids, 2]
        x_back = getattr(self.cfg, 'foot_target_rect_x_back', 0.08)
        x_forward = getattr(self.cfg, 'foot_target_rect_x_forward', 0.3)
        y_outward = getattr(self.cfg, 'foot_target_rect_y_outward', 0.15)
        y_inward = getattr(self.cfg, 'foot_target_rect_y_inward', 0.02)
        local_x = sample_uniform(-x_back, x_forward, (num_generate,), self.device)
        is_left_foot = swing_foot_indices == 0
        y_min = torch.where(is_left_foot, torch.full((num_generate,), -y_inward, device=self.device, dtype=center_xy.dtype), torch.full((num_generate,), -y_outward, device=self.device, dtype=center_xy.dtype))
        y_max = torch.where(is_left_foot, torch.full((num_generate,), y_outward, device=self.device, dtype=center_xy.dtype), torch.full((num_generate,), y_inward, device=self.device, dtype=center_xy.dtype))
        rand_y = torch.rand(num_generate, device=self.device, dtype=center_xy.dtype)
        local_y = y_min + rand_y * (y_max - y_min)
        min_dist = getattr(self.cfg, 'foot_target_min_distance', 0.1)
        dist = torch.sqrt(local_x ** 2 + local_y ** 2).clamp(min=1e-06)
        too_close = dist < min_dist
        safe_r = sample_uniform(min_dist, x_forward, (num_generate,), self.device)
        scale = safe_r / dist
        local_x = torch.where(too_close, local_x * scale, local_x)
        local_y = torch.where(too_close, local_y * scale, local_y)
        local_x = torch.clamp(local_x, -x_back, x_forward)
        local_y = torch.max(torch.min(local_y, y_max), y_min)
        local_positions = torch.stack([local_x, local_y, torch.zeros(num_generate, device=self.device, dtype=center_xy.dtype)], dim=1)
        world_offset = quat_apply_yaw(root_quat, local_positions)
        target_positions = torch.zeros(num_generate, 3, device=self.device, dtype=center_xy.dtype)
        target_positions[:, :2] = center_xy + world_offset[:, :2]
        target_positions[:, 2] = target_z
        self._foot_target_positions[env_ids] = target_positions
        self._target_foot_indices[env_ids] = swing_foot_indices

    def _check_and_generate_targets(self) -> None:
        """点位生成器入口：首次生成、踩中延迟到期后或踩中立即（delay=0）刷新。用户模式下跳过自动生成。"""
        if self._foot_target_positions is None:
            return
        skip_auto = self._user_target_mode if self._user_target_mode is not None else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        t_now = self.episode_length_buf.float() * self.step_dt
        if self._target_regenerate_deadline is not None:
            delay_ready = torch.isfinite(self._target_regenerate_deadline) & (t_now >= self._target_regenerate_deadline) & ~skip_auto
            env_ids_delay = torch.where(delay_ready)[0]
            if env_ids_delay.numel() > 0:
                self._target_regenerate_deadline[env_ids_delay] = float('nan')
                use_pd = self._path_driven_target_env_mask(env_ids_delay)
                env_pd = env_ids_delay[use_pd]
                env_rd = env_ids_delay[~use_pd]
                if env_pd.numel() > 0:
                    dtype = self.robot.data.root_pos_w.dtype
                    vx = float(getattr(self.cfg, 'ref_path_speed_m_s', 0.5))
                    n = int(env_pd.numel())
                    desired_velocity = torch.tensor([[vx, 0.0]], device=self.device, dtype=dtype).repeat(n, 1)
                    self._generate_path_driven_target(env_pd, desired_velocity)
                    self._target_foot_indices[env_pd] = 1 - self._target_foot_indices[env_pd]
                if env_rd.numel() > 0:
                    self._generate_random_target(env_rd, alternate_swing=True)
        not_generated = torch.isinf(self._target_generation_time) & (self._target_generation_time < 0) & ~skip_auto
        env_ids_init = torch.where(not_generated)[0]
        if env_ids_init.numel() > 0:
            use_pd = self._path_driven_target_env_mask(env_ids_init)
            env_pd = env_ids_init[use_pd]
            env_rd = env_ids_init[~use_pd]
            if env_pd.numel() > 0:
                v_w = self._ref_path_command_world()
                self._generate_path_driven_target(env_pd, v_w[env_pd, :2])
            if env_rd.numel() > 0:
                self._generate_random_target(env_rd, alternate_swing=False)
        if self._target_hit is not None:
            env_ids_hit = torch.where(self._target_hit & ~skip_auto)[0]
            if env_ids_hit.numel() > 0:
                use_pd = self._path_driven_target_env_mask(env_ids_hit)
                env_pd = env_ids_hit[use_pd]
                env_rd = env_ids_hit[~use_pd]
                if env_pd.numel() > 0:
                    self._target_foot_indices[env_pd] = 1 - self._target_foot_indices[env_pd]
                    v_w = self._ref_path_command_world()
                    self._generate_path_driven_target(env_pd, v_w[env_pd, :2])
                if env_rd.numel() > 0:
                    self._generate_random_target(env_rd, alternate_swing=True)
            env_ids_user_hit = torch.where(self._target_hit & skip_auto)[0]
            if len(env_ids_user_hit) > 0:
                self._clear_user_target(env_ids_user_hit)

    def _clear_user_target(self, env_ids: torch.Tensor) -> None:
        """清除用户目标（踩中后调用）。保持用户模式，等待用户再次按 T 设置新目标。"""
        if len(env_ids) == 0:
            return
        if self._target_regenerate_deadline is not None:
            self._target_regenerate_deadline[env_ids] = float('nan')
        self._foot_target_positions[env_ids] = 0.0
        self._target_generation_time[env_ids] = float('-inf')
        self._target_hit[env_ids] = False
        if self._swing_air_accum_s is not None:
            self._swing_air_accum_s[env_ids] = 0.0
        if self._path_start_time is not None:
            self._path_start_time[env_ids] = float('-inf')
        if self._path_start_updated_for_target is not None:
            self._path_start_updated_for_target[env_ids] = False
        if self._aerial_hold_start_time is not None:
            self._aerial_hold_start_time[env_ids] = float('-inf')

    def set_user_target(self, env_id: int, position_world: tuple[float, float, float] | list[float], swing_foot_index: int | None=None) -> bool:
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
        if self._swing_foot_path_start is not None:
            self._swing_foot_path_start[env_ids] = feet_pos_w[0, swing_foot_index, :].clone().unsqueeze(0)
        if self._path_start_time is not None:
            self._path_start_time[env_ids] = self.episode_length_buf[env_ids].float() * self.step_dt
        if self._path_start_updated_for_target is not None:
            self._path_start_updated_for_target[env_ids] = False
        if self._aerial_hold_start_time is not None:
            self._aerial_hold_start_time[env_ids] = float('-inf')
        self._target_generation_time[env_ids] = self.episode_length_buf[env_ids].float() * self.step_dt
        self._target_hit[env_ids] = False
        self._user_target_mode[env_ids] = True
        if self._target_regenerate_deadline is not None:
            self._target_regenerate_deadline[env_ids] = float('nan')
        if self._swing_air_accum_s is not None:
            self._swing_air_accum_s[env_ids] = 0.0
        self._next_is_follow[env_ids] = False
        self._update_target_markers()
        self._update_ref_path_trajectory_markers()
        return True

    def _reward_curriculum_components(self, step: int, curriculum: list, mode: str='step', steps_per_iter: int=24) -> list[str] | None:
        """解析 reward_curriculum，返回当前阶段启用的奖励组件名列表；无匹配则 None（表示全部启用）。
        mode: 'step' 用 step_start/step_end；'iteration' 用 iter_start/iter_end，且 step→iter = step // steps_per_iter。"""
        if mode == 'iteration':
            tick = step // max(1, steps_per_iter)
            (start_key, end_key) = ('iter_start', 'iter_end')
        else:
            tick = step
            (start_key, end_key) = ('step_start', 'step_end')
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
        comp = phase.get('components')
        return comp if isinstance(comp, (list, tuple)) else None

    def _expand_deprecated_reward_curriculum_components(self, names: list[str] | tuple[str, ...]) -> list[str]:
        """将课程里已废弃的组件名映射为当前实现键，保证 ``comp`` 合分与 TensorBoard 日志一致。"""
        legacy = frozenset({'rew_swing_velocity'})
        seen: dict[str, None] = {}
        out: list[str] = []
        for n in names:
            if n == 'rew_swing_velocity':
                n = 'rew_foot_path_tracking'
            if n in legacy:
                continue
            elif n not in seen:
                seen[n] = None
                out.append(n)
        return out

    def _foot_hit_curriculum_radius_xy_m(self) -> float:
        """地面踩点：水平命中半径（米）。iter < iter_start 固定 thresh_start；iter_start～iter_end 线性收到 thresh_end；iter >= iter_end 固定 thresh_end。"""
        step = getattr(self, 'common_step_counter', 0)
        steps_per_iter = getattr(self.cfg, 'reward_curriculum_steps_per_iteration', 24)
        iter_cur = step // max(1, steps_per_iter)
        iter_start = getattr(self.cfg, 'foot_target_hit_threshold_iter_start', 3000)
        iter_end = getattr(self.cfg, 'foot_target_hit_threshold_iter_end', 8000)
        thresh_start = float(getattr(self.cfg, 'foot_target_hit_threshold', 0.08))
        thresh_end = float(getattr(self.cfg, 'foot_target_hit_threshold_end', 0.03))
        if iter_cur < iter_start:
            return thresh_start
        if iter_cur >= iter_end:
            return thresh_end
        t = (iter_cur - iter_start) / max(1, iter_end - iter_start)
        return thresh_start + t * (thresh_end - thresh_start)

    def _foot_path_tracking_sigma_m(self) -> float:
        """轨迹跟踪高斯 sigma（米）。iter < iter_start 固定 sigma_start；iter_start～iter_end 线性收到 sigma_end；iter >= iter_end 固定 sigma_end。"""
        step = getattr(self, 'common_step_counter', 0)
        steps_per_iter = getattr(self.cfg, 'reward_curriculum_steps_per_iteration', 24)
        iter_cur = step // max(1, steps_per_iter)
        iter_start = getattr(self.cfg, 'foot_path_tracking_sigma_iter_start', 3000)
        iter_end = getattr(self.cfg, 'foot_path_tracking_sigma_iter_end', 13000)
        sigma_start = float(getattr(self.cfg, 'foot_path_tracking_sigma_start', 0.075))
        sigma_end = float(getattr(self.cfg, 'foot_path_tracking_sigma_end', getattr(self.cfg, 'foot_path_tracking_sigma', 0.025)))
        if iter_cur < iter_start:
            return sigma_start
        if iter_cur >= iter_end:
            return sigma_end
        t = (iter_cur - iter_start) / max(1, iter_end - iter_start)
        return sigma_start + t * (sigma_end - sigma_start)

    def _foot_path_tracking_scale_curriculum(self) -> float:
        """轨迹追踪奖励权重（Scale）：线性退火衰减。"""
        step = getattr(self, 'common_step_counter', 0)
        steps_per_iter = getattr(self.cfg, 'reward_curriculum_steps_per_iteration', 24)
        iter_cur = step // max(1, steps_per_iter)
        iter_start = getattr(self.cfg, 'rew_scale_foot_path_tracking_iter_start', 8000)
        iter_end = getattr(self.cfg, 'rew_scale_foot_path_tracking_iter_end', 15000)
        scale_start = float(getattr(self.cfg, 'rew_scale_foot_path_tracking_start', 5.0))
        scale_end = float(getattr(self.cfg, 'rew_scale_foot_path_tracking_end', 0.5))

        if iter_cur < iter_start:
            return scale_start
        if iter_cur >= iter_end:
            return scale_end
        t = (iter_cur - iter_start) / max(1, iter_end - iter_start)
        return scale_start + t * (scale_end - scale_start)

    def _get_rewards(self) -> torch.Tensor:
        """踩点、动态步态引导（滞空/离地/距离吸引）、稳定性/平滑正则（含宇树式 hip_pos）；可选奖励课程按 iteration 屏蔽部分项。"""
        root_quat = self.robot.data.root_quat_w
        root_ang_vel_w = self.robot.data.root_ang_vel_w
        (roll_angle, pitch_angle, _) = euler_xyz_from_quat(root_quat)
        roll_angle = torch.atan2(torch.sin(roll_angle), torch.cos(roll_angle))
        pitch_angle = torch.atan2(torch.sin(pitch_angle), torch.cos(pitch_angle))
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
        self._check_and_generate_targets()
        foot_path_tracking_reward = torch.zeros(num_envs, device=self.device)
        is_swing_foot_in_air = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        env_indices = torch.arange(num_envs, device=self.device)
        touchdown_event = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        has_target = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        if self._foot_target_positions is not None:
            has_target = torch.any(self._foot_target_positions != 0, dim=1)
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
            if self._target_foot_indices is not None and has_target_valid and (swing_foot_contact is not None):
                self._swing_foot_lifted = self._swing_foot_lifted | ~swing_foot_contact
                is_swing_foot_in_air = ~swing_foot_contact
                if self._swing_foot_contact_prev is not None:
                    contact_to_no_contact = self._swing_foot_contact_prev & ~swing_foot_contact
                    if self._path_start_updated_for_target is not None and self._swing_foot_path_start is not None and (feet_pos_w is not None) and getattr(self.cfg, 'foot_path_start_on_lift', True):
                        update_start = contact_to_no_contact & ~self._path_start_updated_for_target & has_target
                        if torch.any(update_start):
                            env_ids_update = torch.where(update_start)[0]
                            swing_pos_now = feet_pos_w[env_ids_update, self._target_foot_indices[env_ids_update], :]
                            self._swing_foot_path_start[env_ids_update] = swing_pos_now.clone()
                            current_time_lift = self.episode_length_buf[env_ids_update].float() * self.step_dt
                            if self._path_start_time is not None:
                                self._path_start_time[env_ids_update] = current_time_lift
                            self._path_start_updated_for_target[env_ids_update] = True
                    touchdown_event = ~self._swing_foot_contact_prev & swing_foot_contact
                    self._swing_foot_contact_prev = swing_foot_contact.clone()
                else:
                    touchdown_event = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
            foot_hit_reward = torch.zeros(num_envs, device=self.device)
            if self._target_foot_indices is not None and self._foot_target_positions is not None and (len(feet_body_ids) >= 2) and (feet_pos_w is not None) and has_target_valid:
                swing_foot_pos = feet_pos_w[env_indices, self._target_foot_indices, :]
                ankle_ground_h = float(getattr(self.cfg, 'foot_ankle_ground_height', 0.07))
                target_xy = self._foot_target_positions[:, :2]
                target_z_ground = self._foot_target_positions[:, 2]
                expected_ankle_z = target_z_ground + ankle_ground_h
                d_xy = torch.norm(swing_foot_pos[:, :2] - target_xy, dim=1)
                swing_foot_z = swing_foot_pos[:, 2]
                d_z_err = torch.abs(swing_foot_z - expected_ankle_z)
                z_tolerance = float(getattr(self.cfg, 'foot_target_hit_z_tolerance', 0.04))
                z_contact_slack = float(getattr(self.cfg, 'foot_target_hit_z_contact_slack', 0.02))
                thresh_xy = self._foot_hit_curriculum_radius_xy_m()
                xy_ok = d_xy < thresh_xy
                z_ok_contact = swing_foot_contact & (swing_foot_z <= target_z_ground + ankle_ground_h + z_contact_slack) if swing_foot_contact is not None else torch.zeros(num_envs, dtype=torch.bool, device=self.device)
                z_ok = (d_z_err < z_tolerance) | z_ok_contact
                geometry_ok = xy_ok & z_ok
                # ==========================================================
                # 终极物理判定 (修复版)
                # ==========================================================
                # 1. 物理引擎是否真实感受到了触地反作用力
                is_contact_active = swing_foot_contact if swing_foot_contact is not None else torch.zeros(num_envs, dtype=torch.bool, device=self.device)

                # 2. 基本落地证据：有物理碰撞 + 之前抬过腿（防滑步）
                # 注意：这里不再包含 ~self._foot_land_rewarded，防止提前误锁
                landing_valid = has_target & is_contact_active & self._swing_foot_lifted

                # 3. 核心判定：落地合法 + 几何位置精确 (XY和Z都对)
                hit_target = landing_valid & geometry_ok
                d_eff = torch.sqrt(torch.square(d_xy / max(thresh_xy, 1e-06)) + torch.square(d_z_err / max(ankle_ground_h, 1e-06))).clamp(min=1e-06)
                sigma_hit = getattr(self.cfg, 'foot_hit_sigma', 0.03)
                sigma_eff = sigma_hit / max(thresh_xy, 1e-06)
                hit_reward_base = torch.exp(-0.5 * torch.square(d_eff / sigma_eff))

                # 4. 奖励分发：只有真的中了 && 这一步还没领过，才发奖
                reward_frame = hit_target & ~self._foot_land_rewarded
                foot_hit_reward = torch.where(reward_frame, hit_reward_base, torch.zeros_like(hit_reward_base))

                # 5. 状态锁定：只有在 hit_target 为 True 时才标记 rewarded，防止在路边碰一下地就失效
                if self._foot_land_rewarded is not None:
                    self._foot_land_rewarded = self._foot_land_rewarded | hit_target
                # =====================================================================
                # 核心优化：动态跟步延迟 (Dynamic Follow-up Delay)
                # =====================================================================
                delay_base = float(getattr(self.cfg, 'foot_target_regenerate_delay_s', 0.5))

                # 如果下一步是跟步(is_next_follow为True)，给一个极短的0.05s缓冲让物理引擎结算，然后立刻出脚；
                # 如果双腿已经合拢，准备迈出全新的随机步，则正常停顿 delay_base 秒
                is_next_follow = self._next_is_follow if hasattr(self, '_next_is_follow') else torch.zeros(num_envs, dtype=torch.bool, device=self.device)
                delay_tensor = torch.where(is_next_follow, torch.tensor(0.05, device=self.device), torch.tensor(delay_base, device=self.device))
                skip_tm = self._user_target_mode if self._user_target_mode is not None else torch.zeros(num_envs, dtype=torch.bool, device=self.device)
                # 6. 统一更新 self._target_hit，确保 _check_and_generate_targets 能感知到
                # 无论用户模式还是自动模式，只要中了就置 True
                self._target_hit = self._target_hit | hit_target
                # ==========================================================
                # 刷新计时逻辑
                # ==========================================================
                current_time = self.episode_length_buf.float() * self.step_dt
                if self._target_regenerate_deadline is not None:
                    # 如果是自动模式，且刚踩中且没排期，就排期刷新
                    auto_hit = hit_target & ~skip_tm
                    sched_ready = auto_hit & torch.isnan(self._target_regenerate_deadline)
                    self._target_regenerate_deadline = torch.where(sched_ready, current_time + delay_tensor, self._target_regenerate_deadline)
                if self._last_touchdown_time is not None:
                    self._last_touchdown_time = torch.where(hit_target, current_time, self._last_touchdown_time)
                aerial_thresh = getattr(self.cfg, 'foot_target_aerial_ground_threshold', 0.075)
                target_z = self._foot_target_positions[:, 2]
                aerial_target = (target_z > aerial_thresh) & has_target
                reach_xy_th = getattr(self.cfg, 'foot_target_aerial_reach_xy_threshold', 0.08)
                reach_z_tol = getattr(self.cfg, 'foot_target_aerial_reach_z_tolerance', 0.05)
                dist_to_target_3d = swing_foot_pos - self._foot_target_positions
                d_xy_aerial = torch.norm(dist_to_target_3d[:, :2], dim=1)
                d_z_aerial = torch.abs(dist_to_target_3d[:, 2])
                aerial_reach = is_swing_foot_in_air & aerial_target & (d_xy_aerial < reach_xy_th) & (d_z_aerial < reach_z_tol)
                if self._aerial_hold_start_time is not None:
                    first_reach = aerial_reach & (self._aerial_hold_start_time < -1000000000.0)
                    self._aerial_hold_start_time = torch.where(first_reach, current_time, self._aerial_hold_start_time)
                hold_duration = getattr(self.cfg, 'foot_target_aerial_hold_duration_s', 0.5)
                aerial_hold_done = aerial_target & (self._aerial_hold_start_time > -1000000000.0) & (current_time - self._aerial_hold_start_time >= hold_duration) if self._aerial_hold_start_time is not None else torch.zeros_like(hit_target)
                hit_target = hit_target | aerial_hold_done
                in_hold = aerial_target & (self._aerial_hold_start_time > -1000000000.0) & (current_time - self._aerial_hold_start_time < hold_duration) & is_swing_foot_in_air if self._aerial_hold_start_time is not None else torch.zeros_like(hit_target)
                if self._swing_foot_path_start is not None:
                    start_xy = self._swing_foot_path_start[:, :2]
                    end_xy = self._foot_target_positions[:, :2]
                    initial_dist_xy = torch.norm(end_xy - start_xy, dim=1).clamp(min=0.0001)
                    progress_mode = getattr(self.cfg, 'foot_path_progress_mode', 'time')
                    use_start_on_lift = getattr(self.cfg, 'foot_path_start_on_lift', True)
                    if progress_mode == 'time':
                        t_now = self.episode_length_buf.float() * self.step_dt
                        if use_start_on_lift and self._path_start_time is not None:
                            elapsed = (t_now - self._path_start_time).clamp(min=0.0)
                            path_started = self._path_start_time > float('-inf')
                        else:
                            elapsed = (t_now - self._target_generation_time).clamp(min=0.0)
                            path_started = torch.ones(num_envs, dtype=torch.bool, device=self.device)
                        base_s = getattr(self.cfg, 'foot_path_duration_base_s', 0.55)
                        per_m = getattr(self.cfg, 'foot_path_duration_per_m', 0.8)
                        expected_duration = (base_s + per_m * initial_dist_xy).clamp(min=0.2)
                        t = (elapsed / expected_duration).clamp(0.0, 1.0)
                    else:
                        swing_foot_xy = feet_pos_w[env_indices, self._target_foot_indices, :2]
                        current_dist_to_target_xy = torch.norm(swing_foot_xy - end_xy, dim=1)
                        t = (1.0 - current_dist_to_target_xy / initial_dist_xy).clamp(0.0, 1.0)
                        path_started = torch.ones(num_envs, dtype=torch.bool, device=self.device)
                    base_h = getattr(self.cfg, 'foot_ankle_ground_height', 0.07)
                    sigma = self._foot_path_tracking_sigma_m()
                    aerial_target_path = (target_z > aerial_thresh) & has_target
                    # ==========================================================
                    # 终极仿生轨迹生成器 (Quintic XY + Bezier Z)
                    # ==========================================================
                    dist_min = getattr(self.cfg, 'foot_target_min_distance', 0.25)
                    dist_max = getattr(self.cfg, 'foot_target_max_distance', 0.40)
                    peak_min = getattr(self.cfg, 'foot_path_peak_height_min', 0.10)
                    peak_max = getattr(self.cfg, 'foot_path_peak_height_max', 0.15)
                    dist_ratio = ((initial_dist_xy - dist_min) / (dist_max - dist_min + 1e-06)).clamp(0.0, 1.0)
                    peak_h = peak_min + (peak_max - peak_min) * dist_ratio

                    # 1. XY 轴：五次多项式插值 (Quintic Spline)
                    # 公式：s(t) = 10t^3 - 15t^4 + 6t^5
                    # 效果：t=0 和 t=1 时，一阶导(速度)和二阶导(加速度)均为 0，极度平滑
                    t_xy = 10 * t**3 - 15 * t**4 + 6 * t**5
                    ref_xy = (1 - t_xy).unsqueeze(1) * start_xy + t_xy.unsqueeze(1) * end_xy

                    # 2. Z 轴：三次贝塞尔曲线 (Cubic Bezier)
                    # 公式：P(t) = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3
                    p0_z = torch.full((num_envs, 1), base_h, device=self.device)
                    # 终点 P3_z：区分地面点和空中目标点
                    p3_z = torch.where(aerial_target_path.unsqueeze(1), target_z.unsqueeze(1), p0_z)

                    # 核心魔法：控制点 P1 和 P2
                    # 将它们上拉，形成类似人类跨栏时的“倒 U 型 / 平顶抛物线”
                    pull_factor = 1.6
                    p1_z = p0_z + (peak_h - base_h).unsqueeze(1) * pull_factor
                    p2_z = p3_z + (peak_h - base_h).unsqueeze(1) * pull_factor

                    # 贝塞尔计算
                    t_z = t.unsqueeze(1)
                    inv_t = 1.0 - t_z
                    ref_z = (inv_t**3) * p0_z + 3 * (inv_t**2) * t_z * p1_z + 3 * inv_t * (t_z**2) * p2_z + (t_z**3) * p3_z

                    # 组合成最终的 3D 参考坐标
                    ref_pos = torch.cat([ref_xy, ref_z], dim=1)
                    dist_to_ref = torch.norm(swing_foot_pos - ref_pos, dim=1)
                    raw_reward = torch.exp(-0.5 * torch.square(dist_to_ref / sigma))
                    if use_start_on_lift and self._path_start_time is not None:
                        raw_reward = torch.where(path_started, raw_reward, torch.zeros_like(raw_reward))
                    foot_path_tracking_reward = torch.where(has_target, raw_reward, foot_path_tracking_reward)
                if torch.any(hit_target):
                    self._check_and_generate_targets()
                    self._update_target_markers()
                    self._update_ref_path_trajectory_markers()
        self._initialize_joint_limits()
        joint_limit_violation = torch.zeros(num_envs, device=self.device)
        buffer_width = self.cfg.joint_limit_buffer
        if self._joint_limit_map:
            for (joint_id, (min_limit, max_limit)) in self._joint_limit_map.items():
                q = joint_pos[:, joint_id]
                lower_buffer_start = min_limit - buffer_width
                lower_penalty = torch.where(q >= min_limit, torch.zeros_like(q), torch.where(q >= lower_buffer_start, min_limit - q, min_limit - q))
                upper_buffer_end = max_limit + buffer_width
                upper_penalty = torch.where(q <= max_limit, torch.zeros_like(q), torch.where(q <= upper_buffer_end, q - max_limit, q - max_limit))
                joint_limit_violation += lower_penalty + upper_penalty
        root_height = self.robot.data.root_pos_w[:, 2]
        height_target = self.cfg.height_target
        height_band = self.cfg.height_penalty_band
        height_violation = (torch.abs(root_height - height_target) > height_band).float()
        swing_extra = getattr(self.cfg, 'rew_scale_pitch_roll_swing_extra', 0.0)
        pitch_roll_swing_multiplier = 1.0 + swing_extra * is_swing_foot_in_air.float() * has_target.float()
        scale_ang_vel = getattr(self.cfg, 'rew_scale_pitch_roll_ang_vel', -0.001)
        pitch_roll_ang_vel_sq = torch.square(root_ang_vel_w[:, 0]) + torch.square(root_ang_vel_w[:, 1])
        rew_pitch_roll_ang_vel = scale_ang_vel * pitch_roll_ang_vel_sq
        rew_action_rate = torch.zeros(num_envs, device=self.device)
        if self._prev_actions is not None and hasattr(self, 'actions') and (self.actions is not None):
            action_diff = self.actions - self._prev_actions
            scale_action_rate = getattr(self.cfg, 'rew_scale_action_rate', -0.0001)
            rew_action_rate = scale_action_rate * torch.sum(torch.square(action_diff), dim=1)
        rew_contact_no_vel = torch.zeros(num_envs, device=self.device)
        if len(feet_body_ids) >= 2:
            fc = self.get_feet_contact_state()
            if fc is not None:
                body_lin_vel_w = self.robot.data.body_lin_vel_w
                # 严格对齐宇树：惩罚 3D 接触速度 (包含 Z 轴弹跳)
                feet_vel_xyz = body_lin_vel_w[:, feet_body_ids[:2], :3]
                contact_expanded = fc.unsqueeze(-1)
                contact_vel_sq = torch.sum(torch.square(feet_vel_xyz * contact_expanded.float()), dim=(1, 2))
                scale_contact_no_vel = getattr(self.cfg, 'rew_scale_contact_no_vel', -2.0)
                rew_contact_no_vel = scale_contact_no_vel * contact_vel_sq
        self._initialize_hip_pos_penalty_joints()
        rew_hip_pos = torch.zeros(num_envs, device=self.device)
        if self._hip_pos_joint_ids is not None and self._hip_pos_joint_ids.numel() > 0:
            q_hip = joint_pos[:, self._hip_pos_joint_ids]
            scale_hip = float(getattr(self.cfg, 'rew_scale_hip_pos', -0.5))
            rew_hip_pos = scale_hip * torch.sum(torch.square(q_hip), dim=1)
        # 获取当前 step 对应的动态 tracking 权重
        current_tracking_scale = self._foot_path_tracking_scale_curriculum()

        reward_result = compute_rewards(
            self.cfg.rew_scale_pitch_roll_angle,
            self.cfg.rew_scale_height,
            self.cfg.rew_scale_joint_velocity,
            self.cfg.rew_scale_joint_acceleration,
            self.cfg.rew_scale_foot_hit,
            current_tracking_scale,  # 使用动态退火权重替换原有的 self.cfg.rew_scale_foot_path_tracking
            self.cfg.rew_scale_joint_limit,
            pitch_angle,
            roll_angle,
            joint_vel,
            joint_acc,
            foot_hit_reward,
            foot_path_tracking_reward,
            joint_limit_violation,
            height_violation,
            pitch_roll_swing_multiplier,
            rew_pitch_roll_ang_vel,
            rew_action_rate,
            rew_contact_no_vel,
            rew_hip_pos,
        )
        (total_reward, rew_pitch_roll_angle, rew_height, rew_joint_velocity, rew_joint_acceleration, rew_foot_hit, rew_foot_path_tracking, rew_joint_limit) = reward_result
        curriculum = getattr(self.cfg, 'reward_curriculum', None)
        active_components = None
        if curriculum and len(curriculum) > 0:
            step = getattr(self, 'common_step_counter', 0)
            mode = getattr(self.cfg, 'reward_curriculum_mode', 'step')
            steps_per_iter = getattr(self.cfg, 'reward_curriculum_steps_per_iteration', 24)
            active = self._reward_curriculum_components(step, curriculum, mode=mode, steps_per_iter=steps_per_iter)
            if active:
                active = self._expand_deprecated_reward_curriculum_components(active)
            active_components = active
            if active:
                comp = {'rew_pitch_roll_angle': rew_pitch_roll_angle, 'rew_pitch_roll_ang_vel': rew_pitch_roll_ang_vel, 'rew_height': rew_height, 'rew_joint_velocity': rew_joint_velocity, 'rew_joint_acceleration': rew_joint_acceleration, 'rew_foot_hit': rew_foot_hit, 'rew_foot_path_tracking': rew_foot_path_tracking, 'rew_joint_limit': rew_joint_limit, 'rew_action_rate': rew_action_rate, 'rew_contact_no_vel': rew_contact_no_vel, 'rew_hip_pos': rew_hip_pos}
                total_reward = sum((comp[n] for n in active if n in comp))
            elif active is not None:
                total_reward = torch.zeros_like(rew_pitch_roll_angle)
        if hasattr(self, '_episode_reward_sums'):

            def _add_if_active(name: str, value: torch.Tensor) -> None:
                if active_components is None or (active_components and name in active_components):
                    self._episode_reward_sums[name] += value
                else:
                    self._episode_reward_sums[name] += torch.zeros_like(value)
            _add_if_active('rew_pitch_roll_angle', rew_pitch_roll_angle)
            _add_if_active('rew_pitch_roll_ang_vel', rew_pitch_roll_ang_vel)
            _add_if_active('rew_height', rew_height)
            _add_if_active('rew_joint_velocity', rew_joint_velocity)
            _add_if_active('rew_joint_acceleration', rew_joint_acceleration)
            _add_if_active('rew_foot_hit', rew_foot_hit)
            _add_if_active('rew_foot_path_tracking', rew_foot_path_tracking)
            _add_if_active('rew_joint_limit', rew_joint_limit)
            _add_if_active('rew_action_rate', rew_action_rate)
            _add_if_active('rew_contact_no_vel', rew_contact_no_vel)
            _add_if_active('rew_hip_pos', rew_hip_pos)
        self._check_and_generate_targets()
        self._update_target_markers()
        self._update_ref_path_trajectory_markers()
        return total_reward

    def get_reward_breakdown(self) -> dict:
        if not hasattr(self, '_last_episode_reward_means'):
            return {'error': '奖励统计尚未初始化'}
        total = sum(self._last_episode_reward_means.values())
        percentages = {}
        for (name, value) in self._last_episode_reward_means.items():
            if abs(total) > 1e-06:
                percentages[name] = value / total * 100
            else:
                percentages[name] = 0.0
        return {'components': self._last_episode_reward_means.copy(), 'percentages': percentages, 'total': total}

    def print_reward_breakdown(self):
        breakdown = self.get_reward_breakdown()
        if 'error' in breakdown:
            print(f"[WARNING] {breakdown['error']}")
            return
        print('\n' + '=' * 60)
        print('奖励分量统计（Episode平均值）')
        print('=' * 60)
        print(f"{'奖励分量':<30} {'平均值':>12} {'占比':>10}")
        print('-' * 60)
        sorted_items = sorted(breakdown['components'].items(), key=lambda x: abs(x[1]), reverse=True)
        for (name, value) in sorted_items:
            pct = breakdown['percentages'].get(name, 0.0)
            sign = '+' if value >= 0 else ''
            print(f'{name:<30} {sign}{value:>11.2f} {pct:>9.1f}%')
        print('-' * 60)
        print(f"{'总计':<30} {breakdown['total']:>12.2f} {'100.0':>9}%")
        print('=' * 60 + '\n')

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        root_height = self.robot.data.root_pos_w[:, 2]
        fallen = root_height < self.cfg.reset_root_height * 0.5
        return (fallen, time_out)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if hasattr(self, '_episode_reward_sums') and len(env_ids) > 0:
            env_ids_tensor = torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
            curriculum = getattr(self.cfg, 'reward_curriculum', None)
            active_components = None
            if curriculum and len(curriculum) > 0:
                step = getattr(self, 'common_step_counter', 0)
                mode = getattr(self.cfg, 'reward_curriculum_mode', 'step')
                steps_per_iter = getattr(self.cfg, 'reward_curriculum_steps_per_iteration', 24)
                active_components = self._reward_curriculum_components(step, curriculum, mode=mode, steps_per_iter=steps_per_iter)
                if active_components:
                    active_components = self._expand_deprecated_reward_curriculum_components(active_components)
            total_sum = 0.0
            for name in self._reward_components:
                mean_val = torch.mean(self._episode_reward_sums[name][env_ids_tensor]).item()
                self._last_episode_reward_means[name] = mean_val
                if active_components is None or (active_components and name in active_components):
                    total_sum += mean_val
            if not hasattr(self, 'extras') or self.extras is None:
                self.extras = {}
            if 'log' not in self.extras:
                self.extras['log'] = {}
            for name in self._reward_components:
                mean_val = self._last_episode_reward_means[name]
                if active_components is not None and active_components and (name not in active_components):
                    mean_val = 0.0
                self.extras['log'][f'reward/{name}'] = mean_val
            self.extras['log']['reward/total'] = total_sum
            for name in self._reward_components:
                self._episode_reward_sums[name][env_ids_tensor] = 0.0
        super()._reset_idx(env_ids)
        if hasattr(self, '_persistent_feet_contact'):
            env_ids_t = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._persistent_feet_contact[env_ids_t] = False
        if self._ankle_contact_consecutive_count is not None and len(env_ids) > 0:
            env_ids_t = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._ankle_contact_consecutive_count[env_ids_t] = 0
        self._initialize_locked_joints()
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        noise = sample_uniform(-0.02, 0.02, joint_pos.shape, joint_pos.device)
        if joint_pos.shape[1] == len(self.robot.joint_names):
            leg_substr = ('hip_', 'knee_', 'ankle_', 'torso_joint')
            for (j, name) in enumerate(self.robot.joint_names):
                if any((s in name for s in leg_substr)):
                    noise[:, j] = 0.0
        joint_pos += noise
        if self._locked_joint_ids and len(self._locked_joint_ids) > 0:
            default_pos = self.robot.data.default_joint_pos[env_ids].clone()
            apply_locked_joint_targets(self.robot, joint_pos, default_pos, self._locked_joint_ids)
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        default_root_state[:, 2] = self.cfg.reset_root_height
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
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        if self._prev_joint_pos_target is not None:
            self._prev_joint_pos_target[env_ids] = joint_pos
        if self._foot_target_positions is not None:
            env_ids_tensor = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._target_generation_time[env_ids_tensor] = float('-inf')
            self._target_hit[env_ids_tensor] = False
            if self._target_regenerate_deadline is not None:
                self._target_regenerate_deadline[env_ids_tensor] = float('nan')
            if self._swing_air_accum_s is not None:
                self._swing_air_accum_s[env_ids_tensor] = 0.0
        if self._foot_land_rewarded is not None:
            self._foot_land_rewarded[env_ids] = False
        if self._swing_foot_contact_prev is not None:
            self._swing_foot_contact_prev[env_ids] = True
        if self._swing_foot_lifted is not None:
            self._swing_foot_lifted[env_ids] = False
        if self._ankle_contact_consecutive_count is not None:
            self._ankle_contact_consecutive_count[env_ids, :] = 0
        if self._target_foot_indices is not None:
            self._target_foot_indices[env_ids] = 0
        if self._last_touchdown_time is not None:
            self._last_touchdown_time[env_ids] = float('-inf')
        if self._swing_foot_path_start is not None:
            self._swing_foot_path_start[env_ids] = 0.0
        if self._path_start_time is not None:
            self._path_start_time[env_ids] = float('-inf')
        if self._path_start_updated_for_target is not None:
            self._path_start_updated_for_target[env_ids] = False
        if self._aerial_hold_start_time is not None:
            self._aerial_hold_start_time[env_ids] = float('-inf')
        if hasattr(self, '_next_is_follow'):
            self._next_is_follow[env_ids] = False
        if getattr(self.cfg, 'event_curriculum_enabled', False) and self._event_next_push_at is not None:
            self._apply_event_curriculum_mass(env_ids)
            self._reschedule_event_push_times_on_reset(env_ids)
        ref_need_pose = getattr(self.cfg, 'ref_path_velocity_command_enabled', False) or getattr(self.cfg, 'ref_path_visualization_enabled', False)
        env_ids_tensor = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ref_need_pose and self._ref_path_psi0 is not None and (self._ref_path_origin_xy is not None) and env_ids_tensor.numel() > 0:
            dq = default_root_state[:, 3:7]
            (_, _, yaw0) = euler_xyz_from_quat(dq)
            self._ref_path_psi0[env_ids_tensor] = yaw0
            self._ref_path_origin_xy[env_ids_tensor] = default_root_state[:, :2].clone()
        if getattr(self.cfg, 'ref_path_velocity_command_enabled', False) and self._ref_path_progress_m is not None and env_ids_tensor.numel() > 0:
            self._ref_path_progress_m[env_ids_tensor] = 0.0

@torch.jit.script
def compute_rewards(rew_scale_pitch_roll_angle: float, rew_scale_height: float, rew_scale_joint_velocity: float, rew_scale_joint_acceleration: float, rew_scale_foot_hit: float, rew_scale_foot_path_tracking: float, rew_scale_joint_limit: float, pitch_angle: torch.Tensor, roll_angle: torch.Tensor, joint_vel: torch.Tensor, joint_acc: torch.Tensor, foot_hit: torch.Tensor, foot_path_tracking_reward: torch.Tensor, joint_limit_violation: torch.Tensor, height_violation: torch.Tensor, pitch_roll_swing_multiplier: torch.Tensor, rew_pitch_roll_ang_vel: torch.Tensor, rew_action_rate: torch.Tensor, rew_contact_no_vel: torch.Tensor, rew_hip_pos: torch.Tensor):
    rew_height = rew_scale_height * height_violation
    pitch_roll_angle_penalty = torch.square(pitch_angle) + torch.square(roll_angle)
    joint_acceleration_penalty = torch.sum(torch.square(joint_acc), dim=1)
    rew_pitch_roll_angle = rew_scale_pitch_roll_angle * pitch_roll_angle_penalty * pitch_roll_swing_multiplier
    rew_joint_acceleration = rew_scale_joint_acceleration * joint_acceleration_penalty
    rew_joint_velocity = rew_scale_joint_velocity * torch.sum(torch.square(joint_vel), dim=1)
    rew_foot_hit = rew_scale_foot_hit * foot_hit
    rew_foot_path_tracking = rew_scale_foot_path_tracking * foot_path_tracking_reward
    rew_joint_limit = rew_scale_joint_limit * joint_limit_violation
    total_reward = rew_height + rew_pitch_roll_angle + rew_pitch_roll_ang_vel + rew_joint_velocity + rew_joint_acceleration + rew_foot_hit + rew_foot_path_tracking + rew_joint_limit + rew_action_rate + rew_contact_no_vel + rew_hip_pos
    return (total_reward, rew_pitch_roll_angle, rew_height, rew_joint_velocity, rew_joint_acceleration, rew_foot_hit, rew_foot_path_tracking, rew_joint_limit)
