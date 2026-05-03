from pathlib import Path
from isaaclab_assets.robots.unitree import G1_CFG
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
_UNITREE_PD_LEGS = G1_CFG.actuators['legs'].replace(stiffness={'.*_hip_yaw_joint': 100.0, '.*_hip_roll_joint': 100.0, '.*_hip_pitch_joint': 100.0, '.*_knee_joint': 150.0, 'torso_joint': 150.0}, damping={'.*_hip_yaw_joint': 2.0, '.*_hip_roll_joint': 2.0, '.*_hip_pitch_joint': 2.0, '.*_knee_joint': 4.0, 'torso_joint': 4.0})
_HOST_PD_FEET = G1_CFG.actuators['feet'].replace(stiffness=40.0, damping=2.0)
_HOST_PD_ARMS = G1_CFG.actuators['arms'].replace(stiffness=100.0, damping=20.0)
_HOST_PD_ACTUATORS = {'legs': _UNITREE_PD_LEGS, 'feet': _HOST_PD_FEET, 'arms': _HOST_PD_ARMS}
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
_LOCAL_G1_USD = Path(__file__).resolve().parents[5] / 'assets' / 'g1_cad' / 'g1.usd'
_G1_USD_PATH = str(_LOCAL_G1_USD) if _LOCAL_G1_USD.is_file() else G1_CFG.spawn.usd_path

@configclass
class G1PlacingEnvCfg(DirectRLEnvCfg):
    """G1机器人踩点环境配置"""
    decimation = 2
    episode_length_s = 15.0
    # 纯净的 106 维本体感受 + 目标相对坐标 (去除了参考速度)
    observation_space = 106
    action_space = 37
    policy_observation_enabled = True
    state_space = 0
    force_zero_action = False
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    robot_cfg: ArticulationCfg = G1_CFG.replace(
        prim_path='/World/envs/env_.*/Robot',
        spawn=G1_CFG.spawn.replace(
            usd_path=_G1_USD_PATH,
            rigid_props=G1_CFG.spawn.rigid_props.replace(max_depenetration_velocity=10.0),
            articulation_props=G1_CFG.spawn.articulation_props.replace(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        actuators=_HOST_PD_ACTUATORS,
        init_state=G1_CFG.init_state.replace(
            joint_pos={
                **dict(G1_CFG.init_state.joint_pos),
                # 微蹲腿部默认（覆盖 G1 原站立腿角）；其余关节保留官方 cfg
                '.*_hip_yaw_joint': 0.0,
                '.*_hip_roll_joint': 0.0,
                '.*_hip_pitch_joint': -0.1,
                '.*_knee_joint': 0.3,
                '.*_ankle_pitch_joint': -0.2,
                '.*_ankle_roll_joint': 0.0,
                'torso_joint': 0.0,
            }
        ),
    )
    ground_plane_cfg: sim_utils.GroundPlaneCfg = sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=2.0, replicate_physics=True)
    events = None
    event_curriculum_enabled: bool = False
    event_curriculum_steps_per_iteration: int = 24
    event_curriculum_base_iteration: int = 0
    event_push_interval_range_s: tuple[float, float] = (2.0, 4.0)
    event_torso_body_names: tuple[str, ...] = ('^left_elbow_roll_link$', '^right_elbow_roll_link$')
    event_curriculum: tuple = ({'iter_start': 0, 'iter_end': 5000, 'mass_add_kg': (0.5, 1.5), 'push_enabled': False}, {'iter_start': 5000, 'iter_end': 10000, 'mass_add_kg': (1.0, 2.5), 'push_enabled': True, 'push_xy': 0.15, 'push_z': 0.05}, {'iter_start': 10000, 'iter_end': None, 'mass_add_kg': (0.5, 2.5), 'push_enabled': True, 'push_xy': 0.25, 'push_z': 0.05})
    action_scale = 0.25
    action_rate_limit_rad_per_step = 0.05
    torso_joint_limit_rad = 0.35
    soft_dof_pos_limit = 0.9
    # ---- 1. 解除姿态封印，允许身体前倾 (治僵硬) ----
    rew_scale_pitch_roll_angle = -1.0  # (原 -5.0) 对齐宇树 orientation
    rew_scale_pitch_roll_ang_vel = -0.05  # (原 -0.001) 对齐宇树的 ang_vel_xy
    rew_scale_pitch_roll_swing_extra = 0.0  # (原 1.0) 废弃此惩罚，允许单腿支撑时自然倾斜
    rew_scale_height = -10.0  # (原 -5.0) 对齐宇树的 base_height
    height_target = 0.74
    height_penalty_band = 0.10
    rew_scale_joint_velocity = -0.001
    # ---- 2. 重锤打击高频抖动 (治抽搐) ----
    rew_scale_joint_acceleration = -2.5e-7  # (原 -1e-07) 对齐宇树 dof_acc
    rew_scale_action_rate = -0.01  # (原 -0.001) 对齐宇树 action_rate
    # ---- 3. 其他对齐项 ----
    rew_scale_contact_no_vel = -0.2  # (原 -0.05) 对齐宇树 contact_no_vel
    rew_scale_hip_pos = -1.0  # (原 -2.0) 对齐宇树 hip_pos
    rew_scale_joint_limit = -5.0  # (原 -1.0) 对齐宇树的 dof_pos_limits
    joint_limit_buffer = 0.1
    # 极大幅度提高命中奖励，用重赏打破机器人悬空不踩的局部最优
    rew_scale_foot_hit = 200.0
    # 轨迹追踪奖励权重：前期高权重辅助引导，后期线性退火至极低值，逼迫网络独立行走
    rew_scale_foot_path_tracking_start = 5.0
    rew_scale_foot_path_tracking_end = 0.5
    rew_scale_foot_path_tracking_iter_start = 8000
    rew_scale_foot_path_tracking_iter_end = 15000
    rew_scale_foot_hold = 2.0
    foot_hit_sigma = 0.03
    foot_ankle_ground_height = 0.07
    foot_path_peak_height_min = 0.10
    foot_path_peak_height_max = 0.10
    foot_path_tracking_sigma = 0.025
    foot_path_tracking_sigma_start = 0.075
    foot_path_tracking_sigma_end = 0.025
    foot_path_tracking_sigma_iter_start = 3000
    foot_path_tracking_sigma_iter_end = 13000
    foot_path_progress_mode: str = 'time'
    foot_path_duration_base_s = 0.55
    foot_path_duration_per_m = 0.8
    foot_path_start_on_lift = True
    foot_path_lift_phase_ratio = 0.35
    foot_path_extend_phase_ratio = 0.75
    foot_path_lift_height = 0.1
    foot_target_aerial_ground_threshold = 0.075
    foot_target_aerial_hold_duration_s = 0.5
    foot_target_aerial_reach_xy_threshold = 0.08
    foot_target_aerial_reach_z_tolerance = 0.05
    foot_hold_sigma = 0.04
    com_over_support_sigma = 0.03
    penetration_threshold = 0.02
    foot_contact_use_physics_sensor: bool = True
    # ---- 4. 放宽判定期，逼迫大步幅 (治原地踏步与蜻蜓点水) ----
    foot_contact_touchdown_force: float = 30.0  # (原 50.0) 降低力控要求
    foot_contact_liftoff_force: float = 10.0  # (原 20.0)
    foot_contact_height_threshold = 0.075
    foot_contact_consecutive_frames = 2
    foot_contact_velocity_threshold = 0.08
    foot_target_hit_threshold = 0.08
    foot_target_hit_threshold_end = 0.08  # (原 0.03) 彻底放宽命中圈
    foot_target_hit_threshold_iter_start = 3000
    foot_target_hit_threshold_iter_end = 8000
    foot_target_hit_z_tolerance = 0.05
    foot_target_hit_z_contact_slack = 0.05
    foot_target_hit_z_max = 0.09
    foot_target_hit_min_elapsed_s = 0.05
    foot_target_regenerate_delay_s = 0.5
    foot_target_rect_x_forward = 0.35  # (原 0.12) 逼迫大步幅，把重心扔出去
    foot_target_rect_x_back = 0.05
    foot_target_rect_y_outward = 0.15
    foot_target_rect_y_inward = 0.02
    foot_target_min_distance = 0.1
    foot_target_max_distance = 0.40  # (原 0.15) 允许 Planner 指示更远的点
    foot_spacing_stand_m = 0.24
    reset_root_height = 0.74
    reward_curriculum_mode = 'iteration'
    reward_curriculum_steps_per_iteration = 24
    # 阶段1：iter 0..3499 仅正则；阶段2：iter >=3500 加入踩点与轨迹跟踪
    reward_curriculum = [{'iter_start': 0, 'iter_end': 3500, 'components': ['rew_pitch_roll_angle', 'rew_pitch_roll_ang_vel', 'rew_height', 'rew_joint_velocity', 'rew_joint_acceleration', 'rew_joint_limit', 'rew_action_rate', 'rew_contact_no_vel', 'rew_hip_pos']}, {'iter_start': 3500, 'iter_end': None, 'components': ['rew_pitch_roll_angle', 'rew_pitch_roll_ang_vel', 'rew_height', 'rew_joint_velocity', 'rew_joint_acceleration', 'rew_joint_limit', 'rew_foot_hit', 'rew_foot_path_tracking', 'rew_action_rate', 'rew_contact_no_vel', 'rew_hip_pos']}]
    # 参考路径：驱动弧长累计与路径驱动落点；期望速度不再进入 policy 观测（关闭则进度不推进且内部速度指令为 0）
    ref_path_velocity_command_enabled: bool = True
    ref_path_speed_m_s: float = 0.5
    ref_path_straight_m: float = 5.0  # 地面平面内沿重置航向的直线段长度 [m]
    ref_path_quarter_arc_length_m: float = 10.0  # 紧随其后的 90° 圆弧的弧长 [m]（非半径；R=2L/π）
    ref_path_turn_left: bool = True
    # 视口绘制上述**贴地**参考路径（橙）；USD 带状网格连续路面；与青色「抬脚空间曲线」不同
    ref_path_visualization_enabled: bool = True
    ref_path_visualize_env_id: int = 0
    ref_path_visualize_z_offset_m: float = 0.005  # 相对 env 地面略抬高，减轻与地片 z-fight
    ref_path_visualize_n_straight: int = 48
    ref_path_visualize_n_arc: int = 36
    ref_path_visualize_road_half_width_m: float = 0.15  # 路面半宽（m），总宽约 2× 该值
    # Raibert 式路径驱动落脚点：为 True 时在非用户模式下用期望速度分解落点；play/测试时与 ref_path 一起开
    path_driven_target_enabled: bool = True
    path_driven_raibert_kv: float = 0.05
    path_driven_prob: float = 0.0  # 0.0：纯随机点位；1.0 为路径驱动落点
