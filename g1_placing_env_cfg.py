from pathlib import Path
from isaaclab_assets.robots.unitree import G1_CFG
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
_UNITREE_PD_LEGS = G1_CFG.actuators['legs'].replace(stiffness={'.*_hip_yaw_joint': 100.0, '.*_hip_roll_joint': 100.0, '.*_hip_pitch_joint': 100.0, '.*_knee_joint': 150.0, 'torso_joint': 150.0}, damping={'.*_hip_yaw_joint': 2.0, '.*_hip_roll_joint': 2.0, '.*_hip_pitch_joint': 2.0, '.*_knee_joint': 4.0, 'torso_joint': 4.0})
_HOST_PD_FEET = G1_CFG.actuators['feet'].replace(stiffness=40.0, damping=20.0)
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
    observation_space = 106
    action_space = 37
    policy_observation_enabled = True
    state_space = 0
    force_zero_action = False
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    robot_cfg: ArticulationCfg = G1_CFG.replace(prim_path='/World/envs/env_.*/Robot', spawn=G1_CFG.spawn.replace(usd_path=_G1_USD_PATH, rigid_props=G1_CFG.spawn.rigid_props.replace(max_depenetration_velocity=10.0), articulation_props=G1_CFG.spawn.articulation_props.replace(solver_position_iteration_count=8, solver_velocity_iteration_count=4), collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0)), actuators=_HOST_PD_ACTUATORS)
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
    rew_scale_pitch_roll_angle = -0.5
    rew_scale_pitch_roll_ang_vel = -0.001
    rew_scale_pitch_roll_swing_extra = 0.5
    rew_scale_height = -5.0
    height_target = 0.74
    height_penalty_band = 0.05
    rew_scale_joint_velocity = -0.001
    rew_scale_joint_acceleration = -1e-07
    rew_scale_action_rate = -0.001
    rew_scale_action_smoothness = -1e-4
    rew_scale_contact_no_vel = -0.05
    rew_scale_hip_pos = -2.0
    rew_scale_joint_limit = -1.0
    joint_limit_buffer = 0.1
    rew_scale_foot_hit = 50.0
    rew_scale_foot_path_tracking = 5.0
    rew_scale_foot_hold = 2.0
    foot_hit_sigma = 0.03
    foot_ankle_ground_height = 0.07
    foot_path_peak_height_min = 0.10
    foot_path_peak_height_max = 0.14
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
    foot_contact_force_threshold: float = 1.0
    foot_contact_height_threshold = 0.075
    foot_contact_consecutive_frames = 2
    foot_contact_velocity_threshold = 0.08
    foot_target_hit_threshold = 0.08
    foot_target_hit_threshold_end = 0.03
    foot_target_hit_threshold_iter_start = 3000
    foot_target_hit_threshold_iter_end = 8000
    foot_target_hit_z_tolerance = 0.04
    foot_target_hit_z_contact_slack = 0.02
    foot_target_hit_z_max = 0.09
    foot_target_hit_min_elapsed_s = 0.05
    foot_target_regenerate_delay_s = 0.5
    foot_target_rect_x_forward = 0.3
    foot_target_rect_x_back = 0.12
    foot_target_rect_y_outward = 0.15
    foot_target_rect_y_inward = 0.02
    foot_target_min_distance = 0.1
    foot_target_max_distance = 0.25
    foot_spacing_stand_m = 0.24
    reset_root_height = 0.74
    reward_curriculum_mode = 'iteration'
    reward_curriculum_steps_per_iteration = 24
    reward_curriculum = [{'iter_start': 0, 'iter_end': 3000, 'components': ['rew_pitch_roll_angle', 'rew_pitch_roll_ang_vel', 'rew_height', 'rew_joint_velocity', 'rew_joint_acceleration', 'rew_joint_limit', 'rew_action_rate', 'rew_action_smoothness', 'rew_contact_no_vel', 'rew_hip_pos']}, {'iter_start': 3000, 'iter_end': None, 'components': ['rew_pitch_roll_angle', 'rew_pitch_roll_ang_vel', 'rew_height', 'rew_joint_velocity', 'rew_joint_acceleration', 'rew_joint_limit', 'rew_foot_hit', 'rew_foot_path_tracking', 'rew_foot_hold', 'rew_action_rate', 'rew_action_smoothness', 'rew_contact_no_vel', 'rew_hip_pos']}]
