from pathlib import Path

from isaaclab_assets.robots.unitree import G1_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

# 宇树官方轻量化 PD 增益 (适用于空载纯走路/步态训练)
_UNITREE_PD_LEGS = G1_CFG.actuators["legs"].replace(
    stiffness={
        ".*_hip_yaw_joint": 100.0,
        ".*_hip_roll_joint": 100.0,
        ".*_hip_pitch_joint": 100.0,
        ".*_knee_joint": 150.0,
        "torso_joint": 150.0,
    },
    damping={
        ".*_hip_yaw_joint": 2.0,
        ".*_hip_roll_joint": 2.0,
        ".*_hip_pitch_joint": 2.0,
        ".*_knee_joint": 4.0,
        "torso_joint": 4.0,
    },
)
_HOST_PD_FEET = G1_CFG.actuators["feet"].replace(stiffness=40.0, damping=20.0)
_HOST_PD_ARMS = G1_CFG.actuators["arms"].replace(stiffness=100.0, damping=20.0)
_HOST_PD_ACTUATORS = {"legs": _UNITREE_PD_LEGS, "feet": _HOST_PD_FEET, "arms": _HOST_PD_ARMS}

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg

_LOCAL_G1_USD = Path(__file__).resolve().parents[5] / "assets" / "g1_cad" / "g1.usd"
_G1_USD_PATH = str(_LOCAL_G1_USD) if _LOCAL_G1_USD.is_file() else G1_CFG.spawn.usd_path


@configclass
class G1PlacingEnvCfg(DirectRLEnvCfg):
    """G1机器人踩点环境配置"""
    
    # 环境基本参数
    decimation = 2  # 动作更新频率（每N个物理步更新一次）
    episode_length_s = 15.0  # Episode长度（秒）
    
    # 观察空间和动作空间
    observation_space = 106  # 观察空间维度：root(13) + joints(74) + target(3) + feet(16)=接触(2)+位置(6)+朝向(8)
    action_space = 37  # 动作空间维度（37个驱动关节）
    policy_observation_enabled = True  # True=打开训练用；False=暂时关闭 policy 观测（检修用）
    state_space = 0  # 状态空间维度（不使用）
    force_zero_action = False  # True 时每步 action=0，仅看 PD 跟踪默认姿态效果（调试用）
    
    # 仿真参数
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,  # 物理步长（120Hz）
        render_interval=decimation
    )
    
    # 机器人配置（增强碰撞检测精度，防止穿模）；腿部 PD 使用宇树官方轻量化增益，RL 动作语义不变
    robot_cfg: ArticulationCfg = G1_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=G1_CFG.spawn.replace(
            usd_path=_G1_USD_PATH,
            rigid_props=G1_CFG.spawn.rigid_props.replace(
                max_depenetration_velocity=10.0,  # 提高去穿透速度，防止脚部卡地
            ),
            articulation_props=G1_CFG.spawn.articulation_props.replace(
                solver_position_iteration_count=8,  # 提高位置求解精度
                solver_velocity_iteration_count=4,  # 速度求解迭代次数
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,  # 提前 5mm 触发碰撞检测，减少穿模
                rest_offset=0.0,
            ),
        ),
        actuators=_HOST_PD_ACTUATORS,
    )

    # 地面配置（摩擦系数参考 HoST G1：0.5 偏低易打滑，0.8/0.7 更稳）
    ground_plane_cfg: sim_utils.GroundPlaneCfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.7,
        ),
    )

    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,  # 并行环境数（训练 10000 iteration 用 2048）
        env_spacing=2.0,  # 环境间距（米）
        replicate_physics=True  # 共享物理属性
    )

    # 物理干扰课程：由 G1PlacingEnv 内根据训练 iteration 调用 mdp（不挂 EventManager，避免开局地狱难度）
    events = None

    # True：启用「躯干负重 + 定时推力」课程；False：关闭，与未加干扰时一致
    event_curriculum_enabled: bool = False
    # 与 PPO num_steps_per_env 一致：common_step_counter // 该值 = 当前训练 iteration（与 reward_curriculum 一致）
    event_curriculum_steps_per_iteration: int = 24
    # 续训时：若 runner 恢复了与 checkpoint 一致的全局 iteration（common_step_counter≈iter×24），设为 checkpoint 的 iteration，
    # 则有效 iteration = max(0, 当前−base)，使课程从阶段 1 重新跑。新进程且 counter 从 0 开始则保持 0。
    event_curriculum_base_iteration: int = 0
    # 两次推力之间的间隔（秒），按环境独立采样
    event_push_interval_range_s: tuple[float, float] = (2.0, 4.0)
    # 附加质量作用的 body 名称正则（与 randomize_rigid_body_mass 一致）；原先为 (".*torso.*",)。
    # 升级为高保真物理（将附加质量加在小臂/肘侧，贴近抬板载荷；推力仍对整机根施加，见 env）。
    # 灵巧手 USD（与单机任务一致）body 名为 left_elbow_roll_link / right_elbow_roll_link；与 g1_23dof.urdf 的 left_elbow_link 不同。
    event_torso_body_names: tuple[str, ...] = (
        "^left_elbow_roll_link$",
        "^right_elbow_roll_link$",
    )
    # 单机抗干扰特训：与三机抬板约 1kg/人载荷对齐，降低过大负重与推力导致策略崩溃或僵直步态的风险
    event_curriculum: tuple = (
        # 阶段 1 [0, 5000)：适应基础负重。模拟 1kg 平分重量及轻微动态起伏。无推力。
        {"iter_start": 0, "iter_end": 5000, "mass_add_kg": (0.5, 1.5), "push_enabled": False},

        # 阶段 2 [5000, 10000)：引入轻微拉扯。质量上限放宽至 2.5kg（模拟重心偏移），加入模拟队友步态不同步的轻微拉扯（±0.15 m/s）。
        {"iter_start": 5000, "iter_end": 10000, "mass_add_kg": (1.0, 2.5), "push_enabled": True, "push_xy": 0.15, "push_z": 0.05},

        # 阶段 3 [10000, ∞)：实战极限抗压。质量下限放宽，推力提升至极限值（±0.25 m/s），训练抗极端协同失误的能力。
        {"iter_start": 10000, "iter_end": None, "mass_add_kg": (0.5, 2.5), "push_enabled": True, "push_xy": 0.25, "push_z": 0.05},
    )

    # 动作缩放
    action_scale = 0.25  # 动作值范围：[-0.25, 0.25] 弧度

    # 关节目标变化率限制（抑制高频运动）：每步关节目标相对上步的最大变化（弧度），0=不限制
    action_rate_limit_rad_per_step = 0.02  # 约 1.1°/step，decimation=2 时约 0.6 rad/s

    torso_joint_limit_rad = 0.35  # 躯干关节（名称含 torso）奖励限位 ±rad；髋类限位见 URDF+soft_dof_pos_limit
    # 与 unitree_rl_gym ``g1_config.py`` 的 ``rewards.soft_dof_pos_limit`` 一致：在 URDF 全行程内缩窄后再计越界惩罚
    soft_dof_pos_limit = 0.9

    # 奖励权重 - 稳定性组（学习阶段降低惩罚，鼓励探索）
    rew_scale_pitch_roll_angle = -0.5  # 放宽姿态惩罚，允许跨步过程中的合理前倾微调
    rew_scale_pitch_roll_ang_vel = -1e-3  # 俯仰/横滚角速度惩罚：scale * (pitch_ang_vel² + roll_ang_vel²)
    rew_scale_lin_vel_z = 0.0  # 关闭躯干 Z 轴速度惩罚
    rew_scale_height = -2.0  # 根高度波动惩罚：偏离目标超过 height_penalty_band 时
    height_target = 0.74  # 目标根高度（米）；与 G1_CFG 默认根高一致
    height_penalty_band = 0.05  # 允许波动范围（米），74cm±5cm 内无惩罚，即 [0.69, 0.79] m
    
    # 奖励权重 - 平滑性组
    rew_scale_joint_velocity = -1e-3  # 关节速度惩罚（加重以促平滑，贴近 Unitree 量级）
    rew_scale_joint_acceleration = -1e-6  # 关节加速度惩罚（与宇树 legged_gym dof_acc 同型：控制步内 ‖Δq̇‖²）

    # 宇树官方移植的平滑与姿态惩罚
    rew_scale_action_rate = -0.005  # 惩罚动作突变（进一步加重，强抑制抽搐）
    rew_scale_contact_no_vel = -2.0  # 加重防滑步惩罚，禁止贴地蹭
    # 宇树 legged_gym ``g1_env._reward_hip_pos``：左右髋 roll + yaw 绝对角平方和；12dof 对应 dof 索引 [1,2,7,8]
    rew_scale_hip_pos = -0.5

    # 奖励权重 - 关节限制组（越界时 joint_limit_violation > 0，惩罚 = rew_scale_joint_limit * violation）
    rew_scale_joint_limit = -5.0  # 与宇树 ``dof_pos_limits = -5.0`` 对齐
    joint_limit_buffer = 0.1  # 缓冲带宽度（弧度）
    # 髋 pitch/roll/yaw 边界：g1_12dof.urdf + soft_dof_pos_limit；躯干为 torso_joint_limit_rad 对称区间。

    
    # 奖励参数 - 踩点组（目标点 z 为各 env 地面标高；命中时期望踝 z = 地面 + foot_ankle_ground_height）
    rew_scale_foot_hit = 15.0  # 踩点奖励权重（严格命中时一次发放）
    foot_hit_sigma = 0.03  # 高斯标准差（等效距离空间），越小越鼓励精确踩中
    foot_ankle_ground_height = 0.07  # 相对目标地面 z 的踝关节期望高度（米）

    # ==========================================================
    # 新版：动态摆动相引导组
    # ==========================================================
    rew_scale_feet_air_time = 8.0  # 滞空时间奖：仅落地瞬间发放 time_score，再乘此项（与踩点几何无关）
    rew_scale_foot_clearance = -20.0  # 惩罚：摆动脚贴地滑行
    rew_scale_distance_attraction = 1.0  # 摆动相：脚踝到目标 XY 距离的高斯吸引（越远分越低）
    rew_scale_swing_knee = -2.0  # 摆动脚膝关节弯曲不足惩罚（目标弯曲角 0.6rad）
    feet_air_time_target = 0.8  # 目标滞空时间（秒），严格以 0.8s 为峰
    feet_air_time_sigma = 0.35  # 时间容差（标准差）；放宽后对偏离 0.8s 的容忍度更高
    # 摆动相脚踝相对 env 地面高度目标（米）；偏离该高度的平方计入 clearance惩罚（越远越大）
    foot_clearance_height = 0.09

    # 触地判定：默认与 H1 速度任务一致，用 PhysX ContactSensor 的法向净接触力范数（需 spawn 的 activate_contact_sensors）
    foot_contact_use_physics_sensor: bool = True
    foot_contact_force_threshold: float = 1.0  # ‖net_contact_force‖ 超过该值视为触地；写入 ContactSensorCfg.force_threshold
    # foot_contact_use_physics_sensor=False 时使用下列启发式（高度 + 连续帧 + 速度）
    foot_contact_height_threshold = 0.075  # 脚踝高度阈值（米），低于此值视为“低高度”
    foot_contact_consecutive_frames = 2  # 需连续满足的帧数
    foot_contact_velocity_threshold = 0.08  # 脚踝线速度阈值（m/s），低于此值视为“速度很小”

    # 踩点任务参数（放宽以提高刷新可靠性）
    foot_target_hit_threshold = 0.08  # 踩点判定：课程起始阈值（米，8cm）
    foot_target_hit_threshold_end = 0.03  # 课程结束阈值（米，3cm）
    foot_target_hit_threshold_iter_start = 1500  # iter <1500 用起始阈值；≥1500 起进入线性段
    foot_target_hit_threshold_iter_end = 5000  # iter≥5000 固定为 3cm；1500–5000 内由 8cm 线性收紧到 3cm
    foot_target_hit_z_tolerance = 0.04  # 踩点判定：|踝 z − (目标地面z+foot_ankle_ground_height)| 允许偏差（米）
    foot_target_hit_z_contact_slack = 0.02  # 已触地时竖直放宽：踝 z ≤ 地面+foot_ankle_ground_height+此项 亦算 z 合格
    foot_target_hit_z_max = 0.09  # 相对目标地面 z 的踝高上界（米）：高度备用判定时 swing_踝 z < 地面+此项
    foot_target_hit_min_elapsed_s = 0.05  # 高度备用判定：目标生成后至少经过此时长才允许判定踩中，避免误触发
    # 自动踩点模式：判定踩中后延迟该时长再生成下一目标（跟随点/随机点）；0 表示与旧版一致「当步刷新」
    foot_target_regenerate_delay_s = 0.5
    
    # ==========================================================
    # 优化版：目标点位生成边界
    # ==========================================================
    foot_target_rect_x_forward = 0.30
    foot_target_rect_x_back = 0.12
    # 区分向外(Outward)和向内(Inward)的横向边界
    foot_target_rect_y_outward = 0.15  # 向外侧可迈 15cm（保持平衡）
    foot_target_rect_y_inward = 0.02  # 向内侧不超过 2cm（防剪刀步）
    foot_target_min_distance = 0.10

    foot_spacing_stand_m = 0.24  # 跟随点双腿间距（米）：一脚踩中后，另一脚跟上的目标距离

    # 重置条件
    reset_root_height = 0.74  # 与 G1_CFG init_state.pos[2]、height_target 一致

    # 分阶段奖励（奖励课程）：None = 不启用，所有奖励始终有效
    # reward_curriculum_mode: "step" 按 env 步数分阶段；"iteration" 按训练 iteration 分阶段（与 TensorBoard 一致）
    # "iteration" 时用 iter_start/iter_end，且 iteration = common_step_counter // reward_curriculum_steps_per_iteration（需与 PPO num_steps_per_env 一致，如 24）
    reward_curriculum_mode = "iteration"
    reward_curriculum_steps_per_iteration = 24  # 与 PPO num_steps_per_env 一致
    # 每项 dict：step 模式用 step_start/step_end；iteration 模式用 iter_start/iter_end。*_end 为 None 表示无上界。
    reward_curriculum = [
        # 将探索期从 1500 延长到 3000，给它足够时间适应抗拉扯负重
        {
            "iter_start": 0,
            "iter_end": 3_000,
            "components": [
                "rew_pitch_roll_angle",
                "rew_pitch_roll_ang_vel",
                "rew_lin_vel_z",
                "rew_height",
                "rew_joint_velocity",
                "rew_joint_acceleration",
                "rew_joint_limit",
                "rew_action_rate",
                "rew_contact_no_vel",
                "rew_hip_pos",
            ],
        },
        {
            "iter_start": 3_000,
            "iter_end": None,
            "components": [
                "rew_pitch_roll_angle",
                "rew_pitch_roll_ang_vel",
                "rew_lin_vel_z",
                "rew_height",
                "rew_joint_velocity",
                "rew_joint_acceleration",
                "rew_joint_limit",
                "rew_foot_hit",
                "rew_feet_air_time",
                "rew_foot_clearance",
                "rew_distance_attraction",
                "rew_swing_knee",
                "rew_action_rate",
                "rew_contact_no_vel",
                "rew_hip_pos",
            ],
        },
    ]
