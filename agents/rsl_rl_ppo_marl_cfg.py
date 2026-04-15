# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO：三机抬板 MARL（单智能体拼接 obs=318, act=111）。"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1PlacingMarlPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """三机协同：拼接后 policy 观测 109×3=327（高层脚目标模式），动作 40×3=120；若 use_high_level_foot_target=False 则为 318/111。"""

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "g1_placing_marl"
    empirical_normalization = False
    logger = "tensorboard"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        # 续训/微调：较低 LR 避免洗掉已学步态；从零训练可用 CLI 覆盖为 1e-3（或更保守续训用 3e-5）
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
