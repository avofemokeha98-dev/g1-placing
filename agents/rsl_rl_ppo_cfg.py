# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1PlacingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for PPO runner for G1 Placing."""

    num_steps_per_env = 24
    # 与 event_curriculum 三阶段总长度一致：5000+5000+10000；续训请用 CLI 覆盖或提高总 iter
    max_iterations = 20000
    save_interval = 500     # 每 500 iteration 保存一次 checkpoint
    experiment_name = "g1_placing"
    empirical_normalization = False
    logger = "tensorboard"  # 启用TensorBoard监控（默认已启用，显式设置确保启用）
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
