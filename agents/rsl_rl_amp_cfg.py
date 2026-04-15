# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL + AMP 配置：在现有 PPO 基础上加入 AMP 判别器。"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from .rsl_rl_ppo_cfg import G1PlacingPPORunnerCfg


@configclass
class G1PlacingAMPRunnerCfg(G1PlacingPPORunnerCfg):
    """RSL-RL + AMP 配置：沿用 PPO 结构，加入 AMP 相关参数。"""

    experiment_name = "g1_placing_amp"

    # AMP 参数（通过 train.py 或此处 override）
    amp_task_scale: float = 0.8   # task reward 权重
    amp_scale: float = 0.2       # style reward 权重（reward = task_scale*task + amp_scale*style）
    amp_discriminator_hidden_dims: tuple[int, ...] = (1024, 512)
    amp_discriminator_reward_scale: float = 2.0
    amp_expert_buffer_size: int = 50000
