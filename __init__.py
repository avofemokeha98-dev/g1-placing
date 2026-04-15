# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
G1 Placing 环境注册。
- 单机踩点: ``Isaac-G1-Placing-Direct-v0``
- 三机抬六边形板 MARL: ``Isaac-G1-Placing-MARL-Direct-v0``
"""

import gymnasium as gym

from . import agents
from .g1_placing_env import G1PlacingEnv
from .g1_placing_env_cfg import G1PlacingEnvCfg


##
# Register Gym environments.
##

gym.register(
    id="Isaac-G1-Placing-Direct-v0",
    entry_point="isaaclab_tasks.direct.placing.g1_placing_env:G1PlacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.placing.g1_placing_env_cfg:G1PlacingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PlacingPPORunnerCfg",
        "rsl_rl_amp_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:G1PlacingAMPRunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-G1-Placing-MARL-Direct-v0",
    entry_point="isaaclab_tasks.direct.placing.g1_placing_marl_env:G1PlacingMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.placing.g1_placing_marl_env_cfg:G1PlacingMarlEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_marl_cfg:G1PlacingMarlPPORunnerCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
    },
)
