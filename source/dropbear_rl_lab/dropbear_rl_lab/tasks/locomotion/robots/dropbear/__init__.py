# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Dropbear humanoid robot locomotion environments.
"""

import gymnasium as gym

from . import agents, velocity_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Velocity-Dropbear-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:DropbearVelocityRoughEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:DropbearVelocityRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:DropbearVelocityRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Dropbear-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:DropbearVelocityRoughEnvCfg_PLAY",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:DropbearVelocityRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:DropbearVelocityRoughPPORunnerCfg",
    },
)

