# Copyright (c) 2025, Hyperspawn Robotics.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dropbear RL Lab: Reinforcement Learning Environments for Dropbear Humanoid Robot.

This package provides comprehensive reinforcement learning environments and configurations
for the Dropbear humanoid robot, built on NVIDIA's Isaac Lab platform.

Key Components:
    - Robot configurations and assets
    - Locomotion tasks and environments
    - MDP components (rewards, observations, commands)
    - Training agents and utilities

Developed by Hyperspawn Robotics in partnership with NVIDIA.
"""

__version__ = "0.1.0"
__author__ = "Hyperspawn Robotics"
__email__ = "support@hyperspawn.com"
__license__ = "Apache-2.0"

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]

# Lazy import to avoid Isaac Lab initialization issues
def get_dropbear_cfg():
    """Get the Dropbear robot configuration.
    
    Returns:
        DropbearArticulationCfg: The Dropbear robot configuration.
    """
    from dropbear_rl_lab.assets.robots.dropbear import DROPBEAR_CFG
    return DROPBEAR_CFG
