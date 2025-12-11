# Copyright (c) 2025, Hyperspawn Technologies.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Environment builder for remote workers without IsaacLab."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from typing import Any, Dict, Optional, Tuple


class StubVecEnv:
    """Lightweight stub environment for remote tensor math workers.

    This provides the minimal gym-like interface needed by RSL-RL's OnPolicyRunner
    without requiring IsaacLab or RTX GPU access. The actual environment state and
    observations come from the controller via network payloads.
    """

    def __init__(
        self,
        num_envs: int = 1,
        num_obs: int = 48,
        num_actions: int = 12,
        device: str = "cuda:0",
    ):
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.device = torch.device(device)

        # Define observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_obs,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_actions,),
            dtype=np.float32
        )

        # Initialize stub state
        self._obs = torch.zeros(num_envs, num_obs, device=self.device, dtype=torch.float32)
        self._rewards = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self._dones = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self._info = {}

    def reset(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reset the environment.

        Returns:
            Tuple of (observations, info_dict)
        """
        self._obs.zero_()
        self._dones.zero_()
        return self._obs, {}

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            actions: Action tensor of shape (num_envs, num_actions)

        Returns:
            Tuple of (observations, rewards, dones, info_dict)
        """
        # In a real remote setup, this would send actions to controller
        # and receive back observations, rewards, dones
        # For now, return stub data
        return self._obs, self._rewards, self._dones, self._info

    def get_observations(self) -> torch.Tensor:
        """Get current observations.

        Returns:
            Observation tensor of shape (num_envs, num_obs)
        """
        return self._obs

    def close(self) -> None:
        """Clean up environment resources."""
        pass


def build_stub_env(
    task_name: str,
    num_envs: int = 1,
    num_obs: int = 48,
    num_actions: int = 12,
    device: str = "cuda:0",
    **kwargs
) -> StubVecEnv:
    """Build a stub environment for remote workers.

    Args:
        task_name: Name of the task (currently ignored, for future extension)
        num_envs: Number of parallel environments
        num_obs: Observation space dimension
        num_actions: Action space dimension
        device: Device to run on (cuda or cpu)
        **kwargs: Additional arguments (ignored for stub env)

    Returns:
        StubVecEnv instance compatible with RSL-RL
    """
    print(f"[remote] Building stub environment for task: {task_name}")
    print(f"[remote] Configuration: num_envs={num_envs}, obs_dim={num_obs}, act_dim={num_actions}, device={device}")

    return StubVecEnv(
        num_envs=num_envs,
        num_obs=num_obs,
        num_actions=num_actions,
        device=device,
    )
