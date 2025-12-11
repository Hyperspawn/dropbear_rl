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

        # Minimal config dict required by RSL-RL
        self.cfg = {
            "num_envs": num_envs,
            "num_observations": num_obs,
            "num_actions": num_actions,
        }

        # Auto-select best available GPU or fall back to CPU
        selected_device = device
        if torch.cuda.is_available():
            # Find the most capable GPU (highest compute capability)
            best_gpu = None
            best_capability = (0, 0)

            for i in range(torch.cuda.device_count()):
                capability = torch.cuda.get_device_capability(i)
                gpu_name = torch.cuda.get_device_name(i)
                print(f"[stub_env] GPU {i}: {gpu_name} (compute {capability[0]}.{capability[1]})")

                if capability > best_capability:
                    best_capability = capability
                    best_gpu = i

            if best_gpu is not None and best_capability >= (7, 0):
                selected_device = f"cuda:{best_gpu}"
                gpu_name = torch.cuda.get_device_name(best_gpu)
                print(f"[stub_env] Selected GPU {best_gpu}: {gpu_name} (compute {best_capability[0]}.{best_capability[1]})")
            else:
                print(f"[stub_env] No compatible GPU found (need compute >= 7.0), using CPU")
                selected_device = "cpu"
        else:
            print("[stub_env] CUDA not available, using CPU")
            selected_device = "cpu"

        try:
            self.device = torch.device(selected_device)
            # Test device
            test_tensor = torch.zeros(1, device=self.device)
            del test_tensor
        except RuntimeError as e:
            print(f"[stub_env] Device {selected_device} error: {e}, falling back to CPU")
            self.device = torch.device("cpu")

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
        # RSL-RL expects observations as a dict with a "policy" key
        self._obs_dict = {
            "policy": torch.zeros(num_envs, num_obs, device=self.device, dtype=torch.float32)
        }
        self._rewards = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self._dones = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self._info = {}

        # Additional attributes required by RSL-RL OnPolicyRunner
        self.max_episode_length = 1000  # Default episode length
        self.episode_length_buf = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.reset_buf = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.common_step_counter = 0

    def reset(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Reset the environment.

        Returns:
            Tuple of (observations_dict, info_dict)
        """
        self._obs_dict["policy"].zero_()
        self._dones.zero_()

        # Wrap obs_dict with ObsDict for .to() support
        class ObsDict(dict):
            """Dict subclass that supports .to() method for RSL-RL compatibility."""
            def to(self, device):
                """Move all tensors in the dict to the specified device."""
                return ObsDict({k: v.to(device) if isinstance(v, torch.Tensor) else v
                               for k, v in self.items()})

        return ObsDict(self._obs_dict), {}

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            actions: Action tensor of shape (num_envs, num_actions)

        Returns:
            Tuple of (observations_dict, rewards, dones, info_dict)
        """
        # In a real remote setup, this would send actions to controller
        # and receive back observations, rewards, dones
        # For now, return stub data

        # Wrap obs_dict with ObsDict for .to() support
        class ObsDict(dict):
            """Dict subclass that supports .to() method for RSL-RL compatibility."""
            def to(self, device):
                """Move all tensors in the dict to the specified device."""
                return ObsDict({k: v.to(device) if isinstance(v, torch.Tensor) else v
                               for k, v in self.items()})

        return ObsDict(self._obs_dict), self._rewards, self._dones, self._info

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get current observations.

        Returns:
            Dictionary with "policy" key containing observation tensor
        """
        # RSL-RL calls .to(device) on the result, so we need to wrap in a subclass
        class ObsDict(dict):
            """Dict subclass that supports .to() method for RSL-RL compatibility."""
            def to(self, device):
                """Move all tensors in the dict to the specified device."""
                return ObsDict({k: v.to(device) if isinstance(v, torch.Tensor) else v
                               for k, v in self.items()})

        return ObsDict(self._obs_dict)

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
