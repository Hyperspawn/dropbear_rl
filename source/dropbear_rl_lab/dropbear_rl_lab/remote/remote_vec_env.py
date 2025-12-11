# Copyright (c) 2025, Hyperspawn Technologies.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""RemoteVecEnv: Network-connected environment for A100 workers."""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

# Add project root to path for remote_protocol_rl import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from remote_protocol_rl import (
    MSG_ACTION_BATCH,
    MSG_OBS_BATCH,
    MessageEnvelope,
    MessageSequencer,
    TensorSerializer,
    create_action_batch_message,
)


class RemoteVecEnv:
    """Vectorized environment that receives obs from controller via network.

    This environment runs on A100 workers and communicates with the RTX controller
    to get observations and send back actions. It's compatible with RSL-RL's
    OnPolicyRunner.
    """

    def __init__(
        self,
        num_envs: int,
        num_obs: int,
        num_actions: int,
        nkn_bridge,  # NKNSidecar instance
        controller_address: str,
        device: str = "cuda:0",
        timeout: float = 30.0,
    ):
        """Initialize remote environment.

        Args:
            num_envs: Number of parallel environments
            num_obs: Observation dimension
            num_actions: Action dimension
            nkn_bridge: NKNSidecar instance for network communication
            controller_address: NKN address of controller
            device: Device to run on
            timeout: Timeout for waiting for obs (seconds)
        """
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.controller_address = controller_address
        self.timeout = timeout

        # Auto-select best GPU
        selected_device = device
        if torch.cuda.is_available():
            best_gpu = None
            best_capability = (0, 0)

            for i in range(torch.cuda.device_count()):
                capability = torch.cuda.get_device_capability(i)
                gpu_name = torch.cuda.get_device_name(i)
                print(f"[remote_env] GPU {i}: {gpu_name} (compute {capability[0]}.{capability[1]})")

                if capability > best_capability:
                    best_capability = capability
                    best_gpu = i

            if best_gpu is not None and best_capability >= (7, 0):
                selected_device = f"cuda:{best_gpu}"
                gpu_name = torch.cuda.get_device_name(best_gpu)
                print(f"[remote_env] Selected GPU {best_gpu}: {gpu_name}")
            else:
                print(f"[remote_env] No compatible GPU (need >= 7.0), using CPU")
                selected_device = "cpu"
        else:
            selected_device = "cpu"

        self.device = torch.device(selected_device)

        # RSL-RL compatibility
        self.cfg = {
            "num_envs": num_envs,
            "num_observations": num_obs,
            "num_actions": num_actions,
        }

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(num_actions,), dtype=np.float32
        )

        # Network communication
        self.nkn_bridge = nkn_bridge
        self.sequencer = MessageSequencer()
        self.obs_queue: queue.Queue[MessageEnvelope] = queue.Queue(maxsize=100)
        self.step_counter = 0

        # Current state
        self._current_obs: Optional[Dict[str, torch.Tensor]] = None
        self._current_rewards: Optional[torch.Tensor] = None
        self._current_dones: Optional[torch.Tensor] = None

        # Additional attributes required by RSL-RL OnPolicyRunner
        self.max_episode_length = 1000  # Default episode length
        self.episode_length_buf = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.reset_buf = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.common_step_counter = 0

        # Register message handler
        self._original_on_message = getattr(nkn_bridge, "on_message", None)
        nkn_bridge.on_message = self._handle_network_message

        print(f"[remote_env] Initialized for {num_envs} envs, waiting for obs from {controller_address}")

    def _handle_network_message(self, src: str, body: Dict[str, Any]) -> None:
        """Handle incoming network messages."""
        # Only process messages from controller
        if src != self.controller_address:
            # Pass to original handler
            if self._original_on_message:
                self._original_on_message(src, body)
            return

        try:
            envelope = MessageEnvelope.from_dict(body)
            processed = self.sequencer.process_message(envelope)

            if processed and processed.msg_type == MSG_OBS_BATCH:
                # Queue observation message
                self.obs_queue.put(processed, block=False)

        except Exception as e:
            print(f"[remote_env] Error processing message: {e}")

    def reset(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Reset environment by waiting for initial observations from controller.

        Returns:
            Tuple of (observations_dict, info_dict)
        """
        print("[remote_env] Waiting for initial observations from controller...")

        # Wait for first observation batch
        try:
            msg = self.obs_queue.get(timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError(
                f"[remote_env] Timeout waiting for initial obs from controller (waited {self.timeout}s)"
            )

        # Deserialize observations
        payload = msg.payload
        obs_serialized = payload["obs"]
        obs_dict = TensorSerializer.deserialize_obs_dict(obs_serialized)

        # Move to correct device
        obs_dict = {
            key: tensor.to(self.device) for key, tensor in obs_dict.items()
        }

        self._current_obs = obs_dict
        self.step_counter = payload["step_id"]

        # Wrap obs_dict with ObsDict for .to() support
        class ObsDict(dict):
            """Dict subclass that supports .to() method for RSL-RL compatibility."""
            def to(self, device):
                """Move all tensors in the dict to the specified device."""
                return ObsDict({k: v.to(device) if isinstance(v, torch.Tensor) else v
                               for k, v in self.items()})

        print(f"[remote_env] Received initial obs for step {self.step_counter}")
        return ObsDict(obs_dict), {}

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Execute one step: send actions, wait for next observations.

        Args:
            actions: Action tensor (num_envs, num_actions)

        Returns:
            Tuple of (observations_dict, rewards, dones, info_dict)
        """
        # Send actions to controller
        action_msg = create_action_batch_message(
            self.sequencer,
            step_id=self.step_counter,
            actions=actions.cpu(),  # Send from CPU to avoid device issues
        )

        try:
            self.nkn_bridge.send_dm(
                self.controller_address,
                action_msg.to_dict()
            )
        except Exception as e:
            print(f"[remote_env] Error sending actions: {e}")
            raise

        # Wait for next observation batch
        try:
            msg = self.obs_queue.get(timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError(
                f"[remote_env] Timeout waiting for obs at step {self.step_counter} (waited {self.timeout}s)"
            )

        # Deserialize
        payload = msg.payload
        obs_dict = TensorSerializer.deserialize_obs_dict(payload["obs"])
        obs_dict = {k: v.to(self.device) for k, v in obs_dict.items()}

        rewards = TensorSerializer.deserialize_tensor(payload["rewards"]).to(self.device)
        dones = TensorSerializer.deserialize_tensor(payload["dones"]).to(self.device)

        self._current_obs = obs_dict
        self._current_rewards = rewards
        self._current_dones = dones
        self.step_counter = payload["step_id"]

        # Wrap obs_dict with ObsDict for .to() support
        class ObsDict(dict):
            """Dict subclass that supports .to() method for RSL-RL compatibility."""
            def to(self, device):
                """Move all tensors in the dict to the specified device."""
                return ObsDict({k: v.to(device) if isinstance(v, torch.Tensor) else v
                               for k, v in self.items()})

        return ObsDict(obs_dict), rewards, dones, {}

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get current observations.

        Returns:
            Dictionary with "policy" key containing observation tensor
        """
        if self._current_obs is None:
            raise RuntimeError("[remote_env] No observations available, call reset() first")

        # RSL-RL calls .to(device) on the result, so we need to wrap in a subclass
        class ObsDict(dict):
            """Dict subclass that supports .to() method for RSL-RL compatibility."""
            def to(self, device):
                """Move all tensors in the dict to the specified device."""
                return ObsDict({k: v.to(device) if isinstance(v, torch.Tensor) else v
                               for k, v in self.items()})

        return ObsDict(self._current_obs)

    def close(self) -> None:
        """Clean up resources."""
        print("[remote_env] Closing remote environment")
        # Restore original message handler
        if getattr(self, "_original_on_message", None):
            self.nkn_bridge.on_message = self._original_on_message
