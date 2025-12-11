#!/usr/bin/env python3
"""Controller-side remote environment wrapper.

This wraps an IsaacLab environment and sends observations to A100 workers,
receives actions back, and applies them locally for simulation.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Dict, Optional

import torch

from remote_protocol_rl import (
    MSG_ACTION_BATCH,
    MSG_OBS_BATCH,
    MessageEnvelope,
    MessageSequencer,
    create_obs_batch_message,
)


class ControllerRemoteEnvWrapper:
    """Wraps IsaacLab env to send obs to remote worker and receive actions.

    This runs on the RTX controller and coordinates with RemoteVecEnv on A100.
    """

    def __init__(
        self,
        base_env,  # IsaacLab VecEnv
        nkn_bridge,  # NKNSidecar instance
        worker_address: str,
        timeout: float = 10.0,
    ):
        """Initialize controller-side wrapper.

        Args:
            base_env: IsaacLab environment instance
            nkn_bridge: NKN bridge for communication
            worker_address: NKN address of A100 worker
            timeout: Timeout for action responses (seconds)
        """
        self.base_env = base_env
        self.nkn_bridge = nkn_bridge
        self.worker_address = worker_address
        self.timeout = timeout

        # Message handling
        self.sequencer = MessageSequencer()
        self.action_queue: queue.Queue[MessageEnvelope] = queue.Queue(maxsize=100)
        self.step_counter = 0

        # Register message handler
        self._original_on_message = getattr(nkn_bridge, "on_message", None)
        nkn_bridge.on_message = self._handle_network_message

        # Delegate all attributes to base_env
        self.num_envs = base_env.num_envs
        self.device = base_env.device
        self.cfg = base_env.cfg if hasattr(base_env, 'cfg') else {}

        print(f"[controller_env] Initialized wrapper for remote worker: {worker_address}")
        print(f"[controller_env] Will send obs and receive actions via NKN")

    def _handle_network_message(self, src: str, body: Dict[str, Any]) -> None:
        """Handle incoming network messages."""
        # Only process action messages from our worker
        if src != self.worker_address:
            if self._original_on_message:
                self._original_on_message(src, body)
            return

        try:
            envelope = MessageEnvelope.from_dict(body)
            processed = self.sequencer.process_message(envelope)

            if processed and processed.msg_type == MSG_ACTION_BATCH:
                # Queue action message
                self.action_queue.put(processed, block=False)
                print(f"[controller_env] Received actions for step {processed.payload['step_id']}")

        except Exception as e:
            print(f"[controller_env] Error processing message: {e}")

    def reset(self):
        """Reset environment and send initial observations to worker."""
        print("[controller_env] Resetting environment...")

        # Reset base environment
        obs_dict, extras = self.base_env.reset()

        # Send initial observations to worker
        self._send_observations(obs_dict, rewards=None, dones=None)

        return obs_dict, extras

    def step(self, actions: Optional[torch.Tensor] = None):
        """Step environment.

        If actions not provided, waits for actions from remote worker.

        Args:
            actions: Actions to apply (optional, will wait for worker if None)

        Returns:
            Tuple of (obs, rewards, dones, extras)
        """
        # If no actions provided, wait for them from worker
        if actions is None:
            print(f"[controller_env] Waiting for actions from worker for step {self.step_counter}...")
            try:
                action_msg = self.action_queue.get(timeout=self.timeout)
            except queue.Empty:
                raise TimeoutError(
                    f"[controller_env] Timeout waiting for actions at step {self.step_counter}"
                )

            # Deserialize actions
            from remote_protocol_rl import TensorSerializer
            actions_bytes = bytes.fromhex(action_msg.payload["actions"])
            actions = TensorSerializer.deserialize_tensor(actions_bytes)
            actions = actions.to(self.device)

            print(f"[controller_env] Received actions, executing step {self.step_counter}")

        # Execute step in base environment
        obs_dict, rewards, dones, extras = self.base_env.step(actions)

        # Send observations to worker for next step
        self._send_observations(obs_dict, rewards, dones)

        return obs_dict, rewards, dones, extras

    def _send_observations(
        self,
        obs_dict: Dict[str, torch.Tensor],
        rewards: Optional[torch.Tensor],
        dones: Optional[torch.Tensor],
    ) -> None:
        """Send observation batch to remote worker.

        Args:
            obs_dict: Observation tensors
            rewards: Reward tensor (None for initial reset)
            dones: Done flags (None for initial reset)
        """
        # Move tensors to CPU for serialization
        obs_cpu = {k: v.cpu() for k, v in obs_dict.items()}
        rewards_cpu = rewards.cpu() if rewards is not None else None
        dones_cpu = dones.cpu() if dones is not None else None

        # Create message
        msg = create_obs_batch_message(
            self.sequencer,
            step_id=self.step_counter,
            obs_dict=obs_cpu,
            rewards=rewards_cpu,
            dones=dones_cpu,
        )

        # Send via NKN
        try:
            self.nkn_bridge.send_dm(self.worker_address, msg.to_dict())
            print(f"[controller_env] Sent observations for step {self.step_counter}")
            self.step_counter += 1
        except Exception as e:
            print(f"[controller_env] Error sending observations: {e}")
            raise

    def close(self):
        """Clean up resources."""
        print("[controller_env] Closing controller wrapper")
        if getattr(self, "_original_on_message", None):
            self.nkn_bridge.on_message = self._original_on_message
        if hasattr(self.base_env, 'close'):
            self.base_env.close()

    def __getattr__(self, name):
        """Delegate unknown attributes to base environment."""
        return getattr(self.base_env, name)
