#!/usr/bin/env python3
"""RL-specific network protocol for controller ↔ worker communication.

This module handles:
- Message ordering via sequence numbers
- Deduplication via message IDs
- Tensor serialization for network transport
- Bidirectional obs/action/reward exchange
"""

from __future__ import annotations

import hashlib
import io
import time
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch


# Message type constants
MSG_TRAIN_START = "train_start"
MSG_OBS_BATCH = "obs_batch"
MSG_ACTION_BATCH = "action_batch"
MSG_REWARD_BATCH = "reward_batch"
MSG_CHECKPOINT = "checkpoint"
MSG_METRICS = "metrics"
MSG_TRAIN_DONE = "train_done"
MSG_HEARTBEAT = "heartbeat"


@dataclass
class MessageEnvelope:
    """Wrapper for all RL protocol messages with ordering/dedup metadata."""
    msg_type: str
    msg_id: str
    sequence: int
    timestamp: float
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "msg_type": self.msg_type,
            "msg_id": self.msg_id,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> MessageEnvelope:
        """Reconstruct from dict."""
        return MessageEnvelope(
            msg_type=data["msg_type"],
            msg_id=data["msg_id"],
            sequence=data["sequence"],
            timestamp=data["timestamp"],
            payload=data["payload"],
        )


class MessageSequencer:
    """Handles message ordering and deduplication."""

    def __init__(self):
        self.next_seq = 0
        self.received_msgs: Dict[str, float] = {}  # msg_id -> timestamp
        self.expected_seq = 0
        self.out_of_order_buffer: Dict[int, MessageEnvelope] = {}

    def create_message(
        self,
        msg_type: str,
        payload: Dict[str, Any],
    ) -> MessageEnvelope:
        """Create a new outbound message with sequence number."""
        msg_id = self._generate_msg_id(msg_type, payload)
        msg = MessageEnvelope(
            msg_type=msg_type,
            msg_id=msg_id,
            sequence=self.next_seq,
            timestamp=time.time(),
            payload=payload,
        )
        self.next_seq += 1
        return msg

    def process_message(
        self,
        msg: MessageEnvelope,
    ) -> Optional[MessageEnvelope]:
        """Process incoming message with dedup and ordering.

        Returns:
            Message if valid and in-order, None if duplicate or out-of-order
        """
        # Deduplication check
        if msg.msg_id in self.received_msgs:
            age = time.time() - self.received_msgs[msg.msg_id]
            if age < 60.0:  # Dedupe window: 60 seconds
                return None  # Duplicate, ignore

        # Mark as received
        self.received_msgs[msg.msg_id] = time.time()

        # Clean old entries (> 60s)
        cutoff = time.time() - 60.0
        self.received_msgs = {
            mid: ts for mid, ts in self.received_msgs.items() if ts > cutoff
        }

        # Ordering check
        if msg.sequence == self.expected_seq:
            # In-order message
            self.expected_seq += 1

            # Check if buffered messages are now ready
            while self.expected_seq in self.out_of_order_buffer:
                buffered = self.out_of_order_buffer.pop(self.expected_seq)
                self.expected_seq += 1
                # Note: In production, yield buffered messages too
                # For now, just advance expected_seq

            return msg

        elif msg.sequence > self.expected_seq:
            # Out-of-order, buffer it
            self.out_of_order_buffer[msg.sequence] = msg
            return None

        else:
            # Old message (seq < expected), ignore
            return None

    def _generate_msg_id(self, msg_type: str, payload: Dict[str, Any]) -> str:
        """Generate unique message ID based on content."""
        content = f"{msg_type}:{time.time_ns()}:{id(payload)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class TensorSerializer:
    """Serialize/deserialize PyTorch tensors for network transport."""

    @staticmethod
    def serialize_tensor(tensor: torch.Tensor, compress: bool = True) -> bytes:
        """Convert tensor to bytes.

        Args:
            tensor: PyTorch tensor
            compress: Whether to apply zlib compression

        Returns:
            Serialized bytes
        """
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        data = buffer.getvalue()

        if compress:
            data = zlib.compress(data, level=6)

        return data

    @staticmethod
    def deserialize_tensor(data: bytes, decompress: bool = True) -> torch.Tensor:
        """Reconstruct tensor from bytes.

        Args:
            data: Serialized bytes
            decompress: Whether to decompress first

        Returns:
            PyTorch tensor
        """
        if decompress:
            data = zlib.decompress(data)

        buffer = io.BytesIO(data)
        tensor = torch.load(buffer, weights_only=True)
        return tensor

    @staticmethod
    def serialize_obs_dict(obs_dict: Dict[str, torch.Tensor]) -> Dict[str, bytes]:
        """Serialize observation dictionary.

        Args:
            obs_dict: Dict of observation tensors (e.g., {"policy": tensor})

        Returns:
            Dict with same keys but serialized tensor values
        """
        return {
            key: TensorSerializer.serialize_tensor(tensor)
            for key, tensor in obs_dict.items()
        }

    @staticmethod
    def deserialize_obs_dict(serialized: Dict[str, bytes]) -> Dict[str, torch.Tensor]:
        """Deserialize observation dictionary.

        Args:
            serialized: Dict with serialized tensor values

        Returns:
            Dict of PyTorch tensors
        """
        return {
            key: TensorSerializer.deserialize_tensor(data)
            for key, data in serialized.items()
        }


def create_train_start_message(
    sequencer: MessageSequencer,
    task_config: Dict[str, Any],
    agent_config: Dict[str, Any],
) -> MessageEnvelope:
    """Create training start message with configs."""
    payload = {
        "task_config": task_config,
        "agent_config": agent_config,
    }
    return sequencer.create_message(MSG_TRAIN_START, payload)


def create_train_done_message(
    sequencer: MessageSequencer,
    iterations: int,
    log_dir: str = "",
) -> MessageEnvelope:
    """Create training completion message."""
    payload = {
        "iterations": iterations,
    }
    if log_dir:
        payload["log_dir"] = log_dir
    return sequencer.create_message(MSG_TRAIN_DONE, payload)


def create_obs_batch_message(
    sequencer: MessageSequencer,
    step_id: int,
    obs_dict: Dict[str, torch.Tensor],
    rewards: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
) -> MessageEnvelope:
    """Create observation batch message.

    Args:
        sequencer: Message sequencer
        step_id: Global step counter
        obs_dict: Observation tensors {"policy": tensor}
        rewards: Reward tensor (optional, for step > 0)
        dones: Done flags (optional, for step > 0)

    Returns:
        Message envelope
    """
    payload = {
        "step_id": step_id,
        "obs": TensorSerializer.serialize_obs_dict(obs_dict),
    }

    if rewards is not None:
        payload["rewards"] = TensorSerializer.serialize_tensor(rewards)

    if dones is not None:
        payload["dones"] = TensorSerializer.serialize_tensor(dones)

    return sequencer.create_message(MSG_OBS_BATCH, payload)


def create_action_batch_message(
    sequencer: MessageSequencer,
    step_id: int,
    actions: torch.Tensor,
) -> MessageEnvelope:
    """Create action batch message.

    Args:
        sequencer: Message sequencer
        step_id: Global step counter
        actions: Action tensor

    Returns:
        Message envelope
    """
    payload = {
        "step_id": step_id,
        "actions": TensorSerializer.serialize_tensor(actions),
    }
    return sequencer.create_message(MSG_ACTION_BATCH, payload)


def create_checkpoint_message(
    sequencer: MessageSequencer,
    checkpoint_data: bytes,
    iteration: int,
) -> MessageEnvelope:
    """Create checkpoint transfer message.

    Args:
        sequencer: Message sequencer
        checkpoint_data: Serialized model checkpoint
        iteration: Training iteration number

    Returns:
        Message envelope
    """
    payload = {
        "iteration": iteration,
        "checkpoint": checkpoint_data.hex(),  # Hex encode for JSON
    }
    return sequencer.create_message(MSG_CHECKPOINT, payload)


def create_metrics_message(
    sequencer: MessageSequencer,
    iteration: int,
    metrics: Dict[str, float],
) -> MessageEnvelope:
    """Create training metrics message.

    Args:
        sequencer: Message sequencer
        iteration: Training iteration
        metrics: Dict of metric name -> value

    Returns:
        Message envelope
    """
    payload = {
        "iteration": iteration,
        "metrics": metrics,
    }
    return sequencer.create_message(MSG_METRICS, payload)


def create_heartbeat_message(
    sequencer: MessageSequencer,
    role: str,
    info: Optional[Dict[str, Any]] = None,
) -> MessageEnvelope:
    """Create lightweight heartbeat to confirm liveness."""
    payload = {"role": role}
    if info:
        payload["info"] = info
    return sequencer.create_message(MSG_HEARTBEAT, payload)
