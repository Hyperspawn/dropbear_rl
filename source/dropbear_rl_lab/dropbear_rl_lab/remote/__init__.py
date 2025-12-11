# Copyright (c) 2025, Hyperspawn Technologies.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Remote adapter module for IsaacLab-free tensor workers.

This module provides environment builders and helpers for running RL training
on remote A100 workers without IsaacLab dependencies.
"""

from .env_builder import build_stub_env
from .config_helpers import load_task_config, apply_hydra_overrides
from .remote_vec_env import RemoteVecEnv

__all__ = [
    "build_stub_env",
    "load_task_config",
    "apply_hydra_overrides",
    "RemoteVecEnv",
]
