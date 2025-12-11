# Copyright (c) 2025, Hyperspawn Technologies.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Remote adapter module for IsaacLab-free tensor workers.

This module provides environment builders and helpers for running RL training
on remote A100 workers without IsaacLab dependencies.
"""

from .env_builder import build_stub_env
from .config_helpers import (
    RemoteAgentCfg,
    RemoteEnvCfg,
    apply_hydra_overrides,
    hydra_task_config,
    load_task_config,
)
from .logging_utils import build_log_paths, dump_json_file, dump_pickle_file, ensure_log_directory
from .remote_vec_env import RemoteVecEnv

__all__ = [
    "build_stub_env",
    "RemoteVecEnv",
    "load_task_config",
    "apply_hydra_overrides",
    "hydra_task_config",
    "RemoteEnvCfg",
    "RemoteAgentCfg",
    "build_log_paths",
    "ensure_log_directory",
    "dump_json_file",
    "dump_pickle_file",
]
