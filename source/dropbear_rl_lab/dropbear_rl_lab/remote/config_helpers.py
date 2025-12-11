# Copyright (c) 2025, Hyperspawn Technologies.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration helpers for remote workers without IsaacLab."""

from __future__ import annotations

from typing import Any, Dict, Optional


def load_task_config(task_name: str) -> Dict[str, Any]:
    """Load task configuration without IsaacLab dependencies.

    This is a minimal implementation that returns default config.
    In production, this could load configs from shared JSON/YAML files
    or receive them from the controller.

    Args:
        task_name: Name of the task

    Returns:
        Dictionary with task configuration
    """
    print(f"[remote] Loading config for task: {task_name}")

    # Default configuration for Dropbear velocity task
    default_config = {
        "task_name": task_name,
        "num_envs": 4,
        # Use full policy obs dim from IsaacLab env (policy group ~193)
        "num_obs": 193,
        # Action dim for Dropbear velocity task (IsaacLab ActionManager reports 22)
        "num_actions": 22,
        "device": "cuda:0",
        "episode_length_s": 20.0,
        "dt": 0.02,
    }

    return default_config


def apply_hydra_overrides(config: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    """Apply Hydra-style overrides to configuration.

    Args:
        config: Base configuration dictionary
        overrides: List of override strings in format "key=value" or "+key=value"

    Returns:
        Updated configuration dictionary
    """
    for override in overrides:
        override = override.strip()
        if not override:
            continue

        # Handle + prefix for new keys
        if override.startswith("+"):
            override = override[1:]

        # Split on first = only
        if "=" not in override:
            continue

        key, value = override.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Parse value type
        parsed_value: Any = value
        if value.lower() in ("true", "false"):
            parsed_value = value.lower() == "true"
        elif value.replace(".", "", 1).replace("-", "", 1).isdigit():
            if "." in value:
                parsed_value = float(value)
            else:
                parsed_value = int(value)

        # Navigate nested keys (e.g., "agent_cfg.policy.init_noise_std")
        keys = key.split(".")
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = parsed_value
        print(f"[remote] Applied override: {key}={parsed_value}")

    return config
