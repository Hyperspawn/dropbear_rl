# Copyright (c) 2025, Hyperspawn Technologies.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration helpers for remote workers without IsaacLab."""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_AGENT_ALGORITHM: Dict[str, Any] = {
    "class_name": "PPO",
    "value_loss_coef": 1.0,
    "use_clipped_value_loss": True,
    "clip_param": 0.2,
    "entropy_coef": 0.01,
    "num_learning_epochs": 5,
    "num_mini_batches": 4,
    "learning_rate": 1.0e-3,
    "schedule": "adaptive",
    "gamma": 0.99,
    "lam": 0.95,
    "desired_kl": 0.01,
    "max_grad_norm": 1.0,
}

DEFAULT_AGENT_POLICY: Dict[str, Any] = {
    "class_name": "ActorCritic",
    "init_noise_std": 1.0,
    "actor_hidden_dims": [256, 256, 256],
    "critic_hidden_dims": [256, 256, 256],
    "activation": "elu",
}

DEFAULT_REMOTE_TASKS: Dict[str, Dict[str, Any]] = {
    "default": {
        "env": {
            "task_name": "Isaac-Velocity-Dropbear-v0",
            "num_envs": 4,
            "num_obs": 48,
            "num_actions": 12,
            "device": "cuda:0",
            "seed": 42,
        },
        "agent": {
            "experiment_name": "dropbear_remote",
            "run_name": None,
            "seed": 42,
            "device": "cuda:0",
            "num_steps_per_env": 24,
            "max_iterations": 1,
            "empirical_normalization": False,
            "save_interval": 50,
            "log_interval": 1,
            "algorithm": DEFAULT_AGENT_ALGORITHM,
            "policy": DEFAULT_AGENT_POLICY,
            "obs_groups": {},
            "privileged_obs_groups": {},
        },
    },
    "Isaac-Velocity-Dropbear-v0": {
        "env": {
            "task_name": "Isaac-Velocity-Dropbear-v0",
            "num_envs": 4,
            "num_obs": 48,
            "num_actions": 12,
            "device": "cuda:0",
            "seed": 42,
        },
        "agent": {
            "experiment_name": "dropbear_velocity",
            "run_name": None,
            "seed": 42,
            "device": "cuda:0",
            "num_steps_per_env": 32,
            "max_iterations": 1000,
            "empirical_normalization": False,
            "save_interval": 50,
            "log_interval": 1,
            "algorithm": DEFAULT_AGENT_ALGORITHM,
            "policy": DEFAULT_AGENT_POLICY,
            "obs_groups": {},
            "privileged_obs_groups": {},
        },
    },
}


@dataclass
class RemoteEnvSceneCfg:
    num_envs: int = 4


@dataclass
class RemoteEnvSimCfg:
    device: str = "cuda:0"


@dataclass
class RemoteEnvCfg:
    task_name: str
    scene: RemoteEnvSceneCfg
    sim: RemoteEnvSimCfg
    seed: int
    num_obs: int
    num_actions: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RemoteEnvCfg":
        return cls(
            task_name=str(data.get("task_name", "")),
            scene=RemoteEnvSceneCfg(num_envs=int(data.get("num_envs", 4))),
            sim=RemoteEnvSimCfg(device=str(data.get("device", "cuda:0"))),
            seed=int(data.get("seed", 42)),
            num_obs=int(data.get("num_obs", 48)),
            num_actions=int(data.get("num_actions", 12)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "num_envs": self.scene.num_envs,
            "device": self.sim.device,
            "seed": self.seed,
            "num_obs": self.num_obs,
            "num_actions": self.num_actions,
        }


@dataclass
class RemoteAgentCfg:
    experiment_name: str = "dropbear_remote"
    run_name: Optional[str] = None
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 1
    empirical_normalization: bool = False
    save_interval: int = 50
    log_interval: int = 1
    algorithm: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_AGENT_ALGORITHM))
    policy: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_AGENT_POLICY))
    obs_groups: Dict[str, Any] = field(default_factory=dict)
    privileged_obs_groups: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RemoteAgentCfg":
        agent = cls(
            experiment_name=str(data.get("experiment_name", "dropbear_remote")),
            run_name=data.get("run_name"),
            seed=int(data.get("seed", 42)),
            device=str(data.get("device", "cuda:0")),
            num_steps_per_env=int(data.get("num_steps_per_env", 24)),
            max_iterations=int(data.get("max_iterations", 1)),
            empirical_normalization=bool(data.get("empirical_normalization", False)),
            save_interval=int(data.get("save_interval", 50)),
            log_interval=int(data.get("log_interval", 1)),
            algorithm=dict(data.get("algorithm", DEFAULT_AGENT_ALGORITHM)),
            policy=dict(data.get("policy", DEFAULT_AGENT_POLICY)),
            obs_groups=dict(data.get("obs_groups", {})),
            privileged_obs_groups=dict(data.get("privileged_obs_groups", {})),
        )
        return agent

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "seed": self.seed,
            "device": self.device,
            "num_steps_per_env": self.num_steps_per_env,
            "max_iterations": self.max_iterations,
            "empirical_normalization": self.empirical_normalization,
            "save_interval": self.save_interval,
            "log_interval": self.log_interval,
            "algorithm": dict(self.algorithm),
            "policy": dict(self.policy),
            "obs_groups": dict(self.obs_groups),
            "privileged_obs_groups": dict(self.privileged_obs_groups),
        }

    def update_from_cli(self, args_cli: Any) -> None:
        if hasattr(args_cli, "seed") and args_cli.seed is not None:
            self.seed = args_cli.seed
        if hasattr(args_cli, "device") and args_cli.device is not None:
            self.device = args_cli.device
        if hasattr(args_cli, "max_iterations") and args_cli.max_iterations is not None:
            self.max_iterations = args_cli.max_iterations
        if hasattr(args_cli, "run_name") and args_cli.run_name:
            self.run_name = args_cli.run_name


@dataclass
class RemoteTaskConfig:
    env: RemoteEnvCfg
    agent: RemoteAgentCfg

    def to_dict(self) -> Dict[str, Any]:
        return {"env": self.env.to_dict(), "agent": self.agent.to_dict()}


def load_task_config(task_name: str) -> Dict[str, Any]:
    """Load a remote-friendly configuration for the desired task."""
    base = DEFAULT_REMOTE_TASKS.get(task_name, DEFAULT_REMOTE_TASKS["default"])
    return copy.deepcopy(base)


def apply_hydra_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply simple Hydra-style overrides to nested dictionaries."""
    for override in overrides:
        override = override.strip()
        if not override:
            continue
        if override.startswith("+"):
            override = override[1:]
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        key = key.strip()
        value = value.strip()
        parsed_value: Any = value
        if value.lower() in ("true", "false"):
            parsed_value = value.lower() == "true"
        elif value.replace(".", "", 1).replace("-", "", 1).isdigit():
            if "." in value:
                parsed_value = float(value)
            else:
                parsed_value = int(value)
        keys = key.split(".")
        current: Dict[str, Any] = config
        for part in keys[:-1]:
            entry = current.get(part)
            if not isinstance(entry, dict):
                entry = {}
                current[part] = entry
            current = entry
        current[keys[-1]] = parsed_value
        print(f"[remote] Applied override: {key}={parsed_value}")
    return config


def hydra_task_config(task_name: str, entry_point: str):
    """Decorate a function so it receives remote-friendly env and agent configs."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            base_config = load_task_config(task_name)
            overrides = sys.argv[1:]
            updated = apply_hydra_overrides(base_config, overrides)
            env_cfg = RemoteEnvCfg.from_dict(updated.get("env", {}))
            agent_cfg = RemoteAgentCfg.from_dict(updated.get("agent", {}))
            return func(env_cfg, agent_cfg, *args, **kwargs)

        return wrapper

    return decorator
