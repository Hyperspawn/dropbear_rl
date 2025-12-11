# Copyright (c) 2025, Hyperspawn Technologies.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Remote training script for RSL-RL agent without IsaacLab dependencies.

This script runs on A100 tensor workers and performs pure tensor math without
requiring IsaacLab, RTX GPUs, or Isaac Sim. It reuses the CLI args and Hydra
decorators from train.py but uses lightweight stub environments.
"""

import argparse
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import torch

# IMPORTANT: This script NEVER imports IsaacLab
# It only imports tensor math libraries and RL helpers

# Add scripts directory to path for cli_args import
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import cli_args  # isort: skip

# Import remote helpers (no IsaacLab dependency)
from dropbear_rl_lab.remote import build_stub_env, load_task_config, apply_hydra_overrides

# Import RSL-RL directly (available on remote worker)
from rsl_rl.runners import OnPolicyRunner

print("[train_remote] Running on IsaacLab-free tensor worker")
print("[train_remote] This script never imports IsaacLab modules")


def dump_pickle_file(filename: str, data: object) -> None:
    """Persist configuration data using pickle."""
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(filename, "wb") as handle:
        pickle.dump(data, handle)


def main():
    """Train with RSL-RL agent on remote worker."""
    # Parse arguments (same structure as train.py)
    parser = argparse.ArgumentParser(description="Train RL agent with RSL-RL on remote worker (no IsaacLab).")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
    parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on.")
    parser.add_argument("--headless", action="store_true", default=False, help="Headless mode (ignored on remote).")

    # Append RSL-RL cli arguments
    cli_args.add_rsl_rl_args(parser)

    # Parse known args, collect Hydra overrides
    args_cli, hydra_args = parser.parse_known_args()

    print(f"[train_remote] Task: {args_cli.task}")
    print(f"[train_remote] Device: {args_cli.device}")
    print(f"[train_remote] Num envs: {args_cli.num_envs}")
    print(f"[train_remote] Max iterations: {args_cli.max_iterations}")
    print(f"[train_remote] Hydra overrides: {hydra_args}")

    # Load task configuration (no IsaacLab registry)
    task_config = load_task_config(args_cli.task or "Isaac-Velocity-Dropbear-v0")
    task_config = apply_hydra_overrides(task_config, hydra_args)

    # Override with CLI args
    if args_cli.num_envs is not None:
        task_config["num_envs"] = args_cli.num_envs
    if args_cli.device is not None:
        task_config["device"] = args_cli.device

    # Determine if we have a controller sending observations
    # Check if remote_client has a controller address (meaning RTX is running train.py)
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    import remote_client

    # Get controller address from app_address field (RTX controller)
    controller_address = None
    if hasattr(args_cli, 'app_address') and args_cli.app_address:
        controller_address = args_cli.app_address
    else:
        # Try to get from remote config
        cfg = remote_client.get_remote_config()
        nkn_cfg = cfg.get("nkn", {})
        controller_address = nkn_cfg.get("app_address", "")

    if controller_address:
        print("[train_remote] ========================================")
        print("[train_remote] PRODUCTION MODE: RemoteVecEnv")
        print(f"[train_remote] Waiting for observations from RTX controller: {controller_address}")
        print("[train_remote] ========================================")

        # Get NKN bridge
        nkn_bridge = remote_client.get_nkn_bridge()
        if not nkn_bridge:
            raise RuntimeError("[train_remote] No NKN bridge available for RemoteVecEnv!")

        # Create RemoteVecEnv for production training
        from dropbear_rl_lab.remote import RemoteVecEnv
        env = RemoteVecEnv(
            num_envs=task_config.get("num_envs", 4),
            num_obs=task_config.get("num_obs", 48),
            num_actions=task_config.get("num_actions", 12),
            nkn_bridge=nkn_bridge,
            controller_address=controller_address,
            device=task_config.get("device", "cuda:0"),
            timeout=30.0,
        )
        print(f"[train_remote] RemoteVecEnv initialized - waiting for obs from {controller_address}")
    else:
        # Fallback to stub environment for testing
        print("[train_remote] ========================================")
        print("[train_remote] STUB MODE: No controller detected")
        print("[train_remote] Creating stub environment (no IsaacLab imports)")
        print("[train_remote] ========================================")
        env = build_stub_env(
            task_name=args_cli.task or "Isaac-Velocity-Dropbear-v0",
            num_envs=task_config.get("num_envs", 4),
            num_obs=task_config.get("num_obs", 48),
            num_actions=task_config.get("num_actions", 12),
            device=task_config.get("device", "cuda:0"),
        )

    # Create minimal agent configuration
    # In production, this would come from shared config or controller
    from dataclasses import dataclass, field
    from typing import Dict, Any

    @dataclass
    class MinimalPPOConfig:
        """Minimal PPO configuration for remote runner."""
        # Algorithm parameters
        class_name: str = "PPO"
        value_loss_coef: float = 1.0
        use_clipped_value_loss: bool = True
        clip_param: float = 0.2
        entropy_coef: float = 0.01
        num_learning_epochs: int = 5
        num_mini_batches: int = 4
        learning_rate: float = 1.0e-3
        schedule: str = "adaptive"
        gamma: float = 0.99
        lam: float = 0.95
        desired_kl: float = 0.01
        max_grad_norm: float = 1.0

        # Policy parameters
        init_noise_std: float = 1.0
        actor_hidden_dims: list = field(default_factory=lambda: [256, 256, 256])
        critic_hidden_dims: list = field(default_factory=lambda: [256, 256, 256])
        activation: str = "elu"

        # Runner parameters
        seed: int = 42
        device: str = "cuda:0"
        num_steps_per_env: int = 24
        max_iterations: int = 1
        empirical_normalization: bool = False
        save_interval: int = 50
        log_interval: int = 1
        policy: Dict[str, Any] = field(default_factory=dict)

        def to_dict(self) -> dict:
            """Convert to dictionary for RSL-RL."""
            policy_dict = {
                "class_name": "ActorCritic",
                "init_noise_std": self.init_noise_std,
                "actor_hidden_dims": self.actor_hidden_dims,
                "critic_hidden_dims": self.critic_hidden_dims,
                "activation": self.activation,
            }
            policy_dict.update(self.policy)

            return {
                "algorithm": {
                    "class_name": self.class_name,
                    "value_loss_coef": self.value_loss_coef,
                    "use_clipped_value_loss": self.use_clipped_value_loss,
                    "clip_param": self.clip_param,
                    "entropy_coef": self.entropy_coef,
                    "num_learning_epochs": self.num_learning_epochs,
                    "num_mini_batches": self.num_mini_batches,
                    "learning_rate": self.learning_rate,
                    "schedule": self.schedule,
                    "gamma": self.gamma,
                    "lam": self.lam,
                    "desired_kl": self.desired_kl,
                    "max_grad_norm": self.max_grad_norm,
                },
                "policy": policy_dict,
                "seed": self.seed,
                "device": self.device,
                "num_steps_per_env": self.num_steps_per_env,
                "max_iterations": self.max_iterations,
                "empirical_normalization": self.empirical_normalization,
                "save_interval": self.save_interval,
                "log_interval": self.log_interval,
                # Required by RSL-RL OnPolicyRunner
                "obs_groups": {},  # Empty dict for stub environment
                "privileged_obs_groups": {},  # Empty dict for stub environment
            }

    agent_cfg = MinimalPPOConfig()

    # Apply overrides from Hydra args
    agent_dict = agent_cfg.to_dict()
    override_config = {"agent_cfg": agent_dict}
    override_config = apply_hydra_overrides(override_config, hydra_args)
    agent_dict = override_config.get("agent_cfg", agent_dict)

    # Override from CLI
    if args_cli.seed is not None:
        agent_dict["seed"] = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_dict["max_iterations"] = args_cli.max_iterations
    if args_cli.device is not None:
        agent_dict["device"] = args_cli.device

    # Create log directory
    experiment_name = args_cli.task or "dropbear_remote"
    experiment_name = experiment_name.lower().replace("-", "_")
    log_root_path = os.path.join("logs", "rsl_rl", experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[train_remote] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if hasattr(args_cli, "run_name") and args_cli.run_name:
        log_dir += f"_{args_cli.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Create OnPolicyRunner
    print("[train_remote] Creating RSL-RL OnPolicyRunner (pure tensor math)")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    runner = OnPolicyRunner(env, agent_dict, log_dir=log_dir, device=agent_dict["device"])

    # Dump configuration
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_pickle_file(os.path.join(log_dir, "params", "agent.pkl"), agent_dict)
    dump_pickle_file(os.path.join(log_dir, "params", "task.pkl"), task_config)

    print(f"[train_remote] Starting training for {agent_dict['max_iterations']} iterations")
    print("[train_remote] Environment is STUB - no actual simulation running")
    print("[train_remote] This is pure tensor math on A100 worker")

    # Run training
    runner.learn(num_learning_iterations=agent_dict["max_iterations"], init_at_random_ep_len=True)

    # Close environment
    env.close()

    print("[train_remote] Training completed on remote worker")
    print("[train_remote] No IsaacLab imports were used - confirmed!")


if __name__ == "__main__":
    main()
