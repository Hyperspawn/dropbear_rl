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
import json
import os
import pickle
import secrets
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

from remote_protocol_rl import (
    MessageSequencer,
    create_train_done_message,
    create_train_start_message,
    create_metrics_message,
)

from nkn_sidecar import NKNSidecar
print("[train_remote] Running on IsaacLab-free tensor worker")
print("[train_remote] This script never imports IsaacLab modules")


def dump_pickle_file(filename: str, data: object) -> None:
    """Persist configuration data using pickle."""
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(filename, "wb") as handle:
        pickle.dump(data, handle)


def main(provided_sidecar=None, argv=None):
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
    parser.add_argument("--app_address", type=str, default=None, help="NKN address of controller/app (alias).")
    parser.add_argument("--controller_address", type=str, default=None, help="NKN address of controller/app.")
    parser.add_argument("--nkn_seed", type=str, default=None, help="Override NKN seed for worker sidecar.")
    parser.add_argument("--nkn_identifier", type=str, default=None, help="Identifier for worker sidecar.")

    # Append RSL-RL cli arguments
    cli_args.add_rsl_rl_args(parser)

    # Parse known args, collect Hydra overrides
    args_cli, hydra_args = parser.parse_known_args(argv)

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
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    controller_address = (
        args_cli.controller_address
        or args_cli.app_address
        or os.environ.get("DROPBEAR_CONTROLLER_NKN_ADDRESS")
        or ""
    )
    controller_address = controller_address.strip().rstrip("\\/")

    # Build or reuse NKN sidecar for worker dataplane
    nkn_bridge = None
    if provided_sidecar is not None:
        nkn_bridge = provided_sidecar
        print(f"[train_remote] Reusing existing NKN sidecar at {nkn_bridge.address}")
    else:
        nkn_seed = (
            (args_cli.nkn_seed or "").strip()
            or os.environ.get("DROPBEAR_REMOTE_NKN_SEED", "").strip()
        )
        nkn_identifier = (
            args_cli.nkn_identifier
            or os.environ.get("DROPBEAR_REMOTE_NKN_IDENTIFIER", "").strip()
            or "dropbear_remote_worker"
        )
        nkn_seed_ws = ""
        nkn_num_subclients = 2

        # Attempt to load config for defaults (if present)
        config_file = PROJECT_ROOT / "isaaclab_remote_connection.json"
        if config_file.exists():
            try:
                config = json.loads(config_file.read_text(encoding="utf-8"))
                nkn_cfg = config.get("nkn", {})
                nkn_seed = nkn_seed or str(nkn_cfg.get("seed", "")).strip()
                controller_address = controller_address or str(nkn_cfg.get("app_address", "")).strip()
                nkn_identifier = args_cli.nkn_identifier or str(nkn_cfg.get("identifier", "dropbear_remote_worker"))
                nkn_seed_ws = str(nkn_cfg.get("seed_ws", "")).strip()
                try:
                    nkn_num_subclients = max(1, int(nkn_cfg.get("num_subclients", nkn_num_subclients)))
                except Exception:
                    nkn_num_subclients = 2
            except Exception as exc:  # pragma: no cover
                print(f"[train_remote] Failed to read NKN config: {exc}")

        if not nkn_seed:
            nkn_seed = secrets.token_hex(32)
            print(f"[train_remote] Generated transient NKN seed for worker: {nkn_seed}")

        if controller_address:
            try:
                nkn_bridge = NKNSidecar(
                    seed_hex=nkn_seed,
                    identifier=nkn_identifier,
                    num_subclients=nkn_num_subclients,
                    seed_ws=nkn_seed_ws,
                )
                nkn_bridge.start()
                nkn_bridge.wait_ready(timeout=30.0)
                print(f"[train_remote] NKN bridge ready at {nkn_bridge.address}")
            except Exception as exc:
                print(f"[train_remote] Failed to start NKN sidecar: {exc}")
                nkn_bridge = None
        if controller_address and nkn_bridge:
            print(f"[train_remote] Controller address: {controller_address}")
            print(f"[train_remote] Worker address: {nkn_bridge.address}")

    if controller_address:
        print("[train_remote] ========================================")
        print("[train_remote] PRODUCTION MODE: RemoteVecEnv")
        print(f"[train_remote] Waiting for observations from RTX controller: {controller_address}")
        print("[train_remote] ========================================")

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
            timeout=120.0,
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

    train_sequencer = getattr(env, "sequencer", None) or MessageSequencer()

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

    # Align agent/device with the selected env device (e.g., cuda:best_gpu)
    agent_dict["device"] = str(env.device)

    # If running against a controller, push a heavier workload to use the A100.
    # These are intentionally high, but not extreme, and keep shapes consistent.
    if controller_address:
        agent_dict["num_steps_per_env"] = 384
        agent_dict["num_mini_batches"] = 12
        agent_dict["num_learning_epochs"] = 10
        agent_dict.setdefault("policy", {})
        agent_dict["policy"]["actor_hidden_dims"] = [768, 768, 768]
        agent_dict["policy"]["critic_hidden_dims"] = [768, 768, 768]

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
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass
    # Ensure initial observations are available before constructing the runner
    try:
        env.reset()
        print("[train_remote] Initial observations received from controller.")
    except Exception as exc:
        print(f"[train_remote] Failed to receive initial observations: {exc}")
        raise

    runner = OnPolicyRunner(env, agent_dict, log_dir=log_dir, device=agent_dict["device"])

    # Dump configuration
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_pickle_file(os.path.join(log_dir, "params", "agent.pkl"), agent_dict)
    dump_pickle_file(os.path.join(log_dir, "params", "task.pkl"), task_config)

    print(f"[train_remote] Starting training for {agent_dict['max_iterations']} iterations")
    if controller_address:
        print("[train_remote] PRODUCTION MODE: Training with real observations from RTX")
    else:
        print("[train_remote] STUB MODE: Training with zero observations (testing only)")

    # Setup checkpoint transfer if we have a controller
    checkpoint_sender = None
    if controller_address and nkn_bridge:
        from checkpoint_transfer_protocol import CheckpointSender
        checkpoint_sender = CheckpointSender(nkn_bridge, env.sequencer if hasattr(env, 'sequencer') else None)
        print("[train_remote] Checkpoint auto-transfer enabled → RTX controller")
        try:
            start_msg = create_train_start_message(
                train_sequencer,
                task_config,
                agent_dict,
                worker_address=str(nkn_bridge.address or ""),
            )
            nkn_bridge.send_dm(controller_address, start_msg.to_dict())
            print("[train_remote] train_start sent to controller")
        except Exception as exc:
            print(f"[train_remote] Failed to send train_start: {exc}")

    # Run training with checkpoint callback
    save_interval = agent_dict.get("save_interval", 50)

    for iteration in range(agent_dict["max_iterations"]):
        # Run one iteration of training
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=(iteration == 0))

        # Check if we should save checkpoint
        #current_it = runner.tot_iter
        current_it = iteration + 1
        if current_it % save_interval == 0:
            # Save checkpoint locally
            checkpoint_path = os.path.join(log_dir, f"model_{current_it}.pt")
            runner.save(checkpoint_path)
            print(f"[train_remote] Saved checkpoint: {checkpoint_path}")

            # Transfer to controller if available
            if checkpoint_sender and controller_address:
                print(f"[train_remote] Transferring checkpoint to RTX controller...")
                checkpoint_id = checkpoint_sender.send_checkpoint(
                    file_path=Path(checkpoint_path),
                    destination=controller_address,
                    iteration=current_it,
                )
                print(f"[train_remote] Transfer initiated: {checkpoint_id}")

        # Emit lightweight metrics to controller for progress
        if controller_address and nkn_bridge:
            try:
                metrics_msg = create_metrics_message(train_sequencer, iteration=current_it, metrics={})
                nkn_bridge.send_dm(controller_address, metrics_msg.to_dict())
            except Exception as exc:
                print(f"[train_remote] Failed to send metrics: {exc}")

    # Close environment
    env.close()

    print("[train_remote] Training completed on remote worker")
    if controller_address:
        print("[train_remote] All checkpoints transferred to RTX controller")
        try:
            done_msg = create_train_done_message(
                train_sequencer,
                iterations=agent_dict.get("max_iterations", 0),
                log_dir=log_dir,
            )
            nkn_bridge.send_dm(controller_address, done_msg.to_dict())
            print("[train_remote] train_done sent to controller")
        except Exception as exc:
            print(f"[train_remote] Failed to send train_done: {exc}")
    else:
        print("[train_remote] No controller - checkpoints remain on A100")


if __name__ == "__main__":
    main()
