# Copyright (c) 2025, Hyperspawn Technologies.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train RL agent with RSL-RL for Dropbear robot."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL for Dropbear robot.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--remote_worker_address", type=str, default=None, help="NKN address of remote A100 worker for policy execution."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# DEBUG: Show what we received
print("=" * 80, flush=True)
print("[train.py] ARGUMENT PARSING DEBUG", flush=True)
print("=" * 80, flush=True)
print(f"[train.py] sys.argv = {sys.argv}", flush=True)
print(f"[train.py] args_cli.remote_worker_address = {args_cli.remote_worker_address}", flush=True)
print(f"[train.py] Type: {type(args_cli.remote_worker_address)}", flush=True)
print(f"[train.py] Truthy? {bool(args_cli.remote_worker_address)}", flush=True)
print("=" * 80, flush=True)

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import inspect
import os
import shutil
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner


def dump_pickle_file(filename: str, data: object) -> None:
    """Persist configuration data using pickle (replicates the legacy helper)."""
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(filename, "wb") as handle:
        pickle.dump(data, handle)


import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import dropbear_rl_lab.tasks  # noqa: F401
from dropbear_rl_lab.remote import build_log_paths, ensure_log_directory

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # DEBUG: Check if remote_worker_address is still accessible inside main()
    print("=" * 80, flush=True)
    print("[train.py] INSIDE main() - Checking remote_worker_address", flush=True)
    print(f"[train.py] args_cli.remote_worker_address = {args_cli.remote_worker_address}", flush=True)
    print(f"[train.py] Truthy? {bool(args_cli.remote_worker_address)}", flush=True)
    print("=" * 80, flush=True)

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # create consistent logging directories
    log_root, log_dir = build_log_paths(agent_cfg.experiment_name, agent_cfg.run_name)
    ensure_log_directory(log_root, log_dir)
    log_root_path = str(log_root)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"Exact experiment name requested from command line: {log_dir.name}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Check if remote mode is enabled for offloading policy to A100 workers
    print(f"[train.py] ========================================", flush=True)
    print(f"[train.py] CHECKING REMOTE MODE", flush=True)
    print(f"[train.py] args_cli.remote_worker_address = '{args_cli.remote_worker_address}'", flush=True)
    print(f"[train.py] Type: {type(args_cli.remote_worker_address)}", flush=True)
    print(f"[train.py] Evaluates to: {bool(args_cli.remote_worker_address)}", flush=True)
    print(f"[train.py] Will enter remote mode: {bool(args_cli.remote_worker_address)}", flush=True)
    print(f"[train.py] ========================================", flush=True)

    if args_cli.remote_worker_address:
        print("[train.py] ========================================")
        print("[train.py] REMOTE MODE ACTIVE")
        print("[train.py] RTX: Running IsaacLab simulation locally")
        print("[train.py] A100: Policy inference via NKN")
        print("[train.py] ========================================")

        worker_address = args_cli.remote_worker_address
        print(f"[train.py] NKN worker address: {worker_address}", flush=True)
        print(f"[train.py] Wrapping environment for remote policy execution...", flush=True)

        try:
            # Import and wrap with ControllerRemoteEnvWrapper
            # NOTE: We need to create NKN bridge here in Isaac Sim environment
            import sys
            from pathlib import Path
            PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
            sys.path.insert(0, str(PROJECT_ROOT))

            from nkn_sidecar import NKNSidecar
            import json

            # Load NKN config from isaaclab_remote_connection.json
            config_file = PROJECT_ROOT / "isaaclab_remote_connection.json"
            print(f"[train.py] Loading config from: {config_file}", flush=True)
            with open(config_file, 'r') as f:
                config = json.load(f)

            nkn_cfg = config.get("nkn", {})
            print(f"[train.py] Config loaded successfully", flush=True)
        except Exception as e:
            print(f"[train.py] ERROR during setup: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise

        # CRITICAL: Use a DIFFERENT identifier than app.py to avoid address conflicts
        # app.py uses "dropbear_app", we use "dropbear_train"
        # This ensures two separate NKN addresses for the two processes
        train_identifier = "dropbear_train"

        nkn_bridge = NKNSidecar(
            seed_hex=str(nkn_cfg.get("seed", "")),
            identifier=train_identifier,
            num_subclients=max(1, int(nkn_cfg.get("num_subclients", 2))),
            seed_ws=str(nkn_cfg.get("seed_ws", "")),
        )
        nkn_bridge.start()
        if not nkn_bridge.wait_ready(timeout=30.0):
            raise RuntimeError("[train.py] NKN bridge failed to become ready")

        train_nkn_address = nkn_bridge.address
        print(f"[train.py] NKN bridge started in Isaac Sim environment")
        print(f"[train.py] Train NKN address: {train_nkn_address}")
        print(f"[train.py] Worker will send actions TO this address")

        # Update config file with train.py's address (local only)
        nkn_cfg["train_address"] = train_nkn_address
        config["nkn"] = nkn_cfg
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        # CRITICAL: Send train_address directly to A100 worker via NKN
        # The config file update above only updates RTX's local file,
        # but A100 has a separate config file on a different machine!
        print(f"[train.py] ========================================", flush=True)
        print(f"[train.py] SENDING TRAIN_ADDRESS ANNOUNCEMENT", flush=True)
        print(f"[train.py] My train_address: {train_nkn_address}", flush=True)
        print(f"[train.py] Sending to worker: {worker_address}", flush=True)
        message_to_send = {
            "type": "train_address_announcement",
            "train_address": train_nkn_address,
            "description": "train.py NKN bridge address for receiving actions"
        }
        print(f"[train.py] Message payload: {message_to_send}", flush=True)
        nkn_bridge.send_dm(worker_address, message_to_send)
        print(f"[train.py] ✓ send_dm() completed successfully", flush=True)
        print(f"[train.py] ========================================", flush=True)

        from controller_remote_env import ControllerRemoteEnvWrapper
        env = ControllerRemoteEnvWrapper(
            base_env=env,
            nkn_bridge=nkn_bridge,
            worker_address=worker_address,
            timeout=30.0,
        )
        print("[train.py] ========================================")
        print("[train.py] ✓ Environment wrapped with ControllerRemoteEnvWrapper!")
        print("[train.py] ✓ Simulation: RTX (IsaacLab)")
        print("[train.py] ✓ Policy: A100 (via NKN)")
        print(f"[train.py] ✓ Worker address: {worker_address}")
        print("[train.py] ========================================")

        # Setup checkpoint receiver to get trained models from A100
        from checkpoint_transfer_protocol import CheckpointReceiver, MSG_CHECKPOINT_START, MSG_CHECKPOINT_CHUNK, MSG_CHECKPOINT_REQUEST_RETRY, MSG_CHECKPOINT_ACK
        from remote_protocol_rl import MessageSequencer, MessageEnvelope

        checkpoint_receiver = CheckpointReceiver(
            nkn_bridge=nkn_bridge,
            sequencer=MessageSequencer(),
            save_dir=Path(log_root_path),
        )

        # Register message handler for checkpoints
        original_on_message = getattr(nkn_bridge, "on_message", None)

        def checkpoint_message_handler(src: str, body: dict):
            """Handle checkpoint transfer messages."""
            if not isinstance(body, dict):
                if original_on_message:
                    original_on_message(src, body)
                return
            try:
                envelope = MessageEnvelope.from_dict(body)
            except Exception as e:
                print(f"[train.py] Error parsing checkpoint envelope: {e}")
                if original_on_message:
                    original_on_message(src, body)
                return
            msg_type = getattr(envelope, "msg_type", None)
            if not msg_type:
                if original_on_message:
                    original_on_message(src, body)
                return

            if msg_type == MSG_CHECKPOINT_START:
                checkpoint_receiver.handle_checkpoint_start(src, envelope.payload)
            elif msg_type == MSG_CHECKPOINT_CHUNK:
                checkpoint_receiver.handle_checkpoint_chunk(src, envelope.payload)
            elif msg_type == MSG_CHECKPOINT_REQUEST_RETRY:
                # Forward to ControllerRemoteEnvWrapper if needed
                pass
            elif msg_type == MSG_CHECKPOINT_ACK:
                # A100 acknowledged receipt
                pass
            else:
                if original_on_message:
                    original_on_message(src, body)

        nkn_bridge.on_message = checkpoint_message_handler
        print("[train.py] Checkpoint receiver enabled - will save models from A100")

        _run_remote_controller_loop(env)
        env.close()
        _wait_for_remote_checkpoint(log_root_path)
        return
    print("[train.py] Local mode: simulation and policy both on RTX")

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle_file(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle_file(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()

    # finished local training


def _run_remote_controller_loop(env) -> None:
    """Keep the IsaacLab controller stepping while the A100 worker provides actions."""
    print("[train.py] ===== Starting controller loop for remote policy =====")
    step_count = 0
    try:
        env.reset()
        while True:
            env.step()
            step_count += 1
    except TimeoutError:
        print("[train.py] Remote policy appear to have finished (timeout waiting for new actions).")
    except KeyboardInterrupt:
        print("[train.py] Remote controller loop interrupted by user.")
    finally:
        print(f"[train.py] Controller loop ending after {step_count} steps.")


def _wait_for_remote_checkpoint(log_root_path: str) -> None:
    """Wait until the checkpoint receiver saves the final artifacts."""
    print("[train.py] ========================================")
    print("[train.py] Training completed!")
    print("[train.py] Waiting for final checkpoint from A100...")
    print("[train.py] ========================================")
    time.sleep(5)

    checkpoint_dirs = list(Path(log_root_path).glob("iteration_*"))
    if checkpoint_dirs:
        latest_checkpoint_dir = max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)
        checkpoint_files = list(latest_checkpoint_dir.glob("model_*.pt"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            print(f"[train.py] Latest checkpoint: {latest_checkpoint}")
            print(f"[train.py] Ready to visualize!")
            print(f"[train.py]")
            print(f"[train.py] To visualize the trained policy, run:")
            print(f"[train.py]   python3 app.py")
            print(f"[train.py]   Then select dropbear_play (with remote mode OFF)")
            print(f"[train.py] Or run directly:")
            print(f"[train.py]   ./isaaclab.sh -p scripts/rsl_rl/play.py \\")
            print(f"[train.py]       --task {args_cli.task} \\")
            print(f"[train.py]       --checkpoint {latest_checkpoint}")
        else:
            print(f"[train.py] No checkpoint files found in {latest_checkpoint_dir}")
    else:
        print(f"[train.py] No checkpoint directories found in {log_root_path}")

    print(f"[train.py]")
    print(f"[train.py] Press Ctrl+C to exit or close this window")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"[train.py] Exiting...")


if __name__ == "__main__":
    # run the main function
    print("=" * 80, flush=True)
    print("[train.py] About to call main() function", flush=True)
    print(f"[train.py] args_cli.remote_worker_address before main() = {args_cli.remote_worker_address}", flush=True)
    print("=" * 80, flush=True)
    main()
    print("=" * 80, flush=True)
    print("[train.py] main() function returned", flush=True)
    print("=" * 80, flush=True)
    # close sim app
    simulation_app.close()
