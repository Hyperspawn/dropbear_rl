# Copyright (c) 2025, Hyperspawn Technologies.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Remote training script for RSL-RL agents without IsaacLab dependencies."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from rsl_rl.runners import OnPolicyRunner

import cli_args  # isort: skip
from dropbear_rl_lab.remote import (
    RemoteAgentCfg,
    RemoteEnvCfg,
    RemoteVecEnv,
    build_log_paths,
    build_stub_env,
    dump_json_file,
    dump_pickle_file,
    ensure_log_directory,
    hydra_task_config,
)
from nkn_sidecar import NKNSidecar

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_FILE = PROJECT_ROOT / "isaaclab_remote_connection.json"
DEFAULT_TASK = "Isaac-Velocity-Dropbear-v0"


def _load_connection_config() -> Dict[str, Any]:
    if not CONFIG_FILE.exists():
        return {}
    try:
        print(f"[train_remote] Loading config from {CONFIG_FILE}", flush=True)
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _start_nkn_bridge(controller_address: str) -> NKNSidecar:
    if not controller_address:
        raise RuntimeError("Controller address required for remote NKNSidecar.")
    cfg = _load_connection_config()
    nkn_cfg = cfg.get("nkn", {})
    seed = str(nkn_cfg.get("seed", "")).strip()
    identifier = str(nkn_cfg.get("remote_address") or nkn_cfg.get("identifier") or "dropbear_remote")
    num_subclients = max(1, int(nkn_cfg.get("num_subclients", 2)))
    seed_ws = str(nkn_cfg.get("seed_ws") or "")
    bridge = NKNSidecar(
        seed_hex=seed,
        identifier=identifier,
        num_subclients=num_subclients,
        seed_ws=seed_ws,
        on_ready=lambda addr: print(f"[train_remote] NKN bridge ready at {addr}"),
    )
    bridge.start()
    if not bridge.wait_ready(timeout=30.0):
        raise RuntimeError("NKN bridge failed to become ready.")
    bridge.send_dm(controller_address, {"type": "start", "description": "train_remote ready"})
    return bridge


def _dump_run_configs(log_dir: Path, env_cfg: RemoteEnvCfg, agent_cfg: RemoteAgentCfg) -> None:
    params_dir = log_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    dump_json_file(params_dir / "env.json", env_cfg.to_dict())
    dump_json_file(params_dir / "agent.json", agent_cfg.to_dict())
    dump_pickle_file(params_dir / "env.pkl", env_cfg.to_dict())
    dump_pickle_file(params_dir / "agent.pkl", agent_cfg.to_dict())


def _wait_for_train_address_from_config(timeout: float = 40.0, poll: float = 0.5) -> Optional[str]:
    """Poll the shared config file until train_address is populated."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        cfg = _load_connection_config()
        cfg = _load_connection_config()
        print(f"[train_remote] DEBUG: Config snapshot: {cfg.get('nkn', {})}", flush=True)
        nkn_cfg = cfg.get("nkn", {})
        train_address = str(nkn_cfg.get("train_address") or "").strip()
        if train_address:
            print(f"[train_remote] ✓ Config contains train_address: {train_address}", flush=True)
            return train_address
        elapsed = time.time() - start_time
        remaining = max(0.0, timeout - elapsed)
        print(f"[train_remote] Still waiting for train_address in config ({int(elapsed)}s elapsed, {int(remaining)}s remaining)", flush=True)
        time.sleep(poll)
    print(f"[train_remote] ⚠ Timeout waiting for train_address in config ({timeout}s)", flush=True)
    return None


# Create debug log file
DEBUG_LOG = Path("/tmp/train_remote_debug.log")
with open(DEBUG_LOG, "w") as f:
    f.write(f"[train_remote] SCRIPT INVOKED at {time.time()}\n")
    f.write(f"[train_remote] sys.argv = {sys.argv}\n")

print("=" * 80, flush=True)
print("[train_remote] SCRIPT INVOKED - Parsing arguments...", flush=True)
print(f"[train_remote] sys.argv = {sys.argv}", flush=True)
print(f"[train_remote] Debug log: {DEBUG_LOG}", flush=True)
print("=" * 80, flush=True)

parser = argparse.ArgumentParser(description="Train RL agent with RSL-RL on remote worker (no IsaacLab).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs.")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on.")
parser.add_argument("--headless", action="store_true", default=False, help="Headless mode (ignored on remote).")
parser.add_argument("--app-address", type=str, default=None, help="Controller/app NKN address for handshake.")
parser.add_argument("--train-address", type=str, default=None, help="Controller train NKN address for receiving actions.")
cli_args.add_rsl_rl_args(parser)
args_cli, hydra_args = parser.parse_known_args()

print(f"[train_remote] Parsed --app-address: {args_cli.app_address}", flush=True)

# Reset argv so Hydra decorator sees only overrides
sys.argv = [sys.argv[0]] + hydra_args


@hydra_task_config(args_cli.task or DEFAULT_TASK, "rsl_rl_cfg_entry_point")
def main(env_cfg: RemoteEnvCfg, agent_cfg: RemoteAgentCfg) -> None:
    """Train with RSL-RL agent on remote worker."""
    # Enable unbuffered output for remote logging
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print("=" * 80, flush=True)
    print("[train_remote] SCRIPT STARTED", flush=True)
    print("=" * 80, flush=True)

    if args_cli.device:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.update_from_cli(args_cli)
    env_cfg.seed = agent_cfg.seed

    # Start NKN bridge first (we need it to receive train_address announcement)
    print("=" * 80)
    print("[train_remote] STARTUP - Initializing NKN bridge")
    print("=" * 80)

    # We need app_address (from CLI arg) to send the initial handshake
    app_address = args_cli.app_address
    controller_address = None
    train_address_arg = args_cli.train_address
    if train_address_arg:
        controller_address = train_address_arg.strip()
        print(f"[train_remote] Received --train-address CLI override: {controller_address}")
    if not app_address:
        print("[train_remote] ⚠ ERROR: --app-address not provided!")
        print("[train_remote] ⚠ Cannot proceed without controller handshake")
        print("[train_remote] ⚠ Falling back to STUB environment (no real training)")
        bridge = None
    else:
        print(f"[train_remote] Controller app_address (for handshake): {app_address}")
        # Start our NKN bridge
        bridge = _start_nkn_bridge(app_address)
        print(f"[train_remote] ✓ Our NKN bridge started: {bridge.address}")

        if not controller_address:
            controller_address = _wait_for_train_address_from_config(timeout=40.0)
        if controller_address:
            print(f"[train_remote] ✓ Using controller address: {controller_address}")
        else:
            print("[train_remote] ⚠ Timeout waiting for train_address in config!")
            print("[train_remote] ⚠ Falling back to app_address (may cause conflicts)")
            controller_address = app_address

    print("=" * 80)
    print("[train_remote] STARTUP CONFIGURATION")
    print("=" * 80)
    print(f"[train_remote] Controller address for actions: {controller_address or '<NONE - will use stub env>'}")

    if controller_address:
        print("[train_remote] ✓ Running in PRODUCTION mode (Controller-driven observations)")
        print(f"[train_remote] ✓ Will send actions to: {controller_address}")
    else:
        print("[train_remote] ⚠ Controller address not available!")
        print("[train_remote] ⚠ Falling back to STUB environment (no real training)")
        print("[train_remote] ⚠ This means NO observations from RTX controller!")
    print("=" * 80)

    log_root, log_dir = build_log_paths(agent_cfg.experiment_name, agent_cfg.run_name)
    ensure_log_directory(log_root, log_dir)
    print(f"[train_remote] Logging to {log_dir}")
    _dump_run_configs(log_dir, env_cfg, agent_cfg)

    # Build environment (bridge already created above if needed)
    if controller_address and bridge:
        print(f"[train_remote] Creating RemoteVecEnv connected to {controller_address}")
        env = RemoteVecEnv(
            num_envs=env_cfg.scene.num_envs,
            num_obs=env_cfg.num_obs,
            num_actions=env_cfg.num_actions,
            nkn_bridge=bridge,
            controller_address=controller_address,
            device=agent_cfg.device,
            timeout=30.0,
        )
    else:
        print("[train_remote] Creating stub environment (no remote connection)")
        env = build_stub_env(
            task_name=env_cfg.task_name,
            num_envs=env_cfg.scene.num_envs,
            num_obs=env_cfg.num_obs,
            num_actions=env_cfg.num_actions,
            device=agent_cfg.device,
        )
        bridge = None

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    checkpoint_sender = None
    if controller_address and bridge and hasattr(env, "sequencer"):
        from checkpoint_transfer_protocol import CheckpointSender

        checkpoint_sender = CheckpointSender(bridge, env.sequencer)
        print("[train_remote] Checkpoint sender ready (controller is handling artifacts).")

    save_interval = agent_cfg.save_interval
    for iteration in range(agent_cfg.max_iterations):
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=(iteration == 0))

        current_iteration = iteration + 1
        if save_interval > 0 and current_iteration % save_interval == 0:
            checkpoint_path = log_dir / f"model_{current_iteration}.pt"
            runner.save(str(checkpoint_path))
            print(f"[train_remote] Saved checkpoint: {checkpoint_path}")
            if checkpoint_sender and controller_address:
                print("[train_remote] Dispatching checkpoint to controller...")
                checkpoint_sender.send_checkpoint(
                    file_path=checkpoint_path,
                    destination=controller_address,
                    iteration=current_iteration,
                )

    env.close()
    if bridge:
        bridge.stop()
    print("[train_remote] Remote training completed.")


if __name__ == "__main__":
    main()
