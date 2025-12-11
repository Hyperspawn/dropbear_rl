#!/usr/bin/env python3
"""NKN-backed remote executor for Dropbear RL."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from app import (
    DEFAULT_ISAACSIM_VERSION,
    DEFAULT_NVIDIA_PYPI,
    DEFAULT_TORCHAUDIO_VERSION,
    DEFAULT_TORCHVISION_VERSION,
    DEFAULT_TORCH_VERSION,
    DROPBEAR_EXTENSION_DIR,
    create_venv_with_python,
    ensure_actor_critic_std,
    ensure_dropbear_installed,
    find_python_311,
    prepend_path,
    run_cmd,
    venv_bin_dir,
    venv_pip,
    venv_python,
    write_marker,
)
from nkn_sidecar import NKNSidecar

PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_VENV_NAME = "env_remote"
REMOTE_VENV_DIR = PROJECT_ROOT / REMOTE_VENV_NAME
REMOTE_MARKER = REMOTE_VENV_DIR / ".remote_bootstrap_ok"
INSIDE_FLAG = "--_inside-remote"


def _prepare_env() -> Dict[str, str]:
    return prepend_path(dict(os.environ), venv_bin_dir(REMOTE_VENV_DIR))


def resolve_command(cmd: Iterable[str]) -> List[str]:
    resolved: List[str] = []
    for part in cmd:
        path = Path(part)
        if path.is_absolute():
            resolved.append(str(path))
        else:
            candidate = PROJECT_ROOT / part
            if candidate.exists():
                resolved.append(str(candidate))
            else:
                resolved.append(part)
    return resolved


def _stream_command(
    cmd: List[str],
    description: Optional[str],
    env: Dict[str, str],
    send_fn: Callable[[Dict[str, object]], None],
    session_id: Optional[str] = None,
) -> None:
    desc = description or "run started"

    def _send(payload: Dict[str, object]) -> None:
        if session_id:
            payload = dict(payload)
            payload["session_id"] = session_id
        send_fn(payload)

    _send({"type": "start", "description": desc})
    try:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception as exc:
        _send({"type": "error", "message": f"Failed to start command: {exc}"})
        _send({"type": "exit", "code": 1})
        return
    assert process.stdout is not None
    for line in process.stdout:
        _send({"type": "log", "message": line.rstrip()})
    return_code = process.wait()
    _send({"type": "exit", "code": return_code})


def boot_remote_venv() -> None:
    if INSIDE_FLAG in sys.argv:
        sys.argv.remove(INSIDE_FLAG)
        return
    py311 = find_python_311()
    if not py311:
        raise RuntimeError("Python 3.11 is required to bootstrap the remote environment.")
    create_venv_with_python(py311, REMOTE_VENV_DIR)
    remote_python = venv_python(REMOTE_VENV_DIR)
    if not remote_python.exists():
        raise RuntimeError("Failed to create remote Python binary.")
    env = prepend_path(dict(os.environ), venv_bin_dir(REMOTE_VENV_DIR))
    if not REMOTE_MARKER.exists():
        run_cmd([str(remote_python), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"], env=env)
        torch_index = "https://download.pytorch.org/whl/cu128"
        torch_pkgs = [
            f"torch=={DEFAULT_TORCH_VERSION}",
            f"torchvision=={DEFAULT_TORCHVISION_VERSION}",
            f"torchaudio=={DEFAULT_TORCHAUDIO_VERSION}",
            "--index-url",
            torch_index,
        ]
        run_cmd([str(remote_python), "-m", "pip", "install", "-U"] + torch_pkgs, env=env)
        isaacsim_spec = f"isaacsim[all,extscache]=={DEFAULT_ISAACSIM_VERSION}"
        run_cmd([str(remote_python), "-m", "pip", "install", isaacsim_spec, "--extra-index-url", DEFAULT_NVIDIA_PYPI], env=env)
        ensure_dropbear_installed(remote_python, venv_pip(REMOTE_VENV_DIR), DROPBEAR_EXTENSION_DIR)
        ensure_actor_critic_std(REMOTE_VENV_DIR)
        write_marker(REMOTE_VENV_DIR, REMOTE_MARKER.name)
    os.execv(
        str(remote_python),
        [str(remote_python), str(__file__), INSIDE_FLAG] + [arg for arg in sys.argv[1:]],
    )


class NKNRemoteAgent:
    def __init__(self, seed_hex: str, identifier: str, num_subclients: int, controller_address: str) -> None:
        if not seed_hex:
            raise ValueError("NKN seed hex is required for the remote agent.")
        self.env = _prepare_env()
        self.controller_address = controller_address.strip()
        self.bridge = NKNSidecar(
            seed_hex=seed_hex,
            identifier=identifier,
            num_subclients=max(1, num_subclients),
            on_ready=self._on_ready,
            on_status=self._on_status,
            on_message=self._on_message,
            on_error=self._on_error,
        )
        self._running = False
        self._handshake_lock = threading.Lock()
        self._handshake_sent = False

    def start(self) -> None:
        self.bridge.start()
        if not self.bridge.wait_ready(timeout=30.0):
            raise RuntimeError("NKN bridge failed to become ready.")
        self._running = True
        print("[remote] NKN remote agent is up; waiting for commands.")
        self._send_handshake()

    def stop(self) -> None:
        if self._running:
            self._running = False
            self.bridge.stop()
            print("[remote] NKN remote agent shutting down.")

    def run_blocking(self) -> None:
        try:
            self.start()
            while self._running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _send_handshake(self) -> None:
        if not self.controller_address:
            return
        with self._handshake_lock:
            if self._handshake_sent:
                return
            addr = self.bridge.address
            if not addr:
                return
            payload = {
                "type": "handshake",
                "description": f"NKN remote agent {socket.gethostname()}",
                "address": addr,
                "ts": int(time.time() * 1000),
            }
            try:
                self.bridge.send_dm(self.controller_address, payload)
                print(f"[remote] Sent handshake to controller at {self.controller_address}")
                self._handshake_sent = True
            except Exception as exc:  # pragma: no cover
                print(f"[remote] Failed to send handshake: {exc}")

    def _on_ready(self, address: str) -> None:
        print(f"[remote] NKN bridge ready at {address}")
        self._send_handshake()

    def _on_status(self, message: str) -> None:
        if message:
            print(f"[remote] NKN status: {message}")

    def _on_error(self, message: str) -> None:
        if message:
            print(f"[remote] NKN error: {message}")

    def _on_message(self, src: str, body: Dict[str, Any]) -> None:
        if not isinstance(body, dict) or body.get("type") != "command":
            return
        threading.Thread(target=self._execute_command, args=(src, body), daemon=True).start()

    def _execute_command(self, src: str, body: Dict[str, Any]) -> None:
        session_id = body.get("session_id")

        def send(payload: Dict[str, object]) -> None:
            if session_id:
                payload = dict(payload)
                payload["session_id"] = session_id
            self.bridge.send_dm(src, payload)

        self._send_handshake()
        send({"type": "ack", "message": "Remote agent ready and awaiting commands."})
        cmd = body.get("cmd")
        if not isinstance(cmd, list):
            send({"type": "error", "message": "Invalid command payload."})
            return
        resolved_cmd = resolve_command(cmd)
        _stream_command(
            resolved_cmd,
            body.get("description"),
            self.env,
            send,
            session_id=session_id,
        )


def main() -> None:
    boot_remote_venv()
    parser = argparse.ArgumentParser(description="NKN-backed remote Dropbear executor.")
    parser.add_argument("--nkn-seed", type=str, default=os.environ.get("DROPBEAR_REMOTE_NKN_SEED", ""), help="NKN seed hex for this remote agent.")
    parser.add_argument("--nkn-identifier", type=str, default=f"dropbear_remote_{socket.gethostname()}", help="Identifier for the NKN bridge.")
    parser.add_argument("--nkn-num-subclients", type=int, default=2, help="Number of NKN sub-clients.")
    parser.add_argument("--app-address", type=str, default=os.environ.get("DROPBEAR_APP_NKN_ADDRESS", ""), help="Controller/app NKN address for handshake.")
    args = parser.parse_args()

    seed = args.nkn_seed.strip().lower().replace("0x", "")
    if not seed:
        raise RuntimeError("NKN seed hex is required to start the remote agent.")

    agent = NKNRemoteAgent(seed_hex=seed, identifier=args.nkn_identifier, num_subclients=args.nkn_num_subclients, controller_address=args.app_address)
    agent.run_blocking()


if __name__ == "__main__":
    main()
