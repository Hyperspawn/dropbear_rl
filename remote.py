"""Remote IsaacLab runner that listens for commands from the local controller."""

import argparse
import json
import os
import socket
import socketserver
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import remote_protocol

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

PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_VENV_NAME = "env_remote"
REMOTE_VENV_DIR = PROJECT_ROOT / REMOTE_VENV_NAME
REMOTE_MARKER = REMOTE_VENV_DIR / ".remote_bootstrap_ok"
INSIDE_FLAG = "--_inside-remote"


class RemoteRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline()
        if not raw:
            return
        try:
            request = json.loads(raw.decode("utf-8").strip())
        except Exception:
            return
        cmd = request.get("cmd")
        if not isinstance(cmd, list):
            self._send({"type": "error", "message": "Invalid command payload."})
            return
        description = request.get("description", "remote run")
        self._send({"type": "ack", "message": "Remote server received request and is preparing IsaacLab."})
        self._send({"type": "start", "description": description})
        resolved_cmd = self._resolve_command(cmd)
        try:
            env = prepend_path(dict(os.environ), venv_bin_dir(REMOTE_VENV_DIR))
            process = subprocess.Popen(
            resolved_cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        except Exception as exc:
            self._send({"type": "error", "message": f"Failed to start command: {exc}"})
            self._send({"type": "exit", "code": 1})
            return

        assert process.stdout is not None
        for line in process.stdout:
            self._send_log(line.rstrip())
        return_code = process.wait()
        self._send({"type": "exit", "code": return_code})

    def _resolve_command(self, cmd: Iterable[str]) -> List[str]:
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

    def _send(self, payload: Dict[str, object]) -> None:
        self.wfile.write(remote_protocol.encode_message(payload))
        self.wfile.flush()

    def _send_log(self, line: str) -> None:
        self._send({"type": "log", "message": line})


class RemoteServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


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


def main() -> None:
    boot_remote_venv()
    parser = argparse.ArgumentParser(description="Remote Dropbear RL server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host/interface to listen on.")
    parser.add_argument("--port", type=int, default=8721, help="Port to listen on.")
    args = parser.parse_args()
    server = RemoteServer((args.host, args.port), RemoteRequestHandler)
    advertised_host = args.host
    if args.host in ("0.0.0.0", ""):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as test_sock:
                test_sock.connect(("8.8.8.8", 80))
                advertised_host = test_sock.getsockname()[0]
        except Exception:
            advertised_host = "0.0.0.0"
    print(f"[remote] Listening on {args.host}:{args.port} (reachable via {advertised_host}:{args.port})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
