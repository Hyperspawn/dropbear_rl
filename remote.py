"""Remote IsaacLab runner that listens for commands from the local controller."""

import argparse
import json
import os
import socket
import socketserver
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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

def _send_payload(writer, payload: Dict[str, object]) -> None:
    writer.write(remote_protocol.encode_message(payload))
    writer.flush()


def _stream_command(
    cmd: List[str],
    description: Optional[str],
    env: Dict[str, str],
    writer,
) -> None:
    desc = description or "run started"
    _send_payload(writer, {"type": "start", "description": desc})
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
        _send_payload(writer, {"type": "error", "message": f"Failed to start command: {exc}"})
        _send_payload(writer, {"type": "exit", "code": 1})
        return
    assert process.stdout is not None
    for line in process.stdout:
        _send_payload(writer, {"type": "log", "message": line.rstrip()})
    return_code = process.wait()
    _send_payload(writer, {"type": "exit", "code": return_code})


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
        env = _prepare_env()
        resolved_cmd = resolve_command(cmd)
        _stream_command(resolved_cmd, description, env, self.wfile)

    def _send(self, payload: Dict[str, object]) -> None:
        self.wfile.write(remote_protocol.encode_message(payload))
        self.wfile.flush()


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
    parser.add_argument("--target-host", type=str, default="", help="Local app host to connect to for reverse remote.")
    parser.add_argument("--target-port", type=int, default=8765, help="Local app port for reverse remote.")
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Prompt for the reverse listener host:port before connecting.",
    )
    args = parser.parse_args()
    if args.reverse or args.target_host:
        target_host = args.target_host
        target_port = args.target_port
        if args.reverse or not target_host:
            target_host, target_port = _prompt_reverse_target(target_port)
        run_reverse_agent(target_host, target_port)
        return
        run_reverse_agent(args.target_host, args.target_port)
        return
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


def _prompt_reverse_target(default_port: int) -> Tuple[str, int]:
    prompt = (
        "\nPaste the reverse listener target that the RTX app is exposing (host[:port]).\n"
        f"Leave blank to cancel. Default port is {default_port}: "
    )
    try:
        response = input(prompt).strip()
    except EOFError:
        raise RuntimeError("Reverse target prompt aborted.")
    if not response:
        raise RuntimeError("No reverse listener provided.")
    parts = response.split(":")
    host = parts[0].strip()
    port = default_port
    if len(parts) > 1:
        try:
            port = int(parts[1])
        except ValueError:
            raise RuntimeError(f"Invalid port: {parts[1]}")
    if not host:
        raise RuntimeError("Reverse listener host is empty.")
    return host, port


def run_reverse_agent(target_host: str, target_port: int) -> None:
    if not target_host or target_port <= 0:
        raise RuntimeError("Reverse target host and port must be provided.")
    env = _prepare_env()
    try:
        with socket.create_connection((target_host, target_port), timeout=5) as sock:
            reader = sock.makefile("rb")
            writer = sock.makefile("wb")
            _send_payload(
                writer,
                {
                    "type": "handshake",
                    "session_id": str(uuid.uuid4()),
                    "description": f"reverse agent {socket.gethostname()}",
                },
            )
            _send_payload(writer, {"type": "ack", "message": "Remote agent ready and awaiting commands."})
            for msg in remote_protocol.iter_messages(reader):
                if msg.get("type") != "command":
                    continue
                cmd = msg.get("cmd")
                if not isinstance(cmd, list):
                    _send_payload(writer, {"type": "error", "message": "Invalid command payload."})
                    continue
                resolved_cmd = resolve_command(cmd)
                _stream_command(resolved_cmd, msg.get("description"), env, writer)
    except Exception as exc:
        print(f"[remote] Reverse agent connection failed: {exc}", flush=True)
        sys.exit(1)
if __name__ == "__main__":
    main()
