"""Remote IsaacLab runner that listens for commands from the local controller."""

import argparse
import contextlib
import curses
import json
import os
import socket
import socketserver
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

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
from nkn_sidecar import NKNSidecar

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
    send_payload: Callable[[Dict[str, object]], None],
    session_id: Optional[str] = None,
) -> None:
    desc = description or "run started"
    def _send(payload: Dict[str, object]) -> None:
        if session_id:
            payload = dict(payload)
            payload["session_id"] = session_id
        send_payload(payload)

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


def _pick_accessible_port(preferred: int, max_port: int = 5006) -> int:
    upper = min(max_port, 65535)
    start = preferred
    if preferred > upper:
        start = max(5003, upper - (max_port - 5003))
    for candidate in range(start, upper + 1):
        with contextlib.suppress(OSError):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tester:
                tester.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                tester.bind(("0.0.0.0", candidate))
                return candidate
    raise RuntimeError(f"Could not bind any port between {preferred} and {max_port}.")


def _run_direct_server(host: str, port: int) -> None:
    accessible_port = _pick_accessible_port(port, max_port=5006)
    if accessible_port != port:
        print(f"[remote] Port {port} busy; using {accessible_port} instead.")
    advertised_host = host
    if host in ("0.0.0.0", ""):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as test_sock:
                test_sock.connect(("8.8.8.8", 80))
                advertised_host = test_sock.getsockname()[0]
        except Exception:
            advertised_host = host
    responder = DiscoveryResponder(advertised_host, accessible_port)
    responder.start()
    server = RemoteServer((host, accessible_port), RemoteRequestHandler)
    print(f"[remote] Listening on {host}:{accessible_port} (reachable via {advertised_host}:{accessible_port})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        responder.stop()
        server.shutdown()


def _parse_host_port(value: str, default_port: int) -> Tuple[str, int]:
    parts = value.strip().split(":")
    host = parts[0].strip()
    if not host:
        raise ValueError("Host cannot be empty.")
    port = default_port
    if len(parts) > 1 and parts[1].strip():
        port = int(parts[1].strip())
    return host, port


DISCOVERY_PORT = 5005


class DiscoveryResponder(threading.Thread):
    def __init__(self, host: str, port: int) -> None:
        super().__init__(daemon=True)
        self._host = host
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._running = threading.Event()

    def run(self) -> None:
        self._running.set()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(0.5)
        sock.bind(("0.0.0.0", DISCOVERY_PORT))
        self._sock = sock
        while self._running.is_set():
            try:
                raw, addr = sock.recvfrom(1024)
            except socket.timeout:
                continue
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                continue
            if payload.get("type") != "discover":
                continue
            response = json.dumps(
                {"type": "discover_response", "host": self._host, "port": self._port}
            ).encode("utf-8")
            sock.sendto(response, addr)

    def stop(self) -> None:
        self._running.clear()
        if self._sock:
            self._sock.close()


def _curses_prompt(stdscr: Any, prompt: str, default: str) -> Optional[str]:
    curses.echo()
    curses.curs_set(1)
    stdscr.erase()
    stdscr.addstr(0, 0, prompt)
    stdscr.addstr(2, 0, f"(default: {default})")
    stdscr.refresh()
    try:
        line = stdscr.getstr(4, 0, 64)
    finally:
        curses.noecho()
        curses.curs_set(0)
    if not line:
        return None
    return line.decode("utf-8", errors="ignore").strip()


def run_curses_menu(stdscr: Any, args) -> Tuple[Optional[str], dict]:
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)

    options = ["Start direct server", "Start reverse agent", "Quit"]
    selected = 0
    reverse_target = args.target_host
    reverse_port = args.target_port
    error_msg: Optional[str] = None
    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        stdscr.addstr(0, 0, "Remote runner menu", curses.color_pair(1) | curses.A_BOLD)
        for idx, desc in enumerate(options):
            attr = curses.A_REVERSE if idx == selected else curses.A_NORMAL
            stdscr.addstr(2 + idx, 2, f"{'> ' if idx == selected else '  '}{desc}", attr)
        stdscr.addstr(6, 2, f"Direct listen: {args.host}:{args.port}")
        stdscr.addstr(7, 2, f"Reverse target: {reverse_target or 'not set'}:{reverse_port}")
        if error_msg:
            stdscr.addstr(9, 2, error_msg[: max(0, width - 4)], curses.color_pair(2))
        stdscr.addstr(height - 2, 2, "Use ↑/↓ to choose, Enter to activate, q to exit.", curses.A_DIM)
        stdscr.refresh()

        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            selected = (selected - 1) % len(options)
        elif key in (curses.KEY_DOWN, ord("j")):
            selected = (selected + 1) % len(options)
        elif key in (10, 13):
            if selected == 0:
                return "server", {}
            elif selected == 1:
                prompt = "Enter reverse listener target (host[:port]) or leave blank to cancel:"
                response = _curses_prompt(stdscr, prompt, f"{reverse_target or 'host'}:{reverse_port}")
                if not response:
                    continue
                try:
                    host, port = _parse_host_port(response, reverse_port)
                except ValueError as exc:
                    error_msg = f"Invalid input: {exc}"
                    continue
                reverse_target = host
                reverse_port = port
                return "reverse", {"host": host, "port": port}
            else:
                return None, {}
        elif key in (ord("q"), 27):
            return None, {}
        else:
            error_msg = None
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
        session_id = request.get("session_id")
        ack_payload = {"type": "ack", "message": "Remote server received request and is preparing IsaacLab."}
        if session_id:
            ack_payload["session_id"] = session_id
        self._send(ack_payload)
        env = _prepare_env()
        resolved_cmd = resolve_command(cmd)
        _stream_command(
            resolved_cmd,
            description,
            env,
            lambda payload: self._send(payload),
            session_id=session_id,
        )

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
    parser.add_argument("--port", type=int, default=5003, help="Port to listen on.")
    parser.add_argument("--target-host", type=str, default="", help="Local app host to connect to for reverse remote.")
    parser.add_argument(
        "--target-port",
        type=int,
        default=remote_client.get_reverse_port(),
        help="Local app port to connect to for reverse remote (defaults to the reverse listener port).",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Prompt for the reverse listener host:port before connecting.",
    )
    parser.add_argument("--menu", dest="menu", action="store_true", help="Use the curses menu.")
    parser.add_argument("--no-menu", dest="menu", action="store_false", help="Skip the curses menu.")
    parser.add_argument(
        "--nkn",
        action="store_true",
        help="Run the remote agent over NKN instead of raw sockets.",
    )
    parser.add_argument("--nkn-seed", type=str, default="", help="Seed hex for the NKN bridge (required with --nkn).")
    parser.add_argument("--nkn-identifier", type=str, default="dropbear_remote", help="Identifier for the NKN bridge.")
    parser.add_argument(
        "--nkn-num-subclients",
        type=int,
        default=2,
        help="Number of NKN sub-clients to spawn under the bridge.",
    )
    parser.set_defaults(menu=sys.stdin.isatty())
    args = parser.parse_args()

    if args.nkn:
        seed = args.nkn_seed or os.environ.get("DROPBEAR_NKN_SEED", "")
        if not seed:
            raise RuntimeError("NKN seed hex is required to run in --nkn mode.")
        agent = NKNRemoteAgent(seed, args.nkn_identifier, max(1, args.nkn_num_subclients))
        agent.run_blocking()
        return

    if args.menu and sys.stdin.isatty():
        action, payload = curses.wrapper(run_curses_menu, args)
        if action == "server":
            _run_direct_server(args.host, args.port)
            return
        if action == "reverse":
            run_reverse_agent(payload["host"], payload["port"])
            return
        return

    if args.reverse or args.target_host:
        target_host = args.target_host
        target_port = args.target_port
        if args.reverse or not target_host:
            target_host, target_port = _prompt_reverse_target(target_port)
        run_reverse_agent(target_host, target_port)
        return

    _run_direct_server(args.host, args.port)


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
    while True:
        try:
            with socket.create_connection((target_host, target_port), timeout=5) as sock:
                reader = sock.makefile("rb")
                writer = sock.makefile("wb")
                session_id = str(uuid.uuid4())
                _send_payload(
                    writer,
                    {
                        "type": "handshake",
                        "session_id": session_id,
                        "description": f"reverse agent {socket.gethostname()}",
                    },
                )
                _send_payload(writer, {"type": "ack", "message": "Remote agent ready and awaiting commands.", "session_id": session_id})
                send_fn = lambda payload: _send_payload(writer, payload)
                for msg in remote_protocol.iter_messages(reader):
                    if msg.get("type") != "command":
                        continue
                    cmd = msg.get("cmd")
                    if not isinstance(cmd, list):
                        _send_payload(writer, {"type": "error", "message": "Invalid command payload."})
                        continue
                    resolved_cmd = resolve_command(cmd)
                    _stream_command(
                        resolved_cmd,
                        msg.get("description"),
                        env,
                        send_fn,
                        session_id=msg.get("session_id"),
                    )
            except Exception as exc:
                print(f"[remote] Reverse agent connection failed: {exc}, retrying in 2s...", flush=True)
                time.sleep(2)


class NKNRemoteAgent:
    def __init__(self, seed_hex: str, identifier: str, num_subclients: int) -> None:
        if not seed_hex:
            raise ValueError("NKN seed hex is required for the remote agent.")
        self.env = _prepare_env()
        self.bridge = NKNSidecar(
            seed_hex=seed_hex,
            identifier=identifier,
            num_subclients=num_subclients,
            on_ready=self._on_ready,
            on_status=self._on_status,
            on_message=self._on_message,
            on_error=self._on_error,
        )
        self._running = False

    def start(self) -> None:
        self.bridge.start()
        if not self.bridge.wait_ready(timeout=30.0):
            raise RuntimeError("NKN bridge failed to become ready.")
        self._running = True
        print("[remote] NKN remote agent is up; waiting for commands.")

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

    def _on_ready(self, address: str) -> None:
        print(f"[remote] NKN bridge ready at {address}")

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

        send({"type": "handshake", "message": "Remote agent over NKN", "address": self.bridge.address})
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
if __name__ == "__main__":
    main()
