#!/usr/bin/env python3
"""NKN-backed remote executor for Dropbear RL."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

try:
    import curses
except Exception:  # pragma: no cover
    curses = None

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
SEED_FILE = PROJECT_ROOT / ".remote_nkn_seed"


def _prepare_env() -> Dict[str, str]:
    env = prepend_path(dict(os.environ), venv_bin_dir(REMOTE_VENV_DIR))
    project_path = str(PROJECT_ROOT)
    isaaclab_path = str(PROJECT_ROOT / "IsaacLab")
    dropbear_path = str(DROPBEAR_EXTENSION_DIR)
    env.setdefault("ISAACLAB_PATH", isaaclab_path)
    pythonpath_parts = []
    raw_pythonpath = env.get("PYTHONPATH", "")
    if raw_pythonpath:
        pythonpath_parts.append(raw_pythonpath)
    pythonpath_parts.extend([project_path, isaaclab_path, dropbear_path])
    env["PYTHONPATH"] = os.pathsep.join(part for part in pythonpath_parts if part)
    return env


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


def _load_persistent_seed() -> str:
    if not SEED_FILE.exists():
        return ""
    try:
        return SEED_FILE.read_text(encoding="utf-8").strip().lower().replace("0x", "")
    except Exception:
        return ""


def _save_persistent_seed(seed: str) -> None:
    try:
        SEED_FILE.write_text(seed.lower(), encoding="utf-8")
    except Exception:
        pass


def _load_controller_address_from_config() -> str:
    try:
        config = PROJECT_ROOT / "isaaclab_remote_connection.json"
        if not config.exists():
            return ""
        payload = json.loads(config.read_text(encoding="utf-8"))
        nkn_cfg = payload.get("nkn", {})
        address = nkn_cfg.get("app_address") or ""
        return str(address).strip()
    except Exception:
        return ""


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
    return return_code


class _NullDisplay:
    def record_incoming(self, *_args, **_kwargs):  # pragma: no cover
        pass

    def record_outgoing(self, *_args, **_kwargs):  # pragma: no cover
        pass

    def record_log(self, *_args, **_kwargs):  # pragma: no cover
        pass

    def set_address(self, *_args, **_kwargs):  # pragma: no cover
        pass

    def set_status(self, *_args, **_kwargs):  # pragma: no cover
        pass

    def stop(self):  # pragma: no cover
        pass


class RemoteCursesUI:
    LOG_LIMIT = 8

    def __init__(self) -> None:
        self.address: str = ""
        self.status: str = "starting"
        self.incoming: str = ""
        self.outgoing: str = ""
        self.notification: str = ""
        self.notification_ts: float = 0.0
        self.logs: "deque[str]" = deque(maxlen=self.LOG_LIMIT)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        has_input = bool(getattr(sys.stdin, "isatty", lambda: False)())
        has_output = bool(sys.stdout.isatty())
        self._visible = bool(curses and has_input and has_output)
        self._thread: Optional[threading.Thread] = None
        if self._visible:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        try:
            curses.wrapper(self._curses_loop)
        except Exception as exc:  # pragma: no cover
            self._push_notification(f"UI error: {exc}")

    def _curses_loop(self, stdscr: "curses._CursesWindow") -> None:
        curses.curs_set(0)
        stdscr.nodelay(True)
        while not self._stop_event.is_set():
            self._draw(stdscr)
            try:
                key = stdscr.getch()
                if key in (ord("q"), ord("Q")):
                    self._stop_event.set()
                    break
            except curses.error:
                pass
            time.sleep(0.3)

    def _draw(self, stdscr: "curses._CursesWindow") -> None:
        stdscr.erase()
        try:
            height, width = stdscr.getmaxyx()
        except Exception:
            return
        with self._lock:
            address = self.address or "pending..."
            status = self.status or "starting"
            incoming = self.incoming or "<waiting>"
            outgoing = self.outgoing or "<waiting>"
            logs = list(self.logs)
            note = self.notification
            if note and (time.time() - self.notification_ts) > 8:
                note = ""
        max_width = max(1, width - 1)
        lines = [
            "Dropbear RL Remote Agent",
            "",
            f"NKN address: {address}",
            f"Status: {status}",
            "",
            f"Incoming: {incoming}",
            f"Outgoing: {outgoing}",
            "",
        ]
        for idx, line in enumerate(lines):
            if idx >= height - 2:
                break
            try:
                stdscr.addstr(idx, 0, line[:max_width])
            except curses.error:
                pass
        log_start = len(lines)
        for idx, log_line in enumerate(logs):
            row = log_start + idx
            if row >= height - 2:
                break
            try:
                stdscr.addstr(row, 0, log_line[:max_width])
            except curses.error:
                pass
        if note and height - 1 >= 0:
            try:
                stdscr.addstr(height - 1, 0, note[:max_width], curses.A_REVERSE)
            except curses.error:
                pass
        stdscr.refresh()

    def _push_notification(self, message: str) -> None:
        if not message:
            return
        with self._lock:
            self.notification = message
            self.notification_ts = time.time()

    def record_log(self, message: str) -> None:
        if not message:
            return
        text = message.strip()
        if not text:
            return
        with self._lock:
            self.logs.append(text)
            self._push_notification(text)
        if not self._visible:
            print(text)

    def set_address(self, address: str) -> None:
        with self._lock:
            self.address = address or ""
        self._push_notification(f"Local address: {address}")

    def set_status(self, status: str) -> None:
        with self._lock:
            self.status = status or ""
        self._push_notification(status)

    def record_incoming(self, summary: str) -> None:
        with self._lock:
            self.incoming = summary or ""
        self._push_notification(f"Incoming RL payload: {summary}")

    def record_outgoing(self, summary: str) -> None:
        with self._lock:
            self.outgoing = summary or ""
        self._push_notification(f"Outgoing RL payload: {summary}")


def detect_cuda_version() -> str:
    """Detect CUDA version on the system."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        driver = result.stdout.strip().split('\n')[0]
        print(f"[remote] Detected NVIDIA driver: {driver}")
        print("[remote] Using cu128 PyTorch build (CUDA 12.8 support)")
        return "cu128"
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[remote] Warning: Could not detect CUDA, defaulting to cu128")
        return "cu128"


def verify_torch_cuda(python_exe: Path, env: Dict[str, str]) -> bool:
    """Verify PyTorch CUDA is working."""
    test_script = """
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    x = torch.randn(3, 3, device='cuda')
    y = x @ x
    print("✓ GPU tensor ops work")
"""
    try:
        subprocess.check_call([str(python_exe), "-c", test_script], env=env)
        return True
    except subprocess.CalledProcessError:
        return False


def boot_remote_venv() -> None:
    if INSIDE_FLAG in sys.argv:
        sys.argv.remove(INSIDE_FLAG)
        return

    print("[remote] ============================================")
    print("[remote] Remote A100 Worker Bootstrap")
    print("[remote] ============================================")

    py311 = find_python_311()
    if not py311:
        raise RuntimeError("Python 3.11 required")
    print(f"[remote] Python: {py311}")

    create_venv_with_python(py311, REMOTE_VENV_DIR)
    remote_python = venv_python(REMOTE_VENV_DIR)
    if not remote_python.exists():
        raise RuntimeError("Failed to create venv")

    env = prepend_path(dict(os.environ), venv_bin_dir(REMOTE_VENV_DIR))

    if not REMOTE_MARKER.exists():
        print("[remote] Installing: PyTorch+CUDA, rsl-rl, dropbear_rl_lab[remote]")
        print("[remote] NOT installing: Isaac Sim, IsaacLab")

        print("\n[1/5] Upgrading pip...")
        run_cmd([str(remote_python), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"], env=env)

        cuda_ver = detect_cuda_version()
        print(f"\n[2/5] Installing PyTorch ({cuda_ver})...")
        torch_index = f"https://download.pytorch.org/whl/{cuda_ver}"
        run_cmd([str(remote_python), "-m", "pip", "install",
                 f"torch=={DEFAULT_TORCH_VERSION}",
                 f"torchvision=={DEFAULT_TORCHVISION_VERSION}",
                 f"torchaudio=={DEFAULT_TORCHAUDIO_VERSION}",
                 "--index-url", torch_index], env=env)

        if not verify_torch_cuda(remote_python, env):
            raise RuntimeError("PyTorch CUDA verification failed")

        print("\n[3/5] Installing rsl-rl...")
        run_cmd([str(remote_python), "-m", "pip", "install", "rsl-rl-lib>=2.3.1"], env=env)

        print("\n[4/5] Installing dropbear_rl_lab[remote]...")
        run_cmd([str(remote_python), "-m", "pip", "install", "-e", f"{DROPBEAR_EXTENSION_DIR}[remote]"], env=env)

        print("\n[5/5] Verifying...")
        for mod in ["gymnasium", "numpy", "rsl_rl", "dropbear_rl_lab.remote"]:
            try:
                subprocess.check_call([str(remote_python), "-c", f"import {mod}"], env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"[remote] ✓ {mod}")
            except subprocess.CalledProcessError:
                raise RuntimeError(f"{mod} not installed")

        try:
            subprocess.check_call([str(remote_python), "-c", "import isaaclab"], env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            raise RuntimeError("IsaacLab detected - environment contaminated!")
        except subprocess.CalledProcessError:
            print("[remote] ✓ No IsaacLab (correct)")

        ensure_actor_critic_std(REMOTE_VENV_DIR)
        write_marker(REMOTE_VENV_DIR, REMOTE_MARKER.name)
        print("\n[remote] ============================================")
        print("[remote] Bootstrap Complete - Ready for RL!")
        print("[remote] ============================================\n")

    os.execv(str(remote_python), [str(remote_python), str(__file__), INSIDE_FLAG] + [arg for arg in sys.argv[1:]])


class NKNRemoteAgent:
    def __init__(
        self,
        seed_hex: str,
        identifier: str,
        num_subclients: int,
        controller_address: str,
        enable_ui: bool = True,
    ) -> None:
        if not seed_hex:
            raise ValueError("NKN seed hex is required for the remote agent.")
        self.env = _prepare_env()
        # Propagate NKN info to child commands so they can reuse connectivity
        self.env["DROPBEAR_REMOTE_NKN_SEED"] = seed_hex
        self.env["DROPBEAR_REMOTE_NKN_IDENTIFIER"] = identifier
        if controller_address:
            self.env["DROPBEAR_CONTROLLER_NKN_ADDRESS"] = controller_address
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
        self.display = RemoteCursesUI() if enable_ui else _NullDisplay()
        self._running = False
        self._handshake_lock = threading.Lock()
        self._handshake_sent = False

    def _log(self, message: str) -> None:
        self.display.record_log(message)

    def start(self) -> None:
        self.bridge.start()
        if not self.bridge.wait_ready(timeout=30.0):
            raise RuntimeError("NKN bridge failed to become ready.")
        self._running = True
        self._log("[remote] NKN remote agent is up; waiting for commands.")
        self._send_handshake()

    def stop(self) -> None:
        if self._running:
            self._running = False
        self.bridge.stop()
        self.display.stop()
        self._log("[remote] NKN remote agent shutting down.")

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
                self._log(f"[remote] Sent handshake to controller at {self.controller_address}")
                self._handshake_sent = True
            except Exception as exc:  # pragma: no cover
                self._log(f"[remote] Failed to send handshake: {exc}")

    def _on_ready(self, address: str) -> None:
        self._log(f"[remote] NKN bridge ready at {address}")
        self.display.set_address(address)
        self.display.set_status("bridge ready")
        self._send_handshake()

    def _on_status(self, message: str) -> None:
        if message:
            status = f"[remote] NKN status: {message}"
            self._log(status)
            self.display.set_status(message)

    def _on_error(self, message: str) -> None:
        if message:
            self._log(f"[remote] NKN error: {message}")

    def _on_message(self, src: str, body: Dict[str, Any]) -> None:
        if not isinstance(body, dict):
            return
        typ = body.get("type")
        if typ == "handshake_ack":
            addr = body.get("address") or ""
            status = body.get("status") or "controller ready"
            self._log(f"[remote] Controller handshake ack: {status} ({addr})")
            self.display.set_status(status)
            return
        if typ != "command":
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
        description = body.get("description") or "remote run"
        preview_parts = " ".join(str(part) for part in cmd[:4])
        if len(cmd) > 4:
            preview_parts = f"{preview_parts} ..."
        incoming_summary = f"{description} ({preview_parts or '<no args>'})"
        self.display.record_incoming(incoming_summary)
        self.display.record_outgoing(f"ack {session_id or '<no-id>'}")
        resolved_cmd = resolve_command(cmd)

        # Inline execution for train_remote to reuse the active NKN sidecar
        inline_handled = False
        try:
            if (
                len(resolved_cmd) >= 2
                and resolved_cmd[0] in ("python", "python3")
                and resolved_cmd[1].endswith("scripts/rsl_rl/train_remote.py")
            ):
                inline_handled = True
                send({"type": "start", "description": description})
                try:
                    import importlib.machinery
                    import importlib.util
                    from remote_protocol_rl import MessageSequencer, create_train_start_message

                    self.display.record_log("[remote] Inline train_remote starting...")
                    argv = resolved_cmd[2:]

                    # Extract controller address for an early train_start signal
                    controller_addr = ""
                    for idx, part in enumerate(argv):
                        if part.startswith("--controller_address="):
                            controller_addr = part.split("=", 1)[1]
                            break
                        if part == "--controller_address" and idx + 1 < len(argv):
                            controller_addr = argv[idx + 1]
                            break
                    controller_addr = controller_addr.strip()
                    if controller_addr and self.bridge:
                        try:
                            seq = MessageSequencer()
                            start_msg = create_train_start_message(
                                seq,
                                task_config={},
                                agent_config={},
                                worker_address=self.bridge.address or "",
                            )
                            self.bridge.send_dm(controller_addr, start_msg.to_dict())
                            self.display.record_outgoing(f"train_start → {controller_addr}")
                        except Exception as exc:  # pragma: no cover
                            self.display.record_log(f"[remote] Failed to send early train_start: {exc}")

                    # Load train_remote.py directly from path to avoid package issues
                    tr_path = PROJECT_ROOT / "scripts" / "rsl_rl" / "train_remote.py"
                    loader = importlib.machinery.SourceFileLoader("train_remote_inline", str(tr_path))
                    spec = importlib.util.spec_from_loader(loader.name, loader)
                    if spec is None or spec.loader is None:
                        raise RuntimeError("Could not load train_remote module")
                    train_mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(train_mod)  # type: ignore

                    train_mod.main(provided_sidecar=self.bridge, argv=argv)
                    exit_code = 0
                except SystemExit as exc:  # capture argparse exits
                    exit_code = int(exc.code or 0)
                except Exception as exc:
                    exit_code = 1
                    send({"type": "error", "message": f"Inline train_remote failed: {exc}"})
                    self.display.record_log(f"[remote] Inline train_remote failed: {exc}")
                else:
                    self.display.record_log("[remote] Inline train_remote completed.")
                send({"type": "exit", "code": exit_code})
        except Exception as exc:
            inline_handled = False
            send({"type": "error", "message": f"Inline dispatch check failed: {exc}"})

        if not inline_handled:
            _stream_command(
                resolved_cmd,
                body.get("description"),
                self.env,
                send,
                session_id=session_id,
            )
        self.display.record_outgoing(f"completed {description}")


def main() -> None:
    boot_remote_venv()
    parser = argparse.ArgumentParser(description="NKN-backed remote Dropbear executor.")
    parser.add_argument("--nkn-seed", type=str, default=os.environ.get("DROPBEAR_REMOTE_NKN_SEED", ""), help="NKN seed hex for this remote agent.")
    parser.add_argument("--nkn-identifier", type=str, default=f"dropbear_remote_{socket.gethostname()}", help="Identifier for the NKN bridge.")
    parser.add_argument("--nkn-num-subclients", type=int, default=2, help="Number of NKN sub-clients.")
    parser.add_argument("--app-address", type=str, default=os.environ.get("DROPBEAR_APP_NKN_ADDRESS", ""), help="Controller/app NKN address for handshake.")
    parser.add_argument("--no-ui", action="store_true", help="Disable curses UI (stdout-only logging).")
    args = parser.parse_args()

    seed = args.nkn_seed.strip().lower().replace("0x", "")
    if not seed:
        seed = _load_persistent_seed()
    if not seed:
        seed = secrets.token_hex(32)
        print(f"[remote] Generated NKN seed for remote agent: {seed}")
    _save_persistent_seed(seed)

    controller_address = args.app_address.strip() or _load_controller_address_from_config()
    agent = NKNRemoteAgent(
        seed_hex=seed,
        identifier=args.nkn_identifier,
        num_subclients=args.nkn_num_subclients,
        controller_address=controller_address,
        enable_ui=not args.no_ui,
    )
    agent.run_blocking()


if __name__ == "__main__":
    main()
