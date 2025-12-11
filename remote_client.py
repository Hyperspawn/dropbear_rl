"""Client helpers for dispatching commands to a remote RL executor."""

import json
import queue
import random
import secrets
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import remote_protocol
import reverse_remote
from nkn_sidecar import NKNSidecar

PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_CONFIG_FILE = PROJECT_ROOT / "isaaclab_remote_connection.json"
DEFAULT_REMOTE_CONFIG = {
    "enabled": False,
    "mode": "reverse",
    "host": "127.0.0.1",
    "port": 5003,
    "reverse_port": 5004,
    "nkn": {
        "seed": "",
        "identifier": "dropbear_app",
        "target": "",
        "remote_address": "",
        "num_subclients": 2,
        "seed_ws": "",
    },
}

DISCOVERY_PORT = 5005
DISCOVERY_STATUS_IDLE = "idle"
DISCOVERY_STATUS_SCANNING = "scanning for remote"
DISCOVERY_STATUS_FOUND = "remote discovered"
DISCOVERY_STATUS_FAILED = "discovery failed"

REVERSE_PORT_CANDIDATES = [5004, 5005, 5006]

_remote_config_cache: Optional[Dict[str, object]] = None
_nkn_control: Optional["_NKNControl"] = None


def _load_remote_config() -> Dict[str, object]:
    global _remote_config_cache
    if _remote_config_cache is not None:
        return _remote_config_cache
    data = dict(DEFAULT_REMOTE_CONFIG)
    if REMOTE_CONFIG_FILE.exists():
        try:
            raw = json.loads(REMOTE_CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                data.update(raw)
        except Exception:
            pass
    _ensure_nkn_config(data)
    _remote_config_cache = data
    return data


def _ensure_nkn_config(cfg: Dict[str, object]) -> None:
    defaults = DEFAULT_REMOTE_CONFIG.get("nkn", {})
    merged: Dict[str, object] = dict(defaults)
    raw = cfg.get("nkn")
    if isinstance(raw, dict):
        merged.update(raw)
    cfg["nkn"] = merged


def _ensure_nkn_seed(cfg: Dict[str, object]) -> None:
    nkn_cfg = cfg.get("nkn")
    if not isinstance(nkn_cfg, dict):
        return
    if nkn_cfg.get("seed"):
        return
    nkn_cfg["seed"] = secrets.token_hex(32)
    try:
        save_remote_config(cfg)
    except Exception:
        pass


def get_remote_config() -> Dict[str, object]:
    return dict(_load_remote_config())


def save_remote_config(config: Dict[str, object]) -> None:
    global _remote_config_cache
    data = dict(DEFAULT_REMOTE_CONFIG)
    # copy top-level fields except nested NKN config
    for key, value in config.items():
        if key == "nkn":
            continue
        data[key] = value
    nkn_overrides = config.get("nkn")
    merged_nkn: Dict[str, object] = dict(DEFAULT_REMOTE_CONFIG["nkn"])
    if isinstance(nkn_overrides, dict):
        merged_nkn.update(nkn_overrides)
    data["nkn"] = merged_nkn
    if "port" in data:
        try:
            data["port"] = int(data["port"])
        except Exception:
            data["port"] = DEFAULT_REMOTE_CONFIG["port"]
    if "reverse_port" in data:
        try:
            data["reverse_port"] = int(data["reverse_port"])
        except Exception:
            data["reverse_port"] = DEFAULT_REMOTE_CONFIG["reverse_port"]
    try:
        REMOTE_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        REMOTE_CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        _remote_config_cache = data
    except Exception:
        pass


def is_remote_enabled() -> bool:
    cfg = _load_remote_config()
    if not cfg.get("enabled"):
        return False
    mode = cfg.get("mode", "reverse")
    if mode == "direct":
        return bool(cfg.get("host") and cfg.get("port"))
    if mode == "nkn":
        nkn_cfg = cfg.get("nkn", {})
        return bool(nkn_cfg.get("target"))
    return True


def is_reverse_mode() -> bool:
    cfg = _load_remote_config()
    return cfg.get("mode", "") == "reverse"


def get_reverse_port() -> int:
    cfg = _load_remote_config()
    port = cfg.get("reverse_port", DEFAULT_REMOTE_CONFIG["reverse_port"])
    try:
        return int(port)
    except Exception:
        return DEFAULT_REMOTE_CONFIG["reverse_port"]


def _normalize_command(cmd: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    root = PROJECT_ROOT
    for part in cmd:
        if not isinstance(part, str):
            normalized.append(str(part))
            continue
        p = Path(part)
        if p.is_absolute():
            try:
                rel = p.relative_to(root)
            except Exception:
                normalized.append(part)
            else:
                normalized.append(str(rel))
        else:
            normalized.append(part)
    return normalized


_discovery_status: str = DISCOVERY_STATUS_IDLE


def get_discovery_status() -> str:
    return _discovery_status


def _set_discovery_status(value: str) -> None:
    global _discovery_status
    _discovery_status = value


def discover_remote_agent(timeout: float = 1.0, attempts: int = 3) -> Optional[Tuple[str, int]]:
    msg = json.dumps({"type": "discover"}).encode("utf-8")
    for _ in range(attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.bind(("0.0.0.0", 0))
                sock.settimeout(timeout)
                sock.sendto(msg, ("255.255.255.255", DISCOVERY_PORT))
                raw, _ = sock.recvfrom(2048)
                data = json.loads(raw.decode("utf-8"))
                if data.get("type") == "discover_response":
                    host = data.get("host")
                    port = data.get("port")
                    if host and port:
                        return host, int(port)
        except (socket.timeout, json.JSONDecodeError, OSError):
            continue
    return None


def auto_discover_remote(timeout: float = 1.0) -> Optional[Tuple[str, int]]:
    _set_discovery_status(DISCOVERY_STATUS_SCANNING)
    result = discover_remote_agent(timeout=timeout)
    if result:
        host, port = result
        ok, info = test_remote_connection(host, port, timeout=0.5)
        if ok:
            _set_discovery_status(f"{DISCOVERY_STATUS_FOUND}: {host}:{port}")
            cfg = _load_remote_config()
            cfg["host"] = host
            cfg["port"] = port
            save_remote_config(cfg)
            return host, port
        _set_discovery_status(DISCOVERY_STATUS_FAILED)
        return None
    _set_discovery_status(DISCOVERY_STATUS_FAILED)
    return None
_listener_status: str = "waiting for agent"


def get_listener_status() -> str:
    return _listener_status


def reset_listener_status() -> None:
    global _listener_status
    _listener_status = "waiting for agent"


def _pick_free_port(preferred: int) -> int:
    candidates = [preferred] + [p for p in REVERSE_PORT_CANDIDATES if p != preferred]
    for candidate in candidates:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("0.0.0.0", candidate))
                return sock.getsockname()[1]
        except OSError:
            continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", 0))
        return sock.getsockname()[1]


def _remote_log(msg: str) -> None:
    print(f"[remote] {msg.rstrip()}", flush=True)
    if "Remote handshake" in msg:
        global _listener_status
        _listener_status = msg.strip()


def _remote_ack(msg: str) -> None:
    print(f"[remote] ack: {msg.rstrip()}", flush=True)
    global _listener_status
    _listener_status = msg.strip()


class _NKNClient:
    def __init__(self, cfg: Dict[str, object], config_updater: Callable[[Dict[str, object]], None]) -> None:
        self.cfg = cfg
        self.config_updater = config_updater
        nkn_cfg = self.cfg.setdefault("nkn", {})
        self.target_addr = str(nkn_cfg.get("target") or nkn_cfg.get("remote_address") or "")
        seed = str(nkn_cfg.get("seed") or "")
        identifier = str(nkn_cfg.get("identifier") or "dropbear_app")
        num_subclients = max(1, int(nkn_cfg.get("num_subclients") or 2))
        seed_ws = str(nkn_cfg.get("seed_ws") or "")
        self.sidecar = NKNSidecar(
            seed_hex=seed,
            identifier=identifier,
            num_subclients=num_subclients,
            seed_ws=seed_ws,
            on_ready=self._on_ready,
            on_status=self._on_status,
            on_message=self._on_message,
            on_error=self._on_error,
        )
        self.pending: Dict[str, queue.Queue[Dict[str, object]]] = {}
        self.lock = threading.Lock()
        self.ready_event = threading.Event()
        self.status_text = "starting"
        self.error_text: Optional[str] = None
        self.sidecar.start()
        self.sidecar.wait_ready(timeout=30.0)

    def shutdown(self) -> None:
        self.sidecar.stop()

    def _make_queue(self, session_id: Optional[str]) -> Optional[queue.Queue[Dict[str, object]]]:
        if not session_id:
            return None
        with self.lock:
            return self.pending.pop(session_id, None)

    def _update_remote_address(self, address: Optional[str]) -> None:
        if not address:
            return
        nkn_cfg = self.cfg.setdefault("nkn", {})
        if not nkn_cfg.get("target"):
            nkn_cfg["target"] = address
        self.target_addr = address
        previous_remote = nkn_cfg.get("remote_address")
        if previous_remote != address:
            nkn_cfg["remote_address"] = address
            try:
                self.config_updater(self.cfg)
            except Exception:
                pass

    def _on_ready(self, address: str) -> None:
        self.ready_event.set()
        if address:
            self.status_text = f"ready {address}"

    def _on_status(self, message: str) -> None:
        if message:
            self.status_text = message

    def _on_error(self, message: str) -> None:
        if message:
            self.error_text = message

    def _on_message(self, src: str, body: Dict[str, object]) -> None:
        typ = body.get("type")
        session_id = body.get("session_id")
        if typ == "handshake":
            self._update_remote_address(body.get("address") or body.get("addr"))
            return
        if typ == "log":
            _remote_log(body.get("message", ""))
            return
        if typ == "start":
            _remote_log(body.get("description", "run started"))
            return
        if typ == "ack":
            _remote_ack(body.get("message", "ack"))
            return
        if typ == "error":
            queue_obj = self._make_queue(session_id)
            msg = body.get("message", "remote error")
            if queue_obj:
                queue_obj.put({"type": "exit", "code": 1, "message": msg, "session_id": session_id})
            else:
                _remote_log(msg)
            return
        if typ == "exit":
            queue_obj = self._make_queue(session_id)
            if queue_obj:
                queue_obj.put({"type": "exit", "code": int(body.get("code", 0)), "session_id": session_id})

    def send_command(self, cmd: Iterable[str], description: Optional[str]) -> int:
        if not self.target_addr:
            raise RuntimeError("NKN target address not configured.")
        if not self.ready_event.wait(timeout=30.0):
            raise RuntimeError("NKN bridge not ready.")
        normalized_cmd = _normalize_command(cmd)
        session_id = str(uuid.uuid4())
        queue_obj: queue.Queue[Dict[str, object]] = queue.Queue()
        with self.lock:
            self.pending[session_id] = queue_obj
        payload = {
            "type": "command",
            "session_id": session_id,
            "description": description or "remote run",
            "cmd": normalized_cmd,
        }
        self.sidecar.send_dm(self.target_addr, payload)
        while True:
            msg = queue_obj.get()
            if msg.get("type") == "exit":
                return int(msg.get("code", 0))

    def stats(self) -> Dict[str, int]:
        return {
            "bytes_in": self.sidecar.bytes_in(),
            "bytes_out": self.sidecar.bytes_out(),
            "messages_in": self.sidecar.messages_in(),
            "messages_out": self.sidecar.messages_out(),
        }

    def status(self) -> str:
        result = self.status_text
        if self.error_text:
            return f"{result} (err: {self.error_text})"
        return result


class _NKNControl:
    def __init__(self, cfg: Dict[str, object]) -> None:
        _ensure_nkn_seed(cfg)
        self.cfg = cfg
        self.client = _NKNClient(cfg, save_remote_config)
        self.key = self._make_key(cfg)

    def _make_key(self, cfg: Dict[str, object]) -> tuple:
        nkn = cfg.get("nkn", {})
        return (
            str(nkn.get("seed", "")),
            str(nkn.get("identifier", "")),
            str(nkn.get("target", "")),
            str(nkn.get("num_subclients", "2")),
            str(nkn.get("seed_ws", "")),
        )

    def matches(self, cfg: Dict[str, object]) -> bool:
        return self.key == self._make_key(cfg)

    def send_command(self, cmd: Iterable[str], description: Optional[str]) -> int:
        if not self.client:
            raise RuntimeError("NKN client is not running.")
        return self.client.send_command(cmd, description)

    def shutdown(self) -> None:
        if self.client:
            self.client.shutdown()
            self.client = None

    def stats(self) -> Dict[str, int]:
        if not self.client:
            return {"bytes_in": 0, "bytes_out": 0, "messages_in": 0, "messages_out": 0}
        return self.client.stats()

    def status(self) -> str:
        if not self.client:
            return "inactive"
        return self.client.status()


def _ensure_nkn_control(cfg: Dict[str, object]) -> _NKNControl:
    global _nkn_control
    if _nkn_control and _nkn_control.matches(cfg):
        return _nkn_control
    if _nkn_control:
        _nkn_control.shutdown()
        _nkn_control = None
    _nkn_control = _NKNControl(cfg)
    return _nkn_control


def ensure_nkn_client(cfg: Dict[str, object]) -> _NKNControl:
    return _ensure_nkn_control(cfg)


def stop_nkn_client() -> None:
    global _nkn_control
    if _nkn_control:
        _nkn_control.shutdown()
        _nkn_control = None


def get_nkn_status() -> str:
    if _nkn_control:
        return _nkn_control.status()
    return "inactive"


def get_nkn_stats() -> Dict[str, int]:
    if _nkn_control:
        return _nkn_control.stats()
    return {"bytes_in": 0, "bytes_out": 0, "messages_in": 0, "messages_out": 0}


def get_nkn_target() -> str:
    cfg = _load_remote_config()
    nkn_cfg = cfg.get("nkn", {})
    return str(nkn_cfg.get("target") or "")


def get_nkn_remote_address() -> str:
    cfg = _load_remote_config()
    nkn_cfg = cfg.get("nkn", {})
    return str(nkn_cfg.get("remote_address") or "")


def _dispatch_direct(cmd: Iterable[str], description: Optional[str]) -> subprocess.CompletedProcess:
    cfg = _load_remote_config()
    host = cfg.get("host")
    port = cfg.get("port")
    if not host or not port:
        raise RuntimeError("Remote host or port not configured.")
    normalized_cmd = _normalize_command(cmd)
    request = {
        "session_id": str(uuid.uuid4()),
        "command": "run",
        "description": description or "remote run",
        "cmd": normalized_cmd,
    }
    return_code = None
    try:
        with socket.create_connection((host, int(port)), timeout=5) as sock:
            with sock.makefile("rwb") as stream:
                stream.write(remote_protocol.encode_message(request))
                stream.flush()
                for msg in remote_protocol.iter_messages(stream):
                    typ = msg.get("type")
                    if typ == "log":
                        _remote_log(msg.get("message", ""))
                    elif typ == "start":
                        _remote_log(msg.get("description", "run started"))
                    elif typ == "ack":
                        _remote_ack(msg.get("message", "ack"))
                    elif typ == "exit":
                        return_code = int(msg.get("code", 0))
                        break
    except Exception as exc:
        raise RuntimeError(f"Remote dispatch failed: {exc}") from exc
    if return_code is None:
        raise RuntimeError("Remote host closed without exit code.")
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(cmd))
    return subprocess.CompletedProcess(args=list(cmd), returncode=return_code)


def _ensure_reverse_bridge() -> None:
    cfg = _load_remote_config()
    preferred_port = get_reverse_port()
    reset_listener_status()
    attempted: set[int] = set()
    current_port = preferred_port
    while True:
        try:
            reverse_remote.ensure_bridge("0.0.0.0", current_port, _remote_log, _remote_ack)
            if current_port != preferred_port:
                cfg["reverse_port"] = current_port
                save_remote_config(cfg)
            return
        except OSError:
            reverse_remote.stop_bridge()
            attempted.add(current_port)
            next_port = _pick_free_port(preferred_port)
            if next_port in attempted:
                raise RuntimeError("Failed to bind reverse listener.")
            current_port = next_port


def _dispatch_reverse(cmd: Iterable[str], description: Optional[str]) -> subprocess.CompletedProcess:
    normalized_cmd = _normalize_command(cmd)
    try:
        exit_code = reverse_remote.dispatch_command(normalized_cmd, description)
    except Exception as exc:
        raise RuntimeError(f"Remote dispatch failed: {exc}") from exc
    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, normalized_cmd)
    return subprocess.CompletedProcess(args=list(normalized_cmd), returncode=exit_code)


def dispatch_remote(cmd: Iterable[str], description: Optional[str] = None) -> subprocess.CompletedProcess:
    cfg = _load_remote_config()
    mode = cfg.get("mode", "direct")
    if mode == "reverse":
        _ensure_reverse_bridge()
        return _dispatch_reverse(cmd, description)
    if mode == "nkn":
        control = ensure_nkn_client(cfg)
        exit_code = control.send_command(cmd, description)
        normalized_cmd = _normalize_command(cmd)
        if exit_code != 0:
            raise subprocess.CalledProcessError(exit_code, normalized_cmd)
        return subprocess.CompletedProcess(args=list(normalized_cmd), returncode=exit_code)
    else:
        return _dispatch_direct(cmd, description)


def test_remote_connection(host: str, port: int, timeout: float = 2.0) -> Tuple[bool, str]:
    if not host:
        return False, "Host is empty"
    if not port:
        return False, "Port is not set"
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            pass
    except Exception as exc:
        return False, str(exc)
    return True, "Connection succeeded"


def ensure_reverse_listener() -> Tuple[str, int]:
    _ensure_reverse_bridge()
    return reverse_remote.bridge_address()


def is_reverse_listener_active() -> bool:
    if reverse_remote.is_agent_available():
        return True
    return False


def stop_reverse_listener() -> None:
    reverse_remote.stop_bridge()
    reset_listener_status()
