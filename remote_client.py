"""Client helpers for dispatching commands to a remote RL executor."""

import json
import socket
import subprocess
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import remote_protocol
import reverse_remote

PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_CONFIG_FILE = PROJECT_ROOT / "isaaclab_remote_connection.json"
DEFAULT_REMOTE_CONFIG = {
    "enabled": False,
    "mode": "reverse",
    "host": "127.0.0.1",
    "port": 8721,
    "reverse_port": 8765,
}

_remote_config_cache: Optional[Dict[str, object]] = None


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
    _remote_config_cache = data
    return data


def get_remote_config() -> Dict[str, object]:
    return dict(_load_remote_config())


def save_remote_config(config: Dict[str, object]) -> None:
    global _remote_config_cache
    data = dict(DEFAULT_REMOTE_CONFIG)
    data.update(config)
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
    return bool(cfg.get("enabled") and cfg.get("host") and cfg.get("port"))


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


def _remote_log(msg: str) -> None:
    print(f"[remote] {msg.rstrip()}", flush=True)


def _remote_ack(msg: str) -> None:
    print(f"[remote] ack: {msg.rstrip()}", flush=True)


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
    port = get_reverse_port()
    reverse_remote.ensure_bridge("0.0.0.0", port, _remote_log, _remote_ack)


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
