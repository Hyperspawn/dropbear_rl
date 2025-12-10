"""Client helpers for dispatching commands to a remote RL executor."""

import json
import socket
import subprocess
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import remote_protocol

PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_CONFIG_FILE = PROJECT_ROOT / "isaaclab_remote_connection.json"
DEFAULT_REMOTE_CONFIG = {"enabled": False, "host": "127.0.0.1", "port": 8721}

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
    try:
        REMOTE_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        REMOTE_CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        _remote_config_cache = data
    except Exception:
        pass


def is_remote_enabled() -> bool:
    cfg = _load_remote_config()
    return bool(cfg.get("enabled") and cfg.get("host") and cfg.get("port"))


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


def dispatch_remote(cmd: Iterable[str], description: Optional[str] = None) -> subprocess.CompletedProcess:
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
                        print(f"[remote] {msg.get('message', '').rstrip()}", flush=True)
                    elif typ == "start":
                        print(f"[remote] {msg.get('description', 'run started')}", flush=True)
                    elif typ == "ack":
                        print(f"[remote] ack: {msg.get('message', '').rstrip()}", flush=True)
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
