#!/usr/bin/env python3
"""
bootstrap_isaaclab.py

One-shot installer + runner for:
  1) Python 3.11 venv
  2) Isaac Sim via pip (NVIDIA index)
  3) CUDA-enabled PyTorch wheel (cu128)
  4) Isaac Lab from source (clone + ./isaaclab.sh --install)
  5) Run an Isaac Lab script using the venv

Typical usage:
  python3 bootstrap_isaaclab.py
  python3 bootstrap_isaaclab.py --run train_ant --headless
  python3 bootstrap_isaaclab.py --base ~/work --env env_isaaclab --repo IsaacLab

Notes:
- Linux pip install of Isaac Sim requires GLIBC >= 2.35 and (typically) x86_64.
- Isaac Sim 5.x requires Python 3.11 in the environment.
- First simulator run may prompt for NVIDIA Omniverse EULA and pull extensions.
"""

from __future__ import annotations

import argparse
import ctypes
import curses
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import remote_client
import reverse_remote


DEFAULT_ISAACSIM_VERSION = "5.1.0"
DEFAULT_TORCH_VERSION = "2.7.0"
DEFAULT_TORCHVISION_VERSION = "0.22.0"
DEFAULT_TORCHAUDIO_VERSION = "2.7.0"
DEFAULT_NVIDIA_PYPI = "https://pypi.nvidia.com"
DEFAULT_ISAACLAB_GIT = "https://github.com/isaac-sim/IsaacLab.git"
PROJECT_ROOT = Path(__file__).resolve().parent
DROPBEAR_EXTENSION_DIR = PROJECT_ROOT / "source" / "dropbear_rl_lab"
DROPBEAR_MODULE_PATH = DROPBEAR_EXTENSION_DIR / "dropbear_rl_lab"

DROPBEAR_RUNS: list[str] = [
    "create_empty",
    "train_ant",
    "train_anymal",
    "dropbear_quick_test",
    "dropbear_test_commands",
    "dropbear_train",
    "dropbear_play",
    "none",
]
POLICY_LOG_ROOT = PROJECT_ROOT / "IsaacLab" / "logs" / "rsl_rl"
SAVE_AFTER_FILE = PROJECT_ROOT / "isaaclab_save_after.json"

TRAINING_CONFIG_FILE = PROJECT_ROOT / "isaaclab_training_configs.json"
DEFAULT_TRAINING_CONFIG_NAME = "dropbear_default"

TRAINING_CONFIG_FIELDS = [
    {
        "path": "agent_cfg.policy.init_noise_std",
        "label": "Init noise std",
        "kind": "float",
        "min": 0.01,
        "max": 2.0,
        "step": 0.05,
        "display_format": "{:.2f}",
        "override_format": "{:.4g}",
        "default": 0.5,
        "description": "Exploration noise applied to the policy. Too little = no exploration; too much = jitter.",
        "warn_low": 0.05,
        "warn_low_msg": "Very low noise can freeze exploration.",
        "warn_high": 1.6,
        "warn_high_msg": "Very high noise often yields unstable actions.",
    },
    {
        "path": "agent_cfg.algorithm.entropy_coef",
        "label": "Entropy coef",
        "kind": "float",
        "min": 0.0,
        "max": 0.2,
        "step": 0.01,
        "display_format": "{:.3f}",
        "override_format": "{:.4g}",
        "default": 0.05,
        "description": "Weight on entropy regularization to retain randomness during training.",
        "warn_high": 0.15,
        "warn_high_msg": "Large entropy weights can drown out the reward signal.",
    },
    {
        "path": "agent_cfg.algorithm.clip_param",
        "label": "Clip param",
        "kind": "float",
        "min": 0.01,
        "max": 0.5,
        "step": 0.01,
        "display_format": "{:.3f}",
        "override_format": "{:.4g}",
        "default": 0.1,
        "description": "PPO clipping range for policy updates. Too large allows big jumps; too small is conservative.",
        "warn_high": 0.3,
        "warn_high_msg": "Very large clip parameters may destabilize updates.",
    },
    {
        "path": "agent_cfg.algorithm.learning_rate",
        "label": "Learning rate",
        "kind": "float",
        "min": 1e-5,
        "max": 1e-3,
        "step": 1e-5,
        "display_format": "{:.4g}",
        "override_format": "{:.6g}",
        "default": 1.0e-4,
        "description": "Optimizer step size. High values may diverge; low values slow learning.",
        "warn_high": 5e-4,
        "warn_high_msg": "Very high LR may cause gradients to explode.",
    },
    {
        "path": "agent_cfg.algorithm.max_grad_norm",
        "label": "Max grad norm",
        "kind": "float",
        "min": 0.1,
        "max": 2.0,
        "step": 0.1,
        "display_format": "{:.2f}",
        "override_format": "{:.4g}",
        "default": 0.5,
        "description": "Clips gradients to avoid exploding updates. Too tight = slow learning.",
        "warn_low": 0.2,
        "warn_low_msg": "Very low clipping may starve learning.",
    },
    {
        "path": "agent_cfg.algorithm.num_learning_epochs",
        "label": "Learning epochs",
        "kind": "int",
        "min": 1,
        "max": 10,
        "step": 1,
        "display_format": "{}",
        "override_format": "{}",
        "default": 2,
        "description": "How many passes over each rollout for PPO updates.",
        "warn_high": 6,
        "warn_high_msg": "Too many epochs can overfit on stale data.",
    },
    {
        "path": "agent_cfg.algorithm.value_loss_coef",
        "label": "Value loss coef",
        "kind": "float",
        "min": 0.0,
        "max": 2.0,
        "step": 0.1,
        "display_format": "{:.2f}",
        "override_format": "{:.4g}",
        "default": 1.0,
        "description": "Weight for the critic loss relative to policy loss.",
        "warn_high": 1.6,
        "warn_high_msg": "Very high values overemphasize the critic.",
    },
    {
        "path": "agent_cfg.algorithm.use_clipped_value_loss",
        "label": "Clip value loss",
        "kind": "bool",
        "default": True,
        "description": "Toggle clipping on the value loss to prevent runaway critic targets.",
        "warn_bool_msg": "Disabling value clipping may destabilize the critic.",
    },
    {
        "path": "agent_cfg.algorithm.schedule",
        "label": "LR schedule",
        "kind": "enum",
        "values": ["fixed", "adaptive"],
        "default": "fixed",
        "description": "LR schedule used by RSL-RL. Adaptive follows KL/advantage feedback.",
    },
    {
        "path": "agent_cfg.algorithm.gamma",
        "label": "Discount (gamma)",
        "kind": "float",
        "min": 0.8,
        "max": 0.999,
        "step": 0.005,
        "display_format": "{:.3f}",
        "override_format": "{:.4g}",
        "default": 0.99,
        "description": "Discount factor for future rewards.",
        "warn_low": 0.85,
        "warn_low_msg": "Low gamma shortens the effective horizon.",
    },
    {
        "path": "agent_cfg.algorithm.lam",
        "label": "GAE λ",
        "kind": "float",
        "min": 0.8,
        "max": 0.99,
        "step": 0.01,
        "display_format": "{:.3f}",
        "override_format": "{:.4g}",
        "default": 0.95,
        "description": "GAE λ trades bias vs variance in advantage estimation.",
        "warn_low": 0.85,
        "warn_low_msg": "Low λ reduces temporal credit assignment.",
    },
    {
        "path": "agent_cfg.algorithm.desired_kl",
        "label": "Desired KL",
        "kind": "float",
        "min": 0.001,
        "max": 0.02,
        "step": 0.001,
        "display_format": "{:.4f}",
        "override_format": "{:.4g}",
        "default": 0.005,
        "description": "Target KL divergence per update. Smaller keeps steps conservative.",
        "warn_low": 0.002,
        "warn_low_msg": "Extremely low KL slows down policy improvement.",
        "warn_high": 0.014,
        "warn_high_msg": "High KL can allow overly aggressive updates.",
    },
    {
        "path": "agent_cfg.clip_actions",
        "label": "Clip actions",
        "kind": "bool",
        "default": True,
        "description": "Clip policy outputs to action limits.",
        "warn_bool_msg": "Turning off clipping may send invalid actions to the env.",
    },
    {
        "path": "env_cfg.terminations.base_height.params.minimum_height",
        "label": "Min base height",
        "kind": "float",
        "min": 0.3,
        "max": 1.2,
        "step": 0.02,
        "display_format": "{:.2f}",
        "override_format": "{:.4g}",
        "default": 0.5,
        "description": "Termination height for the robot classed as fallen. Raising it aborts earlier.",
        "warn_low": 0.4,
        "warn_low_msg": "A very low threshold lets the robot tumble longer.",
    },
    {
        "path": "env_cfg.terminations.base_contact.params.threshold",
        "label": "Contact threshold",
        "kind": "float",
        "min": 0.1,
        "max": 5.0,
        "step": 0.1,
        "display_format": "{:.2f}",
        "override_format": "{:.4g}",
        "default": 1.0,
        "description": "Number of illegal contacts tolerated before termination.",
        "warn_low": 0.5,
        "warn_low_msg": "Very strict contact thresholds can terminate normal pushes.",
        "warn_high": 4.0,
        "warn_high_msg": "Too loose thresholds may hide actual collisions.",
    },
    {
        "path": "env_cfg.rewards.undesired_contacts.weight",
        "label": "Undesired contact",
        "kind": "float",
        "min": -3.0,
        "max": 0.0,
        "step": 0.1,
        "display_format": "{:.2f}",
        "override_format": "{:.4g}",
        "default": -1.0,
        "description": "Penalty for undesired contacts (excluding skateboard bodies).",
        "warn_high": -0.4,
        "warn_high_msg": "Reducing this penalty makes collisions cheap.",
    },
    {
        "path": "env_cfg.rewards.feet_slide.weight",
        "label": "Feet slide",
        "kind": "float",
        "min": -1.0,
        "max": 0.0,
        "step": 0.05,
        "display_format": "{:.2f}",
        "override_format": "{:.4g}",
        "default": -0.2,
        "description": "Penalty for foot sliding relative to contact points.",
        "warn_high": -0.05,
        "warn_high_msg": "Weak penalties allow the feet to slip freely.",
    },
]


def _get_primary_ipv4() -> Optional[str]:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    return None


def _get_local_ipv4_candidates() -> list[str]:
    ips: set[str] = set()
    try:
        host = socket.gethostname()
        for info in socket.getaddrinfo(host, None, family=socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass
    primary = _get_primary_ipv4()
    if primary:
        ips.add(primary)
    return sorted(ips)


def _handle_interrupt(signum: int, frame) -> None:
    remote_client.stop_reverse_listener()
    raise KeyboardInterrupt

def _default_training_config_entry() -> dict:
    return {field["path"]: field["default"] for field in TRAINING_CONFIG_FIELDS}


def _coerce_training_value(field: dict, raw_value: object) -> object:
    kind = field.get("kind")
    try:
        if kind == "int":
            return int(raw_value)
        if kind == "float":
            return float(raw_value)
        if kind == "bool":
            if isinstance(raw_value, str):
                return raw_value.strip().lower() not in ("false", "0", "no", "off")
            return bool(raw_value)
        if kind == "enum":
            candidate = str(raw_value)
            values = field.get("values") or []
            return candidate if candidate in values else field["default"]
    except Exception:
        pass
    return field["default"]


def load_training_configs() -> dict:
    data: dict = {"configs": {}, "last_used": DEFAULT_TRAINING_CONFIG_NAME}
    if TRAINING_CONFIG_FILE.exists():
        try:
            raw = json.loads(TRAINING_CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                configs = raw.get("configs")
                if isinstance(configs, dict):
                    data["configs"] = {name: dict(value) for name, value in configs.items() if isinstance(value, dict)}
                last_used = raw.get("last_used")
                if isinstance(last_used, str):
                    data["last_used"] = last_used
        except Exception as exc:
            print(f"[!] Failed to read training config file: {exc}")

    if DEFAULT_TRAINING_CONFIG_NAME not in data["configs"]:
        data["configs"][DEFAULT_TRAINING_CONFIG_NAME] = _default_training_config_entry()

    for name, entry in data["configs"].items():
        for field in TRAINING_CONFIG_FIELDS:
            entry.setdefault(field["path"], field["default"])
            entry[field["path"]] = _coerce_training_value(field, entry[field["path"]])
    if data["last_used"] not in data["configs"]:
        data["last_used"] = DEFAULT_TRAINING_CONFIG_NAME
    return data


def save_training_configs(data: dict) -> None:
    try:
        ensure_dir(TRAINING_CONFIG_FILE.parent)
        TRAINING_CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[!] Failed to persist training configs: {exc}")


def _format_training_value(field: dict, value: object) -> str:
    if field["kind"] == "bool":
        return "ON" if bool(value) else "OFF"
    if field["kind"] == "enum":
        return str(value)
    fmt = field.get("display_format")
    if fmt and isinstance(value, (float, int)):
        return fmt.format(value)
    if field["kind"] == "int":
        return str(int(value))
    return f"{float(value):.4g}"


def _format_override_value(field: dict, value: object) -> str:
    if field["kind"] == "bool":
        return str(bool(value))
    if field["kind"] == "enum":
        return str(value)
    fmt = field.get("override_format")
    if fmt and isinstance(value, (float, int)):
        return fmt.format(value)
    if field["kind"] == "int":
        return str(int(value))
    return f"{float(value):.6g}"


def _training_field_description(field: dict) -> str:
    return field.get("description", "")


def _training_field_warning(field: dict, value: object) -> Optional[str]:
    kind = field["kind"]
    if kind in ("float", "int"):
        val = float(value)
        low = field.get("warn_low")
        high = field.get("warn_high")
        if low is not None and val <= low:
            return field.get("warn_low_msg", f"{field['label']} near {low}.")
        if high is not None and val >= high:
            return field.get("warn_high_msg", f"{field['label']} near {high}.")
    if kind == "bool":
        if not bool(value):
            return field.get("warn_bool_msg")
    return None


def training_config_overrides(entry: dict) -> list[str]:
    overrides: list[str] = []
    for field in TRAINING_CONFIG_FIELDS:
        value = entry.get(field["path"], field["default"])
        overrides.append(f"+{field['path']}={_format_override_value(field, value)}")
    return overrides

# -----------------------------
# helpers
# -----------------------------
def eprint(*a: object) -> None:
    print(*a, file=sys.stderr, flush=True)


def run_cmd(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    cmd_str = " ".join([shlex_quote(x) for x in cmd])
    print(f"+ {cmd_str}", flush=True)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=check)


def shlex_quote(s: str) -> str:
    # minimal, good-enough quoting for printing
    if re.fullmatch(r"[A-Za-z0-9_./:@%+=-]+", s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


def is_venv() -> bool:
    return getattr(sys, "base_prefix", sys.prefix) != sys.prefix


def venv_bin_dir(env_dir: Path) -> Path:
    return env_dir / ("Scripts" if platform.system() == "Windows" else "bin")


def venv_python(env_dir: Path) -> Path:
    b = venv_bin_dir(env_dir)
    return b / ("python.exe" if platform.system() == "Windows" else "python")


def venv_pip(env_dir: Path) -> Path:
    b = venv_bin_dir(env_dir)
    return b / ("pip.exe" if platform.system() == "Windows" else "pip")


def which(exe: str) -> Optional[str]:
    return shutil.which(exe)


def parse_version_tuple(v: str) -> Tuple[int, int, int]:
    m = re.match(r"^\s*(\d+)\.(\d+)(?:\.(\d+))?", v.strip())
    if not m:
        return (0, 0, 0)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))


def get_glibc_version_linux() -> Optional[str]:
    # Works on glibc systems (most Ubuntu/Debian). Returns None on musl or failure.
    try:
        libc = ctypes.CDLL("libc.so.6")
        gnu_get_libc_version = libc.gnu_get_libc_version
        gnu_get_libc_version.restype = ctypes.c_char_p
        v = gnu_get_libc_version().decode("ascii", errors="ignore")
        return v
    except Exception:
        return None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_marker(env_dir: Path, name: str, content: str = "ok\n") -> None:
    ensure_dir(env_dir)
    (env_dir / name).write_text(content, encoding="utf-8")


def has_marker(env_dir: Path, name: str) -> bool:
    return (env_dir / name).exists()


def prepend_path(env: Dict[str, str], p: Path) -> Dict[str, str]:
    env2 = dict(env)
    env2["PATH"] = str(p) + os.pathsep + env2.get("PATH", "")
    return env2


def install_editable_package(py: Path, pip: Path, package_dir: Path) -> None:
    """Install the editable Dropbear RL Lab package into the venv."""
    if pip.name.startswith("python"):
        run_cmd([str(py), "-m", "pip", "install", "-e", str(package_dir)])
    else:
        run_cmd([str(pip), "install", "-e", str(package_dir)])


def run_local_python_script(python: Path, script_path: Path, cwd: Path, env: Dict[str, str], args: List[str]) -> None:
    """Execute a local script via the provided Python interpreter."""
    cmd = [str(python), str(script_path)] + args
    run_cmd(cmd, cwd=str(cwd), env=env)


def git_clone_or_download(repo_url: str, dest: Path, branch: Optional[str] = None) -> None:
    if dest.exists() and (dest / ".git").exists():
        print(f"[i] IsaacLab repo already exists at: {dest}")
        return

    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError(f"Destination exists and is not empty: {dest}")

    git = which("git")
    if git:
        cmd = [git, "clone", "--depth", "1"]
        if branch:
            cmd += ["--branch", branch]
        cmd += [repo_url, str(dest)]
        run_cmd(cmd)
        return

    # fallback: download zipball via curl/wget/python if git is missing
    # We keep it simple: require git unless user installs it via apt.
    raise RuntimeError(
        "git is not available and auto-download fallback is disabled for safety. "
        "Install git (e.g., `sudo apt-get install -y git`) and re-run."
    )


def apt_install_if_requested(pkgs: List[str], mode: str) -> None:
    """
    mode: 'off' | 'auto' | 'on'
    """
    if platform.system() != "Linux":
        return
    if mode == "off":
        return
    if not which("apt-get"):
        if mode == "on":
            raise RuntimeError("Requested --system-deps on, but apt-get not found.")
        return

    # only attempt on Debian/Ubuntu-like
    # this may prompt for sudo password; that's expected.
    sudo = which("sudo")
    is_root = (hasattr(os, "geteuid") and os.geteuid() == 0)
    prefix: List[str] = []
    if not is_root:
        if not sudo:
            if mode == "on":
                raise RuntimeError("Need root to install system deps, but sudo not found.")
            return
        prefix = [sudo]

    env = dict(os.environ)
    env["DEBIAN_FRONTEND"] = "noninteractive"

    run_cmd(prefix + ["apt-get", "update"], env=env)
    run_cmd(prefix + ["apt-get", "install", "-y"] + pkgs, env=env)


def find_python_311() -> Optional[str]:
    # If we're already 3.11, use this interpreter
    if sys.version_info[:2] == (3, 11):
        return sys.executable

    # Common names on Linux/macOS
    for cand in ("python3.11", "python311", "python3"):
        path = which(cand)
        if not path:
            continue
        try:
            out = subprocess.check_output([path, "-c", "import sys; print(sys.version.split()[0])"])
            ver = out.decode().strip()
            if ver.startswith("3.11."):
                return path
        except Exception:
            continue

    # Windows launcher
    if platform.system() == "Windows" and which("py"):
        try:
            out = subprocess.check_output(["py", "-3.11", "-c", "import sys; print(sys.version.split()[0])"])
            ver = out.decode().strip()
            if ver.startswith("3.11."):
                return "py -3.11"  # special token handled later
        except Exception:
            pass

    return None


def create_venv_with_python(py311: str, env_dir: Path) -> None:
    if env_dir.exists() and venv_python(env_dir).exists():
        print(f"[i] venv already exists at: {env_dir}")
        return

    ensure_dir(env_dir.parent)

    if py311 == sys.executable and sys.version_info[:2] == (3, 11):
        # Use stdlib venv if current python is 3.11
        run_cmd([py311, "-m", "venv", str(env_dir)])
        return

    if py311.startswith("py -3.11"):
        # Windows py launcher
        run_cmd(["py", "-3.11", "-m", "venv", str(env_dir)])
        return

    # External python path
    run_cmd([py311, "-m", "venv", str(env_dir)])


def ensure_in_venv(env_dir: Path, passthrough_args: List[str]) -> None:
    """
    If not currently running inside the target venv, re-exec into it.
    """
    if is_venv():
        return
    py = venv_python(env_dir)
    if not py.exists():
        raise RuntimeError(f"venv python not found at: {py}")

    # Re-exec: run this same script inside venv with internal flag
    args = [str(py), str(Path(__file__).resolve()), "--_inside-venv"] + passthrough_args
    print(f"[i] re-exec into venv: {py}")
    os.execv(str(py), args)


def pip_install(pip_path: Path, args: List[str]) -> None:
    run_cmd([str(pip_path), "install"] + args)


def pip_check_import(py_path: Path, module: str, env: Optional[Dict[str, str]] = None) -> bool:
    try:
        cmd_env = add_dropbear_pythonpath(env if env is not None else dict(os.environ))
        subprocess.check_call(
            [str(py_path), "-c", f"import {module}; print({module}.__name__)"], env=cmd_env
        )
        return True
    except Exception:
        return False


def ensure_dropbear_installed(py: Path, pip: Path, package_dir: Path) -> None:
    """Attempt to install `dropbear_rl_lab` so Python can import it."""
    if pip_check_import(py, "dropbear_rl_lab"):
        return
    print("[i] dropbear_rl_lab not available; reinstalling package.")
    install_editable_package(py, pip, package_dir)
    if pip_check_import(py, "dropbear_rl_lab"):
        return
    print("[!] Dropbear RL Lab package still missing after reinstall.")


def ensure_actor_critic_std(env_dir: Path) -> None:
    """Ensure the PPO actor critic uses a positive std when sampling actions."""
    python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    actor_critic_path = (
        env_dir
        / "lib"
        / python_dir
        / "site-packages"
        / "rsl_rl"
        / "modules"
        / "actor_critic.py"
    )
    if not actor_critic_path.exists():
        return
    text = actor_critic_path.read_text(encoding="utf-8")
    old = "        # Create distribution\n        self.distribution = Normal(mean, std)\n"
    if old not in text:
        return
    new = (
        "        # Create distribution\n"
        "        min_std = torch.finfo(std.dtype).eps\n"
        "        std = torch.nan_to_num(std, nan=min_std, posinf=min_std, neginf=min_std)\n"
        "        std = torch.abs(std)\n"
        "        std = torch.clamp(std, min=min_std)\n"
        "        self.distribution = Normal(mean, std)\n"
    )
    actor_critic_path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _gather_policy_checkpoints() -> list[Path]:
    """Return list of available checkpoints under logs/rsl_rl."""
    candidates: list[Path] = []
    if not POLICY_LOG_ROOT.exists():
        return candidates
    for exp in sorted(POLICY_LOG_ROOT.iterdir()):
        if not exp.is_dir():
            continue
        for run in sorted(exp.iterdir()):
            if not run.is_dir():
                continue
            for ckpt in sorted(run.glob("model_*.pt")):
                candidates.append(ckpt)
    return candidates

# Save-run helper
def save_run_config(args: argparse.Namespace, unknown: list[str]) -> None:
    try:
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "run": args.run,
            "dropbear_task": args.dropbear_task,
            "checkpoint": getattr(args, "checkpoint", None),
            "args": unknown,
        }
        SAVE_AFTER_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[i] Saved run metadata to {SAVE_AFTER_FILE}")
    except Exception as exc:
        print(f"[!] Failed to save run metadata: {exc}")

def run_curses_interface() -> Optional[list[str]]:
    """Launch an interactive menu over curses to configure arguments."""

    training_configs = load_training_configs()
    training_active_name = training_configs.get("last_used", DEFAULT_TRAINING_CONFIG_NAME)
    if training_active_name not in training_configs["configs"]:
        training_active_name = DEFAULT_TRAINING_CONFIG_NAME
    remote_settings = remote_client.get_remote_config()
    remote_settings.setdefault("mode", "nkn")
    remote_settings.setdefault("reverse_port", remote_client.get_reverse_port())

    def _format_bytes(value: int) -> str:
        val = float(value)
        for unit in ("B", "KB", "MB", "GB"):
            if val < 1024 or unit == "GB":
                if unit == "B":
                    return f"{int(val)}{unit}"
                return f"{val:.1f}{unit}"
            val /= 1024
        return f"{val:.1f}TB"

    def select_checkpoint(stdscr: "curses._CursesWindow", checkpoints: list[Path]) -> Optional[Path]:
        if not checkpoints:
            stdscr.erase()
            msg = f"No checkpoints found under {POLICY_LOG_ROOT}"
            stdscr.addstr(0, 0, msg, curses.A_BOLD)
            stdscr.addstr(2, 0, "Press any key to return.")
            stdscr.refresh()
            stdscr.getch()
            return None
        selected = 0
        offset = 0
        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            stdscr.addstr(0, 0, f"Select checkpoint ({len(checkpoints)} entries):", curses.A_BOLD)
            available = height - 4
            if available <= 0:
                available = 1
            if selected < offset:
                offset = selected
            elif selected >= offset + available:
                offset = selected - available + 1
            for idx in range(offset, min(len(checkpoints), offset + available)):
                rel = checkpoints[idx].relative_to(POLICY_LOG_ROOT)
                prefix = ">" if idx == selected else " "
                attr = curses.A_REVERSE if idx == selected else curses.A_NORMAL
                stdscr.addstr(1 + idx - offset, 0, f"{prefix} {str(rel)}", attr)
            instructions = "[Enter] Select  [q/Esc] Back  [↑/↓] Navigate"
            stdscr.addstr(height - 2, 0, instructions, curses.color_pair(1))
            stdscr.refresh()
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                selected = (selected - 1) % len(checkpoints)
            elif key in (curses.KEY_DOWN, ord("j")):
                selected = (selected + 1) % len(checkpoints)
            elif key in (10, 13):
                return checkpoints[selected]
            elif key in (ord("q"), 27):
                return None

    def prompt_for_config_name(stdscr: "curses._CursesWindow", prompt: str) -> Optional[str]:
        curses.echo()
        curses.curs_set(1)
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        stdscr.addstr(0, 0, prompt)
        stdscr.addstr(2, 0, "Leave blank to cancel.")
        stdscr.refresh()
        try:
            raw = stdscr.getstr(1, 0, 256)
            name = raw.decode("utf-8", errors="ignore").strip()
        finally:
            curses.noecho()
            curses.curs_set(0)
        return name or None

    def select_training_profile(stdscr: "curses._CursesWindow") -> Optional[str]:
        names = sorted(training_configs["configs"])
        if not names:
            return None
        selected_idx = names.index(training_active_name) if training_active_name in names else 0
        offset = 0
        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            stdscr.addstr(0, 0, "Select training config:", curses.A_BOLD)
            available = height - 4
            if available <= 0:
                available = 1
            if selected_idx < offset:
                offset = selected_idx
            elif selected_idx >= offset + available:
                offset = selected_idx - available + 1
            for idx in range(offset, min(len(names), offset + available)):
                prefix = ">" if idx == selected_idx else " "
                attr = curses.A_REVERSE if idx == selected_idx else curses.A_NORMAL
                stdscr.addstr(1 + idx - offset, 0, f"{prefix} {names[idx]}", attr)
            instructions = "[Enter] Select  [q/Esc] Back  [↑/↓] Navigate"
            stdscr.addstr(height - 2, 0, instructions, curses.color_pair(1))
            stdscr.refresh()
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                selected_idx = (selected_idx - 1) % len(names)
            elif key in (curses.KEY_DOWN, ord("j")):
                selected_idx = (selected_idx + 1) % len(names)
            elif key in (10, 13):
                return names[selected_idx]
            elif key in (ord("q"), 27):
                return None

    def configure_remote_target(stdscr: "curses._CursesWindow") -> None:
        value = prompt_for_config_name(stdscr, "Remote target (host:port):")
        if not value:
            return
        parts = value.split(":")
        host = parts[0].strip()
        if host:
            remote_settings["host"] = host
        if len(parts) > 1:
            try:
                remote_settings["port"] = int(parts[1])
            except Exception:
                pass

        cfg_host = remote_settings.get("host", "")
        cfg_port_value = remote_settings.get("port", remote_client.DEFAULT_REMOTE_CONFIG["port"])
        try:
            cfg_port = int(cfg_port_value)
        except Exception:
            cfg_port = remote_client.DEFAULT_REMOTE_CONFIG["port"]

        spinner_chars = "|/-\\"
        spinner_idx = 0
        result = {"done": False, "success": False, "info": "pending"}

        def _test_connection() -> None:
            success, info = remote_client.test_remote_connection(cfg_host, cfg_port)
            result["success"] = success
            result["info"] = info
            result["done"] = True

        worker = threading.Thread(target=_test_connection, daemon=True)
        worker.start()
        while not result["done"]:
            stdscr.erase()
            stdscr.addstr(0, 0, "Testing remote connection...", curses.A_BOLD)
            stdscr.addstr(
                2,
                0,
                f"{spinner_chars[spinner_idx % len(spinner_chars)]} Connecting to {cfg_host}:{cfg_port}",
            )
            spinner_idx += 1
            stdscr.refresh()
            time.sleep(0.1)
        worker.join()
        success = result["success"]
        info = result["info"]
        remote_client.save_remote_config(remote_settings)

        msg = "Remote target reachable" if success else f"Remote handshake failed: {info}"
        msg_color = curses.color_pair(1) if success else curses.color_pair(2)
        stdscr.erase()
        stdscr.addstr(0, 0, "Remote connection test", curses.A_BOLD)
        stdscr.addstr(2, 0, msg, msg_color)
        stdscr.addstr(4, 0, "Press any key to continue.")
        stdscr.refresh()
        stdscr.getch()

    def training_config_menu(stdscr: "curses._CursesWindow") -> None:
        nonlocal training_active_name
        selected_field = 0
        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            border = "+" + "-" * (width - 2) + "+"
            stdscr.addstr(0, 0, border)
            stdscr.addstr(height - 2, 0, "+" + "-" * (width - 2) + "+")
            for line in range(1, height - 2):
                stdscr.addstr(line, 0, "|")
                stdscr.addstr(line, width - 1, "|")
            title = f" Training Config: {training_active_name} "
            stdscr.addstr(0, max(0, (width - len(title)) // 2), title, curses.color_pair(1) | curses.A_BOLD)
            config = training_configs["configs"][training_active_name]
            content_start = 2
            fields = TRAINING_CONFIG_FIELDS
            max_rows = max(1, height - 8)
            start_idx = max(0, min(selected_field - max_rows + 1, len(fields) - max_rows))
            for idx in range(start_idx, min(len(fields), start_idx + max_rows)):
                field = fields[idx]
                value = config[field["path"]]
                prefix = ">" if idx == selected_field else " "
                attr = curses.A_REVERSE if idx == selected_field else curses.A_NORMAL
                label = f"{prefix} {field['label']}: {_format_training_value(field, value)}"
                stdscr.addstr(content_start + idx - start_idx, 3, label[: max(0, width - 6)], attr)
            current_field = fields[selected_field]
            description = _training_field_description(current_field)
            desc_row = height - 5
            if desc_row > 1 and description:
                stdscr.addstr(desc_row, 2, description[: max(0, width - 4)], curses.A_DIM)
            warn_row = height - 4
            warning = _training_field_warning(current_field, config[current_field["path"]])
            if warn_row > 1:
                if warning:
                    warning_text = "/!\\ " + warning
                    stdscr.addstr(
                        warn_row,
                        2,
                        warning_text[: max(0, width - 4)],
                        curses.color_pair(2),
                    )
                else:
                    # clear the line so stale warnings disappear
                    stdscr.addstr(warn_row, 2, " " * max(0, width - 4))
            instructions = (
                "[+/-] Adjust numeric/enum  [Space] Toggle bool  "
                "[d] Default field  [D] Delete profile  [l] Load  [n] New  [r] Reset  [Enter/q] Back"
            )
            trimmed_instructions = instructions[: max(0, width - 4)]
            stdscr.addstr(height - 3, 2, trimmed_instructions, curses.color_pair(1))
            stdscr.refresh()
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                selected_field = (selected_field - 1) % len(fields)
            elif key in (curses.KEY_DOWN, ord("j")):
                selected_field = (selected_field + 1) % len(fields)
            elif key in (ord("+"), ord("=")):
                field = fields[selected_field]
                path = field["path"]
                kind = field["kind"]
                if kind == "float":
                    current = float(config[path])
                    step = field["step"]
                    new_value = min(field["max"], current + step)
                    config[path] = new_value
                elif kind == "int":
                    current = int(config[path])
                    step = field["step"]
                    new_value = min(field["max"], current + step)
                    config[path] = int(new_value)
                elif kind == "enum":
                    values = field.get("values", [])
                    if values:
                        current = config[path]
                        idx = values.index(current) if current in values else 0
                        idx = min(len(values) - 1, idx + 1)
                        config[path] = values[idx]
            elif key in (ord("-"), ord("_")):
                field = fields[selected_field]
                path = field["path"]
                kind = field["kind"]
                if kind == "float":
                    current = float(config[path])
                    step = field["step"]
                    new_value = max(field["min"], current - step)
                    config[path] = new_value
                elif kind == "int":
                    current = int(config[path])
                    step = field["step"]
                    new_value = max(field["min"], current - step)
                    config[path] = int(new_value)
                elif kind == "enum":
                    values = field.get("values", [])
                    if values:
                        current = config[path]
                        idx = values.index(current) if current in values else 0
                        idx = max(0, idx - 1)
                        config[path] = values[idx]
            elif key == ord(" "):
                field = fields[selected_field]
                path = field["path"]
                if field["kind"] == "bool":
                    config[path] = not bool(config[path])
            elif key == ord("l"):
                chosen = select_training_profile(stdscr)
                if chosen and chosen in training_configs["configs"]:
                    training_active_name = chosen
            elif key == ord("n"):
                name = prompt_for_config_name(stdscr, "Enter name for new profile:")
                if name:
                    training_configs["configs"][name] = dict(config)
                    training_active_name = name
            elif key == ord("d"):
                field = fields[selected_field]
                path = field["path"]
                config[path] = _coerce_training_value(field, field["default"])
            elif key == ord("D"):
                if training_active_name != DEFAULT_TRAINING_CONFIG_NAME and len(training_configs["configs"]) > 1:
                    del training_configs["configs"][training_active_name]
                    training_active_name = sorted(training_configs["configs"])[0]
            elif key == ord("r"):
                training_configs["configs"][training_active_name] = _default_training_config_entry()
            elif key in (10, 13, ord("q"), 27):
                break

    def wrap_menu(stdscr: "curses._CursesWindow") -> Optional[list[str]]:
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        nonlocal remote_settings

        runs = DROPBEAR_RUNS
        selected = runs.index("dropbear_quick_test") if "dropbear_quick_test" in runs else 0
        headless = False
        video = False
        max_iterations = 1
        video_interval = 20
        video_length = 200
        system_deps_choices = ["auto", "on", "off"]
        system_deps_idx = 0
        policy_checkpoints = _gather_policy_checkpoints()
        selected_checkpoint: Optional[Path] = None
        save_after = False
        num_envs = 4
        listener_error: Optional[str] = None

        def _refresh_remote_settings() -> None:
            nonlocal remote_settings
            remote_settings = remote_client.get_remote_config()
            remote_settings.setdefault("mode", "nkn")
            remote_settings.setdefault("reverse_port", remote_client.get_reverse_port())

        def _ensure_nkn_mode_active() -> None:
            nonlocal listener_error
            if remote_settings.get("mode", "nkn") != "nkn":
                remote_client.stop_nkn_client()
                return
            try:
                remote_client.ensure_nkn_client(remote_settings)
                listener_error = None
            except Exception as exc:
                listener_error = f"NKN client error: {exc}"

        def _stop_all_remote_services() -> None:
            remote_client.stop_reverse_listener()
            remote_client.stop_nkn_client()

        def activate_reverse_listener() -> None:
            nonlocal listener_error
            listener_error = None
            try:
                remote_client.ensure_reverse_listener()
                _refresh_remote_settings()
            except Exception as exc:
                listener_error = f"Reverse listener error: {exc}"

        def _start_remote_mode() -> None:
            nonlocal listener_error
            listener_error = None
            mode = remote_settings.get("mode", "nkn")
            if mode == "reverse":
                activate_reverse_listener()
            elif mode == "direct":
                try:
                    remote_client.auto_discover_remote()
                    _refresh_remote_settings()
                except Exception as exc:
                    listener_error = f"Direct remote error: {exc}"
            elif mode == "nkn":
                _ensure_nkn_mode_active()
                _refresh_remote_settings()

        _ensure_nkn_mode_active()
        if remote_settings.get("enabled"):
            _start_remote_mode()

        def _cycle_remote_mode() -> None:
            nonlocal listener_error
            modes = ["reverse", "direct", "nkn"]
            current = remote_settings.get("mode", "nkn")
            try:
                idx = modes.index(current)
            except ValueError:
                idx = 0
            next_mode = modes[(idx + 1) % len(modes)]
            remote_settings["mode"] = next_mode
            remote_client.save_remote_config(remote_settings)
            _refresh_remote_settings()
            listener_error = None
            _stop_all_remote_services()
            _ensure_nkn_mode_active()
            if remote_settings.get("enabled"):
                _start_remote_mode()

        def configure_nkn_target_local(stdscr: "curses._CursesWindow") -> None:
            nonlocal listener_error
            value = prompt_for_config_name(stdscr, "NKN target (remote NKN address):")
            if not value:
                return
            listener_error = None
            nkn_cfg = remote_settings.setdefault("nkn", {})
            nkn_cfg["target"] = value.strip()
            remote_client.save_remote_config(remote_settings)
            _refresh_remote_settings()
            _ensure_nkn_mode_active()
            if remote_settings.get("enabled") and remote_settings.get("mode") == "nkn":
                _stop_all_remote_services()
                _start_remote_mode()

        def configure_nkn_remote_address(stdscr: "curses._CursesWindow") -> None:
            nonlocal listener_error
            value = prompt_for_config_name(stdscr, "Remote agent address (NKN):")
            if not value:
                return
            listener_error = None
            nkn_cfg = remote_settings.setdefault("nkn", {})
            nkn_cfg["remote_address"] = value.strip()
            remote_client.save_remote_config(remote_settings)
            _refresh_remote_settings()
            _ensure_nkn_mode_active()

        def build_args() -> list[str]:
            args = [f"--run={runs[selected]}"]
            if system_deps_choices[system_deps_idx] != "auto":
                args += ["--system-deps", system_deps_choices[system_deps_idx]]
            if headless:
                args.append("--headless")
            args += [f"--dropbear-max-iterations={max_iterations}"]
            if video:
                args.append("--dropbear-video")
            args += [
                f"--dropbear-video-interval={video_interval}",
                f"--dropbear-video-length={video_length}",
            ]
            if selected_checkpoint is not None:
                args.append(f"--checkpoint={selected_checkpoint}")
            if save_after:
                args.append("--save-after")
            if runs[selected] == "dropbear_train":
                args.append(f"--num_envs={num_envs}")
                overrides = training_config_overrides(training_configs["configs"][training_active_name])
                args += overrides
            return args

        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            border = "+" + "-" * (width - 2) + "+"
            stdscr.addstr(0, 0, border)
            stdscr.addstr(height - 2, 0, "+" + "-" * (width - 2) + "+")
            for idx in range(1, height - 2):
                stdscr.addstr(idx, 0, "|")
                stdscr.addstr(idx, width - 1, "|")

            title = " Dropbear RL CLI "
            stdscr.addstr(0, max(0, (width - len(title)) // 2), title, curses.color_pair(1) | curses.A_BOLD)

            stdscr.addstr(2, 3, "Select run:", curses.A_BOLD)
            for idx, run in enumerate(runs):
                prefix = ">" if idx == selected else " "
                attr = curses.A_REVERSE if idx == selected else curses.A_NORMAL
                stdscr.addstr(3 + idx, 5, f"{prefix} {run}", attr)

            info_row = 3 + len(runs) + 1
            if remote_settings.get("mode") == "nkn":
                _ensure_nkn_mode_active()
                _refresh_remote_settings()
            if remote_settings.get("enabled") and remote_settings.get("mode") == "direct":
                remote_client.auto_discover_remote()
                remote_settings = remote_client.get_remote_config()
            remote_enabled = bool(remote_settings.get("enabled"))
            remote_host = remote_settings.get("host", "")
            remote_port = remote_settings.get("port", "")
            reverse_port = remote_settings.get("reverse_port", remote_client.get_reverse_port())
            local_ips = _get_local_ipv4_candidates()
            local_ip = local_ips[0] if local_ips else "127.0.0.1"
            listener_status = "connected" if reverse_remote.is_agent_available() else "waiting"
            listener_note = remote_client.get_listener_status()
            discovery_note = remote_client.get_discovery_status()
            remote_mode = remote_settings.get("mode", "nkn")
            remote_mode_label = "NKN" if remote_mode == "nkn" else remote_mode.capitalize()
            remote_target_line = ""
            remote_status_entries: list[str] = []
            if remote_mode == "direct":
                remote_target_line = f"[R] Remote host: {remote_host}:{remote_port}"
                remote_status_entries.append(f"Discovery: {discovery_note}")
            elif remote_mode == "reverse":
                remote_target_line = f"Reverse listener: {local_ip}:{reverse_port} ({listener_status})"
                remote_status_entries.append(f"Listener status: {listener_note}")
            else:
                nkn_target = remote_client.get_nkn_target()
                nkn_remote_addr = remote_client.get_nkn_remote_address()
                nkn_controller_addr = remote_client.get_nkn_app_address()
                remote_target_line = f"[N] NKN target: {nkn_target or 'unset'}"
                remote_status_entries.append(f"NKN controller addr: {nkn_controller_addr or 'pending...'}")
                remote_status_entries.append(f"NKN remote addr: {nkn_remote_addr or 'unknown'}")
                remote_status_entries.append(f"NKN status: {remote_client.get_nkn_status()}")
                stats = remote_client.get_nkn_stats()
                remote_status_entries.append(
                    f"NKN bytes I/O: {_format_bytes(stats['bytes_in'])}/{_format_bytes(stats['bytes_out'])}"
                )
                remote_status_entries.append(
                    f"NKN messages I/O: {stats['messages_in']}/{stats['messages_out']}"
                )
            options = [
                f"[h] Headless: {'ON' if headless else 'OFF'}",
                f"[v] Video capture: {'ON' if video else 'OFF'}",
                f"[+/=]/[-/_] Iterations: {max_iterations}",
                f"[[]/[]] Video interval: {video_interval}",
                f"[,/.] Video length: {video_length}",
                f"[s] System deps: {system_deps_choices[system_deps_idx]}",
                f"[p] Checkpoint: {selected_checkpoint.name if selected_checkpoint else 'none'}",
                f"[n/m] Num envs: {num_envs}",
                f"[a] Save after: {'ON' if save_after else 'OFF'}",
                f"[t] Training config: {training_active_name}",
                f"[r] Remote compute: {'ON' if remote_enabled else 'OFF'}",
                f"[M] Remote mode: {remote_mode_label}",
                remote_target_line,
                f"[A] Remote agent addr: {remote_client.get_nkn_remote_address() or 'unknown'}",
            ]
            for idx, text in enumerate(options):
                row = info_row + idx
                if row >= height - 3:
                    break
                stdscr.addstr(row, 5, text[: max(0, width - 8)])
            status_row = info_row + len(options)
            for idx, text in enumerate(remote_status_entries):
                row = status_row + idx
                if row >= height - 3:
                    break
                stdscr.addstr(row, 5, text[: max(0, width - 8)])

            footer = "[Enter] Run   [q] Quit"
            stdscr.addstr(height - 1, max(2, (width - len(footer)) // 2), footer, curses.color_pair(1))
            error_row = height - 4
            if error_row > 0:
                if listener_error:
                    stdscr.addstr(error_row, 2, listener_error[: max(0, width - 4)], curses.color_pair(2))
                else:
                    stdscr.addstr(error_row, 2, " " * max(0, width - 4))
            stdscr.addstr(
                height - 3,
                2,
                "Use ↑/↓ to change run; toggle values with highlighted keys "
                "(n/m for envs, t for configs, r/R/M/N for remote).",
                curses.A_DIM,
            )

            stdscr.refresh()
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                selected = (selected - 1) % len(runs)
            elif key in (curses.KEY_DOWN, ord("j")):
                selected = (selected + 1) % len(runs)
            elif key == ord("h"):
                headless = not headless
            elif key == ord("v"):
                video = not video
            elif key in (ord("+"), ord("=")):
                max_iterations = min(max_iterations + 1, 1000)
            elif key in (ord("-"), ord("_")):
                max_iterations = max(1, max_iterations - 1)
            elif key == ord("["):
                video_interval = max(1, video_interval - 1)
            elif key == ord("]"):
                video_interval = min(1000, video_interval + 1)
            elif key == ord(","):
                video_length = max(10, video_length - 10)
            elif key == ord("."):
                video_length = min(2000, video_length + 10)
            elif key == ord("n"):
                num_envs = min(128, num_envs + 1)
            elif key == ord("m"):
                num_envs = max(1, num_envs - 1)
            elif key == ord("s"):
                system_deps_idx = (system_deps_idx + 1) % len(system_deps_choices)
            elif key == ord("a"):
                save_after = not save_after
            elif key == ord("p"):
                ckpt = select_checkpoint(stdscr, policy_checkpoints)
                if ckpt:
                    selected_checkpoint = ckpt
                    if "dropbear_play" in runs:
                        selected = runs.index("dropbear_play")
            elif key == ord("r"):
                remote_settings["enabled"] = not bool(remote_settings.get("enabled"))
                remote_client.save_remote_config(remote_settings)
                _refresh_remote_settings()
                _stop_all_remote_services()
                _ensure_nkn_mode_active()
                if remote_settings.get("enabled"):
                    _start_remote_mode()
            elif key == ord("R"):
                configure_remote_target(stdscr)
                _refresh_remote_settings()
            elif key == ord("M"):
                _cycle_remote_mode()
            elif key == ord("A"):
                configure_nkn_remote_address(stdscr)
                _refresh_remote_settings()
                _ensure_nkn_mode_active()
                if remote_settings.get("enabled") and remote_settings.get("mode") == "nkn":
                    _stop_all_remote_services()
                    _start_remote_mode()
            elif key == ord("N"):
                configure_nkn_target_local(stdscr)
            elif key == ord("t"):
                training_config_menu(stdscr)
            elif key in (10, 13):
                return build_args()
            elif key in (ord("q"), 27):
                return None

    interactive_args: Optional[list[str]]
    handlers: dict[int, object] = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, _handle_interrupt)
    try:
        interactive_args = curses.wrapper(wrap_menu)
    except curses.error:
        interactive_args = None
    finally:
        remote_client.stop_reverse_listener()
        remote_client.stop_nkn_client()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        training_configs["last_used"] = training_active_name
        save_training_configs(training_configs)
    return interactive_args


def _apply_remote_cli_overrides(args: argparse.Namespace) -> None:
    cfg = remote_client.get_remote_config()
    updated = False
    if getattr(args, "remote_mode", None):
        cfg["mode"] = args.remote_mode
        updated = True
    target = getattr(args, "remote_nkn_target", None)
    if target:
        nkn_cfg = cfg.setdefault("nkn", {})
        nkn_cfg["target"] = target
        updated = True
    if updated:
        remote_client.save_remote_config(cfg)


def add_dropbear_pythonpath(env: Dict[str, str]) -> Dict[str, str]:
    env2 = dict(env)
    existing = env2.get("PYTHONPATH", "")
    path_list = [str(DROPBEAR_EXTENSION_DIR)]
    if existing:
        path_list.append(existing)
    env2["PYTHONPATH"] = os.pathsep.join(path_list)
    return env2


# -----------------------------
# main flow
# -----------------------------
def main() -> int:
    interactive_session = os.environ.pop("DROPBEAR_INTERACTIVE_SESSION", "0") == "1"
    interactive_mode = interactive_session or (len(sys.argv) == 1 and sys.stdout.isatty())
    ap = argparse.ArgumentParser(description="Self-contained Isaac Sim (pip) + Isaac Lab (source) venv bootstrapper.")
    ap.add_argument("--base", type=str, default=str(Path.cwd()), help="Workspace directory (default: current dir).")
    ap.add_argument("--env", type=str, default="env_isaaclab", help="Venv directory name (default: env_isaaclab).")
    ap.add_argument("--repo", type=str, default="IsaacLab", help="IsaacLab checkout directory (default: IsaacLab).")
    ap.add_argument("--repo-url", type=str, default=DEFAULT_ISAACLAB_GIT, help="IsaacLab git URL.")
    ap.add_argument("--branch", type=str, default=None, help="Optional git branch/tag to checkout.")
    ap.add_argument("--isaacsim-version", type=str, default=DEFAULT_ISAACSIM_VERSION, help="isaacsim pip version.")
    ap.add_argument("--torch-version", type=str, default=DEFAULT_TORCH_VERSION, help="torch version.")
    ap.add_argument("--torchvision-version", type=str, default=DEFAULT_TORCHVISION_VERSION, help="torchvision version.")
    ap.add_argument("--torchaudio-version", type=str, default=DEFAULT_TORCHAUDIO_VERSION, help="torchaudio version.")
    ap.add_argument("--system-deps", choices=["auto", "on", "off"], default="auto",
                    help="Install system deps via apt-get (cmake, build-essential, git).")
    ap.add_argument("--force", action="store_true", help="Force re-install steps even if markers exist.")
    ap.add_argument(
        "--run",
        choices=DROPBEAR_RUNS,
        default="create_empty",
        help="What to run after install.",
    )
    ap.add_argument("--headless", action="store_true", help="Add --headless to training run (and some scripts).")
    ap.add_argument("--_inside-venv", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument(
        "--dropbear-task",
        type=str,
        default="Isaac-Velocity-Dropbear-v0",
        help="Task ID used for Dropbear training runs.",
    )
    ap.add_argument(
        "--dropbear-play-task",
        type=str,
        default="Isaac-Velocity-Dropbear-Play-v0",
        help="Task ID used for Dropbear playback runs.",
    )
    ap.add_argument(
        "--dropbear-max-iterations",
        type=int,
        default=1,
        help="Iterations for Dropbear quick training run (default: 1).",
    )
    ap.add_argument(
        "--dropbear-video",
        action="store_true",
        help="Record videos during Dropbear training/play runs.",
    )
    ap.add_argument(
        "--dropbear-play-video",
        action="store_true",
        help="Force the Dropbear play command to record video.",
    )
    ap.add_argument(
        "--dropbear-video-interval",
        type=int,
        default=None,
        help="Video recording interval (training only).",
    )
    ap.add_argument(
        "--dropbear-video-length",
        type=int,
        default=None,
        help="Video recording length (training + play).",
    )
    ap.add_argument(
        "--save-after",
        action="store_true",
        help="Save run metadata after the requested command completes.",
    )
    ap.add_argument(
        "--remote-mode",
        choices=["direct", "reverse", "nkn"],
        default="nkn",
        help="Preferred remote compute mode when remote compute is enabled.",
    )
    ap.add_argument(
        "--remote-nkn-target",
        type=str,
        default="",
        help="NKN target address (remote agent identity) for offloading commands.",
    )

    internal_flags = {"--_inside-venv"}
    external_args_present = any(arg not in internal_flags for arg in sys.argv[1:])
    show_interface = interactive_mode and not external_args_present
    interactive_args: Optional[list[str]] = None
    if show_interface:
        interactive_args = run_curses_interface()
        if interactive_args is None:
            return 0
        args_list = interactive_args
        if not interactive_session:
            os.environ["DROPBEAR_INTERACTIVE_SESSION"] = "1"
            interactive_session = True
    else:
        args_list = sys.argv[1:]
    args, unknown = ap.parse_known_args(args_list)
    _apply_remote_cli_overrides(args)

    base = Path(args.base).expanduser().resolve()
    env_dir = base / args.env
    repo_dir = base / args.repo

    # ---- platform checks (esp. Linux glibc / arch) ----
    sysname = platform.system()
    machine = platform.machine().lower()

    if sysname == "Linux":
        glibc = get_glibc_version_linux()
        if glibc is None:
            raise RuntimeError("Could not detect GLIBC version (are you on musl/Alpine?). "
                               "Isaac Sim pip installs generally require GLIBC >= 2.35.")
        if parse_version_tuple(glibc) < (2, 35, 0):
            raise RuntimeError(f"GLIBC {glibc} detected, but Isaac Sim pip requires GLIBC >= 2.35. "
                               "Use the Isaac Sim binaries installation method on older distros.")

        # Isaac Sim pip wheels are typically manylinux_2_35_x86_64; enforce x86_64 for Linux.
        if machine not in ("x86_64", "amd64"):
            raise RuntimeError(f"Linux architecture '{machine}' detected. Isaac Sim pip install is typically x86_64-only. "
                               "Use the Isaac Sim binaries (or supported platform method) instead.")

    # ---- ensure python 3.11 for venv ----
    py311 = find_python_311()
    if not py311:
        # try to install python3.11 via apt if available and allowed
        if sysname == "Linux" and which("apt-get") and args.system_deps in ("auto", "on"):
            eprint("[i] Python 3.11 not found. Attempting to install via apt-get (may prompt for sudo)...")
            apt_install_if_requested(["python3.11", "python3.11-venv", "python3.11-distutils"], mode="on")
            py311 = find_python_311()

        if not py311:
            raise RuntimeError(
                "Python 3.11 interpreter not found. Install Python 3.11 first, then re-run.\n"
                "Examples:\n"
                "  Ubuntu/Debian: sudo apt-get install -y python3.11 python3.11-venv\n"
                "  Or use your distro's recommended Python 3.11 install method."
            )

    # ---- create venv, then re-exec into it ----
    if not args._inside_venv:
        create_venv_with_python(py311, env_dir)
        # Re-exec into that venv, preserving user args except internal flag
        passthrough = [x for x in args_list if x != "--_inside-venv"]
        ensure_in_venv(env_dir, passthrough)

    # From here on: running inside venv
    py = Path(sys.executable)
    pip = venv_pip(env_dir)
    if not pip.exists():
        # fallback to python -m pip
        pip = Path(sys.executable)
    print(f"[i] using venv python: {py}")
    print(f"[i] workspace base: {base}")
    print(f"[i] venv dir: {env_dir}")
    print(f"[i] IsaacLab dir: {repo_dir}")

    # ---- optional system deps ----
    # Needed by Isaac Lab optional deps (e.g., robomimic); also ensure git exists for clone.
    if sysname == "Linux":
        want = args.system_deps
        pkgs = ["cmake", "build-essential", "git"]
        apt_install_if_requested(pkgs, mode=want)

    # ---- pip bootstrap ----
    marker_bootstrap = ".isaaclab_bootstrap_ok"
    if args.force or not has_marker(env_dir, marker_bootstrap):
        # upgrade pip tooling
        if pip.name.startswith("python"):
            run_cmd([str(py), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])
        else:
            run_cmd([str(pip), "install", "-U", "pip", "setuptools", "wheel"])
        write_marker(env_dir, marker_bootstrap)

    # ---- install PyTorch (CUDA 12.8 wheels) ----
    marker_torch = ".isaaclab_torch_ok"
    if args.force or not has_marker(env_dir, marker_torch):
        torch_index = None
        if sysname in ("Linux", "Windows") and machine in ("x86_64", "amd64"):
            torch_index = "https://download.pytorch.org/whl/cu128"
        else:
            raise RuntimeError(f"Unsupported platform for scripted torch install: {sysname} / {machine}")

        torch_pkgs = [
            f"torch=={args.torch_version}",
            f"torchvision=={args.torchvision_version}",
            f"torchaudio=={args.torchaudio_version}",
            "--index-url", torch_index,
        ]
        if pip.name.startswith("python"):
            run_cmd([str(py), "-m", "pip", "install", "-U"] + torch_pkgs)
        else:
            run_cmd([str(pip), "install", "-U"] + torch_pkgs)

        # sanity import
        subprocess.check_call([str(py), "-c", "import torch; import torchvision; print(torch.__version__, torchvision.__version__)"])
        write_marker(env_dir, marker_torch)

    # ---- install Isaac Sim via pip ----
    marker_isaacsim = ".isaaclab_isaacsim_ok"
    if args.force or not has_marker(env_dir, marker_isaacsim):
        isaacsim_spec = f"isaacsim[all,extscache]=={args.isaacsim_version}"
        cmd = [str(py), "-m", "pip", "install", isaacsim_spec, "--extra-index-url", DEFAULT_NVIDIA_PYPI]
        run_cmd(cmd)

        # sanity import (no sim launch yet)
        subprocess.check_call([str(py), "-c", "import isaacsim; print('isaacsim import OK')"])
        write_marker(env_dir, marker_isaacsim)

    marker_dropbear = ".dropbear_extension_ok"
    if args.force or not has_marker(env_dir, marker_dropbear):
        if not DROPBEAR_EXTENSION_DIR.exists():
            raise RuntimeError(f"Dropbear extension not found at: {DROPBEAR_EXTENSION_DIR}")
        install_editable_package(py, pip, DROPBEAR_EXTENSION_DIR)
        write_marker(env_dir, marker_dropbear)
    ensure_dropbear_installed(py, pip, DROPBEAR_EXTENSION_DIR)
    ensure_actor_critic_std(env_dir)

    # ---- clone IsaacLab ----
    marker_repo = ".isaaclab_repo_ok"
    if args.force or not has_marker(env_dir, marker_repo):
        if not repo_dir.exists():
            git_clone_or_download(args.repo_url, repo_dir, branch=args.branch)
        else:
            print(f"[i] repo dir exists: {repo_dir}")
        write_marker(env_dir, marker_repo)

    # ---- run ./isaaclab.sh --install ----
    marker_install = ".isaaclab_install_ok"
    if args.force or not has_marker(env_dir, marker_install):
        if sysname == "Windows":
            bat = repo_dir / "isaaclab.bat"
            if not bat.exists():
                raise RuntimeError(f"Expected {bat} not found.")
            # Ensure venv python first in PATH
            env = prepend_path(dict(os.environ), venv_bin_dir(env_dir))
            run_cmd(["cmd", "/c", str(bat), "--install"], cwd=repo_dir, env=env)
        else:
            sh = repo_dir / "isaaclab.sh"
            if not sh.exists():
                raise RuntimeError(f"Expected {sh} not found.")
            # make executable just in case
            run_cmd(["chmod", "+x", str(sh)], cwd=repo_dir, check=False)
            env = prepend_path(dict(os.environ), venv_bin_dir(env_dir))
            run_cmd(["bash", str(sh), "--install"], cwd=repo_dir, env=env)
        write_marker(env_dir, marker_install)
    ensure_actor_critic_std(env_dir)

    # ---- run something in Isaac Lab / Dropbear scripts ----
    run_env = prepend_path(dict(os.environ), venv_bin_dir(env_dir))
    run_env = add_dropbear_pythonpath(run_env)

    if args.run != "none":

        if sysname == "Windows":
            runner = repo_dir / "isaaclab.bat"
            if not runner.exists():
                raise RuntimeError(f"Expected {runner} not found.")
            base_cmd = ["cmd", "/c", str(runner), "-p"]
        else:
            runner = repo_dir / "isaaclab.sh"
            base_cmd = ["bash", str(runner), "-p"]

        if args.run == "create_empty":
            script_path = "scripts/tutorials/00_sim/create_empty.py"
            cmd = base_cmd + [script_path] + unknown
            # if headless requested and user didn't already pass it, append
            if args.headless and "--headless" not in cmd:
                cmd.append("--headless")
            run_cmd(cmd, cwd=repo_dir, env=run_env)

        elif args.run == "train_ant":
            script_path = "scripts/reinforcement_learning/rsl_rl/train.py"
            cmd = base_cmd + [script_path, "--task=Isaac-Ant-v0"] + unknown
            if args.headless and "--headless" not in cmd:
                cmd.append("--headless")
            run_cmd(cmd, cwd=repo_dir, env=run_env)

        elif args.run == "train_anymal":
            script_path = "scripts/reinforcement_learning/rsl_rl/train.py"
            cmd = base_cmd + [script_path, "--task=Isaac-Velocity-Rough-Anymal-C-v0"] + unknown
            if args.headless and "--headless" not in cmd:
                cmd.append("--headless")
            run_cmd(cmd, cwd=repo_dir, env=run_env)

        elif args.run == "dropbear_quick_test":
            script_path = str(PROJECT_ROOT / "scripts" / "quick_test.py")
            cmd = base_cmd + [script_path] + unknown
            run_cmd(cmd, cwd=repo_dir, env=run_env)

        elif args.run == "dropbear_test_commands":
            script = PROJECT_ROOT / "scripts" / "test_commands.py"
            run_local_python_script(py, script, PROJECT_ROOT, run_env, unknown)

        elif args.run == "dropbear_train":
            script_path = str(PROJECT_ROOT / "scripts" / "rsl_rl" / "train.py")
            train_args: List[str] = [
                f"--task={args.dropbear_task}",
                f"--max_iterations={args.dropbear_max_iterations}",
            ]
            if args.dropbear_video:
                train_args.append("--video")
            if args.dropbear_video_interval is not None:
                train_args.append(f"--video_interval={args.dropbear_video_interval}")
            if args.dropbear_video_length is not None:
                train_args.append(f"--video_length={args.dropbear_video_length}")
            if args.headless and "--headless" not in unknown:
                train_args.append("--headless")
            train_args += unknown
            if remote_client.is_remote_enabled() and "--headless" not in train_args:
                train_args.append("--headless")
            cmd = base_cmd + [script_path] + train_args
            if remote_client.is_remote_enabled():
                cfg = remote_client.get_remote_config()
                if cfg.get("mode") == "nkn":
                    target = remote_client.get_nkn_target()
                    remote_addr = remote_client.get_nkn_remote_address()
                    controller_addr = remote_client.get_nkn_app_address()
                    print(f"[i] NKN target: {target or 'unset'}")
                    if controller_addr:
                        print(f"[i] Controller NKN address: {controller_addr} (provide this to the remote agent)")
                    else:
                        print("[i] Controller NKN address pending sidecar readiness.")
                    if remote_addr:
                        print(f"[i] Remote agent reported address: {remote_addr}")
                    else:
                        print("[i] Remote agent address pending handshake.")
                remote_client.dispatch_remote(cmd, description="dropbear_train")
            else:
                run_cmd(cmd, cwd=repo_dir, env=run_env)

        elif args.run == "dropbear_play":
            script_path = str(PROJECT_ROOT / "scripts" / "rsl_rl" / "play.py")
            play_args: List[str] = [f"--task={args.dropbear_play_task}"]
            if args.dropbear_video or args.dropbear_play_video:
                play_args.append("--video")
            if args.dropbear_video_length is not None:
                play_args.append(f"--video_length={args.dropbear_video_length}")
            if args.headless and "--headless" not in unknown:
                play_args.append("--headless")
            play_args += unknown
            cmd = base_cmd + [script_path] + play_args
            run_cmd(cmd, cwd=repo_dir, env=run_env)

    if args.save_after:
        save_run_config(args, unknown)

    print("\n[i] Done.")
    print("[i] To use the venv later:")
    if platform.system() == "Windows":
        print(f"    {env_dir}\\Scripts\\activate")
    else:
        print(f"    source {env_dir}/bin/activate")
    print(f"[i] IsaacLab checkout: {repo_dir}")
    if interactive_session:
        print("\n[i] Restarting interactive menu...")
        os.environ["DROPBEAR_INTERACTIVE_SESSION"] = "1"
        os.execv(str(sys.executable), [str(sys.executable), str(Path(__file__).resolve()), "--_inside-venv"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as ex:
        eprint(f"\n[!] Command failed with exit code {ex.returncode}")
        raise
    except Exception as ex:
        eprint(f"\n[!] {ex}")
        raise
