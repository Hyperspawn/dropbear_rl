"""Logging helpers shared between local and remote runners."""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def build_log_paths(experiment_name: str, run_name: Optional[str] = None) -> Tuple[Path, Path]:
    """Create consistent log root/run directories for RSL-RL."""
    log_root = Path("logs") / "rsl_rl" / experiment_name
    log_root = log_root.resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if run_name:
        run_suffix = run_name.strip()
        if run_suffix:
            timestamp = f"{timestamp}_{run_suffix}"
    log_dir = log_root / timestamp
    return log_root, log_dir


def ensure_log_directory(log_root: Path, log_dir: Path) -> None:
    """Ensure log directories exist."""
    log_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)


def dump_json_file(path: Path, data: Dict[str, Any]) -> None:
    """Dump a dictionary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def dump_pickle_file(path: Path, data: Any) -> None:
    """Persist arbitrary data via pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(data, handle)
