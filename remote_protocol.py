"""Shared protocol helpers for the Dropbear remote executor."""

import json
from typing import IO, Any, Dict, Iterator


def encode_message(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"


def iter_messages(stream: IO[bytes]) -> Iterator[Dict[str, Any]]:
    for raw in stream:
        if not raw:
            continue
        try:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            yield json.loads(line)
        except json.JSONDecodeError:
            continue
