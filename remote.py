"""Remote IsaacLab runner that listens for commands from the local controller."""

import argparse
import json
import socketserver
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List

import remote_protocol

PROJECT_ROOT = Path(__file__).resolve().parent


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
        self._send({"type": "start", "description": description})
        resolved_cmd = self._resolve_command(cmd)
        try:
            process = subprocess.Popen(
                resolved_cmd,
                cwd=PROJECT_ROOT,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote Dropbear RL server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host/interface to listen on.")
    parser.add_argument("--port", type=int, default=8721, help="Port to listen on.")
    args = parser.parse_args()
    server = RemoteServer((args.host, args.port), RemoteRequestHandler)
    print(f"[remote] Listening on {args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
